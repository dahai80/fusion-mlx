# SPDX-License-Identifier: Apache-2.0
# Video generation API routes for fusion-mlx.
# POST /v1/videos/generate - Text-to-video and image-to-video generation.
# Backend-aware: constraint validation is delegated to the resolved video
# backend (LTX-2, Wan2, ...), so each backend enforces its own frame/dim/I2V
# rules instead of a hardcoded LTX-2 validator.
import base64
import logging
import mimetypes
import os
import tempfile
import time
import urllib.request
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..engines import VideoGenEngine
from ..middleware.auth import verify_api_key, check_rate_limit
from ..engines.video_backends import constraints_for, validate_params
from ..exceptions import ModelNotFoundError
from ..pool import EnginePool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/videos", tags=["videos"])

_pool: EnginePool | None = None


def set_videos_context(pool: EnginePool) -> None:
    global _pool
    _pool = pool


class VideoGenerateRequest(BaseModel):
    prompt: str
    # Number of videos to generate (default 1, max 4)
    n: int = Field(default=1, ge=1, le=4)
    # Number of frames per video (backend-specific: LTX-2 needs 1+8k,
    # Wan2 needs 1+4k; validated against the resolved backend)
    num_frames: int = Field(default=97, ge=1, le=1024)
    # Frame dimensions (backend-specific divisibility; validated per backend)
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    # Frames per second (ignored by backends that control container fps, e.g. Wan2)
    fps: int = Field(default=24, ge=1, le=60)
    # Random seed (None = random)
    seed: int | None = None
    # Response format
    response_format: str = Field(default="url", pattern="^(url|b64_json)$")
    # Model name (default: first available video gen model)
    model: str | None = None
    # Image-to-video input: local path, data URI, or http(s) URL. Only backends
    # with supports_i2v (e.g. Wan2) accept this; others return 422.
    image: str | None = None
    # Optional backend-specific generation knobs (forwarded only when set)
    negative_prompt: str | None = None
    num_inference_steps: int | None = Field(default=None, ge=1, le=200)
    scheduler: str | None = None
    cfg_scale: float | None = None
    guide_scale: float | None = None
    shift: float | None = None
    # Diffusion acceleration knobs (forwarded only when set):
    # tiling: VAE/memory tiling strategy ('auto'/'none'/etc).
    # no_compile: True=mx.compile OFF (safe/verified default); False=compiled
    #   faster path (opt-in, unverified for some backends).
    # enhance_prompt: run backend prompt-enhancer (extra latency, quality boost).
    tiling: str | None = None
    no_compile: bool | None = None
    enhance_prompt: bool | None = None
    # Session ID for multi-shot latent reuse (Phase-2 UMA Radix Latent cache).
    # Same session_id across sequential requests enables tail→first-frame reuse.
    session_id: str | None = None
    # IP-Adapter: subject-driven image-to-video generation.
    # ip_adapter_image: reference image path/URL/data-URI for IP-Adapter.
    # ip_adapter_scale: strength of image conditioning (0.0-2.0, default 1.0).
    ip_adapter_image: str | None = None
    ip_adapter_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    # ControlNet: structural guidance via control image (Canny/depth/pose).
    # controlnet_image: path/URL/data-URI for the control image.
    # controlnet_strength: strength of structural conditioning (0.0-2.0, default 1.0).
    # control_type: type of control image preprocessing ("canny", "depth", "pose", "raw").
    controlnet_image: str | None = None
    controlnet_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    control_type: str = Field(default="canny", pattern="^(canny|depth|pose|raw)$")
    # AnimateDiff: temporal motion module strength (0=off, >0=on, 1.0=normal).
    animatediff_scale: float = Field(default=0.0, ge=0.0, le=2.0)


class VideoOutput(BaseModel):
    url: str | None = None
    b64_json: str | None = None


class VideoGenerateResponse(BaseModel):
    data: list[VideoOutput]
    created: int = Field(default_factory=lambda: int(time.time()))


def _encode_video_output(vid_bytes: bytes, response_format: str) -> VideoOutput:
    b64 = base64.b64encode(vid_bytes).decode()
    if response_format == "b64_json":
        return VideoOutput(b64_json=b64)
    return VideoOutput(url=f"data:video/mp4;base64,{b64}")


def _resolve_image_to_path(image: str) -> tuple[str, bool]:
    # Normalize an image input to a local filesystem path mlx-video can load.
    # Returns (path, is_temp). Caller unlinks temp paths after generation.
    if image.startswith("data:"):
        header, _, payload = image.partition(",")
        is_base64 = "base64" in header.lower()
        mime = header.split(";")[0]
        mime = mime.split(":", 1)[1] if ":" in mime else "image/png"
        ext = mimetypes.guess_extension(mime) or ".png"
        data = base64.b64decode(payload) if is_base64 else payload.encode()
        fd, path = tempfile.mkstemp(prefix="fusion_i2v_", suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path, True
    if image.startswith(("http://", "https://")):
        from ._url_safety import is_safe_url_with_dns
        if not is_safe_url_with_dns(image):
            raise HTTPException(400, "Image URL targets a private/internal address")
        ext = os.path.splitext(urlparse(image).path)[1] or ".png"
        fd, path = tempfile.mkstemp(prefix="fusion_i2v_", suffix=ext)
        os.close(fd)
        urllib.request.urlretrieve(image, path)
        return path, True
    return image, False


@router.post("/generate")
async def generate_video(
    request: VideoGenerateRequest,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
) -> VideoGenerateResponse:
    # Generate videos from a text prompt (and optional image for I2V).
    try:
        if _pool is None:
            raise HTTPException(503, "Engine pool not initialized")

        model_name = request.model or "ltx-2"

        # Backend-aware constraint validation (422 on violation).
        try:
            validate_params(
                constraints_for(model_name),
                num_frames=request.num_frames,
                width=request.width,
                height=request.height,
                n=request.n,
                image=request.image,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc))

        # Resolve I2V image to a local path (400 on resolve failure).
        image_path: str | None = None
        image_is_temp = False
        ip_path: str | None = None
        ip_is_temp = False
        cn_path: str | None = None
        cn_is_temp = False
        if request.image:
            try:
                image_path, image_is_temp = _resolve_image_to_path(request.image)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, f"failed to resolve image input: {exc}")

        try:
            try:
                engine = await _pool.get_engine(model_name)
            except ModelNotFoundError:
                engine = None
            if engine is None or not isinstance(engine, VideoGenEngine):
                raise HTTPException(
                    404,
                    f"Video generation model '{model_name}' not loaded. "
                    "Load a video model first.",
                )

            gen_kwargs: dict = {
                "prompt": request.prompt,
                "num_frames": request.num_frames,
                "width": request.width,
                "height": request.height,
                "fps": request.fps,
                "seed": request.seed,
                "n": request.n,
            }
            if image_path is not None:
                gen_kwargs["image"] = image_path
            if request.negative_prompt is not None:
                gen_kwargs["negative_prompt"] = request.negative_prompt
            if request.num_inference_steps is not None:
                gen_kwargs["num_inference_steps"] = request.num_inference_steps
            if request.scheduler is not None:
                gen_kwargs["scheduler"] = request.scheduler
            if request.cfg_scale is not None:
                gen_kwargs["cfg_scale"] = request.cfg_scale
            if request.guide_scale is not None:
                gen_kwargs["guide_scale"] = request.guide_scale
            if request.shift is not None:
                gen_kwargs["shift"] = request.shift
            if request.tiling is not None:
                gen_kwargs["tiling"] = request.tiling
            if request.no_compile is not None:
                gen_kwargs["no_compile"] = request.no_compile
            if request.enhance_prompt is not None:
                gen_kwargs["enhance_prompt"] = request.enhance_prompt
            if request.session_id is not None:
                gen_kwargs["session_id"] = request.session_id
            if request.ip_adapter_image is not None:
                ip_path, ip_is_temp = _resolve_image_to_path(request.ip_adapter_image)
                gen_kwargs["ip_adapter_image"] = ip_path
            if request.ip_adapter_scale != 1.0:
                gen_kwargs["ip_adapter_scale"] = request.ip_adapter_scale
            if request.controlnet_image is not None:
                cn_path, cn_is_temp = _resolve_image_to_path(request.controlnet_image)
                gen_kwargs["controlnet_image"] = cn_path
            if request.controlnet_strength != 1.0:
                gen_kwargs["controlnet_strength"] = request.controlnet_strength
            if request.control_type != "canny":
                gen_kwargs["control_type"] = request.control_type
            if request.animatediff_scale > 0:
                gen_kwargs["animatediff_scale"] = request.animatediff_scale

            video_bytes_list = await engine.generate(**gen_kwargs)
            outputs = [
                _encode_video_output(vb, request.response_format)
                for vb in video_bytes_list
            ]
            return VideoGenerateResponse(data=outputs)
        finally:
            if image_is_temp and image_path:
                try:
                    os.unlink(image_path)
                except OSError:
                    logger.warning("failed to unlink temp image: %s", image_path)
            if ip_is_temp and ip_path:
                try:
                    os.unlink(ip_path)
                except OSError:
                    logger.warning(
                        "failed to unlink temp ip_adapter image: %s", ip_path
                    )
            if cn_is_temp and cn_path:
                try:
                    os.unlink(cn_path)
                except OSError:
                    logger.warning(
                        "failed to unlink temp controlnet image: %s", cn_path
                    )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video generation failed")
        raise HTTPException(500, "Internal server error")


@router.get("/denoise-stats")
async def video_denoise_stats(model: str | None = None) -> dict:
    # issue #177 Phase-3: expose the last speculative-denoise run's acceptance
    # stats for Fusion-ComfyUI / clients. Additive, default-off: returns
    # available=False (zeroed counters) when spec is off or no run happened -
    # feature surface for when a real distilled draft arrives. No speedup is
    # claimed today (layer-pruned draft was falsified on real 14B, see #177).
    try:
        if _pool is None:
            raise HTTPException(503, "Engine pool not initialized")

        model_name = model or "ltx-2"
        try:
            engine = await _pool.get_engine(model_name)
        except ModelNotFoundError:
            engine = None
        if engine is None or not isinstance(engine, VideoGenEngine):
            raise HTTPException(
                404,
                f"Video generation model '{model_name}' not loaded. "
                "Load a video model first.",
            )

        stats = engine.last_denoise_stats()
        logger.info(
            "denoise-stats: model=%s enabled=%s available=%s",
            model_name,
            stats.get("enabled"),
            stats.get("available"),
        )
        return {"model": model_name, "stats": stats}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("denoise-stats failed")
        raise HTTPException(500, "Internal server error")
