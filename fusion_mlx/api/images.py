# SPDX-License-Identifier: Apache-2.0
"""Image generation API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/images/generate  - Text-to-image / variant image generation
"""

import base64
import json
import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from ..engines import ImageGenEngine
from ..engines.image_gen import VARIANT_MAP, _fast_sibling_model
from ..middleware.auth import check_rate_limit, verify_api_key
from ..pool import EnginePool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/images", tags=["images"])

_pool: EnginePool | None = None


def set_images_context(pool: EnginePool) -> None:
    """Inject engine pool into this module."""
    global _pool
    _pool = pool


class LoRAAdapter(BaseModel):
    # #1376: one adapter for the image path. scale is the mflux lora scale
    # (1.0 = fully applied, 0.0 = no effect).
    path: str
    scale: float = Field(default=1.0, ge=0.0, le=2.0)


class ImageGenerateRequest(BaseModel):
    """Request for image generation."""

    prompt: str
    # Number of images to generate (default 1, max 4)
    n: int = Field(default=1, ge=1, le=4)
    # Image dimensions (default 1024x1024)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    # OpenAI-compatible size string ("WxH", e.g. "512x512"). Mapped to
    # width/height by _parse_size before validation. #0916: without this,
    # OpenAI clients sending {"size":"512x512"} had it silently dropped
    # (extra="ignore") and got 1024x1024 — inflating memory admission and
    # causing spurious InsufficientMemoryError on constrained setups.
    size: str | None = None
    # Diffusion steps (None = variant-aware default; #823: 4 was too few for
    # full-diffusion DiTs like Qwen-Image-2512 which need ~30). 1..50.
    steps: int | None = Field(default=None, ge=1, le=50)
    # Random seed (None = random)
    seed: int | None = None
    # Guidance scale (None = variant default; txt2img=1.0, flux1 variants=4.0).
    # #1407: floor is 0.0, not 1.0 — schnell/turbo LoRA distillation trains at
    # CFG 0 and a hard floor of 1 silently forced those callers away from the
    # setting their weights expect.
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    # Response format
    response_format: str = Field(default="url", pattern="^(url|b64_json)$")
    # Model name (default: first available image gen model)
    model: str | None = None
    # Pipeline variant: txt2img|controlnet_canny|controlnet_upscaler|depth|fill|kontext|redux
    variant: str | None = Field(default=None)
    # Optional diffusion knobs
    scheduler: str | None = None
    negative_prompt: str | None = None
    # #846: stop denoise after this fraction of steps (0-1). Only applies to
    # staged denoise() path (Qwen-Image multi-stage variants); ignored on
    # single-call generate_image variants with a warning.
    denoising_end: float | None = Field(default=None, ge=0.01, le=1.0)
    # ControlNet: input image path for canny/upscaler
    control_image: str | None = None
    controlnet_strength: float | None = Field(default=None, ge=0.0, le=2.0)
    # Redux: reference image paths + strengths
    reference_images: list[str] | None = None
    reference_strengths: list[float] | None = None
    # Fill / Kontext: edit image + mask
    edit_image: str | None = None
    mask_image: str | None = None
    # Depth: depth map image
    depth_image: str | None = None
    # Img2img strength (used by depth/kontext/redux/txt2img i2i)
    image_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    # #1376: per-request LoRA adapters for the FLUX/FLUX.2 mflux variants.
    # Paths must live under FUSION_LORA_ALLOWED_DIRS (same rule as chat LoRA
    # adapters); applied when the worker loads weights, so they need the
    # default subprocess mode.
    lora: list[LoRAAdapter] | None = None
    # Transparent RGBA PNG output (mlx-serve parity). Only Qwen-Image-2.1
    # has a 4-channel VAE; other backends return HTTP 400 when true. The
    # VAE's native 4th channel is preserved in the returned PNG — this does
    # NOT remove a background; use the RGBA prompt convention (see
    # https://github.com/QwenLM/Qwen-Image-2.1#transparent-image-generation-rgba).
    transparent: bool = Field(default=False)
    # #1202/#1251: quality tier. "auto" (default) always uses the requested
    # model. "fast" is an explicit opt-in that routes plain text-to-image
    # requests on a full-diffusion model (FLUX.1-dev / FLUX.2-dev) to the
    # distilled sibling (FLUX.1-schnell 4 steps / FLUX.2-klein). "high"
    # keeps the requested model as-is. Variant or image-input requests
    # always use the requested model (distilled siblings ship no adapters).
    quality: str = Field(default="auto", pattern="^(auto|fast|high)$")

    @model_validator(mode="before")
    @classmethod
    def _parse_size(cls, values):
        # Map OpenAI-style {"size":"WxH"} to width/height before validation.
        if isinstance(values, dict) and values.get("size"):
            parts = str(values["size"]).lower().split("x")
            if len(parts) == 2:
                try:
                    w, h = int(parts[0]), int(parts[1])
                    if "width" not in values:
                        values["width"] = w
                    if "height" not in values:
                        values["height"] = h
                except ValueError:
                    pass
        return values


class ImageOutput(BaseModel):
    """Single generated image output."""

    url: str | None = None
    b64_json: str | None = None


class ImageGenerateResponse(BaseModel):
    """Response from image generation."""

    data: list[ImageOutput]
    created: int = Field(default_factory=lambda: int(__import__("time").time()))
    # #1251: model that actually produced the image. Distinct from the
    # requested id when quality="fast" routed to the distilled sibling.
    model: str | None = None
    requested_model: str | None = None


def _validate_image_loras(items: list[LoRAAdapter]) -> list[dict]:
    # #1376: same containment rule as chat LoRA adapters — a request must not
    # be able to make the worker read an arbitrary safetensors path.
    from ..pool.engine_pool import EnginePool

    allowed = EnginePool._resolve_allowed_adapter_dirs()
    if not allowed:
        raise HTTPException(
            400,
            "per-request image lora adapters are disabled; set "
            "FUSION_LORA_ALLOWED_DIRS to the directory holding the adapters",
        )
    resolved = []
    for item in items:
        real = os.path.realpath(os.path.expanduser(item.path))
        if not any(EnginePool._is_within(real, base) for base in allowed):
            raise HTTPException(
                400, f"lora path is outside FUSION_LORA_ALLOWED_DIRS: {item.path}"
            )
        if not real.endswith(".safetensors") or not os.path.isfile(real):
            raise HTTPException(
                400, f"lora file not found or not safetensors: {item.path}"
            )
        resolved.append({"path": real, "scale": item.scale})
    logger.info("image lora adapters: %s", [(r["path"], r["scale"]) for r in resolved])
    return resolved


@router.post(
    "/generate", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
@router.post(
    "/generations",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def generate_image(request: ImageGenerateRequest) -> ImageGenerateResponse:
    """Generate images from a text prompt using Flux variants.

    Mounted on both /v1/images/generate (legacy) and /v1/images/generations
    (OpenAI-compatible path — FC-2 #0907 audit).
    """
    try:
        if _pool is None:
            raise HTTPException(450, "Engine pool not initialized")

        # Validate variant if provided
        variant = request.variant
        if variant is not None and variant not in VARIANT_MAP:
            raise HTTPException(
                422,
                f"Unknown variant '{variant}'. Available: {list(VARIANT_MAP.keys())}",
            )

        # Find an image gen engine
        from fusion_mlx.exceptions import (
            InsufficientMemoryError,
            ModelNotFoundError,
            ModelTooLargeError,
        )
        from fusion_mlx.server import resolve_model_id

        model_name = request.model
        if not model_name:
            model_name = "flux-2"
        model_name = resolve_model_id(model_name) or model_name

        # #1202: quality-tier routing. quality=fast on a full-diffusion model
        # with a plain text-to-image request resolves the distilled sibling
        # (FLUX.1-dev -> FLUX.1-schnell, FLUX.2-dev -> FLUX.2-klein); missing
        # sibling falls back to the requested model — never fails the request.
        # Variant or image-input requests keep the requested model (distilled
        # siblings ship no adapter variants).
        tier = request.quality
        routed_model = model_name
        substituted = False
        has_image_input = any(
            (
                request.control_image,
                request.reference_images,
                request.edit_image,
                request.depth_image,
            )
        )
        engine: ImageGenEngine | None = None
        if tier == "fast" and not has_image_input and variant in (None, "txt2img"):
            for sibling in _fast_sibling_model(model_name):
                sibling_name = resolve_model_id(sibling) or sibling
                try:
                    sibling_engine = await _pool.get_engine(sibling_name)
                except ModelNotFoundError:
                    logger.info(
                        "quality=fast: distilled sibling '%s' not available",
                        sibling_name,
                    )
                    continue
                if sibling_engine is not None and isinstance(
                    sibling_engine, ImageGenEngine
                ):
                    engine = sibling_engine
                    routed_model = sibling_name
                    substituted = True
                    break
                logger.warning(
                    "quality=fast: sibling '%s' is not an image engine",
                    sibling_name,
                )
        try:
            if engine is None:
                engine = await _pool.get_engine(model_name)
        except ModelNotFoundError as exc:
            avail = (
                ", ".join(exc.available_models) if exc.available_models else "(none)"
            )
            raise HTTPException(
                404,
                f"Image generation model '{model_name}' not found. Available: {avail}. "
                "Load a Flux model first.",
            ) from exc
        if engine is None or not isinstance(engine, ImageGenEngine):
            raise HTTPException(
                404,
                f"Image generation model '{model_name}' not loaded. "
                "Load a Flux model first.",
            )
        if substituted:
            # #1251: substitution is caller-visible (WARNING + response
            # fields) — an explicit dev-model request must never look like a
            # silent downgrade.
            logger.warning(
                "image quality tier: quality=fast routed '%s' -> distilled "
                "'%s' (caller opted in; response echoes actual model)",
                model_name,
                routed_model,
            )
        else:
            logger.debug("image quality tier: %s model '%s'", tier, model_name)
        try:
            from ..server_metrics import get_server_metrics

            get_server_metrics().record_image_quality_tier(tier)
        except Exception:
            logger.debug("image quality tier metric record failed", exc_info=True)

        # If engine was started with a different variant, warn
        if variant is not None and engine.variant != variant:
            logger.warning(
                "Request variant=%s but engine variant=%s; "
                "engine variant is fixed at start time",
                variant,
                engine.variant,
            )

        # Build generate kwargs
        gen_kwargs: dict = dict(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            seed=request.seed,
            guidance=request.guidance,
            n_images=request.n,
        )
        if request.scheduler is not None:
            gen_kwargs["scheduler"] = request.scheduler
        if request.denoising_end is not None:
            gen_kwargs["denoising_end"] = request.denoising_end
        if request.negative_prompt is not None:
            gen_kwargs["negative_prompt"] = request.negative_prompt
        # Variant-specific image inputs
        if request.control_image is not None:
            gen_kwargs["control_image"] = request.control_image
        if request.controlnet_strength is not None:
            gen_kwargs["controlnet_strength"] = request.controlnet_strength
        if request.reference_images is not None:
            gen_kwargs["reference_images"] = request.reference_images
        if request.reference_strengths is not None:
            gen_kwargs["reference_strengths"] = request.reference_strengths
        if request.edit_image is not None:
            gen_kwargs["edit_image"] = request.edit_image
        if request.mask_image is not None:
            gen_kwargs["mask_image"] = request.mask_image
        if request.depth_image is not None:
            gen_kwargs["depth_image"] = request.depth_image
        if request.image_strength is not None:
            gen_kwargs["image_strength"] = request.image_strength
        if request.lora:
            gen_kwargs["lora"] = _validate_image_loras(request.lora)
        if request.transparent:
            # mlx-serve parity: transparent RGBA PNG only on Qwen-Image-2.1
            # (4-channel VAE). Other backends have no alpha channel → 400.
            if engine.variant != "qwen_image_21":
                raise HTTPException(
                    400,
                    "transparent=true requires Qwen-Image-2.1 (4-channel RGBA "
                    "VAE); the loaded image backend has no alpha channel",
                )
            gen_kwargs["transparent"] = True

        image_bytes_list = await engine.generate(**gen_kwargs)

        # Format response
        outputs = []
        for img_bytes in image_bytes_list:
            if request.response_format == "b64_json":
                outputs.append(
                    ImageOutput(b64_json=base64.b64encode(img_bytes).decode())
                )
            else:
                b64 = base64.b64encode(img_bytes).decode()
                outputs.append(ImageOutput(url=f"data:image/png;base64,{b64}"))

        try:
            from ..telemetry import emit
            from ..telemetry.activation_spec import (
                ACTIVATION_FIRST_IMAGE_GENERATION,
                SURFACE_API,
            )

            emit.request(
                endpoint="/v1/images/generate",
                model_alias=routed_model,
                stream=False,
                tool_call_used=False,
                prompt_tokens=0,
                completion_tokens=len(image_bytes_list),
                ttft_ms=0.0,
                tps=0.0,
                status=200,
            )
            emit.activation(
                activation_kind=ACTIVATION_FIRST_IMAGE_GENERATION,
                surface=SURFACE_API,
            )
        except Exception:
            logger.debug(
                "telemetry image-generation activation emit failed",
                exc_info=True,
            )

        return ImageGenerateResponse(
            data=outputs,
            model=routed_model,
            requested_model=model_name,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Image generation validation error: %s", exc)
        raise HTTPException(422, "Invalid request parameters")
    except (ImportError, ModuleNotFoundError) as exc:
        # OP-17 (#0907 audit): the image extra is not installed — a bare
        # 500 "Internal server error" misled operators into thinking the
        # server was broken rather than missing an optional dependency.
        # Surface a 503 with the install hint so the cause is actionable.
        logger.error("Image generation dependency missing: %s", exc)
        raise HTTPException(
            503,
            "Image generation is not available: the optional image extra is "
            "not installed. Install it with: pip install -e '.[image]' "
            "--find-links packaging/_wheels",
        )
    except (InsufficientMemoryError, ModelTooLargeError) as exc:
        # #0916: admission rejection is retryable (free memory / retry), not a
        # generic 500. Map to 507 so clients can back off instead of treating
        # it as a permanent server fault.
        logger.warning("Image generation memory admission failed: %s", exc)
        raise HTTPException(507, str(exc), headers={"Retry-After": "5"}) from exc
    except Exception as exc:
        # #1255: the bare "Internal server error" body gave clients no way to
        # distinguish a retryable worker failure from a permanent fault, and
        # the worker's stderr (OOM, segfault, exit code) never left the
        # server. Echo a machine-readable error_code plus a short cause.
        logger.exception("Image generation failed")
        cause = str(exc) or exc.__class__.__name__
        if isinstance(exc, RuntimeError) and "worker" in cause:
            raise HTTPException(
                500,
                {
                    "error": {
                        "message": f"Image generation failed: {cause[:500]}",
                        "type": "api_error",
                        "code": "image_worker_failed",
                    }
                },
            ) from exc
        raise HTTPException(
            500,
            {
                "error": {
                    "message": "Internal server error",
                    "type": "api_error",
                    "code": "image_generation_failed",
                }
            },
        ) from exc


async def _upload_to_b64(upload: UploadFile | None) -> str | None:
    if upload is None:
        return None
    raw = await upload.read()
    if not raw:
        return None
    return base64.b64encode(raw).decode()


@router.post(
    "/edits",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def edit_image(
    image: list[UploadFile] = File(...),
    prompt: str = Form(...),
    mask: UploadFile | None = File(None),
    model: str = Form("flux-2"),
    n: int = Form(1),
    size: str | None = Form(None),
    response_format: str = Form("b64_json"),
    guidance: float | None = Form(None),
    negative_prompt: str | None = Form(None),
    lora: str | None = Form(None),
) -> ImageGenerateResponse:
    """OpenAI SDK multipart shape for image edits (client.images.edit).

    Accepts one or more ``image[]`` file uploads plus an optional ``mask``
    and a text ``prompt``. Files are base64-encoded and routed through the
    existing edit_image flow. The first image is the edit target / img2img
    base. Extra images become reference inputs only on the ``redux`` variant;
    other variants (flux1_dev/flux2_dev/qwen_image_21/fill/kontext) cannot
    consume them, and the engine logs a warning naming how many were dropped
    (#1374).
    """
    if _pool is None:
        raise HTTPException(450, "Engine pool not initialized")
    if not image:
        raise HTTPException(400, "at least one image file is required")

    images_b64 = []
    for up in image:
        b64 = await _upload_to_b64(up)
        if b64:
            images_b64.append(b64)
    if not images_b64:
        raise HTTPException(400, "image file(s) were empty or unreadable")

    mask_b64 = await _upload_to_b64(mask)

    width, height = 1024, 1024
    if size:
        try:
            w, h = size.lower().split("x")
            width, height = int(w), int(h)
        except Exception:
            raise HTTPException(
                422, f"invalid size '{size}', expect WxH e.g. 1024x1024"
            )

    request = ImageGenerateRequest(
        prompt=prompt,
        model=model,
        n=n,
        width=width,
        height=height,
        guidance=guidance,
        negative_prompt=negative_prompt,
        response_format=response_format,
    )
    request.edit_image = images_b64[0]
    if len(images_b64) > 1:
        request.reference_images = images_b64[1:]
    if mask_b64:
        request.mask_image = mask_b64
    if lora:
        # #1376: multipart cannot carry a JSON body, so the adapter list
        # arrives as a JSON string field: [{"path": "...", "scale": 0.9}].
        try:
            request.lora = [LoRAAdapter(**item) for item in json.loads(lora)]
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                422,
                "invalid lora field, expect JSON like "
                '[{"path": "...", "scale": 0.9}]: ' + str(exc),
            )
    logger.info(
        "images/edit multipart prompt_len=%d images=%d mask=%s model=%s",
        len(prompt),
        len(images_b64),
        bool(mask_b64),
        model,
    )
    return await generate_image(request)
