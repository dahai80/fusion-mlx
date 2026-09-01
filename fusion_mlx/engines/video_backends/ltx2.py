# SPDX-License-Identifier: Apache-2.0
# LTX-2 video backend (pure-MLX port, Phase 4). Generation runs through the
# vendored fusion_mlx.video.ltx2 port (MLX->MLX, replaces mlx-video for LTX-2).
# generate_video loads weights internally on every call, so start() front-loads
# the network download via get_model_path; generation reads weights from the
# local HF cache. Phase 5 ported Wan2 the same way; mlx-video is no longer a
# runtime dependency.

import asyncio
import gc
import logging
import random
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor, get_video_gen_timeout
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE = "distilled"


class LTX2Backend(VideoBackend):
    name = "ltx2"
    supports_i2v = True

    def __init__(
        self,
        model_name: str,
        *,
        text_encoder_repo: str | None = None,
        pipeline: str = _DEFAULT_PIPELINE,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._text_encoder_repo = text_encoder_repo
        self._pipeline = pipeline
        self._loaded = False
        self._model_path = None
        # #738 Surface A: staged VAE encoder (separate from the monolith pipeline,
        # which owns its own VAE internally). load_vae_encoder populates this.
        self._stage_vae_encoder = None
        self._stage_flags = {}

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        # Exclude LTX-2.5 paths (handled by LTX2_5Backend, registered first).
        if "ltx-2.5" in p or "ltx_2.5" in p or "ltx2.5" in p:
            return False
        return "ltx-2" in p or "ltx_2" in p or "ltx-2.3" in p

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting LTX-2 backend (pure-MLX): %s", model_path)

        def _resolve():
            from fusion_mlx.video.ltx2.utils import get_model_path

            return get_model_path(model_path)

        loop = asyncio.get_running_loop()
        resolved = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _resolve), timeout=180.0
        )
        self._model_path = resolved
        self._loaded = True
        logger.info("LTX-2 backend ready: %s", model_path)

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        gc.collect()
        loop = asyncio.get_running_loop()
        # Use the io executor (max_workers=2) rather than the single-worker
        # video executor so a stop() issued while a long video generation
        # (up to FUSION_VIDEO_GEN_TIMEOUT, default 7200s) is running does not
        # queue behind that generation and trip the 5s timeout.
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        if params.on_step is not None:
            logger.debug(
                "ltx2: on_step progress callback accepted but per-step "
                "streaming not yet emitted for this backend (issue #171 follow-up)"
            )
        # #738 Surface B: ControlNet not fabricatable (shared adapter is Wan2-arch,
        # no per-backend model). Fail visible — refuse silent T2V degrade.
        if params.controlnet_image is not None:
            raise RuntimeError(
                "ltx2: ControlNet (Surface B) not available for this backend — "
                "no per-backend ControlNet model (see issue #738 follow-up). "
                "Refusing to silently degrade to T2V (#738)."
            )
        # #738 Surface C: inpaint re-composite handled in the denoise loop
        # (DiT-agnostic latent-space). init_latent/mask threaded verbatim.
        inpaint_mask = params.inpaint_mask
        init_latent = params.init_latent
        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )
        logger.info(
            "ltx2 generate: inpaint=%s controlnet=%s",
            inpaint_mask is not None,
            params.controlnet_image is not None,
        )

        def _generate():
            results: list[bytes] = []
            for i in range(max(1, params.n)):
                mp4_bytes = _generate_one(
                    self._model_name,
                    self._text_encoder_repo,
                    self._pipeline,
                    prompt=params.prompt,
                    num_frames=params.num_frames,
                    width=params.width,
                    height=params.height,
                    fps=params.fps,
                    seed=base_seed + i,
                    num_inference_steps=params.num_inference_steps,
                    cfg_scale=params.cfg_scale,
                    tiling=params.tiling,
                    enhance_prompt=params.enhance_prompt,
                    image=params.image,
                    image_strength=params.image_strength,
                    image_frame_idx=0,
                    session_id=params.session_id,
                    controlnet_image=params.controlnet_image,
                    inpaint_mask=inpaint_mask,
                    init_latent=init_latent,
                )
                results.append(mp4_bytes)
            return results

        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate),
            timeout=get_video_gen_timeout(),
        )

    async def load_vae_encoder(self) -> None:
        if self._stage_vae_encoder is not None:
            return
        from fusion_mlx.video.ltx2.video_vae import VideoEncoder

        if self._model_path is None:
            from fusion_mlx.video.ltx2.utils import get_model_path

            self._model_path = get_model_path(self._model_name)
        enc_dir = Path(self._model_path) / "vae" / "encoder"

        def _load():
            return VideoEncoder.from_pretrained(enc_dir)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags["vae_encoder"] = True
        logger.info(
            "ltx2: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__
        )

    async def encode(self, pixels: mx.array) -> mx.array:
        if self._stage_vae_encoder is None:
            await self.load_vae_encoder()
        vae_enc = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        # #630 numpy-bridge: detach from any GPU stream before the executor hop.
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae_enc(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("ltx2: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_flags.pop("vae_encoder", None)
        self._stage_vae_encoder = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )
        logger.info("ltx2: vae_encoder unload")

    async def encode_control(self, **kwargs: Any) -> Any:
        if kwargs.get("controlnet_image") is not None:
            raise RuntimeError(
                "ltx2: ControlNet (Surface B) not available for this backend — "
                "no per-backend ControlNet model (see issue #738 follow-up). "
                "Refusing to silently degrade to T2V (#738)."
            )
        logger.info("ltx2: encode_control pure-T2V (no controlnet)")
        return None

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=4,
            dim_divisibility=64,
            num_frames_validator=lambda nf: nf % 8 == 1,
            num_frames_hint="num_frames must satisfy num_frames % 8 == 1",
            dim_hint="width and height must be divisible by 64",
        )


def _generate_one(
    model_repo,
    text_encoder_repo,
    pipeline,
    *,
    prompt: str,
    num_frames: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
    num_inference_steps: int | None = None,
    cfg_scale: float | None = None,
    tiling: str | None = None,
    enhance_prompt: bool | None = None,
    image: str | None = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    session_id: str | None = None,
    controlnet_image=None,
    inpaint_mask=None,
    init_latent=None,
) -> bytes:
    from fusion_mlx.video.ltx2.generate import PipelineType, generate_video

    pipe = PipelineType(pipeline) if isinstance(pipeline, str) else pipeline
    # Diffusion wall-time is ~linear in num_inference_steps, so reducing steps
    # is the dominant speed lever (e.g. 40 -> 10 ≈ 4x on the denoise loop).
    # Defaults left to the pure-MLX port when None (distilled pipeline = 40 steps).
    gen_kwargs: dict[str, Any] = dict(
        pipeline=pipe,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        fps=fps,
        output_path=None,
        verbose=False,
    )
    if num_inference_steps is not None:
        gen_kwargs["num_inference_steps"] = num_inference_steps
    if cfg_scale is not None:
        gen_kwargs["cfg_scale"] = cfg_scale
    if tiling is not None:
        gen_kwargs["tiling"] = tiling
    if enhance_prompt is not None:
        gen_kwargs["enhance_prompt"] = enhance_prompt
    if image is not None:
        gen_kwargs["image"] = image
        gen_kwargs["image_strength"] = image_strength
        gen_kwargs["image_frame_idx"] = image_frame_idx
    if session_id is not None:
        gen_kwargs["session_id"] = session_id
    # #738 Surface B/C: thread through to the denoise loop (fail-visible gate
    # + inpaint re-composite live in fusion_mlx.video.ltx2.denoise).
    gen_kwargs["controlnet_image"] = controlnet_image
    gen_kwargs["inpaint_mask"] = inpaint_mask
    gen_kwargs["init_latent"] = init_latent
    with managed_tempfile_path(prefix="fusion_video_", suffix=".mp4") as handle:
        temp_path = handle.path
        gen_kwargs["output_path"] = temp_path
        logger.info(
            "VideoGen generate: prompt_len=%d frames=%d %dx%d@%dfps seed=%d "
            "steps=%s cfg=%s tiling=%s image=%s",
            len(prompt),
            num_frames,
            width,
            height,
            fps,
            seed,
            num_inference_steps,
            cfg_scale,
            tiling,
            image is not None,
        )
        generate_video(model_repo, text_encoder_repo, prompt, **gen_kwargs)
        with open(temp_path, "rb") as f:
            return f.read()
