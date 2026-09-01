# SPDX-License-Identifier: Apache-2.0
# SVD (Stable Video Diffusion) video backend. I2V only.
# Pure-MLX port: CLIP vision + temporal UNet + video VAE.

import asyncio
import gc
import logging
import random
from typing import Any

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...api._url_safety import is_safe_local_path
from ...engine_core import get_executor, get_video_gen_timeout
from .._progress import make_sync_step_callback
from .base import VideoBackend, VideoConstraints, VideoGenParams, validate_params

logger = logging.getLogger(__name__)

_DEFAULT_STEPS = 25
_DEFAULT_CFG = 3.0
_MIN_FPS = 6
_VALID_NUM_FRAMES = {14, 25}


class SVDBackend(VideoBackend):
    name = "svd"
    supports_i2v = True

    def __init__(self, model_name: str, *, dtype: Any = mx.float16, **kwargs: Any):
        self._model_name = model_name
        self._dtype = dtype
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        return "svd" in p or "stable-video-diffusion" in p or "img2vid" in p

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting SVD backend (pure-MLX): %s", model_path)
        self._model_path = model_path
        self._loaded = True
        logger.info("SVD backend ready: %s", model_path)

    async def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("io"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )
        logger.info("SVD backend stopped: %s", self._model_name)

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        if params.on_step is not None:
            logger.debug(
                "svd: on_step progress callback accepted but per-step "
                "streaming not yet emitted for this backend (issue #171 follow-up)"
            )
        validate_params(
            self.constraints(),
            num_frames=params.num_frames,
            width=params.width,
            height=params.height,
            n=params.n,
            image=params.image,
        )
        if self._model_name.startswith(("/", "~")) or ".." in self._model_name:
            if not is_safe_local_path(self._model_name):
                raise ValueError(
                    f"model_path outside allowed directories: {self._model_name}"
                )
        base_seed = (
            params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        )

        def _generate():
            results: list[bytes] = []
            for i in range(max(1, params.n)):
                mp4_bytes = _generate_one(
                    self._model_name,
                    self._dtype,
                    prompt=params.prompt,
                    image=params.image,
                    num_frames=params.num_frames,
                    width=params.width,
                    height=params.height,
                    fps=params.fps,
                    seed=base_seed + i,
                    num_inference_steps=params.num_inference_steps,
                    cfg_scale=params.cfg_scale,
                    negative_prompt=params.negative_prompt,
                    on_step_sync=sync_cb,
                    # #737 Surface B+C: thread controlnet_image (fail-visible in
                    # generate_video — no backend model) and inpaint surfaces.
                    controlnet_image=params.controlnet_image,
                    inpaint_mask=params.inpaint_mask,
                    init_latent=params.init_latent,
                )
                results.append(mp4_bytes)
            return results

        loop = asyncio.get_running_loop()
        sync_cb = make_sync_step_callback(params.on_step, loop)
        return await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _generate),
            timeout=get_video_gen_timeout(),
        )

    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from fusion_mlx.video.svd.vae import SVDVideoVAE

        def _load():
            return SVDVideoVAE.from_pretrained(self._model_path, dtype=self._dtype)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info(
            "svd: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__
        )

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("svd: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("svd: vae_encoder unload")

    async def encode_control(self, **kwargs: Any) -> Any:
        # #737 Surface B: ControlNet conditioning is not available for svd —
        # the shared ControlNet adapter is Wan2-arch (text_dim=4096,
        # patch_size=[1,2,2], token-residual block injection) and no
        # per-backend ControlNet model exists for svd. Fail visibly
        # (Rule 12): a caller asking for ControlNet must NOT silently degrade
        # to T2V.
        if kwargs.get("controlnet_image") is not None:
            raise RuntimeError(
                "svd: ControlNet (Surface B) not available for this backend — "
                "no per-backend ControlNet model (see issue #737 follow-up). "
                "Refusing to silently degrade to T2V (#737)."
            )
        # No conditioning surfaces implemented for svd beyond VAE encode
        # (Surface A, already on load_vae_encoder/encode). Pure T2V -> None.
        logger.info("svd: encode_control pure-T2V (no controlnet)")
        return None

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=4,
            dim_divisibility=64,
            num_frames_validator=lambda nf: nf in _VALID_NUM_FRAMES or nf % 8 == 1,
            num_frames_hint="num_frames must be 14, 25, or satisfy num_frames % 8 == 1",
            dim_hint="width and height must be divisible by 64",
        )


def _generate_one(
    model_repo,
    dtype,
    *,
    prompt: str | None = None,
    image: str | None = None,
    num_frames: int = 25,
    width: int = 576,
    height: int = 1024,
    fps: int = 7,
    seed: int = 0,
    num_inference_steps: int | None = None,
    cfg_scale: float | None = None,
    negative_prompt: str | None = None,
    on_step_sync=None,
    controlnet_image: str | None = None,
    inpaint_mask=None,
    init_latent=None,
) -> bytes:
    from fusion_mlx.video.svd.generate import generate_video

    steps = int(num_inference_steps) if num_inference_steps else _DEFAULT_STEPS
    cfg = float(cfg_scale) if cfg_scale is not None else _DEFAULT_CFG

    logger.info(
        "svd _generate_one: image=%s frames=%d %dx%d@%dfps seed=%d steps=%d cfg=%.2f",
        image is not None,
        num_frames,
        width,
        height,
        fps,
        seed,
        steps,
        cfg,
    )

    with managed_tempfile_path(prefix="fusion_svd_", suffix=".mp4") as handle:
        generate_video(
            model_repo,
            prompt=prompt,
            image=image,
            num_frames=num_frames,
            width=width,
            height=height,
            fps=max(_MIN_FPS, fps),
            seed=seed,
            num_inference_steps=steps,
            cfg_scale=cfg,
            negative_prompt=negative_prompt,
            output_path=handle.path,
            dtype=dtype,
            on_step_sync=on_step_sync,
            controlnet_image=controlnet_image,
            inpaint_mask=inpaint_mask,
            init_latent=init_latent,
        )
        with open(handle.path, "rb") as f:
            return f.read()
