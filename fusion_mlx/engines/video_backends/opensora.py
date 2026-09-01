# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 video backend (MMDiT / Flux-based).

from __future__ import annotations

import asyncio
import gc
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_executor
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)


class OpenSoraBackend(VideoBackend):
    # Open-Sora V2 11B: Flux-style MMDiT with HunyuanVideo VAE.

    name = "opensora"
    supports_i2v = True

    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self._model_dir: str | None = None
        self._text_encoder = None
        self._vae = None
        self._config = None
        # #739 Surface A: staged VAE encoder (separate from the generation
        # vae kwarg, which is lazily None on the default path).
        # load_vae_encoder populates this.
        self._stage_vae_encoder = None
        self._stage_flags = {}

    @classmethod
    def detect(cls, model_name: str) -> bool:
        lower = model_name.lower()
        return any(k in lower for k in ("opensora", "open-sora", "open_sora"))

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=201,
            dim_divisibility=16,
            num_frames_validator=lambda n: (n - 1) % 4 == 0,
            num_frames_hint="(n-1) must be divisible by 4",
        )

    async def start(self, model_dir: str, **kwargs: Any) -> None:
        self._model_dir = model_dir
        from fusion_mlx.video.opensora.config import OpenSoraConfig

        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            import json

            with open(config_path) as f:
                cfg_dict = json.load(f)
            self._config = OpenSoraConfig.from_dict(cfg_dict)
        else:
            self._config = OpenSoraConfig()
        logger.info(f"OpenSoraBackend started: {model_dir}")

    async def stop(self) -> None:
        self._model_dir = None
        self._text_encoder = None
        self._vae = None
        self._config = None

    async def generate(self, params: VideoGenParams) -> list[bytes] | list[Any]:
        from fusion_mlx.video.opensora.generate import generate_video

        if self._model_dir is None:
            raise RuntimeError("OpenSoraBackend not started — call start() first")

        # #739 Surface B: ControlNet not fabricatable (shared MMDiT adapter,
        # no per-backend control model). Fail visible — refuse silent T2V.
        if params.controlnet_image is not None:
            raise RuntimeError(
                "opensora: ControlNet (Surface B) not available for this backend — "
                "no per-backend ControlNet model (see issue #739 follow-up). "
                "Refusing to silently degrade to T2V (#739)."
            )
        # #739 Surface C: inpaint re-composite handled in the denoise loop
        # (packed-latent-space, DiT-agnostic). init_latent/mask threaded.
        inpaint_mask = params.inpaint_mask
        init_latent = params.init_latent
        raw_output = params.output_format == "raw"
        logger.info(
            "opensora generate: inpaint=%s controlnet=%s",
            inpaint_mask is not None,
            params.controlnet_image is not None,
        )

        def _generate() -> list[bytes] | list[Any]:
            video = generate_video(
                model_dir=self._model_dir,
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                image=params.image,
                num_frames=params.num_frames or 51,
                height=params.height or 480,
                width=params.width or 854,
                num_steps=params.num_inference_steps or 30,
                guidance=params.guide_scale or 4.0,
                seed=params.seed,
                text_encoder=self._text_encoder,
                vae=self._vae,
                controlnet_image=params.controlnet_image,
                inpaint_mask=inpaint_mask,
                init_latent=init_latent,
            )
            if raw_output:
                return [video]
            # MP4 path: convert to uint8 frames and encode
            frames_np = (video * 255).clip(0, 255).astype(np.uint8)
            with managed_tempfile_path(
                prefix="fusion_opensora_", suffix=".mp4"
            ) as handle:
                _save_mp4(frames_np, handle.path, fps=params.fps or 24)
                mp4_bytes = Path(handle.path).read_bytes()
            return [mp4_bytes]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_executor("video"), _generate)

    async def load_vae_encoder(self) -> None:
        if self._stage_vae_encoder is not None:
            return
        import os

        from fusion_mlx.video.hunyuanvideo.vae import HunyuanVideoVAE

        model_path = self._model_dir
        vae_path = (
            os.path.join(model_path, "vae")
            if os.path.isdir(os.path.join(model_path, "vae"))
            else model_path
        )

        def _load():
            vae = HunyuanVideoVAE.from_pretrained(vae_path)
            mx.eval(vae.parameters())
            return vae

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags["vae_encoder"] = True
        logger.info(
            "opensora: vae_encoder load vae=%s",
            type(self._stage_vae_encoder).__name__,
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
            lat = vae_enc.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("opensora: vae encode out_shape=%s", tuple(result.shape))
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
        logger.info("opensora: vae_encoder unload")

    async def encode_control(self, **kwargs: Any) -> Any:
        if kwargs.get("controlnet_image") is not None:
            raise RuntimeError(
                "opensora: ControlNet (Surface B) not available for this backend — "
                "no per-backend ControlNet model (see issue #739 follow-up). "
                "Refusing to silently degrade to T2V (#739)."
            )
        logger.info("opensora: encode_control pure-T2V (no controlnet)")
        return None


def _save_mp4(frames: np.ndarray, output_path: str, fps: int = 16):  # noqa: F821
    try:
        import imageio

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except ImportError:
        from pathlib import Path as _P

        from PIL import Image

        out_dir = _P(output_path).parent / _P(output_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
        logger.warning(
            "No video encoder available, saved %d frames to %s/",
            len(frames),
            out_dir,
        )
