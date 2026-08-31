# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from ...engine_core import get_executor
from .base import VideoBackend, VideoConstraints

logger = logging.getLogger(__name__)


class CogVideoBackend(VideoBackend):
    name = "cogvideo"
    aliases = ["cogvideox", "cog_video", "cogvideo-x"]

    def __init__(self, model_path: str | Path, **kwargs):
        self.model_path = Path(model_path)
        self._model_dir = None
        self._kwargs = kwargs

    @classmethod
    def detect(cls, model_path: str | Path) -> bool:
        p = Path(model_path)
        name = p.name.lower()
        if any(k in name for k in ("cogvideo", "cog_video")):
            return True
        if (p / "config.json").exists():
            try:
                import json

                with open(p / "config.json") as f:
                    cfg = json.load(f)
                if cfg.get("_class_name", "").lower().startswith("cogvideo"):
                    return True
                arch = cfg.get("architectures", [])
                if any("cogvideo" in a.lower() for a in arch):
                    return True
            except Exception:
                pass
        return False

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=1,
            dim_divisibility=16,
            num_frames_validator=lambda n: 9 <= n <= 161,
            num_frames_hint="9-161 frames (odd number recommended)",
            dim_hint="720x480 default, must be multiple of 16",
        )

    def start(self, executor=None) -> None:
        from fusion_mlx.video.cogvideox.utils import get_model_path

        self._model_dir = get_model_path(self.model_path)
        logger.info(f"CogVideoBackend started: {self._model_dir}")

    def stop(self) -> None:
        self._model_dir = None
        import gc

        gc.collect()
        mx.clear_cache()

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        image: str | None = None,
        width: int = 720,
        height: int = 480,
        num_frames: int = 49,
        steps: int | None = None,
        guide_scale: float | None = None,
        seed: int = -1,
        output_path: str = "output.mp4",
        no_compile: bool = True,
        on_step_sync=None,
        session_id: str | None = None,
        **kwargs,
    ) -> str:
        from fusion_mlx.video.cogvideox.generate import generate_video

        return generate_video(
            model_dir=str(self._model_dir),
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guide_scale=guide_scale,
            seed=seed,
            output_path=output_path,
            no_compile=no_compile,
            on_step_sync=on_step_sync,
            session_id=session_id,
        )

    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        from fusion_mlx.video.cogvideox.utils import load_config as _load_cog_config
        from fusion_mlx.video.cogvideox.utils import load_vae_encoder as _load_vae_enc

        vae_path = Path(self._model_dir) / "vae.safetensors"

        def _load():
            cfg, _quant = _load_cog_config(Path(self._model_dir))
            return _load_vae_enc(vae_path, cfg)

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info(
            "cogvideox: vae_encoder load vae=%s",
            type(self._stage_vae_encoder).__name__,
        )

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae_enc = self._stage_vae_encoder
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
            lat = vae_enc.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("cogvideox: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("cogvideox: vae_encoder unload")
