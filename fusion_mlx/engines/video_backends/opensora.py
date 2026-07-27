# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 video backend (MMDiT / Flux-based).

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)


class OpenSoraBackend(VideoBackend):
    # Open-Sora V2 11B: Flux-style MMDiT with HunyuanVideo VAE.

    name = "opensora"
    supports_i2v = True

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.model_name = model_name
        self._model_dir: str | None = None
        self._text_encoder = None
        self._vae = None
        self._config = None

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

    async def generate(self, params: VideoGenParams) -> list[bytes] | list:
        from fusion_mlx.video.opensora.generate import generate_video

        if self._model_dir is None:
            raise RuntimeError("OpenSoraBackend not started — call start() first")

        raw_output = params.output_format == "raw"

        video = generate_video(
            model_dir=self._model_dir,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or "",
            image=params.image,
            num_frames=params.num_frames or 51,
            height=params.height or 480,
            width=params.width or 854,
            num_steps=params.num_steps or 30,
            guidance=params.guidance or 4.0,
            seed=params.seed,
            text_encoder=self._text_encoder,
            vae=self._vae,
            output_format=params.output_format,
        )
        if raw_output:
            return [video]
        # MP4 path: convert to uint8 frames and encode
        import numpy as np
        import tempfile

        from fusion_mlx.video.wan2.postprocess import save_video

        frames_np = (video * 255).clip(0, 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            save_video(frames_np, tmp.name, fps=params.fps or 24)
            mp4_bytes = Path(tmp.name).read_bytes()
            Path(tmp.name).unlink(missing_ok=True)
        return [mp4_bytes]
