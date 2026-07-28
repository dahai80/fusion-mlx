# SPDX-License-Identifier: Apache-2.0
# Open-Sora V2 video backend (MMDiT / Flux-based).

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..._tempfile_safe import managed_tempfile_path
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

        raw_output = params.output_format == "raw"

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
        )

        if raw_output:
            return [video]

        # MP4 path: convert to uint8 frames and encode
        import numpy as np

        frames_np = (video * 255).clip(0, 255).astype(np.uint8)
        with managed_tempfile_path(prefix="fusion_opensora_", suffix=".mp4") as handle:
            _save_mp4(frames_np, handle.path, fps=params.fps or 24)
            mp4_bytes = Path(handle.path).read_bytes()
        return [mp4_bytes]


def _save_mp4(frames: "np.ndarray", output_path: str, fps: int = 16):
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
