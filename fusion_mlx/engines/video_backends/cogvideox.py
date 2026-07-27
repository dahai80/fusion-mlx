# SPDX-License-Identifier: Apache-2.0
import logging
import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from .base import VideoBackend

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

    def constraints(self) -> "VideoConstraints":
        from .base import VideoConstraints

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
