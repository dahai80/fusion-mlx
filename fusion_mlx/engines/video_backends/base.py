# SPDX-License-Identifier: Apache-2.0
# Backend abstraction for multi-backend video generation.
# VideoGenEngine delegates to a registered VideoBackend (ltx2, wan2,
# cogvideo, ltx_video_legacy). Each backend owns its generate path,
# mlx-video/direct-MLX signature normalization, and per-backend
# constraints. Backends are plain objects (not engine subclasses) per
# the single-engine + registry architecture.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VideoGenParams:
    prompt: str
    n: int = 1
    num_frames: int = 97
    width: int = 768
    height: int = 512
    fps: int = 24
    seed: int | None = None
    negative_prompt: str | None = None
    image: str | None = None
    image_strength: float = 1.0
    num_inference_steps: int | None = None
    cfg_scale: float | None = None
    guide_scale: float | None = None
    pipeline: str | None = None
    scheduler: str | None = None
    shift: float | None = None
    # Diffusion acceleration knobs. tiling controls VAE/memory tiling ('auto'
    # by default). no_compile toggles mx.compile on backends that support it
    # (wan2); compile is faster but historically unverified, so the safe
    # default is no_compile=True (i.e. compile OFF). enhance_prompt runs the
    # backend's prompt-enhancer LLM (extra latency, quality boost).
    tiling: str | None = None
    no_compile: bool | None = None
    enhance_prompt: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoConstraints:
    supports_i2v: bool = False
    max_n: int = 4
    dim_divisibility: int = 1
    num_frames_validator: Callable[[int], bool] | None = None
    num_frames_hint: str = ""
    dim_hint: str = ""


class VideoBackend(ABC):
    name: str = ""
    supports_i2v: bool = False

    @classmethod
    @abstractmethod
    def detect(cls, model_path: str) -> bool:
        pass

    @abstractmethod
    async def start(self, model_path: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def generate(self, params: VideoGenParams) -> list[bytes]:
        pass

    @abstractmethod
    def constraints(self) -> VideoConstraints:
        pass

    def get_stats(self) -> dict[str, Any]:
        return {"backend": self.name}


def validate_params(
    constraints: VideoConstraints,
    *,
    num_frames: int,
    width: int,
    height: int,
    n: int,
    image: str | None = None,
) -> None:
    # Backend-aware request validation. Raises ValueError with a concrete
    # hint so the API layer can map it to 422 instead of leaking a 500.
    if n < 1 or n > constraints.max_n:
        raise ValueError(f"n must be between 1 and {constraints.max_n}")
    if image is not None and not constraints.supports_i2v:
        raise ValueError("backend does not support image-to-video (I2V)")
    if constraints.dim_divisibility > 1:
        if width % constraints.dim_divisibility != 0:
            raise ValueError(
                f"width must be divisible by {constraints.dim_divisibility}"
            )
        if height % constraints.dim_divisibility != 0:
            raise ValueError(
                f"height must be divisible by {constraints.dim_divisibility}"
            )
    if constraints.num_frames_validator is not None:
        if not constraints.num_frames_validator(num_frames):
            hint = constraints.num_frames_hint or "num_frames constraint violated"
            raise ValueError(hint)
