# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 video backend (R2V 14B / V2V 14B / A2V 19B).

SkyReels-V3 is a reference-to-video / video-to-video / audio-to-video model
family.  The backend delegates to the vendored fusion_mlx.video.skyreels_v3
pipeline.  SkyReels supports image-to-video (I2V / R2V), so
``supports_i2v=True``.
"""

import asyncio
import gc
import logging
import random
from typing import Any

import mlx.core as mx

from ...engine_core import get_executor
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)


class SkyReelsBackend(VideoBackend):
    name = "skyreels"
    supports_i2v = True

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        return "skyreels" in model_path.lower()

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting SkyReels backend: %s", model_path)
        self._loaded = True
        logger.info("SkyReels backend ready: %s", model_path)

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

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        # SkyReels-V3 generation — delegates to the SkyReels pipeline.
        # The actual generation is handled by the VideoGenEngine which loads
        # the model via the SkyReels pipeline.  This backend is a thin
        # constraint + routing layer so the API can validate parameters
        # before attempting to load the model.
        raise NotImplementedError(
            "SkyReels-V3 generation is handled by the DirecVideoGenEngine "
            "loading path; the backend is a constraint-only stub."
        )

    def constraints(self) -> VideoConstraints:
        # SkyReels-V3: VAE stride = (4, 16, 16) → spatial dims divisible by 16,
        # temporal dims: (num_frames - 1) % 4 == 0.  Supports I2V (R2V branch).
        return VideoConstraints(
            supports_i2v=True,
            max_n=1,
            dim_divisibility=16,
            num_frames_validator=lambda nf: (nf - 1) % 4 == 0,
            num_frames_hint="num_frames must satisfy (num_frames - 1) % 4 == 0 "
            "(e.g. 9, 41, 81, 121)",
            dim_hint="width and height must be divisible by 16",
        )