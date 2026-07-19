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
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..._tempfile_safe import managed_tempfile_path
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
        # Detect branch from model name
        model_lower = self._model_name.lower()
        if "r2v" in model_lower or "r2v_14b" in model_lower:
            return await self._generate_r2v(params)
        elif "v2v" in model_lower or "v2v_14b" in model_lower:
            return await self._generate_v2v(params)
        elif "a2v" in model_lower or "a2v_19b" in model_lower:
            return await self._generate_a2v(params)
        else:
            # Default to R2V for SkyReels models
            logger.warning("Unknown SkyReels branch, defaulting to R2V: %s", self._model_name)
            return await self._generate_r2v(params)

    async def _generate_r2v(self, params: VideoGenParams) -> list[bytes]:
        """R2V: 参考图 + Prompt -> 视频."""
        from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsR2VPipeline

        base_seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        duration = max(1, params.num_frames // 24)  # fps=24 → duration in seconds

        def _gen_one() -> bytes:
            pipeline = SkyReelsR2VPipeline(self._model_name)
            ref_images = [params.image] if params.image else None
            video = pipeline.generate(
                prompt=params.prompt,
                ref_images=ref_images,
                duration=duration,
                seed=base_seed,
            )
            with managed_tempfile_path(prefix="fusion_skyreels_", suffix=".mp4") as handle:
                pipeline.save(video, handle.path)
                with open(handle.path, "rb") as f:
                    return f.read()

        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _gen_one), timeout=600.0
        )
        return [results]

    async def _generate_v2v(self, params: VideoGenParams) -> list[bytes]:
        """V2V: 输入视频 -> 续写视频."""
        from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsV2VPipeline

        base_seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        duration = max(1, params.num_frames // 24)

        def _gen_one() -> bytes:
            pipeline = SkyReelsV2VPipeline(self._model_name)
            video = pipeline.generate(
                prompt=params.prompt,
                input_video=params.image,  # V2V 可接受视频路径作为 image
                duration=duration,
                seed=base_seed,
            )
            with managed_tempfile_path(prefix="fusion_skyreels_", suffix=".mp4") as handle:
                pipeline.save(video, handle.path)
                with open(handle.path, "rb") as f:
                    return f.read()

        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _gen_one), timeout=600.0
        )
        return [results]

    async def _generate_a2v(self, params: VideoGenParams) -> list[bytes]:
        """A2V: 音频 + 参考图 -> 数字人视频."""
        from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsA2VPipeline

        base_seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        duration = max(1, params.num_frames // 24)

        def _gen_one() -> bytes:
            pipeline = SkyReelsA2VPipeline(self._model_name)
            # A2V pipeline 需要 audio 和 ref_image 参数
            video = pipeline.generate(
                prompt=params.prompt,
                audio=params.extra.get("audio", ""),
                ref_image=params.image,
                duration=duration,
                seed=base_seed,
            )
            with managed_tempfile_path(prefix="fusion_skyreels_", suffix=".mp4") as handle:
                pipeline.save(video, handle.path)
                with open(handle.path, "rb") as f:
                    return f.read()

        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(get_executor("video"), _gen_one), timeout=600.0
        )
        return [results]

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