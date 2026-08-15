# SPDX-License-Identifier: Apache-2.0
# LTX-2.5 video backend (pure-MLX port). Mirrors LTX2Backend but targets the
# LTX-2.5 22B model (Gemma4-12b text encoder, duration-head, two-stage
# spatial+temporal upsampler). P0-P5 structural port landed (config / text
# encoder / duration-head / temporal upsampler); the full E2E generate path
# (P6 generate.py two-stage + audio mux) is pending real 22B weights, so
# generate() raises a clear NotImplementedError (fail visible, Rule 12).
# Detection runs BEFORE LTX2Backend so an "ltx-2.5" path does not fall through
# the "ltx-2" substring match (AR doc §3.3 / §7 detect-order conflict).
import asyncio
import gc
import logging
from typing import Any

import mlx.core as mx

from ...engine_core import get_executor
from .base import VideoBackend, VideoConstraints, VideoGenParams

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE = "distilled"
_LTX2_5_DIM_DIV = 32


class LTX2_5Backend(VideoBackend):
    name = "ltx2_5"
    supports_i2v = True

    def __init__(
        self,
        model_name: str,
        *,
        text_encoder_repo: str | None = None,
        pipeline: str = _DEFAULT_PIPELINE,
        two_stage: bool = True,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._text_encoder_repo = text_encoder_repo
        self._pipeline = pipeline
        self._two_stage = two_stage
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        # 显式匹配 2.5；不依赖 "ltx-2" 子串（避免与 LTX2Backend 冲突）。
        return "ltx-2.5" in p or "ltx_2.5" in p or "ltx2.5" in p

    async def start(self, model_path: str, **kwargs: Any) -> None:
        if self._loaded:
            return
        logger.info("Starting LTX-2.5 backend (pure-MLX): %s", model_path)

        def _resolve():
            from fusion_mlx.video.ltx2_5.utils import get_model_path

            return get_model_path(model_path)

        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _resolve), timeout=180.0
        )
        self._loaded = True
        logger.info("LTX-2.5 backend ready: %s", model_path)

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
        # P6 E2E path (two-stage distilled + audio mux + duration-head) is not
        # implemented this round — real 22B weights are required. Fail visible.
        logger.error(
            "LTX2_5Backend.generate: E2E generate path not implemented "
            "(P6 pending 22B weights). prompt=%r frames=%s",
            params.prompt[:60] if params.prompt else "",
            params.num_frames,
        )
        raise NotImplementedError(
            "LTX-2.5 generate() E2E path is not implemented in this phase "
            "(P6/P8 pending real 22B weights download via hf-mirror). "
            "Structural port (config/text-encoder/duration-head/temporal-"
            "upsampler) is landed; see fusion_mlx.video.ltx2_5."
        )

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=4,
            dim_divisibility=_LTX2_5_DIM_DIV,
            num_frames_validator=lambda nf: nf % 8 == 1,
            num_frames_hint="num_frames % 8 == 1 (or omit to let duration-head decide)",
            dim_hint="width and height must be divisible by 32",
        )
