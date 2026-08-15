# SPDX-License-Identifier: Apache-2.0
# MiniMax-H3 video backend：33B Omni-Transformer（视频+音频联合生成）。
# Partition：fl2va（t2va/i2va/l2va/fl2va）/ ref2va（多参考素材）。
# 参照 cosmos.py 结构；generate() E2E 在 P6/P8（本轮 P5 仅 scaffolding + 注册 + 约束）。
import asyncio
import gc
import logging

import mlx.core as mx

from ...engine_core import get_executor
from .base import VideoBackend, VideoConstraints, VideoGenParams, validate_params

logger = logging.getLogger(__name__)

# H3 输出分辨率档。
_H3_RESOLUTIONS = ("768p", "2k")
# H3 VAE 空间下采样 16x；时间下采样 4x。
_H3_DIM_DIV = 16
# 33B 模型，单生成（显存约束）。
_H3_MAX_N = 1
# 最高 15s @24fps = 361 帧。
_H3_MAX_FRAMES = 361


class MiniMaxH3Backend(VideoBackend):
    name = "minimax_h3"
    supports_i2v = True

    def __init__(
        self,
        model_name: str,
        *,
        partition: str = "fl2va",
        resolution: str = "768p",
        **kwargs,
    ):
        self._model_name = model_name
        if partition not in ("fl2va", "ref2va"):
            raise ValueError(
                f"partition must be 'fl2va' or 'ref2va', got {partition!r}"
            )
        self._partition = partition
        if resolution not in _H3_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {_H3_RESOLUTIONS}, got {resolution!r}"
            )
        self._resolution = resolution
        self._loaded = False
        # 路径含 ref2va 或显式 partition 时切 Ref2VA partition。
        if "ref2va" in model_name.lower() and partition == "fl2va":
            self._partition = "ref2va"
            logger.info(
                "minimax_h3 backend: auto-switched partition to ref2va (path hint)"
            )
        logger.info(
            "minimax_h3 backend: init model=%s partition=%s resolution=%s",
            self._model_name,
            self._partition,
            self._resolution,
        )

    @classmethod
    def detect(cls, model_path: str) -> bool:
        p = model_path.lower()
        return any(
            kw in p
            for kw in (
                "minimax-h3",
                "minimax_h3",
                "h3-fl2va",
                "h3-ref2va",
                "h3_video",
                "fl2va",
                "ref2va",
            )
        )

    async def start(self, model_path: str = "", **kwargs) -> None:
        if self._loaded:
            logger.info("minimax_h3 backend: already loaded")
            return
        if model_path:
            self._model_name = model_path
        logger.info(
            "minimax_h3 backend: starting model=%s partition=%s resolution=%s",
            self._model_name,
            self._partition,
            self._resolution,
        )
        # 实际 DiT/VAE/text_encoder 加载在 P6/P8（本轮 scaffolding 不加载权重）。
        self._loaded = True

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
        logger.info("minimax_h3 backend: stopped")

    def constraints(self) -> VideoConstraints:
        return VideoConstraints(
            supports_i2v=True,
            max_n=_H3_MAX_N,
            dim_divisibility=_H3_DIM_DIV,
            num_frames_validator=lambda nf: 1 <= nf <= _H3_MAX_FRAMES,
            num_frames_hint=f"num_frames must be 1..{_H3_MAX_FRAMES} (≤15s @24fps)",
            dim_hint=f"Width/Height divisible by {_H3_DIM_DIV}; short side default 768, 2K via in-context regeneration",
        )

    async def generate(self, params: VideoGenParams) -> list[bytes]:
        c = self.constraints()
        validate_params(
            c,
            num_frames=params.num_frames,
            width=params.width,
            height=params.height,
            n=params.n,
            image=params.image,
        )
        if params.resolution not in _H3_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {_H3_RESOLUTIONS}, got {params.resolution!r}"
            )
        if not self._loaded:
            await self.start()
        # P6 condition + packed-sequence 组装 / P8 真实模型 E2E 未落地。
        # fail-visible：明确报未实现，不静默返回空。
        logger.error(
            "minimax_h3 backend: generate() not implemented (P6/P8 pending); "
            "partition=%s resolution=%s frames=%d",
            self._partition,
            params.resolution,
            params.num_frames,
        )
        raise NotImplementedError(
            "MiniMax-H3 generate() E2E path is not implemented in this phase "
            "(P6 condition/packed-sequence + P8 real-model E2E pending). "
            f"partition={self._partition} resolution={params.resolution}"
        )
