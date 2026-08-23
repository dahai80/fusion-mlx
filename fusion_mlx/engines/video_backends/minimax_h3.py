# SPDX-License-Identifier: Apache-2.0
# MiniMax-H3 video backend：33B Omni-Transformer（视频+音频联合生成）。
# Partition：fl2va（t2va/i2va/l2va/fl2va）/ ref2va（多参考素材）。
# 参照 cosmos.py 结构；generate() E2E 在 P6/P8（本轮 P5 仅 scaffolding + 注册 + 约束）。
import asyncio
import gc
import logging

import mlx.core as mx

from ..._tempfile_safe import managed_tempfile_path
from ...api._url_safety import is_safe_local_path
from ...engine_core import get_executor, get_video_gen_timeout
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
    # Issue #589: only t2va is implemented. i2va/l2va/fl2va image/last-frame
    # conditioning is accepted then silently dropped (silent-wrong-video bug).
    # Declare supports_i2v=False so validate_params rejects image= at the API
    # (422) until the image-conditioned DiT path lands. generate() re-checks
    # last_frame_image/reference_audio (not covered by validate_params).
    supports_i2v = False

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
            supports_i2v=False,
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
        # Issue #589: only t2va implemented. Reject image/last-frame/reference-audio
        # conditioning loudly so callers don't get silent-wrong t2va video.
        # validate_params already rejects image= via supports_i2v=False; this is
        # the backstop for last_frame_image/reference_audio (validate_params
        # does not check those fields) and for direct non-API callers.
        if params.image is not None:
            raise ValueError(
                "MiniMax-H3 i2va not implemented (issue #589): image conditioning "
                "is silently dropped by the t2va path. Use a backend with real i2v."
            )
        if params.last_frame_image is not None:
            raise ValueError(
                "MiniMax-H3 l2va/fl2va not implemented (issue #589): "
                "last_frame_image conditioning is not supported by the t2va path."
            )
        if params.reference_audio is not None:
            raise ValueError(
                "MiniMax-H3 ref2va audio not implemented (issue #589): "
                "reference_audio is not supported by the t2va path."
            )
        if params.resolution not in _H3_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {_H3_RESOLUTIONS}, got {params.resolution!r}"
            )
        if self._model_name.startswith(("/", "~")) or ".." in self._model_name:
            if not is_safe_local_path(self._model_name):
                raise ValueError(
                    f"model_path outside allowed directories: {self._model_name}"
                )
        if not self._loaded:
            await self.start()

        from fusion_mlx.video.minimax_h3.generate import generate_video

        results = []
        for i in range(params.n):
            seed = (params.seed + i) if params.seed is not None else None

            with managed_tempfile_path(prefix="fusion_h3_", suffix=".mp4") as handle:
                output_path = handle.path

                try:
                    timeout = get_video_gen_timeout()
                    await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            get_executor("video"),
                            lambda s=seed, op=output_path: generate_video(
                                model_path=self._model_name,
                                prompt=params.prompt,
                                num_frames=params.num_frames,
                                width=params.width,
                                height=params.height,
                                fps=params.fps,
                                seed=s,
                                num_inference_steps=(params.num_inference_steps or 40),
                                output_path=op,
                                quantize=params.quantize,
                                audio=params.audio,
                            ),
                        ),
                        timeout=timeout,
                    )
                    handle.release()
                    with open(output_path, "rb") as f:
                        results.append(f.read())
                except TimeoutError:
                    logger.error("minimax_h3: generation timed out after %ds", timeout)
                    raise
                except Exception as e:
                    logger.error("minimax_h3: generation failed: %s", e)
                    raise

        return results
