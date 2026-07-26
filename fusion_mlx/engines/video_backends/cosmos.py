# SPDX-License-Identifier: Apache-2.0
# Cosmos video backend: 7B T2V + Predict2 2B I2V.

import asyncio
import gc
import logging
import os

import mlx.core as mx

from .base import VideoBackend, VideoConstraints, VideoGenParams, validate_params

logger = logging.getLogger(__name__)


class CosmosBackend(VideoBackend):
    name = "cosmos"
    supports_i2v = True  # Predict2 2B supports I2V

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._loaded = False
        self._is_predict2 = self._detect_predict2(model_path)

    @staticmethod
    def _detect_predict2(model_path: str) -> bool:
        lower = model_path.lower()
        return "predict2" in lower or "video2world" in lower

    @classmethod
    def detect(cls, model_path: str) -> bool:
        lower = model_path.lower()
        return any(kw in lower for kw in ("cosmos", "predict2", "video2world"))

    async def start(self) -> None:
        if self._loaded:
            logger.info("cosmos backend: already loaded")
            return
        logger.info(
            "cosmos backend: starting model=%s predict2=%s",
            self._model_path,
            self._is_predict2,
        )
        self._loaded = True

    async def stop(self) -> None:
        logger.info("cosmos backend: stopping")
        self._loaded = False
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    def constraints(self) -> VideoConstraints:
        dim_div = 8
        if self._is_predict2:
            nf_hint = "Cosmos Predict2: 41, 81, or 121 frames recommended"
            nf_validator = lambda n: n >= 1
        else:
            nf_hint = "Cosmos 7B T2V: 121 frames recommended, must be 4k+1"
            nf_validator = lambda n: n >= 1 and (n - 1) % 4 == 0

        return VideoConstraints(
            supports_i2v=self._is_predict2,
            max_n=1,
            dim_divisibility=dim_div,
            num_frames_validator=nf_validator,
            num_frames_hint=nf_hint,
            dim_hint=f"Width/height must be divisible by {dim_div}",
        )

    async def generate(self, params: VideoGenParams) -> list[str]:
        c = self.constraints()
        validate_params(
            c,
            num_frames=params.num_frames,
            width=params.width,
            height=params.height,
            n=params.n,
            image=params.image,
        )
        if not self._loaded:
            await self.start()

        from fusion_mlx.video.cosmos.generate import generate_video
        from fusion_mlx._tempfile_safe import managed_tempfile_path
        from fusion_mlx.engine_core import get_video_gen_timeout

        results = []
        for i in range(params.n):
            seed = (params.seed + i) if params.seed is not None else None

            with managed_tempfile_path(
                prefix="fusion_cosmos_", suffix=".mp4"
            ) as handle:
                output_path = handle.path

                try:
                    timeout = get_video_gen_timeout()
                    path = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda s=seed, op=output_path: generate_video(
                                model_path=self._model_path,
                                prompt=params.prompt,
                                num_frames=params.num_frames,
                                width=params.width,
                                height=params.height,
                                fps=params.fps,
                                seed=s,
                                image=params.image,
                                cfg_scale=params.cfg_scale,
                                num_inference_steps=params.num_inference_steps or 50,
                                is_predict2=self._is_predict2,
                                on_step=None,
                                output_path=op,
                            ),
                        ),
                        timeout=timeout,
                    )
                    results.append(handle.release())
                except asyncio.TimeoutError:
                    logger.error("cosmos: generation timed out after %ds", timeout)
                    raise
                except Exception as e:
                    logger.error("cosmos: generation failed: %s", e)
                    raise

        return results
