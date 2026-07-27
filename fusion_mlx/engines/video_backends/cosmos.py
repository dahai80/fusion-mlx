# SPDX-License-Identifier: Apache-2.0
# Cosmos video backend: 7B T2V + Predict2 2B I2V.

import asyncio
import gc
import logging

import mlx.core as mx

from ..._tempfile_safe import managed_tempfile_path
from ...api._url_safety import is_safe_local_path
from ...engine_core import get_executor, get_video_gen_timeout
from .base import VideoBackend, VideoConstraints, VideoGenParams, validate_params

logger = logging.getLogger(__name__)


class CosmosBackend(VideoBackend):
    name = "cosmos"
    # supports_i2v is dynamic: see constraints().supports_i2v

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

    async def start(self, model_path: str = "", **kwargs) -> None:
        if self._loaded:
            logger.info("cosmos backend: already loaded")
            return
        if model_path:
            self._model_path = model_path
        logger.info(
            "cosmos backend: starting model=%s predict2=%s",
            self._model_path,
            self._is_predict2,
        )
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
        logger.info("cosmos backend: stopped")

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
        if self._model_path.startswith(("/", "~")) or ".." in self._model_path:
            if not is_safe_local_path(self._model_path):
                raise ValueError(
                    f"model_path outside allowed directories: {self._model_path}"
                )
        if not self._loaded:
            await self.start()

        from fusion_mlx.video.cosmos.generate import generate_video

        results = []
        for i in range(params.n):
            seed = (params.seed + i) if params.seed is not None else None

            with managed_tempfile_path(
                prefix="fusion_cosmos_", suffix=".mp4"
            ) as handle:
                output_path = handle.path

                try:
                    timeout = get_video_gen_timeout()
                    await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            get_executor("video"),
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
                    handle.release()
                    with open(output_path, "rb") as f:
                        results.append(f.read())
                except TimeoutError:
                    logger.error("cosmos: generation timed out after %ds", timeout)
                    raise
                except Exception as e:
                    logger.error("cosmos: generation failed: %s", e)
                    raise

        return results
