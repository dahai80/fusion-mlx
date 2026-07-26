# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo backend: T2V + I2V.

import asyncio
import gc
import logging

import mlx.core as mx

from ..._tempfile_safe import managed_tempfile_path
from ...engine_core import get_video_gen_timeout
from .base import VideoBackend, VideoConstraints, VideoGenParams, validate_params

logger = logging.getLogger(__name__)


class HunyuanVideoBackend(VideoBackend):
    name = "hunyuanvideo"
    supports_i2v = True

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._loaded = False

    @classmethod
    def detect(cls, model_path: str) -> bool:
        lower = model_path.lower()
        return any(
            kw in lower for kw in ("hunyuanvideo", "hunyuan-video", "hunyuan_video")
        )

    async def start(self) -> None:
        if self._loaded:
            logger.info("hunyuanvideo backend: already loaded")
            return
        logger.info("hunyuanvideo backend: starting model=%s", self._model_path)
        self._loaded = True

    async def stop(self) -> None:
        logger.info("hunyuanvideo backend: stopping")
        self._loaded = False
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    def constraints(self) -> VideoConstraints:
        dim_div = 16
        nf_hint = "HunyuanVideo: 33, 65, or 129 frames recommended, must be 4k+1"
        nf_validator = lambda n: n >= 1 and (n - 1) % 4 == 0

        return VideoConstraints(
            supports_i2v=True,
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

        from fusion_mlx.video.hunyuanvideo.generate import generate_video

        results = []
        for i in range(params.n):
            seed = (params.seed + i) if params.seed is not None else None

            with managed_tempfile_path(prefix="fusion_hv_", suffix=".mp4") as handle:
                output_path = handle.path

                try:
                    timeout = get_video_gen_timeout()
                    await asyncio.wait_for(
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
                                on_step=None,
                                output_path=op,
                            ),
                        ),
                        timeout=timeout,
                    )
                    results.append(handle.release())
                except asyncio.TimeoutError:
                    logger.error(
                        "hunyuanvideo: generation timed out after %ds", timeout
                    )
                    raise
                except Exception as e:
                    logger.error("hunyuanvideo: generation failed: %s", e)
                    raise

        return results
