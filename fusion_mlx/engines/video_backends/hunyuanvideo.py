# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo backend: T2V + I2V.

import asyncio
import gc
import logging

import mlx.core as mx
import numpy as np

from ..._tempfile_safe import managed_tempfile_path
from ...api._url_safety import is_safe_local_path
from ...engine_core import get_executor, get_video_gen_timeout
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

    async def start(self, model_path: str = "", **kwargs) -> None:
        if self._loaded:
            logger.info("hunyuanvideo backend: already loaded")
            return
        if model_path:
            self._model_path = model_path
        logger.info("hunyuanvideo backend: starting model=%s", self._model_path)
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
        logger.info("hunyuanvideo backend: stopped")

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

        from fusion_mlx.video.hunyuanvideo.generate import generate_video

        results = []
        for i in range(params.n):
            seed = (params.seed + i) if params.seed is not None else None

            with managed_tempfile_path(prefix="fusion_hv_", suffix=".mp4") as handle:
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
                    logger.error(
                        "hunyuanvideo: generation timed out after %ds", timeout
                    )
                    raise
                except Exception as e:
                    logger.error("hunyuanvideo: generation failed: %s", e)
                    raise

        return results

    async def load_vae_encoder(self) -> None:
        if getattr(self, "_stage_vae_encoder", None) is not None:
            return
        import os

        from fusion_mlx.video.hunyuanvideo.vae import HunyuanVideoVAE

        model_path = self._model_path
        vae_path = (
            os.path.join(model_path, "vae")
            if os.path.isdir(os.path.join(model_path, "vae"))
            else model_path
        )

        def _load():
            vae = HunyuanVideoVAE.from_pretrained(vae_path)
            mx.eval(vae.parameters())
            return vae

        loop = asyncio.get_running_loop()
        self._stage_vae_encoder = await loop.run_in_executor(get_executor("io"), _load)
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags["vae_encoder"] = True
        logger.info(
            "hunyuan: vae_encoder load vae=%s", type(self._stage_vae_encoder).__name__
        )

    async def encode(self, pixels: mx.array) -> mx.array:
        if getattr(self, "_stage_vae_encoder", None) is None:
            await self.load_vae_encoder()
        vae = self._stage_vae_encoder
        ndim = pixels.ndim
        if ndim not in (4, 5):
            raise ValueError(
                f"encode expects (T,H,W,3) or (1,T,H,W,3); got {tuple(pixels.shape)}"
            )
        src_np = np.array(pixels[0] if ndim == 5 else pixels)

        def _encode():
            x = mx.array(src_np)
            if x.ndim == 4:
                x = x[None]
            lat = vae.encode(x)
            lat = lat if lat.ndim == 5 else lat[None]
            mx.eval(lat)
            return lat

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor("video"), _encode)
        logger.info("hunyuan: vae encode out_shape=%s", tuple(result.shape))
        return result

    async def unload_vae_encoder(self) -> None:
        self._stage_vae_encoder = None
        self._stage_flags = getattr(self, "_stage_flags", {})
        self._stage_flags.pop("vae_encoder", None)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        logger.info("hunyuan: vae_encoder unload")
