# SPDX-License-Identifier: Apache-2.0
"""Image generation engine (Flux 2) for fusion-mlx."""

import asyncio
import gc
import logging
import time
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


def _to_latent_size(image_size: tuple[int, int]) -> tuple[int, int]:
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    return (h // 8, w // 8)


class ImageGenEngine(BaseNonStreamingEngine):
    """Image generation engine wrapping Flux pipelines."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._pipeline = None
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._pipeline is not None:
            return
        logger.info("Starting ImageGen engine: %s", self._model_name)

        def _load():
            try:
                from flux import FluxPipeline
            except ImportError as exc:
                raise ImportError(
                    'Flux image generation requires mlx-examples flux module. '
                    'Install with: pip install "fusion-mlx[image]"'
                ) from exc

            pipeline = FluxPipeline(self._model_name, t5_padding=True)
            pipeline.ensure_models_are_loaded()
            return pipeline

        loop = asyncio.get_running_loop()
        self._pipeline = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _load), timeout=120.0)
        logger.info("ImageGen engine loaded: %s", self._model_name)

    async def stop(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("image"), lambda: (mx.synchronize(), mx.clear_cache())), timeout=5.0)

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: Optional[int] = None,
        guidance: float = 4.0,
        n_images: int = 1,
        output_format: str = "PNG",
        **kwargs,
    ) -> list[bytes]:
        """Generate images from a text prompt.

        Returns a list of PNG image bytes, one per requested image.
        """
        if self._pipeline is None:
            raise RuntimeError("ImageGen engine not started.")

        pipeline = self._pipeline
        latent_size = _to_latent_size((height, width))
        t0 = time.monotonic()
        activity_id = self._begin_activity("generating images", metadata={"prompt_len": len(prompt), "n_images": n_images})

        def _generate():
            latents = pipeline.generate_latents(
                prompt, n_images=n_images, num_steps=steps,
                latent_size=latent_size, guidance=guidance, seed=seed or 0,
            )
            mx.eval(next(latents))
            xt = None
            for xt_in in latents:
                mx.eval(xt_in)
                xt = xt_in
            if xt is None:
                raise RuntimeError("Flux produced no latent output")
            decoded = pipeline.decode(xt, latent_size)
            decoded = (decoded * 255).astype(mx.uint8)
            mx.eval(decoded)
            images = []
            for i in range(min(n_images, decoded.shape[0])):
                img = Image.fromarray(np.array(decoded[i]))
                import io
                buf = io.BytesIO()
                img.save(buf, format=output_format)
                images.append(buf.getvalue())
            return images

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("image"), _generate), timeout=120.0)
            elapsed = time.monotonic() - t0
            self._update_activity(activity_id, elapsed_seconds=elapsed)
            return result
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> Dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._pipeline is not None}

    def __repr__(self) -> str:
        status = "running" if self._pipeline is not None else "stopped"
        return f"<ImageGenEngine model={self._model_name} status={status}>"
