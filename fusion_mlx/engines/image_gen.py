# SPDX-License-Identifier: Apache-2.0
# Image generation engine (Flux) for fusion-mlx.
# Loads mlx-community Flux models (diffusers layout) via mflux.
import asyncio
import gc
import io
import logging
import time
from typing import Any

import mlx.core as mx
from PIL import Image

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


def _to_latent_size(image_size: tuple[int, int]) -> tuple[int, int]:
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    return (h // 8, w // 8)


def _infer_model_config_label(model_path: str) -> str:
    name = model_path.lower()
    if "dev" in name and "schnell" not in name:
        return "dev"
    if "schnell" in name or "lite" in name:
        return "schnell"
    return "schnell"


class ImageGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_name
        self._flux = None
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._flux is not None:
            return
        logger.info("Starting ImageGen engine (mflux): %s", self._model_path)

        def _load():
            try:
                from mflux.models.flux.variants.txt2img.flux import Flux1
                from mflux.models.common.config.model_config import ModelConfig
            except ImportError as exc:
                raise ImportError(
                    "Flux image generation requires mflux. "
                    "Install mflux (editable from mlx-examples/mflux or pip install mflux)."
                ) from exc

            label = _infer_model_config_label(self._model_path)
            if label == "dev":
                model_config = ModelConfig.dev()
            else:
                model_config = ModelConfig.schnell()
            logger.info(
                "ImageGen loading model_config=%s path=%s",
                label,
                self._model_path,
            )
            flux = Flux1(model_config=model_config, model_path=self._model_path)
            return flux

        loop = asyncio.get_running_loop()
        self._flux = await asyncio.wait_for(
            loop.run_in_executor(get_executor("io"), _load), timeout=180.0
        )
        logger.info("ImageGen engine loaded: %s", self._model_name)

    async def stop(self) -> None:
        if self._flux is None:
            return
        self._flux = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                get_executor("image"), lambda: (mx.synchronize(), mx.clear_cache())
            ),
            timeout=5.0,
        )

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: int | None = None,
        guidance: float = 4.0,
        n_images: int = 1,
        output_format: str = "PNG",
        **kwargs,
    ) -> list[bytes]:
        if self._flux is None:
            raise RuntimeError("ImageGen engine not started.")

        flux = self._flux
        base_seed = seed if seed is not None else 0
        t0 = time.monotonic()
        activity_id = self._begin_activity(
            "generating images",
            metadata={"prompt_len": len(prompt), "n_images": n_images},
        )

        def _generate():
            images: list[bytes] = []
            for i in range(max(1, n_images)):
                gen = flux.generate_image(
                    seed=base_seed + i,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                )
                buf = io.BytesIO()
                gen.image.save(buf, format=output_format)
                images.append(buf.getvalue())
            return images

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("image"), _generate), timeout=300.0
            )
            elapsed = time.monotonic() - t0
            self._update_activity(activity_id, elapsed_seconds=elapsed)
            logger.info(
                "ImageGen generated %d image(s) in %.2fs",
                len(result),
                elapsed,
            )
            return result
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._flux is not None}

    def __repr__(self) -> str:
        status = "running" if self._flux is not None else "stopped"
        return f"<ImageGenEngine model={self._model_name} status={status}>"
