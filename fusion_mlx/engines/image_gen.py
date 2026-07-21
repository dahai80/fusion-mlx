# SPDX-License-Identifier: Apache-2.0
# Image generation engine (Flux 2) for fusion-mlx.
# Loads FLUX.2-klein via mflux (Flux2Klein uses mx.compile -> faster denoise).
import asyncio
import gc
import io
import logging
import time
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


def _to_latent_size(image_size: tuple[int, int]) -> tuple[int, int]:
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    return (h // 8, w // 8)


def _infer_flux2_config(model_path: str) -> str:
    name = (model_path or "").lower()
    if "base" in name and "4b" in name:
        return "flux2_klein_base_4b"
    if "base" in name and "9b" in name:
        return "flux2_klein_base_9b"
    if "4b" in name:
        return "flux2_klein_4b"
    if "kv" in name and "9b" in name:
        return "flux2_klein_9b_kv"
    return "flux2_klein_9b"


def _flux_quantize_from_env() -> int | None:
    import os

    env = os.environ.get("FUSION_FLUX_QUANT", "").strip().lower()
    if env in ("", "0", "off", "none", "bf16"):
        return None
    if env in ("w8a16", "w8", "int8", "8"):
        logger.info("Flux2Klein 量化: w8a16 (quantize=8)")
        return 8
    if env in ("w4", "nf4", "int4", "4"):
        logger.info("Flux2Klein 量化: w4 (quantize=4)")
        return 4
    logger.warning(
        "FUSION_FLUX_QUANT=%s 未知, 支持 w8a16/w4/off, 跳过量化",
        env,
    )
    return None


class ImageGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_name
        self._flux = None
        self._mflux_missing = False
        self._kwargs = kwargs
        # Load-time quantization (4/8-bit) reduces memory and often speeds up
        # Flux2 klein. None = load at the model's native precision. Passed
        # through to mflux Flux2Klein(quantize=...).
        self._quantize = kwargs.get("quantize")
        if self._quantize is None:
            self._quantize = _flux_quantize_from_env()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._flux is not None:
            return
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        except ImportError as exc:
            logger.warning(
                "ImageGen engine disabled: mflux-fusion not installed. "
                "Install with: pip install mflux-fusion  (%s)",
                exc,
            )
            self._mflux_missing = True
            return
        logger.info("Starting ImageGen engine (mflux Flux2Klein): %s", self._model_path)

        def _load():
            label = _infer_flux2_config(self._model_path)
            model_config = getattr(ModelConfig, label)()
            logger.info(
                "ImageGen loading flux2 config=%s path=%s",
                label,
                self._model_path,
            )
            flux = Flux2Klein(
                model_config=model_config,
                model_path=self._model_path,
                quantize=self._quantize,
            )
            return flux

        loop = asyncio.get_running_loop()
        self._flux = await asyncio.wait_for(
            loop.run_in_executor(get_executor("image"), _load), timeout=600.0
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
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> list[bytes]:
        if self._flux is None:
            if self._mflux_missing:
                raise RuntimeError(
                    "Image generation unavailable: mflux-fusion not installed. "
                    "Install with: pip install mflux-fusion"
                )
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
                gen_kwargs: dict[str, Any] = dict(
                    seed=base_seed + i,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                )
                if scheduler is not None:
                    gen_kwargs["scheduler"] = scheduler
                if negative_prompt is not None:
                    logger.warning(
                        "Flux2Klein does not support negative_prompt; "
                        "ignoring (got %d chars)",
                        len(negative_prompt),
                    )
                gen = flux.generate_image(**gen_kwargs)
                buf = io.BytesIO()
                gen.image.save(buf, format=output_format)
                images.append(buf.getvalue())
            return images

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(get_executor("image"), _generate), timeout=600.0
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
        return {
            "model_name": self._model_name,
            "loaded": self._flux is not None,
            "mflux_missing": self._mflux_missing,
        }

    def __repr__(self) -> str:
        if self._mflux_missing:
            status = "disabled(mflux-missing)"
        elif self._flux is not None:
            status = "running"
        else:
            status = "stopped"
        return f"<ImageGenEngine model={self._model_name} status={status}>"
