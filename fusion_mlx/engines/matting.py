# SPDX-License-Identifier: Apache-2.0
"""Image matting engine (#1663) — BiRefNet-lite background removal.

Loads a BiRefNet-lite MLX checkpoint (``config.json`` with
``model_type: birefnet`` + ``model.safetensors``; the vendored pure-MLX
implementation is ``fusion_mlx.image.matting.vendor_birefnet_lite``,
upstream moonbox-dev/BiRefNet_lite-MLX, MIT) and produces a soft alpha
matte for an input RGB image. The route layer composites the alpha onto
the source to emit a transparent PNG.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import partial
from typing import Any

import numpy as np

from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_MODEL_LOAD_TIMEOUT = 300.0
_MATTING_TIMEOUT = 300.0


class MattingEngine(BaseNonStreamingEngine):
    engine_type = "image_matting"

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_name
        self._kwargs = kwargs
        self._model: Any = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._model_load_duration: float | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return None

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def start(self) -> None:
        if self._loaded:
            return
        started = time.monotonic()
        await self._run_via_executor(
            self._load_sync, executor_name="image", timeout=_MODEL_LOAD_TIMEOUT
        )
        self._model_load_duration = time.monotonic() - started
        logger.info(
            "Matting engine loaded: model=%s duration=%.1fs",
            self._model_path,
            self._model_load_duration,
        )

    def _load_sync(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            from ..image.matting import vendor_birefnet_lite

            self._model = vendor_birefnet_lite.load(self._model_path)
            self._loaded = True
            logger.info("BiRefNet-lite weights loaded from %s", self._model_path)

    async def stop(self) -> None:
        self._model = None
        self._loaded = False
        await self._teardown_cache(executor_name="image")
        logger.info("Matting engine unloaded: model=%s", self._model_name)

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": self.engine_type,
            "model_name": self._model_name,
            "loaded": self._loaded,
            "model_load_duration": self._model_load_duration,
        }

    def _matte_sync(self, rgb: np.ndarray) -> np.ndarray:
        from ..image.matting import vendor_birefnet_lite

        if self._model is None:
            raise RuntimeError("matting model not loaded")
        return vendor_birefnet_lite.matte(self._model, rgb)

    async def matte(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB -> HxW uint8 alpha (255 = foreground)."""
        if not self._loaded:
            await self.start()
        return await self._run_via_executor(
            partial(self._matte_sync, rgb),
            executor_name="image",
            timeout=_MATTING_TIMEOUT,
        )
