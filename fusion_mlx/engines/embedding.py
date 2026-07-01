# SPDX-License-Identifier: Apache-2.0
"""Embedding engine for fusion-mlx."""

import asyncio
import gc
import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingOutput:
    embeddings: list[list[float]]
    total_tokens: int
    dimensions: int


class MLXEmbeddingModel:
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._processor = None

    def load(self):
        try:
            from mlx_embeddings import load
        except ImportError as exc:
            raise ImportError('mlx-embeddings required. Install with: pip install "fusion-mlx[embedding]"') from exc
        self._model, self._processor = load(self._model_name, trust_remote_code=self._trust_remote_code)

    @property
    def processor(self):
        return self._processor

    @property
    def hidden_size(self) -> int | None:
        if self._model is None:
            return None
        try:
            return self._model.embeddings.embedding_dim
        except AttributeError:
            return None

    def embed(self, inputs: list[Any], max_length: int = 512, padding: bool = True, truncation: bool = True) -> "EmbeddingOutput":
        if self._processor is None:
            raise RuntimeError("Model not loaded")
        encoded = self._processor(
            [i if isinstance(i, str) else i.get("text", "") for i in inputs],
            max_length=max_length, padding=padding, truncation=truncation,
        )
        input_ids = mx.array(encoded["input_ids"])
        with mx.inference_mode():
            embeddings = self._model(input_ids)
        if embeddings.ndim == 3:
            mean_emb = [embeddings[i].mean(axis=0).tolist() for i in range(embeddings.shape[0])]
        else:
            mean_emb = [embeddings.mean(axis=0).tolist()]
        total_tokens = sum(len(e) for e in encoded["input_ids"])
        dims = len(mean_emb[0]) if mean_emb else 0
        return EmbeddingOutput(embeddings=mean_emb, total_tokens=total_tokens, dimensions=dims)

    def get_model_info(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "hidden_size": self.hidden_size}


class EmbeddingEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, trust_remote_code: bool = False, batch_size: int | None = None, *, scheduler_config: Any | None = None):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        if batch_size is None:
            batch_size = getattr(scheduler_config, "embedding_batch_size", 32) if scheduler_config is not None else 32
        self._batch_size = max(1, int(batch_size))
        self._model: MLXEmbeddingModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def processor(self) -> Any:
        return self._model.processor if self._model else None

    @property
    def hidden_size(self) -> int | None:
        return self._model.hidden_size if self._model else None

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Starting embedding engine: {self._model_name}")
        self._model = MLXEmbeddingModel(self._model_name, trust_remote_code=self._trust_remote_code)
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("llm"), self._model.load), timeout=120.0)

    async def stop(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from ..scheduler.helpers import _safe_clear_cache_for_non_llm
        await asyncio.wait_for(
            loop.run_in_executor(get_executor("llm"), _safe_clear_cache_for_non_llm), timeout=5.0)

    async def embed(self, texts: list[str] | list[dict[str, str]], max_length: int = 512, padding: bool = True, truncation: bool = True) -> EmbeddingOutput:
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")
        model = self._model
        input_items = [texts] if isinstance(texts, str) else list(texts)
        if not input_items:
            return EmbeddingOutput(embeddings=[], total_tokens=0, dimensions=0)
        batch_size = self._batch_size
        activity_id = self._begin_activity("embedding", detail="Embedding", total_items=len(input_items))
        try:
            loop = asyncio.get_running_loop()
            embeddings: list[list[float]] = []
            total_tokens = 0
            dimensions = 0
            for start in range(0, len(input_items), batch_size):
                batch = input_items[start:start + batch_size]
                def _embed_sync(b=batch):
                    return model.embed(inputs=b, max_length=max_length, padding=padding, truncation=truncation)
                output = await asyncio.wait_for(
                    loop.run_in_executor(get_executor("llm"), _embed_sync), timeout=30.0)
                embeddings.extend(output.embeddings)
                total_tokens += output.total_tokens
                if output.dimensions:
                    dimensions = output.dimensions
            return EmbeddingOutput(embeddings=embeddings, total_tokens=total_tokens, dimensions=dimensions)
        finally:
            self._end_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._model is not None, "hidden_size": self.hidden_size, "batch_size": self._batch_size}

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<EmbeddingEngine model={self._model_name} status={status}>"
