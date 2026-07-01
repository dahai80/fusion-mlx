# SPDX-License-Identifier: Apache-2.0
"""Reranker engine for fusion-mlx."""

import asyncio
import gc
import logging
from typing import Any

import mlx.core as mx

from ..engine_core import get_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass
class RerankOutput:
    scores: list[float]
    indices: list[int]
    total_tokens: int


class MLXRerankerModel:
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._processor = None

    def load(self):
        try:
            from mlx_embeddings import load as load_embedding
        except ImportError:
            from mlx_lm import load
            self._model, self._processor = load(self._model_name, tokenizer_config={"trust_remote_code": self._trust_remote_code})
            return
        self._model, self._processor = load_embedding(self._model_name, trust_remote_code=self._trust_remote_code)

    @property
    def processor(self):
        return self._processor

    @property
    def num_labels(self) -> int | None:
        if self._model is None:
            return None
        try:
            return self._model.score.head.output_dims
        except AttributeError:
            return None

    def rerank(self, query: "str | dict", documents: "list[str] | list[dict]", max_length: int | None = None) -> RerankOutput:
        if isinstance(query, dict):
            query_text = query.get("text", "")
        else:
            query_text = query
        doc_texts = [d if isinstance(d, str) else d.get("text", "") for d in documents]
        pairs = [[query_text, d] for d in doc_texts]
        ml = max_length or (8192 if hasattr(self._model, "layers") else 512)
        encoded = self._processor(pairs, max_length=ml, padding=True, truncation=True)
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        with mx.inference_mode():
            outputs = self._model(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        if logits.ndim == 3:
            scores = [float(logits[i, -1, 0]) for i in range(len(documents))]
        else:
            scores = [float(logits[i, -1]) for i in range(len(documents))]
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        total_tokens = sum(len(e) for e in encoded["input_ids"])
        return RerankOutput(scores=scores, indices=indices, total_tokens=total_tokens)

    def get_model_info(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "num_labels": self.num_labels}


class RerankerEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model: MLXRerankerModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def processor(self) -> Any:
        return self._model.processor if self._model else None

    @property
    def num_labels(self) -> int | None:
        return self._model.num_labels if self._model else None

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Starting reranker engine: {self._model_name}")
        self._model = MLXRerankerModel(self._model_name, trust_remote_code=self._trust_remote_code)
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

    async def rerank(self, query: "str | dict", documents: "list[str] | list[dict]", top_n: int | None = None, max_length: int | None = None) -> RerankOutput:
        if self._model is None:
            raise RuntimeError("Engine not started.")
        activity_id = self._begin_activity("reranking", total_items=len(documents))
        try:
            loop = asyncio.get_running_loop()
            def _rerank():
                return self._model.rerank(query=query, documents=documents, max_length=max_length)
            output = await asyncio.wait_for(
                loop.run_in_executor(get_executor("llm"), _rerank), timeout=30.0)
            self._update_activity(activity_id, token_count=output.total_tokens)
            if top_n is not None and top_n < len(output.indices):
                return RerankOutput(scores=output.scores, indices=output.indices[:top_n], total_tokens=output.total_tokens)
            return output
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._model is not None, "num_labels": self.num_labels}

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<RerankerEngine model={self._model_name} status={status}>"
