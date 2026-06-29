# SPDX-License-Identifier: Apache-2.0
"""Cohere/Jina-compatible Rerank API routes."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ..server_metrics import get_server_metrics
from .rerank_models import (
    RerankRequest,
    RerankResult,
    RerankResponse,
    RerankUsage,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["rerank"])

_pool: Any = None
_server_state: Any = None


def set_rerank_context(pool: Any, server_state: Any) -> None:
    global _pool, _server_state
    _pool = pool
    _server_state = server_state


async def get_reranker_engine(model_id: str) -> Any:
    """Resolve, load, and type-check a reranker engine."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    engine = await _pool.get_engine(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    from ..engines.reranker import RerankerEngine
    if not isinstance(engine, RerankerEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not a reranker model",
        )
    return engine


def normalize_documents(documents: list[str] | list[dict]) -> list[str]:
    """Normalize document input to list of strings."""
    result = []
    for doc in documents:
        if isinstance(doc, str):
            result.append(doc)
        elif isinstance(doc, dict):
            result.append(doc.get("text", ""))
        else:
            result.append(str(doc))
    return result


@router.post("/rerank")
async def create_rerank(request: RerankRequest) -> RerankResponse:
    """
    Rerank documents by relevance to a query.

    Cohere/Jina-compatible endpoint for document reranking.
    """
    oq_manager = getattr(_server_state, "oq_manager", None) if _server_state else None
    if oq_manager and getattr(oq_manager, "is_quantizing", False):
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )

    await get_reranker_engine(request.model)

    documents_raw = request.documents
    documents_text = normalize_documents(documents_raw)

    if not documents_text:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")

    if isinstance(request.query, str) and not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.perf_counter()

    engine = await get_reranker_engine(request.model)
    output = await engine.rerank(
        query=request.query,
        documents=documents_raw,
        top_n=request.top_n,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Rerank: %d docs, %d tokens in %.3fs",
        len(documents_raw), output.total_tokens, elapsed,
    )
    get_server_metrics().record_request_complete(
        prompt_tokens=output.total_tokens,
        completion_tokens=0,
        cached_tokens=0,
        prefill_duration=elapsed,
        model_id=request.model,
    )

    results = []
    for idx in output.indices:
        if request.return_documents:
            orig = documents_raw[idx]
            display_doc = orig if isinstance(orig, dict) else {"text": orig}
        else:
            display_doc = None
        result = RerankResult(
            index=idx,
            relevance_score=output.scores[idx],
            document=display_doc,
        )
        results.append(result)

    return RerankResponse(
        results=results,
        model=request.model,
        usage=RerankUsage(total_tokens=output.total_tokens),
    )
