# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Embeddings API routes."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ..server_metrics import get_server_metrics
from .embedding_models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from .embedding_utils import (
    encode_embedding_base64,
    normalize_embedding_items,
    normalize_input,
    truncate_embedding,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["embeddings"])

_pool: Any = None
_server_state: Any = None


def set_embeddings_context(pool: Any, server_state: Any) -> None:
    global _pool, _server_state
    _pool = pool
    _server_state = server_state


async def get_embedding_engine(model_id: str) -> Any:
    """Resolve, load, and type-check an embedding engine."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    engine = await _pool.get_engine(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    from ..engines.embedding import EmbeddingEngine

    if not isinstance(engine, EmbeddingEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not an embedding model",
        )
    return engine


def get_embedding_max_length(model_id: str, max_length: int | None) -> int:
    if max_length is not None:
        return max_length
    return 512


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for input text(s)."""
    oq_manager = getattr(_server_state, "oq_manager", None) if _server_state else None
    if oq_manager and getattr(oq_manager, "is_quantizing", False):
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again later.",
        )

    await get_embedding_engine(request.model)

    if request.items is not None:
        embedding_inputs = normalize_embedding_items(request.items)
    elif request.input is not None:
        embedding_inputs = normalize_input(request.input)
    else:
        embedding_inputs = []

    if not embedding_inputs:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    max_length = get_embedding_max_length(
        request.model, getattr(request, "max_length", None)
    )
    truncation = getattr(request, "truncation", True)

    start_time = time.perf_counter()
    try:
        engine = await get_embedding_engine(request.model)
        output = await engine.embed(
            embedding_inputs,
            max_length=max_length,
            truncation=truncation,
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Embedding: %d inputs, %d dims, %d tokens, max_length=%d in %.3fs",
        len(embedding_inputs),
        output.dimensions,
        output.total_tokens,
        max_length,
        elapsed,
    )
    get_server_metrics().record_request_complete(
        prompt_tokens=output.total_tokens,
        completion_tokens=0,
        cached_tokens=0,
        prefill_duration=elapsed,
        model_id=request.model,
    )

    data = []
    for i, embedding in enumerate(output.embeddings):
        if request.dimensions and request.dimensions < len(embedding):
            embedding = truncate_embedding(embedding, request.dimensions)

        if request.encoding_format == "base64":
            formatted_embedding = encode_embedding_base64(embedding)
        else:
            formatted_embedding = embedding

        data.append(EmbeddingData(index=i, embedding=formatted_embedding))

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=output.total_tokens,
            total_tokens=output.total_tokens,
        ),
    )
