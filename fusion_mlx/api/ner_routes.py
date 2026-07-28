# SPDX-License-Identifier: Apache-2.0
"""NER API routes for named entity recognition."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..middleware.auth import check_rate_limit, verify_api_key
from ..server_metrics import get_server_metrics
from .ner_models import NEREntity, NERRequest, NERResponse, NERUsage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ner"])

_pool: Any = None
_server_state: Any = None


def set_ner_context(pool: Any, server_state: Any) -> None:
    global _pool, _server_state
    _pool = pool
    _server_state = server_state


async def get_ner_engine(model_id: str) -> Any:
    """Resolve, load, and type-check a NER engine."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    engine = await _pool.get_engine(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    from ..engines.ner import NEREngine

    if not isinstance(engine, NEREngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not a NER model",
        )
    return engine


@router.post(
    "/ner",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_ner(request: NERRequest) -> NERResponse:
    """Extract named entities from text using GLiNER models."""
    engine = await get_ner_engine(request.model)

    texts = request.text if isinstance(request.text, list) else [request.text]

    if not texts:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not request.labels:
        raise HTTPException(status_code=400, detail="Labels cannot be empty")

    start_time = time.perf_counter()
    try:
        output = await engine.ner(
            texts=texts,
            labels=request.labels,
            threshold=request.threshold,
            flat_ner=request.flat_ner,
            multi_label=request.multi_label,
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "NER: %d texts, %d labels, %d entities, threshold=%.2f in %.3fs",
        len(texts),
        len(request.labels),
        sum(len(e) for e in output.entities),
        request.threshold,
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
    for entity_list in output.entities:
        data.append([
            NEREntity(
                start=e["start"],
                end=e["end"],
                text=e["text"],
                label=e["label"],
                score=e["score"],
            )
            for e in entity_list
        ])

    return NERResponse(
        data=data,
        model=request.model,
        usage=NERUsage(
            prompt_tokens=output.total_tokens,
            total_tokens=output.total_tokens,
        ),
    )
