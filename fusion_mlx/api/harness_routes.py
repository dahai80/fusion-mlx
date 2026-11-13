# SPDX-License-Identifier: Apache-2.0
"""Harness consumer routes (#1559).

fusion-harness v0.9.0 left two engine-side seams open (evolution plan
P1-2 / P3-1): an injectable TokenCounter and a CacheAdvisor. Consumers
(agent-studio, cowork, code) run in separate processes and talk to this
engine over HTTP, so the seams are served as routes:

- POST /harness/v1/tokens/count  — real tokenizer counts for text blobs
- POST /harness/v1/cache/probe   — chain-hashed block classification for
  a chat message list (RAM radix trie + SSD hot/disk + cold)

fusion_mlx/harness_bridge.py wraps these into the protocol objects the
harness CompactionConfig expects (token_counter / cache_advisor).
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..cache.probe_core import probe_prompt_cache
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/harness/v1", tags=["harness"])


class TokenCountRequest(BaseModel):
    texts: list[str]
    model_id: str | None = None


class TokenCountResponse(BaseModel):
    model_id: str
    counts: list[int]


class HarnessProbeRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    model_id: str | None = None


def _resolve_engine(engine_pool: Any, model_id: str | None) -> tuple[str, Any]:
    # Returns (model_id, engine_entry) for the requested model or the
    # first loaded entry when no model_id was given.
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")
    if model_id:
        entry = engine_pool._entries.get(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        if entry.engine is None:
            raise HTTPException(status_code=409, detail=f"Model not loaded: {model_id}")
        return model_id, entry
    for mid, entry in engine_pool._entries.items():
        if entry.engine is not None:
            logger.debug("harness routes: resolved default model '%s'", mid)
            return mid, entry
    raise HTTPException(status_code=503, detail="No loaded model available")


@router.post("/tokens/count", response_model=TokenCountResponse)
async def count_tokens(
    request: TokenCountRequest,
    _auth: bool = Depends(verify_api_key),
):
    engine_pool = _get_engine_pool()
    model_id, entry = _resolve_engine(engine_pool, request.model_id)
    tokenizer = getattr(entry.engine, "_tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise HTTPException(
            status_code=400, detail="Model tokenizer unavailable for counting."
        )
    counts: list[int] = []
    try:
        for text in request.texts:
            counts.append(len(tokenizer.encode(text)))
    except Exception as exc:
        logger.warning("harness tokens/count failed for '%s': %s", model_id, exc)
        raise HTTPException(status_code=400, detail=f"Tokenization failed: {exc}")
    return TokenCountResponse(model_id=model_id, counts=counts)


@router.post("/cache/probe")
async def harness_cache_probe(
    request: HarnessProbeRequest,
    _auth: bool = Depends(verify_api_key),
):
    engine_pool = _get_engine_pool()
    model_id, _entry = _resolve_engine(engine_pool, request.model_id)
    result = probe_prompt_cache(
        engine_pool,
        model_id,
        request.messages,
        tools=request.tools,
        chat_template_kwargs=request.chat_template_kwargs,
    )
    if not result.get("model_loaded", False):
        raise HTTPException(status_code=409, detail=result.get("reason", "not loaded"))
    return result


def _get_engine_pool():
    from fusion_mlx.admin.helpers import _get_engine_pool as _f

    return _f()
