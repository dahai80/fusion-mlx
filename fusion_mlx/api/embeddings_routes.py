# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Embeddings API routes."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..middleware.auth import check_rate_limit, verify_api_key
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

_EMBEDDING_BATCH_SIZE = 64
_EMBEDDING_DEDUP_ENABLED = True

router = APIRouter(prefix="/v1", tags=["embeddings"])

_pool: Any = None
_server_state: Any = None


def set_embeddings_context(pool: Any, server_state: Any) -> None:
    global _pool, _server_state
    _pool = pool
    _server_state = server_state


def _is_default_sentinel(model_id: str | None) -> bool:
    return not model_id or model_id == "default"


def _pool_embedding_ids() -> list[str]:
    if _pool is None:
        return []
    try:
        return [
            mid
            for mid in _pool.list_models()
            if getattr(_pool.get_entry(mid), "engine_type", None) == "embedding"
        ]
    except Exception as exc:
        logger.debug("embedding default lookup failed: %s", exc)
        return []


def _default_embedding_model_id() -> str | None:
    # The id the "default" sentinel maps to on this surface. A configured
    # embed id wins; otherwise the pool's single embedding entry (embedding
    # models are lazily discovered, not pinned — embedding_model_locked is
    # never set on the serve path). Ambiguous or absent → None; the caller
    # rejects the sentinel instead of falling back to the chat model.
    from ..config import get_config
    from ..model_aliases import resolve_model

    cfg = get_config()
    for cand in (cfg.embedding_model_locked, cfg.embedding_model):
        if cand:
            try:
                return resolve_model(cand)
            except Exception as exc:
                logger.debug("configured embedding model %r unresolved: %s", cand, exc)
                return cand
    ids = _pool_embedding_ids()
    if len(ids) == 1:
        return ids[0]
    return None


async def get_embedding_engine(model_id: str) -> Any:
    """Resolve, load, and type-check an embedding engine.

    Errors carry the OpenAI ``{"error":{type,code,message}}`` envelope
    (dict-detail passes through the central HTTPException handler) so SDKs
    that key on ``error.code`` (langchain / llamaindex / openai-python) get a
    machine-readable signal instead of substring-matching the message.
    """
    if _pool is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Server not initialized; embeddings pool unavailable.",
                    "type": "server_busy",
                    "code": "server_not_initialized",
                    "param": None,
                }
            },
        )
    from ..service.helpers import _resolve_request_alias_or_default

    resolved_id = _resolve_request_alias_or_default(
        model_id, _default_embedding_model_id()
    )
    if resolved_id is None and _is_default_sentinel(model_id):
        # Never fall through to the chat-aligned resolver for the sentinel:
        # on a box with 2+ embedding models (the normal install) it resolves
        # the served LLM, which then fails the embedding type check — and
        # get_engine would cold-load the LLM on an embeddings call.
        candidates = _pool_embedding_ids()
        logger.info(
            "Embedding sentinel %r ambiguous: %d embedding model(s) available",
            model_id,
            len(candidates),
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "model 'default' is ambiguous: "
                        f"{len(candidates)} embedding model(s) are available"
                        + (f" ({', '.join(sorted(candidates))})" if candidates else "")
                        + "; name one explicitly"
                    ),
                    "type": "invalid_request_error",
                    "code": "embedding_model_ambiguous",
                    "param": "model",
                }
            },
        )
    if resolved_id is None:
        # Not the configured embedding model (or no embed model configured):
        # fall through to the same resolver the chat path uses so short
        # aliases + HF paths still reach the pool.
        from ..server import resolve_model_with_profile

        try:
            resolved_id, _profile_overrides = resolve_model_with_profile(model_id)
        except ValueError as exc:
            logger.info("Embedding model %r did not resolve: %s", model_id, exc)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Model not found: {model_id} ({exc})",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                        "param": "model",
                    }
                },
            ) from exc
    if resolved_id != model_id:
        logger.info("Embedding alias: %s -> %s", model_id, resolved_id)
    engine = await _pool.get_engine(resolved_id)
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model not found: {model_id}",
                    "type": "not_found_error",
                    "code": "model_not_found",
                    "param": "model",
                }
            },
        )
    from ..engines.embedding import EmbeddingEngine

    if not isinstance(engine, EmbeddingEngine):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"No embedding model loaded for '{model_id}' "
                        "(requested model is not an embedding model). "
                        "Boot with --embedding-model <id> "
                        "and install deps via: pip install 'fusion-mlx[embeddings]'"
                    ),
                    "type": "invalid_request_error",
                    "code": "no_embedding_model",
                    "param": "model",
                }
            },
        )
    return engine


def get_embedding_max_length(model_id: str, max_length: int | None) -> int | None:
    if max_length is not None:
        return max_length
    from ..server import get_max_context_window

    ctx = get_max_context_window(model_id)
    return ctx if ctx is not None else 512


def _dedup_inputs(embedding_inputs: list[str]) -> tuple[list[str], list[int]]:
    """Deduplicate text inputs, returning unique texts and original->unique index map."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    mapping: list[int] = []
    for text in embedding_inputs:
        if text not in seen:
            seen[text] = len(unique)
            unique.append(text)
        mapping.append(seen[text])
    return unique, mapping


@router.post(
    "/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
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

    is_text_only = all(isinstance(item, str) for item in embedding_inputs)
    text_inputs: list[str] = embedding_inputs if is_text_only else []

    max_length = get_embedding_max_length(
        request.model, getattr(request, "max_length", None)
    )
    truncation = getattr(request, "truncation", True)

    dedup_mapping: list[int] | None = None
    unique_texts: list[str] | None = None
    if is_text_only and _EMBEDDING_DEDUP_ENABLED and len(text_inputs) > 1:
        unique_texts, dedup_mapping = _dedup_inputs(text_inputs)
        if len(unique_texts) < len(text_inputs):
            logger.info(
                "Embedding dedup: %d inputs -> %d unique (%d duplicates skipped)",
                len(text_inputs),
                len(unique_texts),
                len(text_inputs) - len(unique_texts),
            )
            text_inputs = unique_texts
        else:
            dedup_mapping = None

    start_time = time.perf_counter()
    try:
        engine = await get_embedding_engine(request.model)
        if is_text_only and len(text_inputs) > _EMBEDDING_BATCH_SIZE:
            all_embeddings: list[list[float]] = []
            total_tokens = 0
            for batch_start in range(0, len(text_inputs), _EMBEDDING_BATCH_SIZE):
                batch = text_inputs[batch_start : batch_start + _EMBEDDING_BATCH_SIZE]
                output = await engine.embed(
                    batch,
                    max_length=max_length,
                    truncation=truncation,
                )
                all_embeddings.extend(output.embeddings)
                total_tokens += output.total_tokens
            embeddings_list = all_embeddings
            dimensions = output.dimensions
            tokens = total_tokens
        else:
            inputs_to_embed = text_inputs if is_text_only else embedding_inputs
            output = await engine.embed(
                inputs_to_embed,
                max_length=max_length,
                truncation=truncation,
            )
            embeddings_list = output.embeddings
            dimensions = output.dimensions
            tokens = output.total_tokens
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Embedding: %d inputs, %d dims, %d tokens, max_length=%d in %.3fs",
        len(embedding_inputs),
        dimensions,
        tokens,
        max_length,
        elapsed,
    )
    get_server_metrics().record_request_complete(
        prompt_tokens=tokens,
        completion_tokens=0,
        cached_tokens=0,
        prefill_duration=elapsed,
        model_id=request.model,
    )

    try:
        from ..telemetry import emit

        emit.request(
            endpoint="/v1/embeddings",
            model_alias=request.model,
            stream=False,
            tool_call_used=False,
            prompt_tokens=tokens,
            completion_tokens=0,
            ttft_ms=elapsed * 1000.0,
            tps=0.0,
            status=200,
        )
    except Exception:
        logger.debug("telemetry request emit failed", exc_info=True)

    if dedup_mapping is not None:
        deduped_embeddings = [embeddings_list[idx] for idx in dedup_mapping]
        embeddings_list = deduped_embeddings

    data = []
    for i, embedding in enumerate(embeddings_list):
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
            prompt_tokens=tokens,
            total_tokens=tokens,
        ),
    )
