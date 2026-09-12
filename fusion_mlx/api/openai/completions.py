# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible legacy text completion and model listing routes."""

from __future__ import annotations

import time
from typing import Any

from fastapi import Depends, HTTPException, Request

from ...exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from ...middleware.auth import check_rate_limit, request_principal, verify_api_key
from .._concurrency import acquire_request_slot, release_request_slot
from .._guards import build_model_error_response
from ..openai_models import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelInfo,
    ModelsResponse,
)
from . import _common
from ._common import (
    _get_settings,
    _resolve_capabilities,
    _resolve_modality,
    logger,
    router,
)
from .chat import _run_chat, _stream_chat


@router.post("/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
) -> Any:
    """Handle legacy text completion requests.

    FC-4 (#0907 audit): the ``prompt`` is wrapped into a single user message
    and run through the chat path (``apply_chat_template``), so raw-prompt
    clients (Continue/Cody expecting bare model continuation) receive
    chat-templated output with role/special-token scaffolding. A true
    raw-prompt passthrough that bypasses the chat template needs an engine
    sampling-layer change tracked separately; until then this endpoint is
    documented as chat-wrapped only.
    """
    # FIM suffix guard: no MLX engine implements fill-in-the-middle yet.
    # A non-empty suffix would be silently dropped (we only forward the
    # prompt), producing wrong completions on code-completion clients
    # (Continue, Cody). Fail visibly with 400 so the client can fall back.
    # The empty-string case is harmless — defensive clients always send it.
    if request.suffix:
        logger.info("Rejecting /v1/completions suffix (FIM unsupported)")
        raise HTTPException(
            status_code=400,
            detail="suffix is not supported: this server does not implement "
            "fill-in-the-middle completion",
        )
    # #226 IDOR scope: bind recorded session stats to the authenticated caller.
    principal = request_principal(http_request)
    try:
        # Convert completion to chat format
        chat_req = ChatCompletionRequest(
            model=request.model,
            adapters=request.adapters,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            stop=request.stop,
        )
        if request.stream:
            return await _stream_chat(
                chat_req, _skip_cap_check=True, principal=principal
            )
        await acquire_request_slot()
        try:
            return await _run_chat(chat_req, _skip_cap_check=True, principal=principal)
        finally:
            release_request_slot()
    except HTTPException:
        raise
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ModelLoadingError, ModelBusyError) as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": str(exc), "type": "server_busy"}},
            headers={"Retry-After": "5"},
        ) from exc
    except InsufficientMemoryError as exc:
        raise build_model_error_response(exc, adapter="openai") from exc
    except ModelTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": str(exc), "type": "model_too_large"}},
        ) from exc
    except Exception as exc:
        logger.exception("Completion failed: %s(%s)", type(exc).__name__, exc)
        raise HTTPException(500, "Internal server error")


@router.get("/models")
async def list_models(
    _auth: bool = Depends(verify_api_key),
) -> ModelsResponse:
    """List available models."""
    if _common._pool is None:
        return ModelsResponse(data=[])

    try:
        model_ids = (
            _common._pool.list_models()
            if _common._pool is not None and hasattr(_common._pool, "list_models")
            else []
        )
    except Exception:
        model_ids = []

    models = [
        ModelInfo(
            id=mid,
            object="model",
            created=int(time.time()),
            owned_by="local",
            modality=_resolve_modality(mid),
            capabilities=_resolve_capabilities(mid),
        )
        for mid in model_ids
    ]

    from ..markitdown import MARKITDOWN_MODEL_ID, markitdown_model_visible

    settings = _get_settings()
    global_settings = getattr(settings, "global_settings", None) if settings else None
    if markitdown_model_visible(global_settings):
        models.append(
            ModelInfo(
                id=MARKITDOWN_MODEL_ID,
                object="model",
                created=int(time.time()),
                owned_by="system",
            )
        )

    return ModelsResponse(data=models)
