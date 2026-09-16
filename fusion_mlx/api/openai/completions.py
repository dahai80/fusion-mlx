# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible legacy text completion and model listing routes."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import Depends, HTTPException, Request
from starlette.responses import JSONResponse, StreamingResponse

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
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
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


def _chat_dict_to_completion_dict(chat_dict: dict, model: str) -> dict:
    """Map a chat-completion response dict to the legacy text-completion shape.

    G-1 (#0912 audit): ``/v1/completions`` wrapped the prompt into a chat
    message and returned the chat shape (``choices[0].message.content``)
    directly instead of ``choices[0].text``. OpenAI SDK clients expecting
    the text-completion envelope read ``choices[0].text`` and got ``None``.
    This is the single message→text mapping the audit calls for.
    """
    choices = chat_dict.get("choices") or []
    text = ""
    finish_reason = None
    logprobs = None
    if choices:
        ch = choices[0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        finish_reason = ch.get("finish_reason")
        logprobs = ch.get("logprobs")
    comp_choices = [
        CompletionChoice(
            index=0,
            text=text,
            finish_reason=finish_reason,
        ).model_dump()
    ]
    comp_choice = comp_choices[0]
    if logprobs is not None:
        comp_choice["logprobs"] = logprobs
    else:
        comp_choice["logprobs"] = None
    out = {
        "id": chat_dict.get("id") or f"cmpl-comp-{int(time.time())}",
        "object": "text_completion",
        "created": chat_dict.get("created") or int(time.time()),
        "model": model,
        "choices": [comp_choice],
    }
    if chat_dict.get("usage"):
        out["usage"] = chat_dict["usage"]
    return out


def _to_completion_result(result: Any, model: str):
    """Convert a chat-path result to the text-completion shape.

    Handles both the plain ``ChatCompletionResponse`` return and the
    ``JSONResponse`` wrapper used when context-budget headers are attached.
    """
    if isinstance(result, JSONResponse):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            chat_dict = json.loads(body)
        else:
            chat_dict = json.loads(body)
        comp = _chat_dict_to_completion_dict(chat_dict, model)
        logger.debug("completions non-stream json-response remap model=%s", model)
        # Drop content-length from copied headers — the remapped completion
        # body is a different length than the original chat body, so the stale
        # content-length causes "Too little data for declared Content-Length"
        # (peer closed connection mid-response). Starlette recomputes it.
        new_headers = {
            k: v for k, v in result.headers.items() if k.lower() != "content-length"
        }
        return JSONResponse(content=comp, headers=new_headers)
    if hasattr(result, "model_dump"):
        chat_dict = result.model_dump()
        comp = _chat_dict_to_completion_dict(chat_dict, model)
        logger.debug("completions non-stream remap model=%s", model)
        return CompletionResponse(**comp)
    return result


async def _rewrite_chat_stream_to_completion(body_iter: Any, model: str):
    """Rewrite chat-shaped SSE chunks to text-completion shape.

    Chat chunk: ``choices[0].delta.content`` / ``object: chat.completion.chunk``
    Legacy chunk: ``choices[0].text`` / ``object: text_completion``
    """
    async for raw in body_iter:
        if isinstance(raw, (bytes, bytearray)):
            chunk_str = raw.decode("utf-8", errors="replace")
        else:
            chunk_str = raw
        # Pass through keep-alive / non-data lines untouched.
        for line in chunk_str.splitlines(keepends=True):
            if not line.startswith("data: "):
                yield line
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                yield line
                continue
            try:
                obj = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                yield line
                continue
            choices = obj.get("choices") or []
            text = None
            finish_reason = None
            logprobs = None
            if choices:
                ch = choices[0]
                delta = ch.get("delta") or {}
                text = delta.get("content")
                finish_reason = ch.get("finish_reason")
                logprobs = ch.get("logprobs")
            comp_choice = {
                "index": 0,
                "text": text if text is not None else "",
                "finish_reason": finish_reason,
                "logprobs": logprobs,
            }
            comp = {
                "id": obj.get("id") or f"cmpl-comp-{int(time.time())}",
                "object": "text_completion",
                "created": obj.get("created") or int(time.time()),
                "model": model,
                "choices": [comp_choice],
            }
            if obj.get("usage"):
                comp["usage"] = obj["usage"]
            yield f"data: {json.dumps(comp, ensure_ascii=False)}\n\n"


def _wrap_stream_response(sr: StreamingResponse, model: str) -> StreamingResponse:
    """Wrap a chat StreamingResponse so its body emits completion SSE."""
    sr.body_iterator = _rewrite_chat_stream_to_completion(sr.body_iterator, model)
    return sr


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
            sr = await _stream_chat(chat_req, _skip_cap_check=True, principal=principal)
            # G-1 (#0912): remap chat-shaped SSE → text-completion shape so
            # ``choices[0].delta.content`` surfaces as ``choices[0].text``.
            if isinstance(sr, StreamingResponse):
                return _wrap_stream_response(sr, request.model)
            return sr
        await acquire_request_slot()
        try:
            result = await _run_chat(
                chat_req, _skip_cap_check=True, principal=principal
            )
        finally:
            release_request_slot()
        # G-1 (#0912): remap chat.completion shape → text_completion shape
        # (choices[0].message.content → choices[0].text).
        return _to_completion_result(result, request.model)
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
