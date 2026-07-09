# SPDX-License-Identifier: Apache-2.0
"""
Anthropic-compatible API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/messages         - Anthropic Messages API
- POST /v1/count_tokens     - Token counting
"""

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..api.adapters.anthropic import AnthropicAdapter
from ..api.adapters.base import InternalResponse, StreamChunk
from ..api.anthropic_models import (
    MessagesRequest as AnthropicMessagesRequest,
)
from ..api.anthropic_models import (
    MessagesResponse as AnthropicMessagesResponse,
)
from ..api.anthropic_models import (
    TokenCountRequest,
    TokenCountResponse,
)
from ..api.anthropic_utils import (
    create_content_block_start_event,
    create_content_block_stop_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_stop_event,
    create_tool_name_delta_event,
    map_finish_reason_to_stop_reason,
)
from ..exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from ..pool import EnginePool
from ..request import SamplingParams
from ..server_metrics import record_llm_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["anthropic"])

_pool: Any = None
_adapter = AnthropicAdapter()


def set_anthropic_context(pool: EnginePool) -> None:
    """Inject engine pool into this module."""
    global _pool
    _pool = pool


def _extract_anthropic_text(msg: Any) -> str:
    """Extract plain text from an Anthropic message content."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                pt = part.get("type", "")
                if pt == "text":
                    parts.append(part.get("text", ""))
                elif pt == "tool_result":
                    result_content = part.get("content", "")
                    parts.append(f"Tool result: {result_content}")
                else:
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)
            elif hasattr(part, "text") and part.text:
                parts.append(part.text)
        return "\n".join(parts)
    return str(content) if content else ""


def _anthropic_to_messages(
    req: AnthropicMessagesRequest,
    tokenizer: Any | None = None,
) -> list[dict]:
    """Convert Anthropic MessagesRequest to engine messages.

    Uses convert_anthropic_to_internal() when a tokenizer is available
    so that tool_use/tool_result blocks are preserved rather than
    flattened to plain text.  Falls back to the legacy text-only
    path when no tokenizer is provided.
    """
    if tokenizer is not None:
        from .anthropic_utils import convert_anthropic_to_internal

        messages = convert_anthropic_to_internal(
            req,
            tokenizer=tokenizer,
            preserve_images=True,
            native_reasoning_content=True,
        )
        return messages

    # Legacy text-only fallback (no tokenizer available)
    messages = []
    if hasattr(req, "system") and req.system:
        if isinstance(req.system, str):
            messages.append({"role": "system", "content": req.system})
        elif isinstance(req.system, list):
            parts = []
            for part in req.system:
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text", "")
                    if txt:
                        parts.append(txt)
            if parts:
                messages.append({"role": "system", "content": "\n\n".join(parts)})

    for msg in req.messages or []:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = _extract_anthropic_text(msg)
        messages.append({"role": role, "content": content})
    return messages


def _build_sampling_params(req: AnthropicMessagesRequest) -> SamplingParams:
    """Convert Anthropic request to SamplingParams."""
    max_tokens = getattr(req, "max_tokens", 2048) or 2048
    temperature = (
        getattr(req, "temperature", 0.7) if hasattr(req, "temperature") else 0.7
    )
    top_p = getattr(req, "top_p", None) if hasattr(req, "top_p") else None
    if top_p is None:
        top_p = 0.9
    stop = getattr(req, "stop_sequences", None)
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop if isinstance(stop, list) else ([stop] if stop else None),
    )


async def _run_anthropic_messages(
    req: AnthropicMessagesRequest,
) -> AnthropicMessagesResponse:
    """Execute a non-streaming Anthropic messages request."""
    if _pool is None:
        raise HTTPException(450, "Engine pool not initialized")

    import time as _time

    from ..server import resolve_model_id

    _start = _time.perf_counter()
    model_name = resolve_model_id(req.model)
    adapter_path = getattr(req, "adapters", None)

    async def _release() -> None:
        await _pool.release_engine(model_name, adapter_path=adapter_path)

    engine = await _pool.get_engine(model_name, _lease=True, adapter_path=adapter_path)
    if engine is None:
        await _release()
        raise HTTPException(404, f"Model {model_name} not available")

    # Reject multimodal content on text-only models
    if not getattr(engine, "is_mllm", False):
        for msg in req.messages:
            content = getattr(msg, "content", "") if msg else None
            if isinstance(content, list):
                for part in content:
                    pt = (
                        part.get("type", "")
                        if isinstance(part, dict)
                        else getattr(part, "type", "")
                    )
                    if pt in (
                        "image_url",
                        "image",
                        "video",
                        "video_url",
                        "audio_url",
                        "audio",
                        "input_audio",
                    ):
                        await _release()
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Model '{model_name}' does not support "
                                "image, video, or audio inputs."
                            ),
                        )

    tokenizer = getattr(engine, "_tokenizer", None) or getattr(
        engine, "tokenizer", None
    )
    messages = _anthropic_to_messages(req, tokenizer=tokenizer)
    sampling = _build_sampling_params(req)
    request_id = f"msg-{uuid.uuid4().hex[:12]}"

    try:
        ct_kwargs = dict(getattr(req, "chat_template_kwargs", {}) or {})
        if getattr(req, "tools", None) and "enable_thinking" not in ct_kwargs:
            ct_kwargs["enable_thinking"] = False
        gen = await engine.chat(
            messages=messages,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            tools=getattr(req, "tools", None),
            stop=sampling.stop,
            chat_template_kwargs=ct_kwargs if ct_kwargs else None,
        )

        # Log completion (Ollama-style)
        logger.info(
            "Non-stream done: %s, finish=%s, prompt_tok=%d, "
            "completion_tok=%d, cached=%d",
            request_id,
            gen.finish_reason,
            gen.prompt_tokens,
            gen.completion_tokens,
            gen.cached_tokens,
        )

        # Convert to InternalResponse, then through adapter
        internal = InternalResponse(
            text=gen.text,
            finish_reason=gen.finish_reason,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            cached_tokens=gen.cached_tokens,
            tool_calls=gen.tool_calls,
            request_id=request_id,
            model=model_name,
        )
        record_llm_metrics(
            prompt_tokens=gen.prompt_tokens or 0,
            completion_tokens=gen.completion_tokens or 0,
            cached_tokens=gen.cached_tokens or 0,
            generation_duration=_time.perf_counter() - _start,
            model_id=model_name,
        )
        return _adapter.format_response(internal, req)
    except HTTPException:
        raise
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ModelLoadingError, ModelBusyError) as exc:
        logger.warning("Anthropic: model temporarily unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": str(exc), "type": "server_busy"}},
            headers={"Retry-After": "5"},
        ) from exc
    except InsufficientMemoryError as exc:
        logger.warning("Anthropic: insufficient memory: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": str(exc), "type": "resource_exhausted"}},
            headers={"Retry-After": "10"},
        ) from exc
    except ModelTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": str(exc), "type": "model_too_large"}},
        ) from exc
    except Exception as exc:
        err_msg = str(exc)
        if "Failed to process image" in err_msg or "Failed to process video" in err_msg:
            raise HTTPException(status_code=400, detail=err_msg)
        logger.exception(
            "Anthropic messages failed for %s: %s(%s)",
            request_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    finally:
        await _release()


async def _stream_anthropic_generator(
    req: AnthropicMessagesRequest,
    engine: Any,
    model_name: str,
    adapter_path: str | None,
) -> AsyncIterator[str]:
    """Generate SSE events for a streaming Anthropic messages request.

    Engine must be resolved BEFORE calling this generator (by the route
    handler) so that ModelNotFoundError / ModelLoadingError become proper
    HTTP status codes instead of unhandled ASGI 500s.
    """
    import time as _time

    _start = _time.perf_counter()

    async def _release() -> None:
        await _pool.release_engine(model_name, adapter_path=adapter_path)

    tokenizer = getattr(engine, "_tokenizer", None) or getattr(
        engine, "tokenizer", None
    )
    messages = _anthropic_to_messages(req, tokenizer=tokenizer)
    sampling = _build_sampling_params(req)
    request_id = f"msg-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Stream start: %s, max_tokens=%d, prompt=%r",
        request_id,
        sampling.max_tokens,
        messages[-1].get("content", "")[:120] if messages else "",
    )
    try:
        # Send message_start
        yield _adapter.format_stream_chunk(
            StreamChunk(text="", is_first=True),
            req,
        )

        ct_kwargs_stream = dict(getattr(req, "chat_template_kwargs", {}) or {})
        if getattr(req, "tools", None) and "enable_thinking" not in ct_kwargs_stream:
            ct_kwargs_stream["enable_thinking"] = False
        async for gen in engine.stream_chat(
            messages=messages,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            tools=getattr(req, "tools", None),
            stop=sampling.stop,
            chat_template_kwargs=ct_kwargs_stream if ct_kwargs_stream else None,
        ):
            if gen.new_text:
                chunk = StreamChunk(
                    text=gen.new_text,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    cached_tokens=gen.cached_tokens,
                )
                yield _adapter.format_stream_chunk(chunk, req)

            if gen.finished:
                # Log completion (Ollama-style)
                logger.info(
                    "Stream done: %s, finish=%s, prompt_tok=%d, "
                    "completion_tok=%d, cached=%d",
                    request_id,
                    gen.finish_reason,
                    gen.prompt_tokens,
                    gen.completion_tokens,
                    gen.cached_tokens,
                )

                # Emit tool_use content blocks if tool calls present
                if gen.tool_calls:
                    for i, tc in enumerate(gen.tool_calls):
                        tc_id = tc.get("id", f"call_{i}")
                        tc_name = tc.get("function", {}).get("name", "")
                        tc_args = tc.get("function", {}).get("arguments", "{}")
                        block_index = 1 + i
                        yield create_content_block_start_event(
                            block_index,
                            "tool_use",
                            id=tc_id,
                            name=tc_name,
                        )
                        yield create_tool_name_delta_event(block_index, tc_name)
                        yield create_input_json_delta_event(block_index, tc_args)
                        yield create_content_block_stop_event(block_index)

                # Send message_delta and message_stop
                stop_reason = map_finish_reason_to_stop_reason(
                    gen.finish_reason,
                    bool(gen.tool_calls),
                )
                yield create_message_delta_event(
                    stop_reason=stop_reason,
                    output_tokens=gen.completion_tokens,
                    input_tokens=gen.prompt_tokens,
                )
                yield create_message_stop_event()

                record_llm_metrics(
                    prompt_tokens=gen.prompt_tokens or 0,
                    completion_tokens=gen.completion_tokens or 0,
                    cached_tokens=gen.cached_tokens or 0,
                    generation_duration=_time.perf_counter() - _start,
                    model_id=model_name,
                )

    except ModelNotFoundError as exc:
        yield f'event: error\ndata: {{"error": {str(exc)!r}, "status": 404}}\n\n'
    except (ModelLoadingError, ModelBusyError) as exc:
        logger.warning("Anthropic stream: model temporarily unavailable: %s", exc)
        yield f'event: error\ndata: {{"error": {{"message": {str(exc)!r}, "type": "server_busy"}}, "status": 503}}\n\n'
    except InsufficientMemoryError as exc:
        logger.warning("Anthropic stream: insufficient memory: %s", exc)
        yield f'event: error\ndata: {{"error": {{"message": {str(exc)!r}, "type": "resource_exhausted"}}, "status": 503}}\n\n'
    except ModelTooLargeError as exc:
        yield f'event: error\ndata: {{"error": {{"message": {str(exc)!r}, "type": "model_too_large"}}, "status": 413}}\n\n'
    except Exception as exc:
        err_msg = str(exc)
        logger.error(
            "Stream ERROR %s: %s\n  type=%s\n  model=%s",
            request_id,
            err_msg,
            type(exc).__name__,
            model_name,
            exc_info=True,
        )
        if "Failed to process image" in err_msg or "Failed to process video" in err_msg:
            yield f'event: error\ndata: {{"error": {err_msg!r}, "status": 400}}\n\n'
        else:
            yield f'event: error\ndata: {{"error": {f"{type(exc).__name__}: {exc}"!r}}}\n\n'
    finally:
        await _release()


@router.post("/messages")
async def anthropic_messages(request: AnthropicMessagesRequest) -> Any:
    """Handle Anthropic Messages API requests."""
    # Log request entry (Ollama-style)
    prompt_preview = ""
    if request.messages:
        first = request.messages[0]
        msg_content = getattr(first, "content", "") if first else ""
        if isinstance(msg_content, str):
            prompt_preview = msg_content[:120]
        elif isinstance(msg_content, list):
            for p in msg_content:
                if isinstance(p, dict) and p.get("type") == "text":
                    prompt_preview = p.get("text", "")[:120]
                    break
    logger.info(
        "Anthropic /messages: model=%s, stream=%s, max_tokens=%d, "
        "temp=%s, top_p=%s, prompt=%r",
        request.model,
        request.stream,
        getattr(request, "max_tokens", 2048) or 2048,
        getattr(request, "temperature", None),
        getattr(request, "top_p", None),
        prompt_preview,
    )
    try:
        if request.stream:
            # Resolve engine BEFORE creating StreamingResponse so that
            # ModelNotFoundError / ModelLoadingError become proper HTTP
            # 404/503 instead of unhandled ASGI 500 after stream starts.
            from ..server import resolve_model_id

            model_name = resolve_model_id(request.model)
            adapter_path = getattr(request, "adapters", None)
            engine = await _pool.get_engine(
                model_name, _lease=True, adapter_path=adapter_path
            )
            if engine is None:
                await _pool.release_engine(model_name, adapter_path=adapter_path)
                raise HTTPException(404, f"Model {model_name} not available")

            # Reject multimodal content on text-only models
            if not getattr(engine, "is_mllm", False):
                for msg in request.messages:
                    content = getattr(msg, "content", "") if msg else None
                    if isinstance(content, list):
                        for part in content:
                            pt = (
                                part.get("type", "")
                                if isinstance(part, dict)
                                else getattr(part, "type", "")
                            )
                            if pt in (
                                "image_url",
                                "image",
                                "video",
                                "video_url",
                                "audio_url",
                                "audio",
                                "input_audio",
                            ):
                                await _pool.release_engine(
                                    model_name, adapter_path=adapter_path
                                )
                                raise HTTPException(
                                    status_code=400,
                                    detail=(
                                        f"Model '{model_name}' does not support "
                                        "image, video, or audio inputs."
                                    ),
                                )

            return StreamingResponse(
                _stream_anthropic_generator(
                    request, engine, model_name, adapter_path
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await _run_anthropic_messages(request)
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
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": str(exc), "type": "resource_exhausted"}},
            headers={"Retry-After": "10"},
        ) from exc
    except ModelTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": str(exc), "type": "model_too_large"}},
        ) from exc
    except Exception as exc:
        logger.exception("Anthropic messages failed: %s(%s)", type(exc).__name__, exc)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")


@router.post("/count_tokens")
async def count_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Count tokens for a given input."""
    # Simple token count estimate (~4 chars per token)
    text = ""
    if hasattr(request, "messages"):
        for msg in request.messages or []:
            if hasattr(msg, "content"):
                text += str(msg.content) or ""
    elif hasattr(request, "input"):
        text = str(request.input)
    token_count = max(1, len(text) // 4)
    return TokenCountResponse(input_tokens=token_count)
