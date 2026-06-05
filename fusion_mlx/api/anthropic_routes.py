# SPDX-License-Identifier: Apache-2.0
"""
Anthropic-compatible API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/messages         - Anthropic Messages API
- POST /v1/count_tokens     - Token counting
"""

import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..api.adapters.anthropic import AnthropicAdapter
from ..api.adapters.base import InternalResponse, StreamChunk
from ..api.anthropic_models import (
    MessagesRequest as AnthropicMessagesRequest,
    MessagesResponse as AnthropicMessagesResponse,
    TokenCountRequest,
    TokenCountResponse,
)
from ..engines.base import GenerationOutput
from ..pool import EnginePool
from ..request import SamplingParams

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
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif hasattr(part, "text") and part.text:
                parts.append(part.text)
        return "\n".join(parts)
    return str(content) if content else ""


def _anthropic_to_messages(req: AnthropicMessagesRequest) -> list[dict]:
    """Convert Anthropic MessagesRequest to engine messages."""
    messages = []
    if hasattr(req, "system") and req.system:
        if isinstance(req.system, str):
            messages.append({"role": "system", "content": req.system})
        elif isinstance(req.system, list):
            for part in req.system:
                if isinstance(part, dict) and part.get("type") == "text":
                    messages.append({"role": "system", "content": part.get("text", "")})

    for msg in req.messages or []:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = _extract_anthropic_text(msg)
        messages.append({"role": role, "content": content})
    return messages


def _build_sampling_params(req: AnthropicMessagesRequest) -> SamplingParams:
    """Convert Anthropic request to SamplingParams."""
    max_tokens = getattr(req, "max_tokens", 2048) or 2048
    temperature = getattr(req, "temperature", 0.7) if hasattr(req, "temperature") else 0.7
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


async def _run_anthropic_messages(req: AnthropicMessagesRequest) -> AnthropicMessagesResponse:
    """Execute a non-streaming Anthropic messages request."""
    if _pool is None:
        raise HTTPException(450, "Engine pool not initialized")

    from ..server import resolve_model_id
    model_name = resolve_model_id(req.model)
    engine = await _pool.get_engine(model_name)
    if engine is None:
        raise HTTPException(404, f"Model {model_name} not available")

    messages = _anthropic_to_messages(req)
    sampling = _build_sampling_params(req)
    request_id = f"msg-{uuid.uuid4().hex[:12]}"

    try:
        gen = await engine.chat(
            messages=messages,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            tools=getattr(req, "tools", None),
            stop=sampling.stop,
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
        return _adapter.format_response(internal, req)
    except Exception as exc:
        logger.exception("Anthropic messages failed for %s", request_id)
        raise HTTPException(500, str(exc))


async def _stream_anthropic_generator(req: AnthropicMessagesRequest) -> AsyncIterator[str]:
    """Generate SSE events for a streaming Anthropic messages request."""
    if _pool is None:
        raise HTTPException(450, "Engine pool not initialized")

    from ..server import resolve_model_id
    model_name = resolve_model_id(req.model)
    engine = await _pool.get_engine(model_name)
    if engine is None:
        raise HTTPException(404, f"Model {model_name} not available")

    messages = _anthropic_to_messages(req)
    sampling = _build_sampling_params(req)
    request_id = f"msg-{uuid.uuid4().hex[:12]}"

    try:
        # Send message_start
        yield _adapter.format_stream_chunk(
            StreamChunk(text="", is_first=True),
            req,
        )

        async for gen in engine.stream_chat(
            messages=messages,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            tools=getattr(req, "tools", None),
            stop=sampling.stop,
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
                # Send message_delta and message_stop
                last = StreamChunk(
                    text="", is_last=True,
                    finish_reason=gen.finish_reason,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    cached_tokens=gen.cached_tokens,
                )
                yield _adapter.format_stream_chunk(last, req)
                yield _adapter.format_stream_end(req)

    except Exception as exc:
        logger.exception("Streaming Anthropic messages failed for %s", request_id)
        yield f"event: error\ndata: {{\"error\": {str(exc)!r}}}\n\n"


@router.post("/messages")
async def anthropic_messages(request: AnthropicMessagesRequest) -> Any:
    """Handle Anthropic Messages API requests."""
    try:
        if request.stream:
            return StreamingResponse(
                _stream_anthropic_generator(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
        return await _run_anthropic_messages(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Anthropic messages failed")
        raise HTTPException(500, str(exc))


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
