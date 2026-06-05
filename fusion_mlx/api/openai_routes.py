# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/chat/completions   - Chat completion (streaming + non-streaming)
- POST /v1/completions         - Legacy text completion
- GET   /v1/models              - List available models
"""

import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..api.adapters.base import InternalResponse, StreamChunk
from ..api.adapters.openai import OpenAIAdapter
from ..api.openai_models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ModelInfo,
    ModelsResponse,
    PromptTokensDetails,
    Usage,
)
from ..engines.base import GenerationOutput
from ..pool import EnginePool
from ..router import RequestRouter
from ..request import SamplingParams

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

# Set by server.py during startup
_pool: Any = None
_request_router: Any = None
_adapter = OpenAIAdapter()


def set_openai_context(pool: EnginePool, req_router: RequestRouter) -> None:
    """Inject engine pool and request router into this module."""
    global _pool, _request_router
    _pool = pool
    _request_router = req_router


def _extract_text(msg: Any) -> str:
    """Extract plain text from a message's content field."""
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
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
                elif part.get("type") == "audio_url":
                    parts.append("[audio]")
                elif part.get("type") in ("video", "video_url"):
                    parts.append("[video]")
            elif hasattr(part, "text") and part.text:
                parts.append(part.text)
        return "\n".join(parts)
    return str(content) if content else ""


def _build_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    """Convert ChatCompletionRequest to SamplingParams."""
    return SamplingParams(
        max_tokens=req.max_tokens or 2048,
        temperature=req.temperature if req.temperature is not None else 0.7,
        top_p=req.top_p if req.top_p is not None else 0.9,
        top_k=getattr(req, "top_k", 0) or 0,
        min_p=getattr(req, "min_p", 0.0) or 0.0,
        presence_penalty=req.presence_penalty if req.presence_penalty is not None else 0.0,
        frequency_penalty=req.frequency_penalty if req.frequency_penalty is not None else 0.0,
        stop=req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None),
        stop_token_ids=getattr(req, "stop_token_ids", None),
    )


def _gen_to_internal(gen: GenerationOutput, model: str, request_id: str) -> InternalResponse:
    """Convert GenerationOutput to InternalResponse for the adapter."""
    return InternalResponse(
        text=gen.text,
        finish_reason=gen.finish_reason,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        cached_tokens=gen.cached_tokens,
        tool_calls=gen.tool_calls,
        request_id=request_id,
        model=model,
    )


async def _run_chat(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Execute a non-streaming chat completion."""
    if _pool is None:
        raise HTTPException(450, "Engine pool not initialized")

    from ..server import resolve_model_id
    model_name = resolve_model_id(request.model)
    engine = await _pool.get_engine(model_name)
    if engine is None:
        raise HTTPException(404, f"Model {model_name} not available")

    messages = [
        {"role": m.role, "content": _extract_text(m)} for m in request.messages
    ]
    sampling = _build_sampling_params(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        gen = await engine.chat(
            messages=messages,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            repetition_penalty=getattr(sampling, "repetition_penalty", 1.0),
            presence_penalty=sampling.presence_penalty,
            tools=request.tools,
            stop=sampling.stop,
        )
        internal = _gen_to_internal(gen, model_name, request_id)
        return _adapter.format_response(internal, request)
    except Exception as exc:
        logger.exception("Non-streaming chat failed for %s", request_id)
        raise HTTPException(500, str(exc))


async def _stream_chat_generator(request: ChatCompletionRequest) -> AsyncIterator[str]:
    """Generate SSE events for a streaming chat completion."""
    if _pool is None:
        raise HTTPException(450, "Engine pool not initialized")

    from ..server import resolve_model_id
    model_name = resolve_model_id(request.model)
    engine = await _pool.get_engine(model_name)
    if engine is None:
        raise HTTPException(404, f"Model {model_name} not available")

    messages = [
        {"role": m.role, "content": _extract_text(m)} for m in request.messages
    ]
    sampling = _build_sampling_params(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        # First chunk with role
        first_chunk = StreamChunk(
            text="", is_first=True,
            prompt_tokens=0, completion_tokens=0, cached_tokens=0,
        )
        yield _adapter.format_stream_chunk(first_chunk, request)

        accumulated = ""
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        async for gen in engine.stream_chat(
            messages=messages,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            repetition_penalty=getattr(sampling, "repetition_penalty", 1.0),
            presence_penalty=sampling.presence_penalty,
            tools=request.tools,
            stop=sampling.stop,
        ):
            if gen.new_text:
                accumulated += gen.new_text
                chunk = StreamChunk(
                    text=gen.new_text,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    cached_tokens=gen.cached_tokens,
                )
                yield _adapter.format_stream_chunk(chunk, request)
                prompt_tokens = gen.prompt_tokens or prompt_tokens
                completion_tokens = gen.completion_tokens or completion_tokens
                cached_tokens = gen.cached_tokens or cached_tokens

            if gen.finished:
                finish_reason = gen.finish_reason

        # Final chunk with finish_reason
        last_chunk = StreamChunk(
            text="", is_last=True, finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        yield _adapter.format_stream_chunk(last_chunk, request)
        yield _adapter.format_stream_end(request)

    except Exception as exc:
        logger.exception("Streaming chat failed for %s", request_id)
        yield f"data: {{\"error\": {str(exc)!r}}}\n\n"


async def _stream_chat(request: ChatCompletionRequest) -> StreamingResponse:
    """Execute a streaming chat completion."""
    return StreamingResponse(
        _stream_chat_generator(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """Handle OpenAI-compatible chat completion requests."""
    try:
        if request.stream:
            return await _stream_chat(request)
        else:
            return await _run_chat(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat completion failed")
        raise HTTPException(500, str(exc))


@router.post("/completions")
async def completions(request: CompletionRequest) -> Any:
    """Handle legacy text completion requests."""
    try:
        # Convert completion to chat format
        chat_req = ChatCompletionRequest(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            stop=request.stop,
        )
        if request.stream:
            return await _stream_chat(chat_req)
        return await _run_chat(chat_req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Completion failed")
        raise HTTPException(500, str(exc))


@router.get("/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    if _pool is None:
        return ModelsResponse(data=[])

    try:
        model_ids = _pool.list_models() if hasattr(_pool, "list_models") else []
    except Exception:
        model_ids = []

    models = [
        ModelInfo(
            id=mid,
            object="model",
            created=int(time.time()),
            owned_by="local",
        )
        for mid in model_ids
    ]
    return ModelsResponse(data=models)
