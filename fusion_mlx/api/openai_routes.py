# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/chat/completions   - Chat completion (streaming + non-streaming)
- POST /v1/completions         - Legacy text completion
- GET   /v1/models              - List available models
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..api.adapters.base import InternalResponse, StreamChunk
from ..api.adapters.openai import OpenAIAdapter
from ..api.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ModelInfo,
    ModelsResponse,
)
from ..api.thinking import ThinkingParser
from ..engines.base import GenerationOutput
from ..pool import EnginePool
from ..request import SamplingParams
from ..router import RequestRouter

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
        presence_penalty=(
            req.presence_penalty if req.presence_penalty is not None else 0.0
        ),
        frequency_penalty=(
            req.frequency_penalty if req.frequency_penalty is not None else 0.0
        ),
        stop=(
            req.stop
            if isinstance(req.stop, list)
            else ([req.stop] if req.stop else None)
        ),
        stop_token_ids=getattr(req, "stop_token_ids", None),
    )


def _gen_to_internal(
    gen: GenerationOutput, model: str, request_id: str
) -> InternalResponse:
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
    engine = await _pool.get_engine(model_name, _lease=True)
    if engine is None:
        await _pool.release_engine(model_name)
        raise HTTPException(404, f"Model {model_name} not available")

    # Reject multimodal content on text-only models
    if not getattr(engine, "is_mllm", False):
        for msg in request.messages:
            content = getattr(msg, "content", "")
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
                        await _pool.release_engine(model_name)
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Model '{model_name}' does not support "
                                "image, video, or audio inputs."
                            ),
                        )

    messages = [{"role": m.role, "content": _extract_text(m)} for m in request.messages]
    sampling = _build_sampling_params(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        ct_kwargs = dict(getattr(request, "chat_template_kwargs", {}) or {})
        if request.tools and "enable_thinking" not in ct_kwargs:
            ct_kwargs["enable_thinking"] = False
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
            chat_template_kwargs=ct_kwargs if ct_kwargs else None,
        )
        # Honor parallel_tool_calls=false by capping to 1 call
        tool_calls = gen.tool_calls
        if (
            tool_calls
            and len(tool_calls) > 1
            and getattr(request, "parallel_tool_calls", None) is False
        ):
            tool_calls = tool_calls[:1]
        internal = _gen_to_internal(gen, model_name, request_id)
        if tool_calls is not None:
            internal.tool_calls = tool_calls
        return _adapter.format_response(internal, request)
    except HTTPException:
        raise
    except Exception as exc:
        err_msg = str(exc)
        # VLM image/video fetch failures -> 400
        if "Failed to process image" in err_msg or "Failed to process video" in err_msg:
            raise HTTPException(status_code=400, detail=err_msg)
        logger.exception("Non-streaming chat failed for %s", request_id)
        raise HTTPException(500, str(exc))
    finally:
        await _pool.release_engine(model_name)


async def _stream_chat_generator(request: ChatCompletionRequest) -> AsyncIterator[str]:
    """Generate SSE events for a streaming chat completion."""
    if _pool is None:
        raise HTTPException(450, "Engine pool not initialized")

    from ..server import resolve_model_id

    model_name = resolve_model_id(request.model)
    engine = await _pool.get_engine(model_name, _lease=True)
    if engine is None:
        await _pool.release_engine(model_name)
        raise HTTPException(404, f"Model {model_name} not available")

    # Reject multimodal content on text-only models
    if not getattr(engine, "is_mllm", False):
        for msg in request.messages:
            content = getattr(msg, "content", "")
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
                        await _pool.release_engine(model_name)
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Model '{model_name}' does not support "
                                "image, video, or audio inputs."
                            ),
                        )

    messages = [{"role": m.role, "content": _extract_text(m)} for m in request.messages]
    sampling = _build_sampling_params(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Create StreamingJSONEncoder for fast-path SSE encoding (avoids per-token
    # Pydantic model construction + model_dump_json overhead)
    from .streaming import StreamingJSONEncoder

    encoder = StreamingJSONEncoder(
        response_id=request_id,
        model=request.model,
        object_type="chat.completion.chunk",
    )

    try:
        # First chunk with role
        first_chunk = StreamChunk(
            text="",
            is_first=True,
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
        )
        yield _adapter.format_stream_chunk(first_chunk, request)

        accumulated = ""
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        # Streaming thinking parser: splits <think...</think > blocks into
        # reasoning_content vs content so OpenAI clients can tell thinking
        # from the real answer (issue #21). No-op for tag-free text.
        parser = ThinkingParser()

        ct_kwargs_stream = dict(getattr(request, "chat_template_kwargs", {}) or {})
        if request.tools and "enable_thinking" not in ct_kwargs_stream:
            ct_kwargs_stream["enable_thinking"] = False
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
            chat_template_kwargs=ct_kwargs_stream if ct_kwargs_stream else None,
        ):
            if gen.new_text:
                accumulated += gen.new_text
                thinking_delta, content_delta = parser.feed(gen.new_text)
                if content_delta:
                    chunk = StreamChunk(
                        text=content_delta,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        cached_tokens=gen.cached_tokens,
                    )
                    yield _adapter.format_stream_chunk(chunk, request, encoder=encoder)
                if thinking_delta:
                    rchunk = StreamChunk(
                        text="",
                        reasoning_content=thinking_delta,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        cached_tokens=gen.cached_tokens,
                    )
                    yield _adapter.format_stream_chunk(rchunk, request, encoder=encoder)
                prompt_tokens = gen.prompt_tokens or prompt_tokens
                completion_tokens = gen.completion_tokens or completion_tokens
                cached_tokens = gen.cached_tokens or cached_tokens

            if gen.finished:
                finish_reason = gen.finish_reason or "stop"
                # Emit tool call deltas if present
                if gen.tool_calls:
                    finish_reason = "tool_calls"
                    for idx, tc in enumerate(gen.tool_calls):
                        tc_chunk = StreamChunk(
                            tool_call_delta=[
                                {
                                    "index": idx,
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": tc.get("function", {}).get(
                                            "arguments", "{}"
                                        ),
                                    },
                                }
                            ],
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cached_tokens=cached_tokens,
                        )
                        yield _adapter.format_stream_chunk(
                            tc_chunk, request, encoder=encoder
                        )

        # Flush any buffered thinking/content from the parser (partial tags,
        # malformed recovery). See issue #21.
        t_tail, c_tail = parser.finish()
        if c_tail:
            cchunk = StreamChunk(
                text=c_tail,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
            yield _adapter.format_stream_chunk(cchunk, request, encoder=encoder)
        if t_tail:
            tchunk = StreamChunk(
                text="",
                reasoning_content=t_tail,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
            yield _adapter.format_stream_chunk(tchunk, request, encoder=encoder)

        # Final chunk with finish_reason
        last_chunk = StreamChunk(
            text="",
            is_last=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        yield _adapter.format_stream_chunk(last_chunk, request, encoder=encoder)
        yield _adapter.format_stream_end(request)

    except asyncio.CancelledError:
        logger.info("Client disconnected during streaming: %s", request_id)
        if engine:
            try:
                asyncio.create_task(engine.abort_request(request_id))
            except Exception:
                pass
        raise
    except Exception as exc:
        err_msg = str(exc)
        # VLM image/video fetch failures -> 400
        if "Failed to process image" in err_msg or "Failed to process video" in err_msg:
            yield f'data: {{"error": {{"message": {err_msg!r}, "status": 400}}}}\n\n'
        else:
            logger.exception("Streaming chat failed for %s", request_id)
            yield f'data: {{"error": {{"message": {err_msg!r}}}}}\n\n'
    finally:
        await _pool.release_engine(model_name)


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
    # Log request entry (Ollama-style)
    prompt_preview = ""
    if request.messages:
        last_msg = request.messages[-1]
        c = getattr(last_msg, "content", "") if last_msg else ""
        prompt_preview = str(c)[:120] if c else ""
    logger.info(
        "OpenAI /chat: model=%s, stream=%s, max_tokens=%s, " "temp=%s, prompt=%r",
        request.model,
        request.stream,
        getattr(request, "max_tokens", None),
        getattr(request, "temperature", 0.7) or 0.7,
        getattr(request, "temperature", None),
    )
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
