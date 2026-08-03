# SPDX-License-Identifier: Apache-2.0
"""Ollama-compatible API routes - /api/generate, /api/chat, /api/tags.

Translates Ollama API requests into internal OpenAI chat completion calls
so tools like Open WebUI, LibreChat, and other Ollama clients can use
fusion-mlx as a drop-in replacement.

Endpoints:
  POST /api/generate  - text generation (prompt-based)
  POST /api/chat      - chat with message array
  GET  /api/tags      - list local models
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ollama"])

_pool: Any = None


def set_ollama_context(pool: Any) -> None:
    global _pool
    _pool = pool


# =============================================================================
# Ollama request / response models
# =============================================================================


class OllamaGenerateRequest(BaseModel):
    model: str = "default"
    prompt: str = ""
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = True
    raw: bool = False
    format: str | dict | None = None
    images: list[str] | None = None
    options: dict | None = None
    keep_alive: str | None = None


class OllamaChatMessage(BaseModel):
    role: str = "user"
    content: str = ""
    images: list[str] | None = None


class OllamaChatRequest(BaseModel):
    model: str = "default"
    messages: list[OllamaChatMessage]
    stream: bool = True
    format: str | dict | None = None
    options: dict | None = None
    template: str | None = None
    keep_alive: str | None = None


# =============================================================================
# Helpers
# =============================================================================


def _options_to_params(options: dict | None) -> dict:
    """Map Ollama options dict to OpenAI sampling parameters."""
    if not options:
        return {}
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "num_predict": "max_tokens",
        "num_ctx": "max_tokens",
        "stop": "stop",
        "repeat_penalty": "repetition_penalty",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "min_p": "min_p",
    }
    out = {}
    for ollama_key, openai_key in mapping.items():
        if ollama_key in options and options[ollama_key] is not None:
            out[openai_key] = options[ollama_key]
    return out


def _build_openai_messages_generate(
    prompt: str,
    system: str | None = None,
) -> list[dict]:
    """Build OpenAI messages list from Ollama generate request."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _build_openai_messages_chat(
    messages: list[OllamaChatMessage],
) -> list[dict]:
    """Convert OllamaChatMessage list to OpenAI message dicts."""
    msgs = []
    for m in messages:
        content = m.content
        if m.images:
            parts = [{"type": "text", "text": content}] if content else []
            for img in m.images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                img
                                if img.startswith("data:")
                                else f"data:image/png;base64,{img}"
                            )
                        },
                    }
                )
            msgs.append({"role": m.role, "content": parts})
        else:
            msgs.append({"role": m.role, "content": content})
    return msgs


async def _call_openai_chat(
    model: str,
    messages: list[dict],
    stream: bool,
    params: dict,
) -> Any:
    """Call internal OpenAI chat completion handler."""
    from .models import (
        AssistantMessage,
        ChatCompletionRequest,
        SystemMessage,
        UserMessage,
    )
    from .openai_routes import (
        _resolve_engine,
        _run_chat,
        _stream_chat_generator,
    )

    openai_msgs = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            openai_msgs.append(SystemMessage(role="system", content=content))
        elif role == "assistant":
            openai_msgs.append(AssistantMessage(role="assistant", content=content))
        else:
            openai_msgs.append(UserMessage(role="user", content=content))

    chat_req = ChatCompletionRequest(
        model=model,
        messages=openai_msgs,
        stream=stream,
        **params,
    )

    if not stream:
        return await _run_chat(chat_req, _skip_cap_check=True)

    from ..server import resolve_model_with_profile

    model_name, profile_overrides = resolve_model_with_profile(model)
    engine = await _resolve_engine(model_name)
    if engine is None:
        raise HTTPException(404, f"Model {model_name} not available")
    return _stream_chat_generator(
        chat_req,
        engine,
        model_name,
        None,
        profile_overrides=profile_overrides,
    )


# =============================================================================
# POST /api/generate
# =============================================================================


@router.post("/api/generate")
async def api_generate(
    request: OllamaGenerateRequest,
    _auth: bool = Depends(verify_api_key),
) -> Any:
    """Ollama-compatible text generation endpoint."""
    logger.info(
        "Ollama /api/generate model=%s stream=%s", request.model, request.stream
    )

    messages = _build_openai_messages_generate(request.prompt, request.system)
    params = _options_to_params(request.options)

    if not request.stream:
        result = await _call_openai_chat(request.model, messages, False, params)
        content = result.choices[0].message.content if result.choices else ""
        usage = result.usage
        return JSONResponse(
            {
                "model": request.model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                "response": content,
                "done": True,
                "done_reason": "stop",
                "context": [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": usage.prompt_tokens if usage else 0,
                "prompt_eval_duration": 0,
                "eval_count": usage.completion_tokens if usage else 0,
                "eval_duration": 0,
            }
        )

    # Streaming
    async def _stream_generate():
        gen = await _call_openai_chat(request.model, messages, True, params)
        accumulated = ""
        eval_count = 0
        prompt_eval_count = 0
        async for chunk in gen:
            if isinstance(chunk, str) and chunk.startswith("data: "):
                payload = chunk[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    for c in data.get("choices", []):
                        delta = c.get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            accumulated += text
                            yield (
                                json.dumps(
                                    {
                                        "model": request.model,
                                        "created_at": time.strftime(
                                            "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                                        ),
                                        "response": text,
                                        "done": False,
                                    }
                                )
                                + "\n"
                            )
                        usage_chunk = data.get("usage")
                        if usage_chunk:
                            eval_count = usage_chunk.get(
                                "completion_tokens", eval_count
                            )
                            prompt_eval_count = usage_chunk.get(
                                "prompt_tokens", prompt_eval_count
                            )
                except json.JSONDecodeError:
                    continue
            elif isinstance(chunk, str) and chunk.startswith(": "):
                yield chunk
                continue

        yield (
            json.dumps(
                {
                    "model": request.model,
                    "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                    ),
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    "context": [],
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": prompt_eval_count,
                    "prompt_eval_duration": 0,
                    "eval_count": eval_count,
                    "eval_duration": 0,
                }
            )
            + "\n"
        )

    return StreamingResponse(
        _stream_generate(),
        media_type="application/x-ndjson",
    )


# =============================================================================
# POST /api/chat
# =============================================================================


@router.post("/api/chat")
async def api_chat(
    request: OllamaChatRequest,
    _auth: bool = Depends(verify_api_key),
) -> Any:
    """Ollama-compatible chat endpoint."""
    logger.info(
        "Ollama /api/chat model=%s stream=%s msgs=%d",
        request.model,
        request.stream,
        len(request.messages),
    )

    messages = _build_openai_messages_chat(request.messages)
    params = _options_to_params(request.options)

    if not request.stream:
        result = await _call_openai_chat(request.model, messages, False, params)
        content = result.choices[0].message.content if result.choices else ""
        usage = result.usage
        return JSONResponse(
            {
                "model": request.model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": usage.prompt_tokens if usage else 0,
                "prompt_eval_duration": 0,
                "eval_count": usage.completion_tokens if usage else 0,
                "eval_duration": 0,
            }
        )

    # Streaming
    async def _stream_chat():
        gen = await _call_openai_chat(request.model, messages, True, params)
        eval_count = 0
        prompt_eval_count = 0
        async for chunk in gen:
            if isinstance(chunk, str) and chunk.startswith("data: "):
                payload = chunk[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    for c in data.get("choices", []):
                        delta = c.get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield (
                                json.dumps(
                                    {
                                        "model": request.model,
                                        "created_at": time.strftime(
                                            "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                                        ),
                                        "message": {
                                            "role": "assistant",
                                            "content": text,
                                        },
                                        "done": False,
                                    }
                                )
                                + "\n"
                            )
                        usage_chunk = data.get("usage")
                        if usage_chunk:
                            eval_count = usage_chunk.get(
                                "completion_tokens", eval_count
                            )
                            prompt_eval_count = usage_chunk.get(
                                "prompt_tokens", prompt_eval_count
                            )
                except json.JSONDecodeError:
                    continue
            elif isinstance(chunk, str) and chunk.startswith(": "):
                yield chunk
                continue

        yield (
            json.dumps(
                {
                    "model": request.model,
                    "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                    ),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": prompt_eval_count,
                    "prompt_eval_duration": 0,
                    "eval_count": eval_count,
                    "eval_duration": 0,
                }
            )
            + "\n"
        )

    return StreamingResponse(
        _stream_chat(),
        media_type="application/x-ndjson",
    )


# =============================================================================
# GET /api/tags
# =============================================================================


@router.get("/api/tags")
async def api_tags(
    _auth: bool = Depends(verify_api_key),
) -> JSONResponse:
    """Ollama-compatible model list endpoint."""
    if _pool is None:
        return JSONResponse({"models": []})

    try:
        model_ids = _pool.list_models() if hasattr(_pool, "list_models") else []
    except Exception:
        model_ids = []

    models = []
    for mid in model_ids:
        entry = _pool.get_entry(mid) if hasattr(_pool, "get_entry") else None
        size = 0
        if entry:
            size = (
                getattr(entry, "estimated_size", 0)
                or getattr(entry, "actual_size", 0)
                or 0
            )
        models.append(
            {
                "name": mid,
                "model": mid,
                "modified_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                ),
                "size": size,
                "digest": "sha256:" + uuid.uuid4().hex[:64],
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": mid.split("-")[0] if "-" in mid else mid,
                    "families": [mid.split("-")[0] if "-" in mid else mid],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
        )

    return JSONResponse({"models": models})


# =============================================================================
# GET /api/version
# =============================================================================


@router.get("/api/version")
async def api_version() -> JSONResponse:
    """Ollama-compatible version endpoint."""
    try:
        from .. import __version__

        ver = __version__
    except Exception:
        ver = "0.1.0"
    return JSONResponse({"version": ver})
