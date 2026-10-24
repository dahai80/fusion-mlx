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

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..middleware.auth import verify_api_key, verify_management_access
from ._concurrency import (
    acquire_request_slot,
    concurrency_guarded,
    release_request_slot,
)
from ._disconnect_guard import handle_disconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ollama"])

_pool: Any = None


def set_ollama_context(pool: Any) -> None:
    global _pool
    _pool = pool


def _entry_context_length(entry) -> int | None:
    if entry is None:
        return None
    ctx = getattr(entry, "model_context_length", None)
    if ctx and ctx > 0:
        return ctx
    engine = getattr(entry, "engine", None)
    if engine is not None:
        try:
            from ..service.helpers import get_model_max_context

            ctx = get_model_max_context(engine)
            if ctx and ctx > 0:
                return ctx
        except Exception:
            logger.debug("context_length: engine lookup failed")
    return None


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
    # #1354 audit M19: num_ctx is the context-window size, not an output cap.
    # Mapping it onto max_tokens made `num_ctx: 8192` request an 8192-token
    # generation and silently overwrite num_predict. The served model's
    # context length is the backend's job; ignore it here.
    if options.get("num_ctx") is not None:
        logger.debug(
            "Ollama options.num_ctx=%s ignored (context window is set by the "
            "loaded model; use num_predict for the output cap)",
            options.get("num_ctx"),
        )
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "num_predict": "max_tokens",
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


def _ollama_error_frame(error: Any) -> str:
    # #1354 audit M20: surface an upstream SSE error as the Ollama ndjson
    # error object instead of letting the stream end as a successful done.
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or str(error)
    else:
        message = str(error)
    logger.error("ollama stream: upstream error frame forwarded: %s", message)
    return json.dumps({"error": message})


def _tool_calls_to_ollama(tool_calls: list) -> list[dict]:
    # OpenAI tool_call: {"id", "type", "function": {"name", "arguments": str}}
    # Ollama: {"function": {"name", "arguments": <object>}}
    out = []
    for tc in tool_calls or []:
        fn = (
            tc.get("function")
            if isinstance(tc, dict)
            else getattr(tc, "function", None)
        )
        name = None
        raw_args = ""
        if isinstance(fn, dict):
            name = fn.get("name")
            raw_args = fn.get("arguments") or ""
        elif fn is not None:
            name = getattr(fn, "name", None)
            raw_args = getattr(fn, "arguments", "") or ""
        if not name:
            continue
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                logger.warning(
                    "ollama: tool_call '%s' arguments are not valid JSON: %r",
                    name,
                    raw_args,
                )
                args = {}
        out.append({"function": {"name": name, "arguments": args}})
    return out


def _ollama_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())


def _ollama_chunk_line(model: str, body: dict[str, Any]) -> str:
    # Audit 1019 L13: generate and chat built the same ndjson envelope.
    payload: dict[str, Any] = {"model": model, "created_at": _ollama_ts()}
    payload.update(body)
    return json.dumps(payload) + "\n"


def _ollama_done_body(
    finish_reason: str | None, prompt_eval_count: int, eval_count: int
) -> dict[str, Any]:
    return {
        "done": True,
        "done_reason": "length" if finish_reason == "length" else "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": 0,
        "eval_count": eval_count,
        "eval_duration": 0,
    }


async def _ollama_stream_lines(
    gen,
    *,
    model: str,
    render_delta,
    render_finish,
    log_label: str,
    request_id: str,
    engine: Any,
):
    # Audit 1019 L13: /api/generate and /api/chat carried a ~90-line copy of
    # this frame walk, so every fix (M20 error frames, usage accounting,
    # keepalives) had to land twice. The walk lives here; each endpoint
    # supplies its own chunk shape through the two callbacks.
    eval_count = 0
    prompt_eval_count = 0
    finish_reason = None
    try:
        async for chunk in gen:
            if isinstance(chunk, str) and chunk.startswith("data: "):
                payload = chunk[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # #1354 audit M20: an upstream error frame has no choices;
                # skipping it and then reporting done_reason "stop" told the
                # client the empty generation worked.
                if data.get("error"):
                    yield _ollama_error_frame(data["error"]) + "\n"
                    return
                for c in data.get("choices", []):
                    delta = c.get("delta", {})
                    if c.get("finish_reason"):
                        finish_reason = c["finish_reason"]
                    for line in render_delta(delta):
                        yield line
                    usage_chunk = data.get("usage")
                    if usage_chunk:
                        eval_count = usage_chunk.get("completion_tokens", eval_count)
                        prompt_eval_count = usage_chunk.get(
                            "prompt_tokens", prompt_eval_count
                        )
            elif isinstance(chunk, str) and chunk.startswith(": "):
                yield chunk
                continue
        for line in render_finish(finish_reason, prompt_eval_count, eval_count):
            yield line
    except asyncio.CancelledError:
        logger.info("ollama %s client disconnected: %s", log_label, request_id)
        handle_disconnect(request_id, engine)
        raise


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
    """Call internal OpenAI chat completion handler.

    For stream=True returns ``(gen, request_id, engine)`` so the ollama
    streaming wrapper can wire ``handle_disconnect`` on client cancel —
    the inner openai generator's own CancelledError guard does NOT fire
    when ASGI cancels the *outer* (ollama) frame. For stream=False
    returns the ChatCompletionResponse unchanged.
    """
    from .models import (
        AssistantMessage,
        ChatCompletionRequest,
        SystemMessage,
        UserMessage,
    )
    from .openai import (
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
        await acquire_request_slot()
        try:
            result = await _run_chat(chat_req, _skip_cap_check=True)
            # P0-1 (#0912 audit): _run_chat returns a JSONResponse (not a
            # ChatCompletionResponse) when context-budget headers are set,
            # but ollama callers expect attribute access (.choices/.usage).
            # Unwrap to a ChatCompletionResponse so /api/chat + /api/generate
            # don't crash with "'JSONResponse' object has no attribute
            # 'choices'".
            if isinstance(result, JSONResponse):
                from .openai_models import ChatCompletionResponse

                _body = result.body
                if isinstance(_body, (bytes, bytearray)):
                    import json as _json

                    _body = _json.loads(_body)
                result = ChatCompletionResponse.model_validate(_body)
            return result
        finally:
            release_request_slot()

    from ..server import resolve_model_with_profile

    model_name, profile_overrides = resolve_model_with_profile(model)
    engine = await _resolve_engine(model_name)
    if engine is None:
        raise HTTPException(404, f"Model {model_name} not available")
    import uuid

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    gen = concurrency_guarded(
        _stream_chat_generator(
            chat_req,
            engine,
            model_name,
            None,
            profile_overrides=profile_overrides,
            request_id=request_id,
        )
    )
    return gen, request_id, engine


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
    # #1169: fast-fail if the LLM executor was poisoned by the G2 watchdog.
    from ._guards import check_llm_executor_healthy

    check_llm_executor_healthy(request.model or "", adapter="openai")

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
        gen, request_id, engine = await _call_openai_chat(
            request.model, messages, True, params
        )

        def render_delta(delta):
            text = delta.get("content", "")
            if not text:
                return ()
            return (
                _ollama_chunk_line(request.model, {"response": text, "done": False}),
            )

        def render_finish(finish_reason, prompt_eval_count, eval_count):
            body: dict[str, Any] = {"response": "", "context": []}
            body.update(_ollama_done_body(finish_reason, prompt_eval_count, eval_count))
            return (_ollama_chunk_line(request.model, body),)

        async for line in _ollama_stream_lines(
            gen,
            model=request.model,
            render_delta=render_delta,
            render_finish=render_finish,
            log_label="/api/generate",
            request_id=request_id,
            engine=engine,
        ):
            yield line

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
    # #1169: fast-fail if the LLM executor was poisoned by the G2 watchdog.
    from ._guards import check_llm_executor_healthy

    check_llm_executor_healthy(request.model or "", adapter="openai")

    messages = _build_openai_messages_chat(request.messages)
    params = _options_to_params(request.options)

    if not request.stream:
        result = await _call_openai_chat(request.model, messages, False, params)
        msg = result.choices[0].message if result.choices else None
        content = (msg.content if msg else "") or ""
        usage = result.usage
        message: dict[str, Any] = {"role": "assistant", "content": content}
        # #1354 audit M20: Ollama clients read tool calls from message.tool_calls
        # (arguments as an object), and reasoning from message.thinking.
        if msg is not None:
            if getattr(msg, "reasoning_content", None):
                message["thinking"] = msg.reasoning_content
            tool_calls = _tool_calls_to_ollama(getattr(msg, "tool_calls", None))
            if tool_calls:
                message["tool_calls"] = tool_calls
        return JSONResponse(
            {
                "model": request.model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                "message": message,
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
        gen, request_id, engine = await _call_openai_chat(
            request.model, messages, True, params
        )
        tool_acc: dict[int, dict[str, str]] = {}

        def render_delta(delta):
            lines = []
            text = delta.get("content") or ""
            thinking = delta.get("reasoning_content") or ""
            if text or thinking:
                message = {"role": "assistant", "content": text}
                if thinking:
                    message["thinking"] = thinking
                lines.append(
                    _ollama_chunk_line(
                        request.model, {"message": message, "done": False}
                    )
                )
            # #1354 audit M20: accumulate tool-call deltas by index; Ollama
            # carries them as whole calls in one message with object-form
            # arguments.
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_acc.setdefault(idx, {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            return lines

        def render_finish(finish_reason, prompt_eval_count, eval_count):
            lines = []
            if tool_acc:
                ollama_calls = _tool_calls_to_ollama(
                    [
                        {
                            "function": {
                                "name": tool_acc[i]["name"],
                                "arguments": tool_acc[i]["arguments"],
                            }
                        }
                        for i in sorted(tool_acc)
                    ]
                )
                if ollama_calls:
                    lines.append(
                        _ollama_chunk_line(
                            request.model,
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": ollama_calls,
                                },
                                "done": False,
                            },
                        )
                    )
            body: dict[str, Any] = {"message": {"role": "assistant", "content": ""}}
            body.update(_ollama_done_body(finish_reason, prompt_eval_count, eval_count))
            lines.append(_ollama_chunk_line(request.model, body))
            return lines

        async for line in _ollama_stream_lines(
            gen,
            model=request.model,
            render_delta=render_delta,
            render_finish=render_finish,
            log_label="/api/chat",
            request_id=request_id,
            engine=engine,
        ):
            yield line

    return StreamingResponse(
        _stream_chat(),
        media_type="application/x-ndjson",
    )


# =============================================================================
# GET /api/tags
# =============================================================================


def _model_digest(model_id: str) -> str:
    # Audit 1019 L9: the digest used to be ``uuid.uuid4()`` per call, so the
    # same model reported a different identity on every request and Ollama
    # clients keyed on the digest (Open WebUI caches, chat model binding)
    # saw a new model each refresh. Derive it from the model id instead —
    # stable across calls and restarts.
    return "sha256:" + hashlib.sha256(model_id.encode("utf-8")).hexdigest()


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
        details = {
            "parent_model": "",
            "format": "gguf",
            "family": mid.split("-")[0] if "-" in mid else mid,
            "families": [mid.split("-")[0] if "-" in mid else mid],
            "parameter_size": "",
            "quantization_level": "",
        }
        ctx_len = _entry_context_length(entry)
        if ctx_len is not None and ctx_len > 0:
            details["context_length"] = ctx_len
        models.append(
            {
                "name": mid,
                "model": mid,
                "modified_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                ),
                "size": size,
                "digest": _model_digest(mid),
                "details": details,
            }
        )

    return JSONResponse({"models": models})


# =============================================================================
# GET /api/version
# =============================================================================


@router.get("/api/version")
async def api_version(
    _auth: bool = Depends(verify_api_key),
) -> JSONResponse:
    """Ollama-compatible version endpoint."""
    try:
        from .. import __version__

        ver = __version__
    except Exception:
        ver = "0.1.0"
    return JSONResponse({"version": ver})


# =============================================================================
# POST /api/show — model details (modelfile, parameters, info)
# =============================================================================


class OllamaShowRequest(BaseModel):
    name: str
    model: str | None = None


@router.post("/api/show")
async def api_show(
    request: OllamaShowRequest,
    _auth: bool = Depends(verify_api_key),
) -> JSONResponse:
    name = request.name or request.model or ""
    logger.info("Ollama /api/show name=%s", name)
    if _pool is None:
        raise HTTPException(404, f"model '{name}' not found")
    from ..server import resolve_model_with_profile

    resolved, _ = resolve_model_with_profile(name)
    entry = _pool.get_entry(resolved)
    if entry is None:
        raise HTTPException(404, f"model '{name}' not found")
    family = (
        entry.config_model_type
        or (resolved.split("-")[0] if "-" in resolved else resolved)
        or "llm"
    )
    size = entry.last_observed_size or entry.actual_size or entry.estimated_size or 0
    details = {
        "parent_model": "",
        "format": "mlx",
        "family": family,
        "families": [family],
        "parameter_size": "",
        "quantization_level": "",
    }
    model_info = {
        "general.architecture": family,
        "general.file_type": "mlx",
    }
    ctx_len = _entry_context_length(entry)
    if ctx_len is not None and ctx_len > 0:
        details["context_length"] = ctx_len
        model_info["llm.context_length"] = ctx_len
    return JSONResponse(
        {
            "name": resolved,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
            "size": size,
            "digest": _model_digest(resolved),
            "details": details,
            "model_info": model_info,
            "modelfile": f"# Modelfile for {resolved}\nFROM {resolved}\n",
            "parameters": "",
        }
    )


# =============================================================================
# GET /api/ps — list running (loaded) models
# =============================================================================


@router.get("/api/ps")
async def api_ps(
    _auth: bool = Depends(verify_api_key),
) -> JSONResponse:
    logger.info("Ollama /api/ps")
    if _pool is None:
        return JSONResponse({"models": []})
    try:
        loaded = _pool.get_loaded_model_ids()
    except Exception:
        loaded = []
    models = []
    for mid in loaded:
        entry = _pool.get_entry(mid) if hasattr(_pool, "get_entry") else None
        size = (entry.last_observed_size or entry.estimated_size) if entry else 0
        models.append(
            {
                "name": mid,
                "model": mid,
                "modified_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                ),
                "size": size or 0,
                "digest": _model_digest(mid),
                "expires_at": None,
                "size_vram": size or 0,
                "details": {
                    "parent_model": "",
                    "format": "mlx",
                    "family": mid.split("-")[0] if "-" in mid else mid,
                    "families": [mid.split("-")[0] if "-" in mid else mid],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
        )
    return JSONResponse({"models": models})


# =============================================================================
# POST /api/pull — local-first no-op (models are downloaded via admin/HF mirror)
# =============================================================================


class OllamaPullRequest(BaseModel):
    name: str
    model: str | None = None
    stream: bool = True
    insecure: bool = False


@router.post("/api/pull")
async def api_pull(
    request: OllamaPullRequest,
    _auth: bool = Depends(verify_management_access),
) -> Any:
    name = request.name or request.model or ""
    logger.info("Ollama /api/pull name=%s stream=%s", name, request.stream)
    if _pool is not None:
        from ..server import resolve_model_with_profile

        try:
            resolved, _ = resolve_model_with_profile(name)
        except Exception:
            resolved = name
        if resolved and _pool.get_entry(resolved) is not None:
            msg = f"model '{name}' already available locally"
        else:
            msg = (
                f"pull is a no-op on fusion-mlx; download '{name}' via the "
                f"admin UI / hf_downloader (HF_MIRROR=https://hf-mirror.com)"
            )
    else:
        msg = "pool unavailable"
    if not request.stream:
        return JSONResponse({"status": "success", "message": msg})
    now = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    async def _stream():
        yield (
            json.dumps({"status": "pulling model", "id": name, "created_at": now})
            + "\n"
        )
        yield (json.dumps({"status": "success", "total": 0, "completed": 0}) + "\n")
        yield (json.dumps({"status": "success"}) + "\n")

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


# =============================================================================
# DELETE /api/delete — remove model files from disk (guarded)
# =============================================================================

# #1353 (H6): /api/delete, /api/pull and /api/copy are management surfaces —
# they mutate model state on disk. Guard them with the management-access
# dependency (same boundary as /metrics, /v1/status and the cluster routes),
# not the inference dependency used by chat/generate.


def _models_root() -> Path:
    return Path(os.path.expanduser("~/.fusion-mlx/models"))


def _safe_model_dir(model_path: str) -> Path:
    root = _models_root().resolve()
    target = Path(model_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(
            400,
            f"refusing to delete model outside ~/.fusion-mlx/models: {model_path}",
        )
    if not target.exists():
        raise HTTPException(404, f"model directory not found: {model_path}")
    return target


@router.delete("/api/delete")
async def api_delete(
    name: str = Query(..., description="model name to delete"),
    _auth: bool = Depends(verify_management_access),
) -> JSONResponse:
    logger.warning("Ollama /api/delete name=%s", name)
    if _pool is None:
        raise HTTPException(503, "model pool not initialized")
    from ..server import resolve_model_with_profile

    resolved, _ = resolve_model_with_profile(name)
    entry = _pool.get_entry(resolved)
    if entry is None:
        raise HTTPException(404, f"model '{name}' not found")
    if entry.engine is not None:
        raise HTTPException(
            409,
            f"model '{resolved}' is currently loaded; unload before deleting",
        )
    target = _safe_model_dir(entry.model_path)
    try:
        shutil.rmtree(target)
    except Exception as e:
        logger.exception("api_delete: rmtree failed for %s", target)
        raise HTTPException(500, f"failed to delete model: {e}")
    logger.info("api_delete: removed %s", target)
    return JSONResponse({"status": "success", "message": f"deleted {resolved}"})


# =============================================================================
# POST /api/copy — create an alias directory (symlink) for a model
# =============================================================================


class OllamaCopyRequest(BaseModel):
    source: str
    destination: str


@router.post("/api/copy")
async def api_copy(
    request: OllamaCopyRequest,
    _auth: bool = Depends(verify_management_access),
) -> JSONResponse:
    logger.info(
        "Ollama /api/copy source=%s destination=%s",
        request.source,
        request.destination,
    )
    if _pool is None:
        raise HTTPException(503, "model pool not initialized")
    if not request.source or not request.destination:
        raise HTTPException(400, "source and destination are required")
    if request.source == request.destination:
        raise HTTPException(400, "source and destination must differ")
    if "/" in request.destination or os.sep in request.destination:
        raise HTTPException(400, "destination must be a plain model name, no path")
    from ..server import resolve_model_with_profile

    src_resolved, _ = resolve_model_with_profile(request.source)
    src_entry = _pool.get_entry(src_resolved)
    if src_entry is None:
        raise HTTPException(404, f"source model '{request.source}' not found")
    src_dir = _safe_model_dir(src_entry.model_path)
    dest_dir = _models_root() / request.destination
    if dest_dir.exists():
        raise HTTPException(409, f"destination '{request.destination}' already exists")
    try:
        dest_dir.symlink_to(src_dir)
    except Exception as e:
        logger.exception("api_copy: symlink failed %s -> %s", src_dir, dest_dir)
        raise HTTPException(500, f"failed to copy model: {e}")
    logger.info("api_copy: %s -> %s", src_dir, dest_dir)
    return JSONResponse({"status": "success"})


# =============================================================================
# POST /api/embeddings — alias to the internal embeddings endpoint
# =============================================================================


class OllamaEmbeddingsRequest(BaseModel):
    model: str = "default"
    prompt: str = ""
    options: dict | None = None
    keep_alive: str | None = None


@router.post("/api/embeddings")
async def api_embeddings(
    request: OllamaEmbeddingsRequest,
    _auth: bool = Depends(verify_api_key),
) -> JSONResponse:
    logger.info("Ollama /api/embeddings model=%s", request.model)
    from .embeddings_routes import create_embeddings
    from .models import EmbeddingRequest

    req = EmbeddingRequest(model=request.model, input=request.prompt)
    result = await create_embeddings(req)
    data = result.data[0] if getattr(result, "data", None) else None
    embedding = data.embedding if data is not None else []
    return JSONResponse({"embedding": embedding})
