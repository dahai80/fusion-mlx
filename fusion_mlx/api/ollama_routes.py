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
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
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
    size = (
        entry.last_observed_size
        or entry.actual_size
        or entry.estimated_size
        or 0
    )
    return JSONResponse(
        {
            "name": resolved,
            "modified_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
            ),
            "size": size,
            "digest": "sha256:" + uuid.uuid4().hex[:64],
            "details": {
                "parent_model": "",
                "format": "mlx",
                "family": family,
                "families": [family],
                "parameter_size": "",
                "quantization_level": "",
            },
            "model_info": {
                "general.architecture": family,
                "general.file_type": "mlx",
            },
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
        size = (
            (entry.last_observed_size or entry.estimated_size) if entry else 0
        )
        models.append(
            {
                "name": mid,
                "model": mid,
                "modified_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()
                ),
                "size": size or 0,
                "digest": "sha256:" + uuid.uuid4().hex[:64],
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
    _auth: bool = Depends(verify_api_key),
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
        yield (
            json.dumps({"status": "success", "total": 0, "completed": 0})
            + "\n"
        )
        yield (json.dumps({"status": "success"}) + "\n")

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


# =============================================================================
# DELETE /api/delete — remove model files from disk (guarded)
# =============================================================================


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
    _auth: bool = Depends(verify_api_key),
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
    _auth: bool = Depends(verify_api_key),
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
