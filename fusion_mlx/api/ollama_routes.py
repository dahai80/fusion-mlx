# SPDX-License-Identifier: Apache-2.0
"""Ollama-compatible API routes for fusion-mlx.

Provides drop-in Ollama API compatibility so tools configured for
``http://localhost:11434`` can point at fusion-mlx instead.

Endpoints:
    POST /api/generate  — text generation (Ollama format)
    POST /api/chat      — chat completion (Ollama format)
    GET  /api/tags      — list local models (Ollama format)
    GET  /api/version   — server version

Importers/callers: fusion_mlx/server.py imports ``router as ollama_router``
and ``set_ollama_context``, registers via ``app.include_router(ollama_router)``
and ``set_ollama_context(self.pool)``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ollama"])

_pool: Any = None


def set_ollama_context(pool) -> None:
    global _pool
    _pool = pool


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = True
    raw: bool = False
    format: str | dict | None = None
    options: dict[str, Any] | None = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    stream: bool = True
    format: str | dict | None = None
    options: dict[str, Any] | None = None
    tools: list[dict] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _resolve_engine(model_name: str):
    from ..server import resolve_model_id

    resolved = resolve_model_id(model_name)
    if _pool is not None:
        engine = await _pool.get_engine(resolved, _lease=True)
        return engine, resolved
    from ..service.helpers import get_engine

    return get_engine(resolved), resolved


async def _release_engine(model_name: str):
    if _pool is not None:
        try:
            _pool.release_engine(model_name)
        except Exception:
            logger.debug("release_engine failed for %s", model_name, exc_info=True)


def _ollama_model_info(model_id: str) -> dict[str, Any]:
    return {
        "name": model_id,
        "model": model_id,
        "modified_at": "",
        "size": 0,
        "digest": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "unknown",
            "quantization_level": "Q4_0",
        },
    }


def _options_to_params(options: dict[str, Any] | None) -> dict[str, Any]:
    if not options:
        return {}
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "num_predict": "max_tokens",
        "stop": "stop",
        "repeat_penalty": "repetition_penalty",
        "presence_penalty": "presence_penalty",
        "seed": "seed",
    }
    out = {}
    for ollama_key, openai_key in mapping.items():
        if ollama_key in options:
            out[openai_key] = options[ollama_key]
    return out


def _ndjson_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ---------------------------------------------------------------------------
# POST /api/generate
# ---------------------------------------------------------------------------


@router.post("/generate")
async def ollama_generate(request: OllamaGenerateRequest, http_request: Request):
    logger.info(
        "Ollama /api/generate: model=%s stream=%s", request.model, request.stream
    )

    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    extra_params = _options_to_params(request.options)

    if request.format:
        if isinstance(request.format, dict):
            extra_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": request.format},
            }
        elif request.format == "json":
            extra_params["response_format"] = {"type": "json_object"}

    engine, resolved_model = await _resolve_engine(request.model)
    if engine is None:
        await _release_engine(resolved_model)
        raise HTTPException(404, f"Model {request.model} not available")

    max_tokens = extra_params.pop("max_tokens", 2048)
    temperature = extra_params.pop("temperature", 0.8)
    top_p = extra_params.pop("top_p", 0.9)
    stop = extra_params.pop("stop", [])

    try:
        if request.stream:

            async def _stream_ollama():
                try:
                    async for chunk in engine.stream_chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        **extra_params,
                    ):
                        yield _ndjson_line(
                            {
                                "model": request.model,
                                "created_at": _now_iso(),
                                "response": chunk.text or "",
                                "done": False,
                            }
                        )
                    yield _ndjson_line(
                        {
                            "model": request.model,
                            "created_at": _now_iso(),
                            "response": "",
                            "done": True,
                            "done_reason": "stop",
                            "context": [],
                            "total_duration": 0,
                            "load_duration": 0,
                            "prompt_eval_count": 0,
                            "eval_count": 0,
                        }
                    )
                except Exception as e:
                    logger.error("Ollama generate stream error: %s", e)
                    yield _ndjson_line(
                        {
                            "model": request.model,
                            "error": str(e),
                            "done": True,
                        }
                    )
                finally:
                    await _release_engine(resolved_model)

            return StreamingResponse(
                _stream_ollama(),
                media_type="application/x-ndjson",
            )
        else:
            result = await engine.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **extra_params,
            )
            await _release_engine(resolved_model)
            return JSONResponse(
                {
                    "model": request.model,
                    "created_at": _now_iso(),
                    "response": result.text or "",
                    "done": True,
                    "done_reason": "stop",
                    "context": [],
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": result.prompt_tokens or 0,
                    "eval_count": result.completion_tokens or 0,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ollama generate error: %s", e)
        await _release_engine(resolved_model)
        return JSONResponse(
            {"model": request.model, "error": str(e), "done": True},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------


@router.post("/chat")
async def ollama_chat(request: OllamaChatRequest, http_request: Request):
    logger.info("Ollama /api/chat: model=%s stream=%s", request.model, request.stream)

    extra_params = _options_to_params(request.options)

    if request.format:
        if isinstance(request.format, dict):
            extra_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": request.format},
            }
        elif request.format == "json":
            extra_params["response_format"] = {"type": "json_object"}

    engine, resolved_model = await _resolve_engine(request.model)
    if engine is None:
        await _release_engine(resolved_model)
        raise HTTPException(404, f"Model {request.model} not available")

    max_tokens = extra_params.pop("max_tokens", 2048)
    temperature = extra_params.pop("temperature", 0.8)
    top_p = extra_params.pop("top_p", 0.9)

    try:
        if request.stream:

            async def _stream_ollama_chat():
                try:
                    async for chunk in engine.stream_chat(
                        messages=request.messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        tools=request.tools,
                        **extra_params,
                    ):
                        content = chunk.text or ""
                        yield _ndjson_line(
                            {
                                "model": request.model,
                                "created_at": _now_iso(),
                                "message": {"role": "assistant", "content": content},
                                "done": False,
                            }
                        )
                    yield _ndjson_line(
                        {
                            "model": request.model,
                            "created_at": _now_iso(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "done_reason": "stop",
                            "total_duration": 0,
                            "load_duration": 0,
                            "prompt_eval_count": 0,
                            "eval_count": 0,
                        }
                    )
                except Exception as e:
                    logger.error("Ollama chat stream error: %s", e)
                    yield _ndjson_line(
                        {
                            "model": request.model,
                            "error": str(e),
                            "done": True,
                        }
                    )
                finally:
                    await _release_engine(resolved_model)

            return StreamingResponse(
                _stream_ollama_chat(),
                media_type="application/x-ndjson",
            )
        else:
            result = await engine.chat(
                messages=request.messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=request.tools,
                **extra_params,
            )
            await _release_engine(resolved_model)
            message: dict[str, Any] = {
                "role": "assistant",
                "content": result.text or "",
            }
            if result.tool_calls:
                message["tool_calls"] = result.tool_calls
            return JSONResponse(
                {
                    "model": request.model,
                    "created_at": _now_iso(),
                    "message": message,
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": result.prompt_tokens or 0,
                    "eval_count": result.completion_tokens or 0,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ollama chat error: %s", e)
        await _release_engine(resolved_model)
        return JSONResponse(
            {"model": request.model, "error": str(e), "done": True},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# GET /api/tags
# ---------------------------------------------------------------------------


@router.get("/tags")
async def ollama_tags():
    logger.info("Ollama /api/tags")
    if _pool is not None:
        try:
            model_ids = _pool.get_loaded_model_ids()
        except Exception:
            model_ids = []
        if not model_ids:
            try:
                model_ids = _pool.list_models()
            except Exception:
                model_ids = []
    else:
        model_ids = []

    models = [_ollama_model_info(m) for m in model_ids]
    return JSONResponse({"models": models})


# ---------------------------------------------------------------------------
# GET /api/version
# ---------------------------------------------------------------------------


@router.get("/version")
async def ollama_version():
    from .. import __version__

    return JSONResponse({"version": __version__})
