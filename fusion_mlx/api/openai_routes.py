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

from fastapi import APIRouter, Depends, HTTPException, Request
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
from ..dispatch import RequestRouter
from ..engines.base import GenerationOutput
from ..exceptions import (
    AdapterPathError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from ..middleware.auth import check_rate_limit, request_principal, verify_api_key
from ..pool import EnginePool
from ..request import SamplingParams
from ..server_metrics import record_llm_metrics
from ..sessions import record_chat_session
from ._guards import check_chat_capability, check_multimodal_content
from .grammar import GrammarBackend, resolve_grammar_backend

logger = logging.getLogger(__name__)

# Strong refs for fire-and-forget abort tasks so they are not GC'd before
# completion; entries self-remove via the done-callback.
_pending_abort_tasks: set[asyncio.Task] = set()

router = APIRouter(prefix="/v1", tags=["openai"])

# Set by server.py during startup
_pool: Any = None
_request_router: Any = None
_adapter = OpenAIAdapter()
log = logging.getLogger(__name__)

_MODEL_TYPE_TO_MODALITY: dict[str, str] = {
    "llm": "text",
    "vlm": "text",
    "embedding": "text",
    "reranker": "text",
    "ner": "text",
    "audio_stt": "audio",
    "audio_tts": "audio",
    "audio_sts": "audio",
    "image": "image",
    "video": "video",
}


def _resolve_modality(model_id: str) -> str:
    from ..model_aliases import resolve_profile

    profile = resolve_profile(model_id)
    if profile is not None and profile.modality:
        return profile.modality
    if _pool is not None:
        entry = _pool.get_entry(model_id)
        if entry is not None:
            mt = getattr(entry, "model_type", None)
            if mt and mt in _MODEL_TYPE_TO_MODALITY:
                return _MODEL_TYPE_TO_MODALITY[mt]
    return "text"


def _resolve_capabilities(model_id: str) -> dict:
    caps = {
        "text_generation": False,
        "tool_calling": False,
        "structured_output": False,
        "vision": False,
        "embedding": False,
    }
    if _pool is not None:
        entry = _pool.get_entry(model_id)
        if entry is not None:
            mt = getattr(entry, "model_type", None)
            if mt == "llm":
                caps["text_generation"] = True
                caps["tool_calling"] = True
                caps["structured_output"] = True
            elif mt == "vlm":
                caps["text_generation"] = True
                caps["tool_calling"] = True
                caps["structured_output"] = True
                caps["vision"] = True
            elif mt == "embedding":
                caps["embedding"] = True
    else:
        caps["text_generation"] = True
    return caps


def set_openai_context(pool: EnginePool, req_router: RequestRouter) -> None:
    """Inject engine pool and request router into this module."""
    global _pool, _request_router
    _pool = pool
    _request_router = req_router


async def _resolve_engine(model_name: str, adapter_path=None):
    if _pool is not None:
        engine = await _pool.get_engine(
            model_name, _lease=True, adapter_path=adapter_path
        )
        return engine
    from ..service.helpers import get_engine

    log.debug("_pool None, falling back to cfg.engine for %s", model_name)
    return get_engine(model_name)


async def _release_engine(model_name: str, adapter_path=None):
    if _pool is not None:
        await _pool.release_engine(model_name, adapter_path=adapter_path)


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


def _detect_prefix_cache_boundary(messages: Any) -> int | None:
    """Auto-detect prefix cache boundary from cache_control hints in messages.

    Scans system messages for cache_control markers (Anthropic-compatible).
    Returns the estimated token boundary for KV prefix cache reuse, or None.

    The boundary is set at the end of the last system message that carries
    a cache_control block, enabling the engine to reuse cached KV states
    for shared system prompt prefixes across requests.
    """
    char_boundary = 0
    found = False
    for m in messages:
        role = getattr(m, "role", "")
        if role != "system":
            break
        content = getattr(m, "content", "")
        has_cache_control = False
        if isinstance(content, str):
            has_cache_control = bool(getattr(m, "cache_control", None))
            if has_cache_control:
                char_boundary += len(content)
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    has_cc = bool(part.get("cache_control"))
                else:
                    has_cc = bool(getattr(part, "cache_control", None))
                if has_cc:
                    found = True
                text = ""
                if isinstance(part, dict):
                    text = part.get("text", "")
                elif hasattr(part, "text"):
                    text = part.text or ""
                char_boundary += len(text)
        else:
            break
        if not has_cache_control and found:
            break
    if not found:
        return None
    return max(1, char_boundary // 4)


async def _inject_web_search(request: ChatCompletionRequest) -> None:
    """When request.web_search is True, search DuckDuckGo for the user's last
    message and prepend the results as a system message into the context."""
    if not request.web_search:
        return

    query = None
    for msg in reversed(request.messages):
        role = getattr(msg, "role", "")
        content = getattr(msg, "content", "")
        if role == "user" and content:
            query = _extract_text(msg).strip()
            break

    if not query:
        return

    logger.info("web_search: querying '%s'", query[:80])
    try:
        import httpx as _httpx

        snippets: list[str] = []
        async with _httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36",
                },
            )
            if resp.status_code == 200:
                import re

                text = resp.text
                results = re.findall(
                    r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', text, re.DOTALL
                )
                for i, raw_title in enumerate(results[:6]):
                    title = re.sub(r"<[^>]+>", "", raw_title).strip()
                    if title:
                        snippets.append(f"{i + 1}. {title}")

        if snippets:
            search_ctx = (
                "[Web Search Results]\n"
                + "\n".join(snippets)
                + "\n\nUse the above results to inform your answer when relevant."
            )
            from .models import Message

            search_msg = Message(role="system", content=search_ctx)
            request.messages.insert(0, search_msg)
            logger.info("web_search: injected %d results", len(snippets))
        else:
            logger.info("web_search: no results found for '%s'", query[:60])
    except Exception as exc:
        logger.warning("web_search failed: %s(%s)", type(exc).__name__, exc)


def _messages_for_engine(request_msgs: Any, is_mllm: bool) -> list[dict]:
    """Convert request messages to the dict list engines expect.

    Text-only models (and plain-string content) get the flattened text form
    from _extract_text. Multimodal models keep the structured content parts
    (image_url / video_url blocks) as dicts so the VLM engine can extract the
    media - flattening here would discard the URLs and the model would see
    only "[video]" / "[image]" placeholders.
    """
    out: list[dict] = []
    for m in request_msgs:
        content = getattr(m, "content", "")
        if is_mllm and isinstance(content, list):
            parts: list[dict] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part)
                elif hasattr(part, "model_dump"):
                    parts.append(part.model_dump(exclude_none=True))
                else:
                    parts.append(dict(part))
            out.append({"role": m.role, "content": parts})
        else:
            out.append({"role": m.role, "content": _extract_text(m)})
    return out


def _build_sampling_params(
    req: ChatCompletionRequest,
    profile_overrides: dict | None = None,
) -> SamplingParams:
    """Convert ChatCompletionRequest to SamplingParams.

    profile_overrides: dict from model:profile resolution.
    Request-level params take precedence; profile fills in unset defaults.
    """
    po = profile_overrides or {}
    return SamplingParams(
        max_tokens=req.max_tokens or po.get("max_tokens") or 2048,
        temperature=(
            req.temperature
            if req.temperature is not None
            else po.get("temperature", 0.7)
        ),
        top_p=(req.top_p if req.top_p is not None else po.get("top_p", 0.9)),
        top_k=getattr(req, "top_k", 0) or po.get("top_k") or 0,
        min_p=getattr(req, "min_p", 0.0) or po.get("min_p") or 0.0,
        presence_penalty=(
            req.presence_penalty
            if req.presence_penalty is not None
            else po.get("presence_penalty", 0.0)
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
        logprobs=bool(req.logprobs),
        top_logprobs=req.top_logprobs,
    )


def _compile_grammar_for_request(engine, req: ChatCompletionRequest):
    """Compile grammar from request's structured_outputs / response_format.

    Returns a backend-specific compiled grammar object (xgrammar CompiledGrammar
    or llguidance LLMatcher), or None if no grammar constraint is requested.
    """
    so = getattr(req, "structured_outputs", None)
    grammar_backend_str = getattr(req, "grammar_backend", None)
    if so is None and getattr(req, "response_format", None) is None:
        return None

    backend = resolve_grammar_backend(grammar_backend_str)
    grammar_spec = None

    if so is not None:
        if isinstance(so, dict):
            grammar_spec = so
        else:
            grammar_spec = {}
            if so.json_schema is not None:
                grammar_spec["json_schema"] = so.json_schema
            if so.regex is not None:
                grammar_spec["regex"] = so.regex
            if so.choice is not None:
                grammar_spec["choice"] = so.choice
            if so.grammar is not None:
                grammar_spec["grammar"] = so.grammar

    rf = getattr(req, "response_format", None)
    if rf is not None and grammar_spec is None:
        if isinstance(rf, dict):
            if rf.get("type") == "json_schema":
                schema = rf.get("json_schema", {})
                if isinstance(schema, dict) and "schema" in schema:
                    grammar_spec = {"json_schema": schema["schema"]}
                else:
                    grammar_spec = {"json_schema": schema}
            elif rf.get("type") == "json_object":
                grammar_spec = {"json_schema": "{}"}
        elif hasattr(rf, "type"):
            if rf.type == "json_schema":
                inner = getattr(rf, "json_schema", None)
                if inner and hasattr(inner, "schema_"):
                    grammar_spec = {"json_schema": inner.schema_}
                elif inner and hasattr(inner, "schema"):
                    grammar_spec = {"json_schema": inner.schema}
            elif rf.type == "json_object":
                grammar_spec = {"json_schema": "{}"}

    if grammar_spec is None:
        return None

    if backend == GrammarBackend.LLGUIDANCE:
        from .grammar import create_llguidance_matcher

        vocab_size = None
        if hasattr(engine, "_model"):
            from ..utils.tokenizer import resolve_vocab_size

            vocab_size = resolve_vocab_size(engine._model)
        matcher = create_llguidance_matcher(
            engine._tokenizer, grammar_spec, vocab_size=vocab_size
        )
        if matcher is not None:
            logger.info("compiled grammar via llguidance for request")
            return matcher
        logger.warning("llguidance compilation failed, trying xgrammar fallback")

    if backend in (GrammarBackend.XGRAMMAR, GrammarBackend.LLGUIDANCE):
        compiler = getattr(engine, "grammar_compiler", None)
        if compiler is None:
            logger.debug("no grammar_compiler on engine, skipping grammar compilation")
            return None
        try:
            if "json_schema" in grammar_spec:
                import json

                schema = grammar_spec["json_schema"]
                if isinstance(schema, dict):
                    schema = json.dumps(schema)
                return compiler.compile_json_schema(schema)
            if "regex" in grammar_spec:
                return compiler.compile_regex(grammar_spec["regex"])
            if "choice" in grammar_spec:
                import json

                return compiler.compile_json_schema(
                    json.dumps(
                        {
                            "type": "string",
                            "enum": grammar_spec["choice"],
                        }
                    )
                )
            if "grammar" in grammar_spec:
                return compiler.compile_grammar(grammar_spec["grammar"])
        except Exception as exc:
            logger.warning("xgrammar compilation failed: %s", exc)
    return None


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
        logprobs=getattr(gen, "logprobs", None),
    )


async def _run_chat(
    request: ChatCompletionRequest,
    *,
    _skip_cap_check: bool = False,
    principal: str | None = None,
) -> ChatCompletionResponse:
    """Execute a non-streaming chat completion."""
    from ..server import resolve_model_with_profile

    _start = time.perf_counter()
    model_name, profile_overrides = resolve_model_with_profile(request.model)
    adapter_path = getattr(request, "adapters", None)

    async def _release() -> None:
        await _release_engine(model_name, adapter_path=adapter_path)

    engine = await _resolve_engine(model_name, adapter_path=adapter_path)
    if engine is None:
        await _release()
        raise HTTPException(404, f"Model {model_name} not available")

    # #205 Guard: reject engines without chat capability (e.g. ImageGenEngine)
    if not _skip_cap_check:
        try:
            check_chat_capability(engine, "chat", model_name)
        except HTTPException:
            await _release()
            raise

    # Reject multimodal content on text-only models
    if not _skip_cap_check:
        try:
            check_multimodal_content(engine, request.messages, model_name)
        except HTTPException:
            await _release()
            raise

    await _inject_web_search(request)

    messages = _messages_for_engine(request.messages, getattr(engine, "is_mllm", False))
    sampling = _build_sampling_params(request, profile_overrides=profile_overrides)
    from .utils import cap_max_tokens_to_context

    sampling.max_tokens = cap_max_tokens_to_context(sampling.max_tokens, model_name)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        ct_kwargs = dict(getattr(request, "chat_template_kwargs", {}) or {})
        # AtomCode 专题优化: enable_thinking 默认禁思考收敛单点 (2026-07-19)
        from .utils import resolve_enable_thinking_default

        resolve_enable_thinking_default(ct_kwargs)
        compiled_grammar = _compile_grammar_for_request(engine, request)
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
            prefix_cache_boundary=(
                getattr(request, "prefix_cache_boundary", None)
                or _detect_prefix_cache_boundary(request.messages)
            ),
            compiled_grammar=compiled_grammar,
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
        record_llm_metrics(
            prompt_tokens=getattr(gen, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(gen, "completion_tokens", 0) or 0,
            cached_tokens=getattr(gen, "cached_tokens", 0) or 0,
            generation_duration=time.perf_counter() - _start,
            model_id=model_name,
        )
        record_chat_session(
            getattr(request, "session_id", None),
            prompt_tokens=getattr(gen, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(gen, "completion_tokens", 0) or 0,
            cached_tokens=getattr(gen, "cached_tokens", 0) or 0,
            principal=principal,
        )
        return _adapter.format_response(internal, request)
    except HTTPException:
        raise
    except AdapterPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ModelLoadingError, ModelBusyError) as exc:
        logger.warning("Model temporarily unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": str(exc), "type": "server_busy"}},
            headers={"Retry-After": "5"},
        ) from exc
    except InsufficientMemoryError as exc:
        logger.warning("Insufficient memory: %s", exc)
        detail = {
            "error": {
                "type": "model_unavailable",
                "message": f"Model {exc.model_id} not loaded and insufficient memory",
                "required_memory_mb": exc.required // (1024 * 1024) if exc.required else 0,
                "available_memory_mb": exc.current // (1024 * 1024) if exc.current else 0,
                "loaded_models": exc.loaded_models,
            }
        }
        if exc.loaded_models:
            unloadable = [m for m in exc.loaded_models if not m.get("pinned", False)]
            if unloadable:
                victim = unloadable[0]
                detail["error"]["suggestion"] = (
                    f"Unload model {victim['model_id']} "
                    f"(free ~{victim.get('memory_mb', '?')}MB) then retry"
                )
        raise HTTPException(status_code=503, detail=detail) from exc
    except ModelTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": str(exc), "type": "model_too_large"}},
        ) from exc
    except Exception as exc:
        err_msg = str(exc)
        if "Failed to process image" in err_msg or "Failed to process video" in err_msg:
            raise HTTPException(status_code=400, detail="Invalid media input")
        logger.exception(
            "Non-streaming chat failed for %s: %s(%s)",
            request_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(500, "Internal server error")
    finally:
        await _release()


async def _stream_chat_generator(
    request: ChatCompletionRequest,
    engine: Any,
    model_name: str,
    adapter_path: str | None,
    *,
    principal: str | None = None,
    profile_overrides: dict | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events for a streaming chat completion.

    Engine must be resolved BEFORE calling this generator (by _stream_chat)
    so that ModelNotFoundError / ModelLoadingError become proper HTTP
    status codes instead of unhandled ASGI 500s.
    """
    _start = time.perf_counter()

    async def _release() -> None:
        await _release_engine(model_name, adapter_path=adapter_path)

    await _inject_web_search(request)

    messages = _messages_for_engine(request.messages, getattr(engine, "is_mllm", False))
    sampling = _build_sampling_params(request, profile_overrides=profile_overrides)
    # Context scaling: cap max_tokens to model context window
    from .utils import cap_max_tokens_to_context

    sampling.max_tokens = cap_max_tokens_to_context(sampling.max_tokens, model_name)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # SSE keepalive: prevent client/proxy timeout during long inference
    from ..server import get_settings
    from .streaming import StreamingJSONEncoder

    _keepalive_interval = getattr(get_settings(), "sse_keepalive_seconds", 20.0) or 0.0
    keepalive = None
    if _keepalive_interval > 0:
        from .utils import SSEKeepalive

        keepalive = SSEKeepalive(interval_seconds=_keepalive_interval)
        keepalive.reset()

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
        yield _adapter.format_stream_chunk(first_chunk, request, encoder=encoder)

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
        # AtomCode 专题优化: enable_thinking 默认禁思考收敛单点 (流式路径, 2026-07-19)
        from .utils import resolve_enable_thinking_default

        resolve_enable_thinking_default(ct_kwargs_stream)
        compiled_grammar = _compile_grammar_for_request(engine, request)
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
            prefix_cache_boundary=(
                getattr(request, "prefix_cache_boundary", None)
                or _detect_prefix_cache_boundary(request.messages)
            ),
            compiled_grammar=compiled_grammar,
        ):
            if gen.new_text:
                if keepalive:
                    keepalive.reset()
                accumulated += gen.new_text
                thinking_delta, content_delta = parser.feed(gen.new_text)
                if content_delta:
                    chunk = StreamChunk(
                        text=content_delta,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        cached_tokens=gen.cached_tokens,
                        logprobs=getattr(gen, "logprobs", None),
                    )
                    yield _adapter.format_stream_chunk(chunk, request, encoder=encoder)
                if thinking_delta:
                    rchunk = StreamChunk(
                        text="",
                        reasoning_content=thinking_delta,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        cached_tokens=gen.cached_tokens,
                        logprobs=getattr(gen, "logprobs", None),
                    )
                    yield _adapter.format_stream_chunk(rchunk, request, encoder=encoder)
                prompt_tokens = gen.prompt_tokens or prompt_tokens
                completion_tokens = gen.completion_tokens or completion_tokens
                cached_tokens = gen.cached_tokens or cached_tokens
            else:
                # No new text — maybe emit SSE keepalive ping
                if keepalive:
                    ping = keepalive.maybe_ping()
                    if ping:
                        yield ping

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

        record_llm_metrics(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            cached_tokens=cached_tokens or 0,
            generation_duration=time.perf_counter() - _start,
            model_id=model_name,
        )
        record_chat_session(
            getattr(request, "session_id", None),
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            cached_tokens=cached_tokens or 0,
            principal=principal,
        )

    except asyncio.CancelledError:
        logger.info("Client disconnected during streaming: %s", request_id)
        if engine:
            try:
                _t = asyncio.create_task(engine.abort_request(request_id))
                _pending_abort_tasks.add(_t)
                _t.add_done_callback(_pending_abort_tasks.discard)
                _t.add_done_callback(
                    lambda t: (
                        logger.warning(
                            "abort_request failed for %s: %s",
                            request_id,
                            t.exception(),
                        )
                        if not t.cancelled() and t.exception()
                        else None
                    )
                )
            except Exception:
                pass
        raise
    except AdapterPathError as exc:
        yield f'data: {{"error": {{"message": {str(exc)!r}, "status": 400}}}}\n\n'
    except ModelNotFoundError as exc:
        yield f'data: {{"error": {{"message": {str(exc)!r}, "status": 404}}}}\n\n'
    except (ModelLoadingError, ModelBusyError) as exc:
        logger.warning("Stream: model temporarily unavailable: %s", exc)
        yield f'data: {{"error": {{"message": {str(exc)!r}, "status": 503, "type": "server_busy"}}}}\n\n'
    except InsufficientMemoryError as exc:
        logger.warning("Stream: insufficient memory: %s", exc)
        import json as _json
        err_detail = {
            "message": f"Model {exc.model_id} not loaded and insufficient memory",
            "status": 503,
            "type": "model_unavailable",
            "required_memory_mb": exc.required // (1024 * 1024) if exc.required else 0,
            "available_memory_mb": exc.current // (1024 * 1024) if exc.current else 0,
            "loaded_models": exc.loaded_models,
        }
        if exc.loaded_models:
            unloadable = [m for m in exc.loaded_models if not m.get("pinned", False)]
            if unloadable:
                victim = unloadable[0]
                err_detail["suggestion"] = (
                    f"Unload model {victim['model_id']} "
                    f"(free ~{victim.get('memory_mb', '?')}MB) then retry"
                )
        yield f"data: {_json.dumps({'error': err_detail})}\n\n"
    except ModelTooLargeError as exc:
        yield f'data: {{"error": {{"message": {str(exc)!r}, "status": 413, "type": "model_too_large"}}}}\n\n'
    except Exception as exc:
        err_msg = str(exc)
        if "Failed to process image" in err_msg or "Failed to process video" in err_msg:
            yield 'data: {"error": {"message": "Invalid media input", "status": 400}}\n\n'
        else:
            logger.exception(
                "Streaming chat failed for %s: %s(%s)",
                request_id,
                type(exc).__name__,
                exc,
            )
            yield 'data: {"error": {"message": "Internal server error"}}\n\n'
    finally:
        await _release()


async def _stream_chat(
    request: ChatCompletionRequest,
    *,
    _skip_cap_check: bool = False,
    principal: str | None = None,
) -> StreamingResponse:
    """Execute a streaming chat completion.

    Resolves the engine BEFORE creating the StreamingResponse so that
    ModelNotFoundError / ModelLoadingError / etc. are caught by the route
    handler's exception handlers and become proper HTTP 404/503 responses
    instead of unhandled ASGI 500 errors after the stream has started.
    """
    from ..server import resolve_model_with_profile

    model_name, profile_overrides = resolve_model_with_profile(request.model)
    adapter_path = getattr(request, "adapters", None)

    # Resolve engine first — exceptions propagate to route handler
    engine = await _resolve_engine(model_name, adapter_path=adapter_path)
    if engine is None:
        await _release_engine(model_name, adapter_path=adapter_path)
        raise HTTPException(404, f"Model {model_name} not available")

    # #205 Guard: reject engines without stream_chat capability
    if not _skip_cap_check:
        try:
            check_chat_capability(engine, "stream_chat", model_name)
        except HTTPException:
            await _release_engine(model_name, adapter_path=adapter_path)
            raise

    # Reject multimodal content on text-only models
    if not _skip_cap_check:
        try:
            check_multimodal_content(engine, request.messages, model_name)
        except HTTPException:
            await _release_engine(model_name, adapter_path=adapter_path)
            raise

    return StreamingResponse(
        _stream_chat_generator(
            request,
            engine,
            model_name,
            adapter_path,
            principal=principal,
            profile_overrides=profile_overrides,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _create_markitdown_chat_completion(
    request: ChatCompletionRequest,
) -> Any:
    from .markitdown import (
        MARKITDOWN_MODEL_ID,
        MarkItDownRequestError,
        convert_messages_to_markdown_async,
        markitdown_model_visible,
        stream_messages_to_markdown_async,
    )

    settings = _get_settings()
    global_settings = getattr(settings, "global_settings", None) if settings else None

    if not markitdown_model_visible(global_settings):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {MARKITDOWN_MODEL_ID}",
        )

    if request.stream:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        markdown_chunks = stream_messages_to_markdown_async(
            request.messages,
            global_settings=global_settings,
            engine_pool=_pool,
            settings_manager=None,
            get_sampling_params=None,
            latest_user_only=True,
        )
        return StreamingResponse(
            _stream_markitdown_response(request, markdown_chunks, response_id),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    async def _build_markitdown_completion():
        try:
            markdown = await convert_messages_to_markdown_async(
                request.messages,
                global_settings=global_settings,
                engine_pool=_pool,
                settings_manager=None,
                get_sampling_params=None,
                latest_user_only=True,
            )
        except MarkItDownRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500, detail="Internal server error"
            ) from exc

        if not markdown:
            raise HTTPException(
                status_code=400,
                detail="No text or supported file content found for MarkItDown.",
            )

        logger.info("MarkItDown completion converted request to markdown")
        return _build_markitdown_response(request, markdown, response_id)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    result = await _build_markitdown_completion()
    return result


def _build_markitdown_response(
    request: ChatCompletionRequest,
    markdown: str,
    response_id: str,
) -> ChatCompletionResponse:
    from .openai_models import ChatCompletionChoice, ChatCompletionResponse

    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": markdown},
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


async def _stream_markitdown_response(
    request: ChatCompletionRequest,
    markdown_chunks: AsyncIterator,
    response_id: str,
) -> AsyncIterator[str]:
    import json as _json

    model = request.model
    created = int(time.time())

    role_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {_json.dumps(role_chunk)}\n\n"

    async for chunk in markdown_chunks:
        if not chunk:
            continue
        content_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
            ],
        }
        yield f"data: {_json.dumps(content_chunk)}\n\n"

    done_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {_json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _get_settings() -> Any:
    from ..server import get_settings

    return get_settings()


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
) -> Any:
    """Handle OpenAI-compatible chat completion requests."""
    from .markitdown import is_markitdown_model

    if is_markitdown_model(request.model):
        return await _create_markitdown_chat_completion(request)

    # #226 IDOR scope: bind recorded session stats to the authenticated caller.
    principal = request_principal(http_request)
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

    # Response cache — check before engine dispatch (non-streaming only)
    _cache_status = "MISS"
    _cache_key = None
    _cache_policy = None
    if not request.stream:
        from ..cache.response_cache import CachePolicy, get_response_cache

        cache = get_response_cache()
        http_headers = dict(http_request.headers)
        _cache_policy = cache.resolve_policy(request.temperature, http_headers)
        if _cache_policy != CachePolicy.BYPASS:
            _cache_key = cache.fingerprint(
                model=request.model,
                messages=[m.model_dump() for m in request.messages],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                tools=request.tools,
                response_format=getattr(request, "response_format", None),
                seed=request.seed,
                adapters=getattr(request, "adapters", None),
            )
            cached = cache.get(_cache_key)
            if cached is not None and _cache_policy != CachePolicy.WRITE_ONLY:
                _cache_status = "HIT"
                logger.info("Response cache HIT key=%s", _cache_key[:12])
                from starlette.responses import JSONResponse

                return JSONResponse(
                    content=cached,
                    headers={"X-Cache": "HIT"},
                )
            if _cache_policy == CachePolicy.ONLY_IF_CACHED:
                from starlette.responses import JSONResponse

                return JSONResponse(
                    content={
                        "error": {
                            "message": "Cache MISS and only-if-cached policy active",
                            "type": "cache_miss",
                        }
                    },
                    status_code=504,
                    headers={"X-Cache": "MISS"},
                )

    try:
        if request.stream:
            return await _stream_chat(request, principal=principal)
        else:
            result = await _run_chat(request, principal=principal)

            # Store in response cache on MISS
            if _cache_key and _cache_policy not in (
                None,
                CachePolicy.NO_STORE,
                CachePolicy.BYPASS,
            ):
                from ..cache.response_cache import CachePolicy, get_response_cache

                cache = get_response_cache()
                resp_dict = (
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
                cache.put(_cache_key, resp_dict)

            return result
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
        detail = {
            "error": {
                "type": "model_unavailable",
                "message": f"Model {exc.model_id} not loaded and insufficient memory",
                "required_memory_mb": exc.required // (1024 * 1024) if exc.required else 0,
                "available_memory_mb": exc.current // (1024 * 1024) if exc.current else 0,
                "loaded_models": exc.loaded_models,
            }
        }
        if exc.loaded_models:
            unloadable = [m for m in exc.loaded_models if not m.get("pinned", False)]
            if unloadable:
                victim = unloadable[0]
                detail["error"]["suggestion"] = (
                    f"Unload model {victim['model_id']} "
                    f"(free ~{victim.get('memory_mb', '?')}MB) then retry"
                )
        raise HTTPException(status_code=503, detail=detail) from exc
    except ModelTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": str(exc), "type": "model_too_large"}},
        ) from exc
    except Exception as exc:
        logger.exception("Chat completion failed: %s(%s)", type(exc).__name__, exc)
        raise HTTPException(500, "Internal server error")


@router.post("/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
) -> Any:
    """Handle legacy text completion requests."""
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
        return await _run_chat(chat_req, _skip_cap_check=True, principal=principal)
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
        detail = {
            "error": {
                "type": "model_unavailable",
                "message": f"Model {exc.model_id} not loaded and insufficient memory",
                "required_memory_mb": exc.required // (1024 * 1024) if exc.required else 0,
                "available_memory_mb": exc.current // (1024 * 1024) if exc.current else 0,
                "loaded_models": exc.loaded_models,
            }
        }
        if exc.loaded_models:
            unloadable = [m for m in exc.loaded_models if not m.get("pinned", False)]
            if unloadable:
                victim = unloadable[0]
                detail["error"]["suggestion"] = (
                    f"Unload model {victim['model_id']} "
                    f"(free ~{victim.get('memory_mb', '?')}MB) then retry"
                )
        raise HTTPException(status_code=503, detail=detail) from exc
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
    if _pool is None:
        return ModelsResponse(data=[])

    try:
        model_ids = (
            _pool.list_models()
            if _pool is not None and hasattr(_pool, "list_models")
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

    from .markitdown import MARKITDOWN_MODEL_ID, markitdown_model_visible

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
