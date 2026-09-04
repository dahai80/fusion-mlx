# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API routes for fusion-mlx.

Provides FastAPI routes for:
- POST /v1/chat/completions   - Chat completion (streaming + non-streaming)
- POST /v1/completions         - Legacy text completion
- GET   /v1/models              - List available models
"""

import asyncio
import copy
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..api.adapters.base import InternalResponse, StreamChunk
from ..api.adapters.openai import OpenAIAdapter
from ..api.context_scaling import (
    compute_scale_factor,
    get_context_scaling_settings,
    is_claude_code_request,
    scale_usage,
)
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
from ..server_metrics import record_llm_disconnect_cancel, record_llm_metrics
from ..sessions import record_chat_session
from ._guards import (
    check_chat_capability,
    check_multimodal_content,
    check_tool_choice_support,
)
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
    if not getattr(request, "web_search", False):
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
    # Fallback when neither the request nor a profile sets max_tokens
    # (e.g. OpenAI-compatible clients that omit it, or AI SDK v6 which
    # silently drops the renamed maxTokens param). Use the operator-
    # configured ServerConfig.default_max_tokens, NOT a hard-coded 2048 —
    # 2048 truncates long structured completions (~3900 chars) before the
    # JSON closes, surfacing as finish=length + client-side parse failure.
    from ..config import get_config

    return SamplingParams(
        max_tokens=req.max_tokens
        or po.get("max_tokens")
        or get_config().default_max_tokens,
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


def _extract_strict_json_schema(req: ChatCompletionRequest):
    # #514: extract the JSON schema dict the post-generate validator
    # should validate against, but ONLY when the request asked for a
    # STRICT json_schema (strict=true). Returns None for non-strict /
    # json_object / absent response_format so the R12-4 postgen path
    # does not run on loose-schema requests (constrained decoding or
    # unconstrained generation is the intended behavior there).
    # Mirrors the schema extraction in _compile_grammar_for_request.
    from .tool_calling import is_strict_json_schema

    rf = getattr(req, "response_format", None)
    if rf is None:
        return None
    if not is_strict_json_schema(rf):
        return None
    if isinstance(rf, dict):
        schema = rf.get("json_schema", {})
        if isinstance(schema, dict) and "schema" in schema:
            return schema["schema"]
        return schema
    if hasattr(rf, "type") and rf.type == "json_schema":
        inner = getattr(rf, "json_schema", None)
        if inner and hasattr(inner, "schema_"):
            return inner.schema_
        if inner and hasattr(inner, "schema"):
            return inner.schema
    return None


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
    headers: dict | None = None,
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

    # Reject forced tool_choice on engines that opted out (e.g. DiffusionEngine)
    if not _skip_cap_check:
        try:
            check_tool_choice_support(engine, request, model_name)
        except HTTPException:
            await _release()
            raise

    await _inject_web_search(request)

    messages = _messages_for_engine(request.messages, getattr(engine, "is_mllm", False))
    from ..tool_parsers.ui_tars_tool_parser import inject_ui_tars_sysprompt_for_lane

    messages = inject_ui_tars_sysprompt_for_lane(
        messages,
        model_name=request.model,
        tool_choice=getattr(request, "tool_choice", None),
        tools=getattr(request, "tools", None),
    )
    sampling = _build_sampling_params(request, profile_overrides=profile_overrides)
    from .utils import cap_max_tokens_to_context

    # MLX-2: prompt token pre-check — reject if prompt exceeds 85% of context window
    prompt_token_estimate = 0
    from ..service.helpers import compute_prompt_tokens_for_messages

    prompt_token_estimate = compute_prompt_tokens_for_messages(
        engine, messages, tools=request.tools
    )
    from ..server import get_max_context_window

    _ctx_win = get_max_context_window(model_name)
    if _ctx_win and _ctx_win > 0 and prompt_token_estimate > 0:
        if prompt_token_estimate > _ctx_win * 0.85:
            logger.warning(
                "Prompt pre-check: prompt_tokens=%d > 85%% of context_window=%d, rejecting",
                prompt_token_estimate,
                _ctx_win,
            )
            raise ModelTooLargeError(
                f"Prompt ({prompt_token_estimate} tokens) exceeds 85% of context window ({_ctx_win} tokens). "
                f"Reduce conversation length or use /compact."
            )

    sampling.max_tokens = cap_max_tokens_to_context(
        sampling.max_tokens, model_name, prompt_token_estimate=prompt_token_estimate
    )
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
        # R12-4 (#514): strict json_schema post-generate validation +
        # single repair retry with a context-length guard. The chat
        # surface normally enforces strict mode via constrained
        # decoding (xgrammar/llguidance grammar compiler); but when
        # the guidance compiler is unavailable (tokenizer missing
        # eos_token_id, xgrammar fallback miss, test stubs with
        # supports_guided_generation=False) the model runs
        # UNCONSTRAINED and strict enforcement silently dropped to
        # 200-OK-with-violating-output. Re-validate the buffered
        # output here and - on violation - attempt one repair retry
        # guarded by repair_messages_fit_context (so a repair that
        # blows context surfaces a deterministic 422, NOT 502).
        # Mirrors /v1/responses via the shared
        # apply_strict_postgen_validation helper so the 422 envelope
        # + context guard cannot drift between surfaces.
        _strict_json_schema = _extract_strict_json_schema(request)
        if _strict_json_schema is not None:
            from .response_format_metrics import incr_strict_request
            from .strict_json_schema import (
                apply_strict_postgen_validation,
                strict_enforcement_enabled,
            )

            incr_strict_request()
            if strict_enforcement_enabled() and compiled_grammar is None:
                _chat_kwargs = {
                    "max_tokens": sampling.max_tokens,
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "top_k": sampling.top_k,
                    "min_p": sampling.min_p,
                    "repetition_penalty": getattr(sampling, "repetition_penalty", 1.0),
                    "presence_penalty": sampling.presence_penalty,
                    "tools": request.tools,
                    "stop": sampling.stop,
                    "chat_template_kwargs": ct_kwargs if ct_kwargs else None,
                    "compiled_grammar": compiled_grammar,
                }
                logger.info(
                    "Strict json_schema mode active on /v1/chat/completions "
                    "(no guided grammar compiled) - engaging R12-4 "
                    "post-generate validation + single repair retry."
                )
                gen = await apply_strict_postgen_validation(
                    engine,
                    messages,
                    _chat_kwargs,
                    gen,
                    _strict_json_schema,
                    param="response_format.json_schema",
                    metrics_prefix="chat",
                )
        # R12: route-level fallback tool-call extraction. Real engines
        # self-parse via _fallback_parse_tool_calls (engines/batched.py),
        # but engines that don't (or test harnesses) leave the hermes
        # envelope in gen.text and gen.tool_calls empty — the envelope
        # then leaks into message.content. Parse here as a safety net,
        # gated on tools present + no engine-emitted tool_calls so real
        # engines skip (no double-parse).
        if request.tools and not getattr(gen, "tool_calls", None):
            try:
                from .tool_calling import convert_tools_for_template, parse_tool_calls

                _dict_tools = convert_tools_for_template(request.tools) or request.tools
                _tok = getattr(engine, "_tokenizer", None) or getattr(
                    engine, "tokenizer", None
                )
                _cleaned, _tc_list = parse_tool_calls(gen.text, _tok, _dict_tools)
                if _tc_list:
                    _tc_dicts = []
                    for _tc in _tc_list:
                        _tc_dicts.append(
                            {
                                "id": _tc.id,
                                "type": _tc.type,
                                "function": {
                                    "name": _tc.function.name,
                                    "arguments": _tc.function.arguments,
                                },
                            }
                        )
                    gen = copy.deepcopy(gen)
                    gen.tool_calls = _tc_dicts
                    if (
                        _cleaned.strip()
                        and _cleaned.strip() != (gen.text or "").strip()
                    ):
                        gen.text = _cleaned
                    logger.info(
                        "r12 fallback tool-call parse: model=%s calls=%d",
                        model_name,
                        len(_tc_dicts),
                    )
            except Exception as e:
                logger.debug("r12 fallback tool-call parse failed: %s", e)
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
        try:
            from ..telemetry import emit
            from ..telemetry.activation_spec import (
                ACTIVATION_FIRST_INFERENCE,
                is_successful_inference,
            )
            from ..telemetry.coherence import (
                is_abnormally_short,
                is_empty,
                looks_like_garbage,
            )

            _ct = getattr(gen, "completion_tokens", 0) or 0
            _pt = getattr(gen, "prompt_tokens", 0) or 0
            _gen_dur = time.perf_counter() - _start
            _tps = (_ct / _gen_dur) if _gen_dur > 0 else 0.0
            _ua = headers.get("user-agent") if headers else None
            _out_text = gen.text or ""
            emit.request(
                endpoint="/v1/chat/completions",
                model_alias=model_name,
                stream=False,
                tool_call_used=bool(tool_calls),
                prompt_tokens=_pt,
                completion_tokens=_ct,
                ttft_ms=0.0,
                tps=_tps,
                status=200,
                caller_agent=_ua,
                output_degenerate=looks_like_garbage(_out_text, _ct),
                completion_empty=is_empty(_ct),
                completion_abnormally_short=is_abnormally_short(_out_text, _ct),
            )
            if is_successful_inference(200, _ct):
                emit.activation(
                    activation_kind=ACTIVATION_FIRST_INFERENCE,
                    surface=emit.server_surface(),
                )
        except Exception:
            logger.debug("telemetry request emit failed", exc_info=True)
        resp = _adapter.format_response(internal, request)

        # Context scaling for Claude Code via OpenAI API
        if headers and is_claude_code_request(headers):
            from ..service.helpers import get_model_max_context

            _enabled, _target = get_context_scaling_settings(
                getattr(_get_settings(), "global_settings", {})
            )
            if _enabled:
                _model_ctx = get_model_max_context(engine)
                _factor = compute_scale_factor(_model_ctx, _target)
                if _factor is not None:
                    _usage = resp.usage
                    _scaled = scale_usage(
                        {
                            "prompt_tokens": _usage.prompt_tokens,
                            "prompt_tokens_details": {
                                "cached_tokens": (
                                    _usage.prompt_tokens_details.cached_tokens
                                    if _usage.prompt_tokens_details
                                    else 0
                                ),
                            },
                        },
                        _factor,
                    )
                    _usage.prompt_tokens = _scaled["prompt_tokens"]
                    if _usage.prompt_tokens_details:
                        _usage.prompt_tokens_details.cached_tokens = _scaled.get(
                            "prompt_tokens_details", {}
                        ).get("cached_tokens", 0)
                    _usage.total_tokens = (
                        _usage.prompt_tokens + _usage.completion_tokens
                    )
                    logger.info(
                        "OpenAI context scaling: model_ctx=%d target=%d factor=%.4f",
                        _model_ctx,
                        _target,
                        _factor,
                    )

        # X-Context-Budget response header (#327)
        from ..service.helpers import (
            build_compact_hint,
            build_context_budget_headers,
            get_model_max_context,
        )

        _ctx_window = get_model_max_context(engine)
        _prompt_tok = resp.usage.prompt_tokens if resp.usage else 0
        _ctx_budget_headers = build_context_budget_headers(
            prompt_tokens=_prompt_tok,
            context_window=_ctx_window,
        )

        # MLX-4: compact suggestion hint appended to message content
        _compact_hint = build_compact_hint(_prompt_tok, _ctx_window)
        if _compact_hint and resp.choices:
            _choice = resp.choices[0]
            if _choice.message and _choice.message.content is not None:
                _choice.message.content = _choice.message.content + "\n" + _compact_hint
            elif _choice.message:
                _choice.message.content = _compact_hint

        if _ctx_budget_headers:
            from starlette.responses import JSONResponse

            _resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp
            return JSONResponse(content=_resp_dict, headers=_ctx_budget_headers)

        return resp
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
                "required_memory_mb": (
                    exc.required // (1024 * 1024) if exc.required else 0
                ),
                "used_memory_mb": (exc.current // (1024 * 1024) if exc.current else 0),
                "ceiling_memory_mb": (
                    exc.ceiling // (1024 * 1024) if exc.ceiling else 0
                ),
                "available_memory_mb": (
                    (exc.ceiling - exc.current) // (1024 * 1024)
                    if exc.ceiling and exc.ceiling > exc.current
                    else 0
                ),
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
        if "exceeds the per-batch cap" in err_msg:
            logger.warning(
                "Non-streaming chat hit per-batch prefill cap for %s: %s",
                request_id,
                err_msg,
            )
            raise HTTPException(status_code=400, detail=err_msg)
        logger.exception(
            "Non-streaming chat failed for %s: %s(%s)",
            request_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(500, "Internal server error")
    finally:
        await _release()


def _resolve_streaming_tool_parser(engine: Any, model_name: str) -> Any:
    # #385 增量 tool_call_delta 流式: 为流式路径解析工具解析器。
    # 复用 routes_internal.models.effective_parsers_for 的分层解析
    # (registry live state -> ServerConfig -> alias-profile)，无配置时
    # 回退 "auto" 自动探测。仅在 request.tools 存在时调用，不影响普通流。
    try:
        from ..config import get_config
        from ..model_aliases import resolve_profile
        from ..routes_internal.models import effective_parsers_for
        from ..tool_parsers import ToolParserManager

        cfg = get_config()
        profile = resolve_profile(model_name)
        profile_tool = profile.tool_call_parser if profile else None
        tool_name, _ = effective_parsers_for(model_name, profile_tool, None)
        if not tool_name:
            tool_name = "auto"
        tokenizer = getattr(engine, "_tokenizer", None)
        parser_cls = ToolParserManager.get_tool_parser(tool_name)
        logger.debug(
            "streaming tool parser resolved: parser=%s model=%s", tool_name, model_name
        )
        return parser_cls(tokenizer)
    except Exception as e:
        logger.debug("streaming tool parser resolve failed: %s", e)
        return None


# Channel-based reasoning parsers route their reasoning trace via channel
# tokens (not <think> tags). Their extract_reasoning_streaming correctly
# separates reasoning from content; the tag-based ThinkingParser does NOT
# (it passes channel markers through as content — issue #444). These are
# the parsers that MUST use the named streaming method; qwen3/deepseek are
# tag-based and already handled by ThinkingParser, so they stay on it.
_CHANNEL_REASONING_PARSERS = frozenset({"harmony", "gpt_oss", "gemma4"})


def _resolve_streaming_reasoning_parser(model_name: str) -> Any:
    # #444 流式 reasoning_parser: 解析 model_settings.reasoning_parser。
    # 仅对 channel-based 解析器 (harmony/gpt_oss/gemma4) 返回实例，其余
    # 返回 None (继续走 ThinkingParser 标签路径)。复用 effective_parsers_for
    # 分层解析 + detect_model_config 回退 (同 _apply_reasoning_parser 非流式)。
    try:
        parser_name = None
        try:
            from ..model_aliases import resolve_profile
            from ..routes_internal.models import effective_parsers_for

            profile = resolve_profile(model_name)
            profile_reasoning = profile.reasoning_parser if profile else None
            _, parser_name = effective_parsers_for(model_name, None, profile_reasoning)
        except Exception:
            parser_name = None
        if not parser_name:
            from ..model_auto_config import detect_model_config

            auto = detect_model_config(model_name)
            if auto is not None:
                parser_name = auto.reasoning_parser
        if not parser_name or parser_name not in _CHANNEL_REASONING_PARSERS:
            return None
        from ..reasoning import get_parser

        parser_cls = get_parser(parser_name)
        parser = parser_cls(tokenizer=None)
        parser.reset_state()
        logger.info(
            "streaming reasoning parser resolved: parser=%s model=%s",
            parser_name,
            model_name,
        )
        return parser
    except Exception as e:
        logger.debug("streaming reasoning parser resolve failed: %s", e)
        return None


async def _stream_chat_generator(
    request: ChatCompletionRequest,
    engine: Any,
    model_name: str,
    adapter_path: str | None,
    *,
    principal: str | None = None,
    profile_overrides: dict | None = None,
    headers: dict | None = None,
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
    from ..tool_parsers.ui_tars_tool_parser import inject_ui_tars_sysprompt_for_lane

    messages = inject_ui_tars_sysprompt_for_lane(
        messages,
        model_name=request.model,
        tool_choice=getattr(request, "tool_choice", None),
        tools=getattr(request, "tools", None),
    )
    sampling = _build_sampling_params(request, profile_overrides=profile_overrides)
    # Context scaling: cap max_tokens to model context window
    from .utils import cap_max_tokens_to_context

    # MLX-2: prompt token pre-check — reject if prompt exceeds 85% of context window
    prompt_token_estimate = 0
    from ..service.helpers import compute_prompt_tokens_for_messages

    prompt_token_estimate = compute_prompt_tokens_for_messages(
        engine, messages, tools=request.tools
    )
    from ..server import get_max_context_window

    _ctx_win = get_max_context_window(model_name)
    if _ctx_win and _ctx_win > 0 and prompt_token_estimate > 0:
        if prompt_token_estimate > _ctx_win * 0.85:
            logger.warning(
                "Prompt pre-check: prompt_tokens=%d > 85%% of context_window=%d, rejecting",
                prompt_token_estimate,
                _ctx_win,
            )
            raise ModelTooLargeError(
                f"Prompt ({prompt_token_estimate} tokens) exceeds 85% of context window ({_ctx_win} tokens). "
                f"Reduce conversation length or use /compact."
            )

    sampling.max_tokens = cap_max_tokens_to_context(
        sampling.max_tokens, model_name, prompt_token_estimate=prompt_token_estimate
    )
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # SSE keepalive: prevent client/proxy timeout during long inference
    from ..server import get_settings
    from .streaming import StreamingJSONEncoder

    _keepalive_interval = getattr(get_settings(), "sse_keepalive_seconds", 20.0) or 0.0
    _is_cc = headers and is_claude_code_request(headers)
    if _is_cc and _keepalive_interval > 5.0:
        _keepalive_interval = 5.0
    keepalive = None
    if _keepalive_interval > 0:
        from .utils import SSEKeepalive

        keepalive = SSEKeepalive(interval_seconds=_keepalive_interval)
        keepalive.reset()

    # Context scaling for Claude Code streaming
    _ctx_scale_factor: float | None = None
    if headers and is_claude_code_request(headers):
        from ..service.helpers import get_model_max_context

        _enabled, _target = get_context_scaling_settings(
            getattr(get_settings(), "global_settings", {})
        )
        if _enabled:
            _model_ctx = get_model_max_context(engine)
            _ctx_scale_factor = compute_scale_factor(_model_ctx, _target)
            if _ctx_scale_factor is not None:
                logger.info(
                    "Stream context scaling: model_ctx=%d target=%d factor=%.4f",
                    _model_ctx,
                    _target,
                    _ctx_scale_factor,
                )

    encoder = StreamingJSONEncoder(
        response_id=request_id,
        model=request.model,
        object_type="chat.completion.chunk",
    )

    try:
        # Claude Code: emit connected comment at stream start
        if _is_cc:
            yield ": connected\n\n"

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
        # #444 流式 channel-based reasoning parser: harmony/gpt_oss/gemma4
        # 用 channel token 路由推理，ThinkingParser 会把 channel 标记当
        # content 泄漏。检测到 channel 解析器时，delta 先过其
        # extract_reasoning_streaming 拆 reasoning/content，再走 ThinkingParser
        # 兜底标签。无 channel 解析器时 reasoning_parser=None，路径不变。
        reasoning_parser = _resolve_streaming_reasoning_parser(model_name)
        reasoning_prev_text = ""

        # #385 增量 tool_call_delta 流式: 仅当 request.tools 存在时启用工具解析器。
        # 无 tools 时 tool_parser=None，路径与改造前逐字节一致 (向后兼容)。
        tool_parser = None
        if request.tools:
            tool_parser = _resolve_streaming_tool_parser(engine, model_name)
        tool_text_accumulated = ""
        tool_calls_streamed = 0
        tool_calls_in_stream = False

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

                # #444 channel-based reasoning parser 先于 ThinkingParser 拆分:
                # harmony/gpt_oss/gemma4 的 channel 标记若直接进 ThinkingParser
                # 会被当 content 泄漏。reasoning_parser 存在时，先抽 reasoning，
                # 剩余 content 再走标签兜底。无 reasoning_parser 时 passthrough。
                reasoning_delta_pre = ""
                content_for_downstream = gen.new_text
                if reasoning_parser is not None:
                    try:
                        dmsg = reasoning_parser.extract_reasoning_streaming(
                            reasoning_prev_text,
                            reasoning_prev_text + gen.new_text,
                            gen.new_text,
                        )
                    except Exception as e:
                        logger.debug("streaming reasoning parse failed: %s", e)
                        dmsg = None
                    reasoning_prev_text += gen.new_text
                    if dmsg is not None:
                        reasoning_delta_pre = dmsg.reasoning or ""
                        content_for_downstream = dmsg.content or ""
                    else:
                        content_for_downstream = ""
                if reasoning_delta_pre:
                    rchunk_pre = StreamChunk(
                        text="",
                        reasoning_content=reasoning_delta_pre,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        cached_tokens=gen.cached_tokens,
                        logprobs=getattr(gen, "logprobs", None),
                    )
                    yield _adapter.format_stream_chunk(
                        rchunk_pre, request, encoder=encoder
                    )

                # #385 增量 tool_call_delta: 当启用工具解析器时，由解析器决定
                # 本段 delta 是普通 content、还是已闭合的 tool_call、或需抑制
                # (工具标记未闭合)。解析器返回 {"tool_calls": [...]} 时按 index
                # 去重，仅发射新增调用；返回 None 抑制整段 (处于标记内部)；
                # 否则取 result["content"] 经 ThinkingParser 拆分后发射。
                if tool_parser is not None:
                    if not content_for_downstream:
                        # channel reasoning parser 抑制了本段 content (纯推理)
                        prompt_tokens = gen.prompt_tokens or prompt_tokens
                        completion_tokens = gen.completion_tokens or completion_tokens
                        cached_tokens = gen.cached_tokens or cached_tokens
                    else:
                        prev_tool_text = tool_text_accumulated
                        tool_text_accumulated += content_for_downstream
                        try:
                            tresult = tool_parser.extract_tool_calls_streaming(
                                prev_tool_text,
                                tool_text_accumulated,
                                content_for_downstream,
                            )
                        except Exception as e:
                            logger.debug("streaming tool parse failed: %s", e)
                            tresult = {"content": content_for_downstream}

                        if tresult is None:
                            # 工具标记未闭合，抑制 content，仅更新计数
                            prompt_tokens = gen.prompt_tokens or prompt_tokens
                            completion_tokens = (
                                gen.completion_tokens or completion_tokens
                            )
                            cached_tokens = gen.cached_tokens or cached_tokens
                        else:
                            new_calls = tresult.get("tool_calls")
                            if new_calls:
                                for tc in new_calls:
                                    idx = tc.get("index", tool_calls_streamed)
                                    if idx < tool_calls_streamed:
                                        continue  # 已发射，跳过去重
                                    tool_calls_streamed = idx + 1
                                    tool_calls_in_stream = True
                                    tc_chunk = StreamChunk(
                                        tool_call_delta=[tc],
                                        prompt_tokens=gen.prompt_tokens,
                                        completion_tokens=gen.completion_tokens,
                                        cached_tokens=gen.cached_tokens,
                                        logprobs=getattr(gen, "logprobs", None),
                                    )
                                    yield _adapter.format_stream_chunk(
                                        tc_chunk, request, encoder=encoder
                                    )
                            # 解析器过滤后的 content (已剥离工具标记)，经 ThinkingParser
                            content_piece = tresult.get("content", "")
                            if content_piece:
                                thinking_delta, content_delta = parser.feed(
                                    content_piece
                                )
                                if content_delta:
                                    chunk = StreamChunk(
                                        text=content_delta,
                                        prompt_tokens=gen.prompt_tokens,
                                        completion_tokens=gen.completion_tokens,
                                        cached_tokens=gen.cached_tokens,
                                        logprobs=getattr(gen, "logprobs", None),
                                    )
                                    yield _adapter.format_stream_chunk(
                                        chunk, request, encoder=encoder
                                    )
                                if thinking_delta:
                                    rchunk = StreamChunk(
                                        text="",
                                        reasoning_content=thinking_delta,
                                        prompt_tokens=gen.prompt_tokens,
                                        completion_tokens=gen.completion_tokens,
                                        cached_tokens=gen.cached_tokens,
                                        logprobs=getattr(gen, "logprobs", None),
                                    )
                                    yield _adapter.format_stream_chunk(
                                        rchunk, request, encoder=encoder
                                    )
                            prompt_tokens = gen.prompt_tokens or prompt_tokens
                            completion_tokens = (
                                gen.completion_tokens or completion_tokens
                            )
                            cached_tokens = gen.cached_tokens or cached_tokens
                else:
                    if not content_for_downstream:
                        # channel reasoning parser 抑制了本段 (纯推理 delta)
                        prompt_tokens = gen.prompt_tokens or prompt_tokens
                        completion_tokens = gen.completion_tokens or completion_tokens
                        cached_tokens = gen.cached_tokens or cached_tokens
                    else:
                        thinking_delta, content_delta = parser.feed(
                            content_for_downstream
                        )
                        if content_delta:
                            chunk = StreamChunk(
                                text=content_delta,
                                prompt_tokens=gen.prompt_tokens,
                                completion_tokens=gen.completion_tokens,
                                cached_tokens=gen.cached_tokens,
                                logprobs=getattr(gen, "logprobs", None),
                            )
                            yield _adapter.format_stream_chunk(
                                chunk, request, encoder=encoder
                            )
                        if thinking_delta:
                            rchunk = StreamChunk(
                                text="",
                                reasoning_content=thinking_delta,
                                prompt_tokens=gen.prompt_tokens,
                                completion_tokens=gen.completion_tokens,
                                cached_tokens=gen.cached_tokens,
                                logprobs=getattr(gen, "logprobs", None),
                            )
                            yield _adapter.format_stream_chunk(
                                rchunk, request, encoder=encoder
                            )
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
                # #385: 已在流中增量发射 tool_call 时，仅标注 finish_reason，
                # 不再重复全量发射 (避免双份)。未增量发射时保留原 finalize 回退。
                if tool_calls_in_stream:
                    finish_reason = "tool_calls"
                elif gen.tool_calls:
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

        # MLX-4: compact suggestion hint appended as final content delta
        from ..service.helpers import build_compact_hint, get_model_max_context

        _ctx_window_stream = get_model_max_context(engine)
        _stream_prompt_tok = prompt_tokens
        if _ctx_scale_factor is not None:
            _stream_prompt_tok = int(prompt_tokens * _ctx_scale_factor)
        _compact_hint = build_compact_hint(_stream_prompt_tok, _ctx_window_stream)
        if _compact_hint:
            hint_chunk = StreamChunk(
                text="\n" + _compact_hint,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
            yield _adapter.format_stream_chunk(hint_chunk, request, encoder=encoder)

        # Final chunk with finish_reason
        _final_prompt = prompt_tokens
        _final_cached = cached_tokens
        if _ctx_scale_factor is not None:
            _final_prompt = int(prompt_tokens * _ctx_scale_factor)
            _final_cached = int(cached_tokens * _ctx_scale_factor)
        last_chunk = StreamChunk(
            text="",
            is_last=True,
            finish_reason=finish_reason,
            prompt_tokens=_final_prompt,
            completion_tokens=completion_tokens,
            cached_tokens=_final_cached,
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
        try:
            from ..telemetry import emit
            from ..telemetry.activation_spec import (
                ACTIVATION_FIRST_INFERENCE,
                is_successful_inference,
            )
            from ..telemetry.coherence import is_empty

            _sct = completion_tokens or 0
            _spt = prompt_tokens or 0
            _sgen_dur = time.perf_counter() - _start
            _stps = (_sct / _sgen_dur) if _sgen_dur > 0 else 0.0
            _sua = headers.get("user-agent") if headers else None
            emit.request(
                endpoint="/v1/chat/completions",
                model_alias=model_name,
                stream=True,
                tool_call_used=bool(tool_calls_in_stream),
                prompt_tokens=_spt,
                completion_tokens=_sct,
                ttft_ms=0.0,
                tps=_stps,
                status=200,
                caller_agent=_sua,
                output_degenerate=False,
                completion_empty=is_empty(_sct),
                completion_abnormally_short=False,
            )
            if is_successful_inference(200, _sct):
                emit.activation(
                    activation_kind=ACTIVATION_FIRST_INFERENCE,
                    surface=emit.server_surface(),
                )
        except Exception:
            logger.debug("telemetry streaming request emit failed", exc_info=True)

    except asyncio.CancelledError:
        logger.info("Client disconnected during streaming: %s", request_id)
        record_llm_disconnect_cancel()
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
            "used_memory_mb": exc.current // (1024 * 1024) if exc.current else 0,
            "ceiling_memory_mb": exc.ceiling // (1024 * 1024) if exc.ceiling else 0,
            "available_memory_mb": (
                (exc.ceiling - exc.current) // (1024 * 1024)
                if exc.ceiling and exc.ceiling > exc.current
                else 0
            ),
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
        elif "exceeds the per-batch cap" in err_msg:
            logger.warning(
                "Streaming chat hit per-batch prefill cap for %s: %s",
                request_id,
                err_msg,
            )
            payload = _json.dumps({"error": {"message": err_msg, "status": 400}})
            yield f"data: {payload}\n\n"
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
    headers: dict | None = None,
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

    # Reject forced tool_choice on engines that opted out (e.g. DiffusionEngine)
    if not _skip_cap_check:
        try:
            check_tool_choice_support(engine, request, model_name)
        except HTTPException:
            await _release_engine(model_name, adapter_path=adapter_path)
            raise

    # X-Context-Budget response header (#327)
    _stream_headers: dict[str, str] = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    from ..service.helpers import (
        build_context_budget_headers,
        compute_prompt_tokens_for_messages,
        get_model_max_context,
    )

    _ctx_window = get_model_max_context(engine)
    if _ctx_window > 0:
        _msg_dicts = [
            m.model_dump() if hasattr(m, "model_dump") else m for m in request.messages
        ]
        _est_prompt = compute_prompt_tokens_for_messages(
            engine,
            _msg_dicts,
            tools=request.tools,
        )
        _ctx_budget_headers = build_context_budget_headers(
            prompt_tokens=_est_prompt,
            context_window=_ctx_window,
        )
        _stream_headers.update(_ctx_budget_headers)

    return StreamingResponse(
        _stream_chat_generator(
            request,
            engine,
            model_name,
            adapter_path,
            principal=principal,
            profile_overrides=profile_overrides,
            headers=headers,
        ),
        media_type="text/event-stream",
        headers=_stream_headers,
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
        "OpenAI /chat: model=%s, stream=%s, max_tokens=%s, temp=%s, prompt=%r",
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
                tools=[
                    t.model_dump() if hasattr(t, "model_dump") else t
                    for t in (request.tools or [])
                ],
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
            return await _stream_chat(
                request, principal=principal, headers=dict(http_request.headers)
            )
        else:
            result = await _run_chat(
                request, principal=principal, headers=dict(http_request.headers)
            )

            # Store in response cache on MISS
            if _cache_key and _cache_policy not in (
                None,
                CachePolicy.NO_STORE,
                CachePolicy.BYPASS,
            ):
                from starlette.responses import JSONResponse

                from ..cache.response_cache import CachePolicy, get_response_cache

                cache = get_response_cache()
                if isinstance(result, JSONResponse):
                    resp_dict = result.body
                    if isinstance(resp_dict, bytes):
                        import json

                        resp_dict = json.loads(resp_dict)
                else:
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
                "required_memory_mb": (
                    exc.required // (1024 * 1024) if exc.required else 0
                ),
                "used_memory_mb": (exc.current // (1024 * 1024) if exc.current else 0),
                "ceiling_memory_mb": (
                    exc.ceiling // (1024 * 1024) if exc.ceiling else 0
                ),
                "available_memory_mb": (
                    (exc.ceiling - exc.current) // (1024 * 1024)
                    if exc.ceiling and exc.ceiling > exc.current
                    else 0
                ),
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
                "required_memory_mb": (
                    exc.required // (1024 * 1024) if exc.required else 0
                ),
                "used_memory_mb": (exc.current // (1024 * 1024) if exc.current else 0),
                "ceiling_memory_mb": (
                    exc.ceiling // (1024 * 1024) if exc.ceiling else 0
                ),
                "available_memory_mb": (
                    (exc.ceiling - exc.current) // (1024 * 1024)
                    if exc.ceiling and exc.ceiling > exc.current
                    else 0
                ),
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
