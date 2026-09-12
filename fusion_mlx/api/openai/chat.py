# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat completion routes and helpers."""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any

from fastapi import Depends, HTTPException, Request

from ...exceptions import (
    AdapterPathError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from ...middleware.auth import check_rate_limit, request_principal, verify_api_key
from ...server_metrics import record_llm_metrics
from ...sessions import record_chat_session
from .._concurrency import (
    acquire_request_slot,
    release_request_slot,
)
from .._guards import (
    build_model_error_response,
    check_chat_capability,
    check_multimodal_content,
    check_tool_choice_support,
)
from ..context_scaling import (
    compute_scale_factor,
    get_context_scaling_settings,
    is_claude_code_request,
    scale_usage,
)
from ..openai_models import ChatCompletionRequest, ChatCompletionResponse
from ._common import (
    _adapter,
    _build_sampling_params,
    _detect_prefix_cache_boundary,
    _get_settings,
    _inject_web_search,
    _messages_for_engine,
    _release_engine,
    _resolve_engine,
    logger,
    router,
)
from .grammar import (
    _compile_grammar_for_request,
    _extract_strict_json_schema,
    _gen_to_internal,
)
from .markitdown import _create_markitdown_chat_completion
from .streaming import _stream_chat


async def _run_chat(
    request: ChatCompletionRequest,
    *,
    _skip_cap_check: bool = False,
    principal: str | None = None,
    headers: dict | None = None,
) -> ChatCompletionResponse:
    """Execute a non-streaming chat completion."""
    from ...server import resolve_model_with_profile

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
    from ...tool_parsers.ui_tars_tool_parser import inject_ui_tars_sysprompt_for_lane

    messages = inject_ui_tars_sysprompt_for_lane(
        messages,
        model_name=request.model,
        tool_choice=getattr(request, "tool_choice", None),
        tools=getattr(request, "tools", None),
    )
    sampling = _build_sampling_params(request, profile_overrides=profile_overrides)
    from ..utils import cap_max_tokens_to_context

    # MLX-2: prompt token pre-check — reject if prompt exceeds 85% of context window
    prompt_token_estimate = 0
    from ...service.helpers import compute_prompt_tokens_for_messages

    prompt_token_estimate = compute_prompt_tokens_for_messages(
        engine, messages, tools=request.tools
    )
    from ...server import get_max_context_window

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
        from ..utils import resolve_enable_thinking_default

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
            from ..response_format_metrics import incr_strict_request
            from ..strict_json_schema import (
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
                from ..tool_calling import convert_tools_for_template, parse_tool_calls

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
            from ...telemetry import emit
            from ...telemetry.activation_spec import (
                ACTIVATION_FIRST_INFERENCE,
                is_successful_inference,
            )
            from ...telemetry.coherence import (
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
            from ...service.helpers import get_model_max_context

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
        from ...service.helpers import (
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
        raise build_model_error_response(exc, adapter="openai") from exc
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
        try:
            from ...telemetry import emit

            emit.error(
                category="request_failure",
                exc=exc,
                phase="request",
            )
        except Exception:
            logger.debug("telemetry error emit failed", exc_info=True)
        raise HTTPException(500, "Internal server error")
    finally:
        await _release()


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
) -> Any:
    """Handle OpenAI-compatible chat completion requests."""
    from ..markitdown import is_markitdown_model

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
        from ...cache.response_cache import CachePolicy, get_response_cache

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

                _hit_headers = {"X-Cache": "HIT"}
                _ci = getattr(request, "__pydantic_extra__", None) or {}
                if isinstance(_ci, dict) and _ci:
                    _hit_headers["X-Fusion-Ignored-Params"] = ",".join(
                        sorted(_ci.keys())
                    )
                return JSONResponse(
                    content=cached,
                    headers=_hit_headers,
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
        # §6.2: collect unrecognized params (extra="allow" on
        # ChatCompletionRequest) and surface them via
        # X-Fusion-Ignored-Params so Cursor/Claude Code/Ollama clients
        # know what was accepted-but-not-forwarded, without 400-ing.
        _ignored = getattr(request, "__pydantic_extra__", None) or {}
        _ignored_header = ""
        if isinstance(_ignored, dict) and _ignored:
            _ignored_header = ",".join(sorted(_ignored.keys()))
            logger.debug("X-Fusion-Ignored-Params: %s", _ignored_header)

        if request.stream:
            return await _stream_chat(
                request, principal=principal, headers=dict(http_request.headers)
            )
        else:
            await acquire_request_slot()
            try:
                result = await _run_chat(
                    request, principal=principal, headers=dict(http_request.headers)
                )
            finally:
                release_request_slot()

            # §6.2: attach X-Fusion-Ignored-Params to non-streaming responses
            if _ignored_header and isinstance(result, JSONResponse):
                result.headers["X-Fusion-Ignored-Params"] = _ignored_header

            # Store in response cache on MISS
            if _cache_key and _cache_policy not in (
                None,
                CachePolicy.NO_STORE,
                CachePolicy.BYPASS,
            ):
                from starlette.responses import JSONResponse

                from ...cache.response_cache import CachePolicy, get_response_cache

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
                cache.put(_cache_key, resp_dict, model=request.model or "")

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
        raise build_model_error_response(exc, adapter="openai") from exc
    except ModelTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": str(exc), "type": "model_too_large"}},
        ) from exc
    except Exception as exc:
        logger.exception("Chat completion failed: %s(%s)", type(exc).__name__, exc)
        raise HTTPException(500, "Internal server error")
