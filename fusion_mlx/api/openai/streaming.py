# SPDX-License-Identifier: Apache-2.0
"""Streaming chat helpers extracted from chat.py."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ...exceptions import (
    AdapterPathError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from ...server_metrics import record_llm_disconnect_cancel, record_llm_metrics
from ...sessions import record_chat_session
from .._concurrency import concurrency_guarded
from .._disconnect_guard import handle_disconnect
from .._guards import (
    _build_insufficient_memory_detail,
    check_chat_capability,
    check_multimodal_content,
    check_tool_choice_support,
)
from ..adapters.base import StreamChunk
from ..context_scaling import (
    compute_scale_factor,
    get_context_scaling_settings,
    is_claude_code_request,
)
from ..openai_models import ChatCompletionRequest
from ..thinking import ThinkingParser
from ._common import (
    _adapter,
    _build_sampling_params,
    _detect_prefix_cache_boundary,
    _inject_web_search,
    _messages_for_engine,
    _release_engine,
    _resolve_engine,
    logger,
)
from .grammar import _compile_grammar_for_request


def _resolve_streaming_tool_parser(engine: Any, model_name: str) -> Any:
    # #385 增量 tool_call_delta 流式: 为流式路径解析工具解析器。
    # 复用 routes_internal.models.effective_parsers_for 的分层解析
    # (registry live state -> ServerConfig -> alias-profile)，无配置时
    # 回退 "auto" 自动探测。仅在 request.tools 存在时调用，不影响普通流。
    try:
        from ...config import get_config
        from ...model_aliases import resolve_profile
        from ...routes_internal.models import effective_parsers_for
        from ...tool_parsers import ToolParserManager

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
# tokens (not  tags). Their extract_reasoning_streaming correctly
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
            from ...model_aliases import resolve_profile
            from ...routes_internal.models import effective_parsers_for

            profile = resolve_profile(model_name)
            profile_reasoning = profile.reasoning_parser if profile else None
            _, parser_name = effective_parsers_for(model_name, None, profile_reasoning)
        except Exception:
            parser_name = None
        if not parser_name:
            from ...model_auto_config import detect_model_config

            auto = detect_model_config(model_name)
            if auto is not None:
                parser_name = auto.reasoning_parser
        if not parser_name or parser_name not in _CHANNEL_REASONING_PARSERS:
            return None
        from ...reasoning import get_parser

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
    resume_prompt_cache: list | None = None,
    resume_cached_tokens: int = 0,
    request_id: str | None = None,
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
    from ...tool_parsers.ui_tars_tool_parser import inject_ui_tars_sysprompt_for_lane

    messages = inject_ui_tars_sysprompt_for_lane(
        messages,
        model_name=request.model,
        tool_choice=getattr(request, "tool_choice", None),
        tools=getattr(request, "tools", None),
    )
    sampling = _build_sampling_params(request, profile_overrides=profile_overrides)
    # Context scaling: cap max_tokens to model context window
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
    request_id = request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # SSE keepalive: prevent client/proxy timeout during long inference
    from ...server import get_settings
    from ..streaming import StreamingJSONEncoder

    _keepalive_interval = getattr(get_settings(), "sse_keepalive_seconds", 20.0) or 0.0
    _is_cc = headers and is_claude_code_request(headers)
    if _is_cc and _keepalive_interval > 5.0:
        _keepalive_interval = 5.0
    keepalive = None
    if _keepalive_interval > 0:
        from ..utils import SSEKeepalive

        keepalive = SSEKeepalive(interval_seconds=_keepalive_interval)
        keepalive.reset()

    # Context scaling for Claude Code streaming
    _ctx_scale_factor: float | None = None
    if headers and is_claude_code_request(headers):
        from ...service.helpers import get_model_max_context

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
        _stream_ttft: float | None = None
        _stream_tps: float | None = None

        ct_kwargs_stream = dict(getattr(request, "chat_template_kwargs", {}) or {})
        # AtomCode 专题优化: enable_thinking 默认禁思考收敛单点 (流式路径, 2026-07-19)
        from ..utils import resolve_enable_thinking_default

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
            resume_prompt_cache=resume_prompt_cache,
            resume_cached_tokens=resume_cached_tokens,
        ):
            if getattr(gen, "time_to_first_token", None) is not None:
                _stream_ttft = gen.time_to_first_token
            if getattr(gen, "generation_tokens_per_second", None) is not None:
                _stream_tps = gen.generation_tokens_per_second
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
        from ...service.helpers import build_compact_hint, get_model_max_context

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
            time_to_first_token=_stream_ttft,
            generation_tokens_per_second=_stream_tps,
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
            from ...telemetry import emit
            from ...telemetry.activation_spec import (
                ACTIVATION_FIRST_INFERENCE,
                is_successful_inference,
            )
            from ...telemetry.coherence import is_empty

            _sct = completion_tokens or 0
            _spt = prompt_tokens or 0
            _sgen_dur = time.perf_counter() - _start
            _stps = (_sct / _sgen_dur) if _sgen_dur > 0 else 0.0
            _sua = headers.get("user-agent") if headers else None
            _sttft_ms = (_stream_ttft * 1000.0) if _stream_ttft else 0.0
            emit.request(
                endpoint="/v1/chat/completions",
                model_alias=model_name,
                stream=True,
                tool_call_used=bool(tool_calls_in_stream),
                prompt_tokens=_spt,
                completion_tokens=_sct,
                ttft_ms=_sttft_ms,
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
            handle_disconnect(request_id, engine)
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

        err_detail = _build_insufficient_memory_detail(exc)
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
    resume_prompt_cache: list | None = None,
    resume_cached_tokens: int = 0,
) -> StreamingResponse:
    """Execute a streaming chat completion.

    Resolves the engine BEFORE creating the StreamingResponse so that
    ModelNotFoundError / ModelLoadingError / etc. are caught by the route
    handler's exception handlers and become proper HTTP 404/503 responses
    instead of unhandled ASGI 500 errors after the stream has started.
    """
    from ...server import resolve_model_with_profile

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
    from ...service.helpers import (
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
        concurrency_guarded(
            _stream_chat_generator(
                request,
                engine,
                model_name,
                adapter_path,
                principal=principal,
                profile_overrides=profile_overrides,
                headers=headers,
                resume_prompt_cache=resume_prompt_cache,
                resume_cached_tokens=resume_cached_tokens,
            )
        ),
        media_type="text/event-stream",
        headers=_stream_headers,
    )
