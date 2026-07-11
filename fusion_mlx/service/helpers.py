# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for route handlers.

Adapted from Rapid-MLX service/helpers.py. Uses fusion-mlx server module
globals instead of get_config() for sampling/thinking/reasoning defaults.
Disconnect logic lives in disconnect_guard.py and is re-exported here.
"""

from __future__ import annotations

import asyncio  # noqa: F401
import hashlib
import inspect
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator  # noqa: F401

from fastapi import HTTPException
from starlette.requests import Request  # noqa: F401

from ..api.constants import (  # noqa: F401
    REASONING_CUTOFF_SENTINEL,
    RESCUE_TAIL_LENGTH,
)
from ..api.models import (
    OPENAI_REASONING_EFFORT_TO_MAX_TOKENS,
    CompletionTokensDetails,
    FunctionCall,
    PromptTokensDetails,
    TokenLogProb,
    ToolCall,
    TopLogProb,
    Usage,
)
from ..api.tool_calling import parse_tool_calls
from ..api.utils import sanitize_output, strip_reasoning_channel_markup
from ..engine.base import BaseEngine, GenerationOutput
from ..tool_parsers import ToolParserManager

logger = logging.getLogger(__name__)

_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9

SSE_RESPONSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


def _get_server_attr(name: str, default=None):
    try:
        from .. import server as _srv

        return getattr(_srv, name, default)
    except Exception:
        return default


def _get_engine_pool():
    try:
        from .. import server as _srv

        return _srv._server_state.get("engine_pool")
    except Exception:
        return None


# ── Admission control ──────────────────────────────────────────────


def _check_admission_or_503(engine) -> None:
    from ..scheduler import BackpressureError

    check = getattr(engine, "check_admission", None)
    if check is None:
        return
    try:
        check()
    except BackpressureError as exc:
        _raise_backpressure_503(exc)


def _release_admission_unless_committed(engine, committed: bool) -> None:
    if committed:
        return
    release = getattr(engine, "release_admission_reservation", None)
    if release is None:
        return
    try:
        release()
    except Exception:
        logger.warning(
            "release_admission_reservation raised on route finally",
            exc_info=True,
        )


def _raise_backpressure_503(exc: Exception) -> None:
    raise HTTPException(
        status_code=503,
        headers={"Retry-After": "1"},
        detail=(
            "Server is busy (max concurrent requests reached). "
            f"Retry after the Retry-After delay. ({exc})"
        ),
    )


# ── Finalize content + reasoning ──────────────────────────────────


def _finalize_content_and_reasoning(
    raw_text: str,
    cleaned_text: str,
    tool_calls: list,
    reasoning_parser,
    engine_reasoning_text: str = "",
    enable_thinking: bool | None = None,
    reasoning_max_tokens: int | None = None,
    finish_reason: str | None = None,
) -> tuple[str, str | None]:
    reasoning_text = None
    if engine_reasoning_text:
        truncated_think = (
            cleaned_text
            and "<think>" in cleaned_text
            and "</think>" not in cleaned_text
        )
        if truncated_think:
            cleaned_text = cleaned_text.partition("<think>")[0].rstrip()
            return cleaned_text, _truncate_reasoning_only(
                engine_reasoning_text, reasoning_max_tokens
            )
        if (
            raw_text
            and "<think>" in raw_text
            and "</think>" not in raw_text
            and not (cleaned_text and cleaned_text.strip())
        ):
            return cleaned_text or "", _truncate_reasoning_only(
                engine_reasoning_text, reasoning_max_tokens
            )
        if (
            finish_reason == "length"
            and cleaned_text
            and cleaned_text.strip()
            and reasoning_parser is not None
        ):
            is_open_in_think_attr = getattr(reasoning_parser, "is_open_in_think", None)
            parser_open = False
            if callable(is_open_in_think_attr):
                try:
                    parser_open = bool(is_open_in_think_attr(cleaned_text))
                except Exception:
                    parser_open = False
            engine_prefix_match = bool(
                engine_reasoning_text
                and isinstance(engine_reasoning_text, str)
                and cleaned_text.strip() == engine_reasoning_text.strip()
            )
            if parser_open or engine_prefix_match:
                return "", _truncate_reasoning_only(
                    engine_reasoning_text, reasoning_max_tokens
                )
        return _apply_reasoning_cap(
            cleaned_text,
            engine_reasoning_text,
            reasoning_max_tokens,
            has_tool_calls=bool(tool_calls),
        )
    if reasoning_parser is None:
        return _apply_reasoning_cap(
            cleaned_text,
            reasoning_text,
            reasoning_max_tokens,
            has_tool_calls=bool(tool_calls),
        )
    if _parser_accepts_enable_thinking(reasoning_parser):
        extract = lambda text: reasoning_parser.extract_reasoning(
            text, enable_thinking=enable_thinking
        )
    else:
        extract = lambda text: reasoning_parser.extract_reasoning(text)
    if tool_calls:
        reasoning_text, _ = extract(raw_text)
    else:
        text_to_parse = cleaned_text or raw_text
        new_reasoning, new_cleaned = extract(text_to_parse)
        first_parse_was_case4 = (
            new_reasoning is not None
            and new_cleaned is None
            and bool(text_to_parse)
            and "<think>" not in text_to_parse
            and "</think>" not in text_to_parse
        )
        first_parse_was_truncated_think = (
            new_reasoning is not None
            and new_cleaned is None
            and bool(text_to_parse)
            and "<think>" in text_to_parse
            and "</think>" not in text_to_parse
        )
        if new_reasoning is None and raw_text and raw_text != text_to_parse:
            retry_reasoning, _ = extract(raw_text)
            if retry_reasoning is not None:
                new_reasoning = retry_reasoning
        reasoning_text = new_reasoning
        if new_cleaned is not None:
            cleaned_text = new_cleaned
        if (
            finish_reason == "length"
            and cleaned_text
            and not first_parse_was_truncated_think
        ):
            is_open_in_think = getattr(reasoning_parser, "is_open_in_think", None)
            open_in_think = False
            if callable(is_open_in_think):
                try:
                    open_in_think = bool(is_open_in_think(cleaned_text))
                except Exception:
                    open_in_think = False
            if not open_in_think and engine_reasoning_text:
                open_in_think = True
            if open_in_think:
                from ..reasoning import finalize_truncation

                if reasoning_text:
                    cleaned_text = ""
                else:
                    routed_reasoning, routed_content = finalize_truncation(
                        True, cleaned_text
                    )
                    cleaned_text = routed_content or ""
                    reasoning_text = routed_reasoning or reasoning_text
                return cleaned_text, _truncate_reasoning_only(
                    reasoning_text, reasoning_max_tokens
                )
        if enable_thinking is True and first_parse_was_case4:
            cleaned_text = ""
        if first_parse_was_truncated_think:
            cleaned_text = cleaned_text.partition("<think>")[0].rstrip()
    return _apply_reasoning_cap(
        cleaned_text,
        reasoning_text,
        reasoning_max_tokens,
        has_tool_calls=bool(tool_calls),
    )


# ── Reasoning cap helpers ─────────────────────────────────────────


def _truncate_reasoning_only(
    reasoning_text: str | None, reasoning_max_tokens: int | None
) -> str | None:
    if not reasoning_text or not reasoning_max_tokens:
        return reasoning_text
    char_budget = reasoning_max_tokens * 4
    if len(reasoning_text) <= char_budget:
        return reasoning_text
    return reasoning_text[:char_budget]


def _apply_reasoning_cap(
    cleaned_text: str,
    reasoning_text: str | None,
    reasoning_max_tokens: int | None,
    has_tool_calls: bool = False,
) -> tuple[str, str | None]:
    if not reasoning_text or not reasoning_max_tokens:
        return cleaned_text, reasoning_text
    char_budget = reasoning_max_tokens * 4
    if len(reasoning_text) <= char_budget:
        return cleaned_text, reasoning_text
    truncated = reasoning_text[:char_budget]
    if has_tool_calls:
        return cleaned_text, truncated
    if not cleaned_text or not cleaned_text.strip():
        suffix = reasoning_text[char_budget:]
        return suffix, truncated
    return cleaned_text, truncated


# ── Silent-drop rescue ─────────────────────────────────────────────


def _is_truncated_mid_think(
    cleaned_text: str | None,
    reasoning_text: str | None,
    finish_reason: str | None,
) -> bool:
    if finish_reason != "length":
        return False
    if not cleaned_text or not cleaned_text.strip():
        return True
    if reasoning_text and reasoning_text.strip():
        return False
    return False


def _should_start_in_thinking(
    chat_template: str,
    enable_thinking: bool | None,
) -> bool:
    if not chat_template:
        return False
    if enable_thinking is False:
        return False
    has_think_tag = "EATURE" in chat_template or "<think>" in chat_template
    if not has_think_tag:
        return False
    return True


_CUTOFF_NOTICE_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_RESCUE_ENV_PRIMARY = "FUSION_REASONING_CUTOFF_NOTICE"
_RESCUE_ENV_LEGACY = "RAPID_MLX_REASONING_CUTOFF_NOTICE"


def _rescue_silent_drop_from_reasoning(
    final_content: str | None,
    reasoning_text: str | None,
    tool_calls: list | None,
    finish_reason: str | None = None,
    raw_text: str | None = None,
    *,
    reasoning_is_case4: bool = False,
    matched_stop: str | None = None,
    prompt_thinking_active: bool = False,
) -> str | None:
    logger.debug(
        "rescue_silent_drop: content=%r reasoning=%r tool_calls=%s finish=%s",
        (final_content or "")[:40],
        (reasoning_text or "")[:40],
        bool(tool_calls),
        finish_reason,
    )
    if final_content and final_content.strip():
        return final_content
    if tool_calls:
        return final_content
    if not reasoning_text or not reasoning_text.strip():
        return final_content
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    truncated_mid_think = (
        (
            finish_reason == "length"
            and raw_text
            and raw_text.lstrip().startswith(THINK_OPEN)
            and THINK_CLOSE not in raw_text
        )
        or (
            finish_reason == "stop"
            and matched_stop is not None
            and raw_text
            and raw_text.lstrip().startswith(THINK_OPEN)
            and THINK_CLOSE not in raw_text
        )
        or (finish_reason == "length" and reasoning_is_case4 and prompt_thinking_active)
        or (
            finish_reason == "stop"
            and reasoning_is_case4
            and matched_stop is not None
            and prompt_thinking_active
        )
    )
    if truncated_mid_think:
        return final_content
    if (
        finish_reason == "length"
        and raw_text
        and "<|channel>thought" in raw_text
        and "<channel|>" not in raw_text[raw_text.rfind("<|channel>thought") :]
    ):
        return final_content
    if (
        raw_text
        and "<|channel|>analysis<|message|>" in raw_text
        and "<|channel|>final<|message|>" not in raw_text
    ):
        return final_content
    return reasoning_text


def _cutoff_notice_enabled() -> bool:
    for env_name in (_RESCUE_ENV_PRIMARY, _RESCUE_ENV_LEGACY):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if raw.strip().lower() in _CUTOFF_NOTICE_DISABLED_VALUES:
            return False
    return True


def _build_reasoning_rescue_payload(reasoning_text: str) -> str:
    stripped = strip_reasoning_channel_markup(reasoning_text.rstrip())
    tail = stripped[-RESCUE_TAIL_LENGTH:]
    sanitized = sanitize_output(tail)
    if not sanitized:
        return REASONING_CUTOFF_SENTINEL
    return f"{REASONING_CUTOFF_SENTINEL}\n\n{sanitized}"


def _apply_reasoning_cutoff_notice(
    final_content: str | None,
    reasoning_text: str | None,
    tool_calls: list | None,
    finish_reason: str | None,
) -> str | None:
    if not _cutoff_notice_enabled():
        return final_content
    if finish_reason != "length":
        return final_content
    if final_content and final_content.strip():
        return final_content
    if tool_calls:
        return final_content
    if not reasoning_text or not reasoning_text.strip():
        return final_content
    return _build_reasoning_rescue_payload(reasoning_text)


# ── Response format validation ─────────────────────────────────────


def _validate_response_format(response_format) -> None:
    if response_format is None:
        return
    rf_type = getattr(response_format, "type", None)
    if rf_type is None:
        return
    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
    if rf_type not in ("text", "json_object", "json_schema"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format type: {rf_type}",
        )


def _is_structured_output_requested(request) -> bool:
    rf = getattr(request, "response_format", None)
    if rf is None:
        return False
    rf_type = getattr(rf, "type", None) or (
        rf.get("type") if isinstance(rf, dict) else None
    )
    return rf_type in ("json_object", "json_schema")


# ── Parser introspection + sampling cascade ───────────────────────


def _parser_accepts_enable_thinking(parser) -> bool:
    try:
        sig = inspect.signature(parser.extract_reasoning)
        return "enable_thinking" in sig.parameters
    except Exception:
        return False


def _cascade(request_value, server_default, fallback):
    if request_value is not None:
        return request_value
    if server_default is not None:
        return server_default
    return fallback


_TOOL_USE_SYSTEM_SUFFIX = (
    "You are a helpful assistant with access to the following tools. "
    "To use a tool, output a JSON object with 'name' and 'arguments' keys."
)

_TOOL_USE_REQUIRED_SUFFIX = (
    "You MUST use one of the provided tools to answer the user's question. "
    "Output a JSON object with 'name' and 'arguments' keys."
)


def _tool_use_required_named_suffix(tool_name: str) -> str:
    return (
        f"You MUST use the tool '{tool_name}' to answer the user's question. "
        f"Output a JSON object with 'name' and 'arguments' keys."
    )


# ── Resolution helpers ─────────────────────────────────────────────


def _resolve_model_name(request_model: str | None) -> str:
    if request_model:
        return request_model
    return _get_server_attr("_model_name") or "default"


def _aliases_match(request_model: str) -> bool:
    model_name = _get_server_attr("_model_name")
    model_alias = _get_server_attr("_model_alias")
    model_path = _get_server_attr("_model_path")
    if request_model == model_name:
        return True
    if model_alias and request_model == model_alias:
        return True
    if model_path and request_model == model_path:
        return True
    return False


def _resolve_request_alias_or_default(request_model: str | None) -> str | None:
    if not request_model or request_model == "default":
        return _get_server_attr("_model_name")
    return request_model


def _resolve_max_tokens(
    request_max_tokens: int | None,
    enable_thinking: bool | None = None,
) -> int | None:
    # Explicit per-request cap is always a hard ceiling - honored verbatim,
    # thinking or not (the operator/user set it deliberately).
    if request_max_tokens is not None:
        return request_max_tokens

    # enable_thinking=None means the route did not resolve the thinking
    # state - preserve the historical pass-through (None = "no cap / engine
    # default") so chat/completions/responses routes keep behaving as before.
    if enable_thinking is None:
        return request_max_tokens

    from ..config import get_config

    cfg = get_config()
    default = cfg.default_max_tokens

    # Thinking on + an IMPLICIT operator default: add CoT headroom
    # (default + thinking_token_budget) so reasoning isn't truncated before
    # the answer. An EXPLICIT operator default is a hard cap - no headroom
    # (the operator chose the ceiling knowingly).
    if enable_thinking and not cfg.default_max_tokens_is_explicit:
        return default + cfg.thinking_token_budget

    return default


def _resolve_temperature(request_temperature: float | None) -> float:
    return _cascade(
        request_temperature,
        _get_server_attr("_default_temperature"),
        _FALLBACK_TEMPERATURE,
    )


def _resolve_top_p(request_top_p: float | None) -> float:
    return _cascade(
        request_top_p,
        _get_server_attr("_default_top_p"),
        _FALLBACK_TOP_P,
    )


def _resolve_top_k(request_top_k: int | None) -> int | None:
    return _cascade(
        request_top_k,
        _get_server_attr("_default_top_k"),
        None,
    )


def _resolve_min_p(request_min_p: float | None) -> float | None:
    return _cascade(
        request_min_p,
        _get_server_attr("_default_min_p"),
        None,
    )


def _resolve_repetition_penalty(request_rp: float | None) -> float | None:
    return _cascade(
        request_rp,
        _get_server_attr("_default_repetition_penalty"),
        None,
    )


def _resolve_presence_penalty(request_pp: float | None) -> float | None:
    return _cascade(
        request_pp,
        _get_server_attr("_default_presence_penalty"),
        None,
    )


def _resolve_frequency_penalty(request_fp: float | None) -> float | None:
    return _cascade(
        request_fp,
        _get_server_attr("_default_frequency_penalty"),
        None,
    )


def _resolve_seed(request_seed: int | None) -> int | None:
    return request_seed


# ── Thinking auto-disable family ───────────────────────────────────


def _extract_thinking_from_request(request) -> bool | None:
    ctk = getattr(request, "chat_template_kwargs", None)
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        return ctk["enable_thinking"]
    return getattr(request, "enable_thinking", None)


def _resolve_enable_thinking(request) -> bool | None:
    no_thinking = _get_server_attr("_no_thinking", False)
    if no_thinking:
        return False
    return _extract_thinking_from_request(request)


def maybe_auto_disable_thinking_for_tools(
    request, enable_thinking: bool | None = None
) -> bool | None:
    tools = getattr(request, "tools", None)
    if not tools:
        return enable_thinking
    if _client_signalled_reasoning_intent(request):
        return False
    tool_choice = getattr(request, "tool_choice", None)
    if tool_choice == "none":
        return enable_thinking
    if enable_thinking is None or enable_thinking is True:
        ctk = getattr(request, "chat_template_kwargs", None)
        if isinstance(ctk, dict):
            ctk["enable_thinking"] = False
        else:
            try:
                request.chat_template_kwargs = {"enable_thinking": False}
            except Exception:
                logger.debug(
                    "maybe_auto_disable_thinking_for_tools: failed to set ctk",
                    exc_info=True,
                )
        _mark_thinking_auto_disabled(request)
        return True
    return enable_thinking


def _client_signalled_reasoning_intent(request, *extra) -> bool:
    for obj in (request, *extra):
        if obj is None:
            continue
        if _extract_thinking_from_request(obj) is not None:
            return True
        if getattr(obj, "reasoning_effort", None) is not None:
            return True
        if getattr(obj, "reasoning_max_tokens", None) is not None:
            return True
        reasoning = getattr(obj, "reasoning", None)
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort is not None:
                return True
    return False


def maybe_apply_reasoning_effort(request) -> bool:
    effort = getattr(request, "reasoning_effort", None)
    if not effort:
        return False
    if effort == "none":
        explicit_thinking = _extract_thinking_from_request(request)
        if explicit_thinking is not None:
            return False
        ctk = getattr(request, "chat_template_kwargs", None)
        if isinstance(ctk, dict):
            ctk["enable_thinking"] = False
        else:
            try:
                request.chat_template_kwargs = {"enable_thinking": False}
            except Exception:
                logger.debug(
                    "maybe_apply_reasoning_effort: failed to set ctk", exc_info=True
                )
        _mark_thinking_auto_disabled(request)
        return True
    max_tokens_map = OPENAI_REASONING_EFFORT_TO_MAX_TOKENS
    mapped = max_tokens_map.get(effort)
    if mapped is None:
        return False
    existing_rmt = getattr(request, "reasoning_max_tokens", None)
    if existing_rmt is not None:
        return False
    try:
        request.reasoning_max_tokens = mapped
    except Exception:
        logger.debug(
            "maybe_apply_reasoning_effort: failed to set reasoning_max_tokens",
            exc_info=True,
        )
    return True


def maybe_auto_disable_thinking_for_casual_chat(
    request, enable_thinking: bool | None
) -> bool | None:
    if enable_thinking is not True:
        return enable_thinking
    tools = getattr(request, "tools", None)
    if tools:
        return enable_thinking
    rf = getattr(request, "response_format", None)
    rf_type = getattr(rf, "type", None) if rf else None
    if rf_type in ("json_object", "json_schema"):
        return enable_thinking
    if _client_signalled_reasoning_intent(request):
        return enable_thinking
    return enable_thinking


def _mark_thinking_auto_disabled(request) -> None:
    try:
        request._auto_disabled_thinking = True
    except Exception:
        pass


_THINKING_FLAG_HONORING_PARSERS: frozenset[str] = frozenset({"qwen3"})


def enable_thinking_warning_header(request, parser_name: str | None) -> dict[str, str]:
    if not parser_name:
        return {}
    if parser_name in _THINKING_FLAG_HONORING_PARSERS:
        return {}
    ctk = getattr(request, "chat_template_kwargs", None)
    if not isinstance(ctk, dict) or "enable_thinking" not in ctk:
        return {}
    if getattr(request, "_auto_disabled_thinking", False):
        return {}
    return {"X-FusionMLX-Warning": f"enable_thinking ignored for parser={parser_name}"}


def _effective_enable_thinking(
    resolved: bool | None, model_name: str | None
) -> bool | None:
    if resolved is not None:
        return resolved
    if not model_name:
        return None
    return "coder" not in model_name.lower()


# ── Extended sampling kwargs ───────────────────────────────────────


def build_extended_sampling_kwargs(request) -> dict:
    kwargs: dict = {}
    for name, resolver in (
        ("top_k", _resolve_top_k),
        ("min_p", _resolve_min_p),
        ("repetition_penalty", _resolve_repetition_penalty),
        ("presence_penalty", _resolve_presence_penalty),
        ("frequency_penalty", _resolve_frequency_penalty),
        ("seed", _resolve_seed),
    ):
        value = resolver(getattr(request, name, None))
        if value is not None:
            kwargs[name] = value
    return kwargs


# ── Usage / logprobs ───────────────────────────────────────────────


def _build_usage(output: GenerationOutput, reasoning_text: str | None) -> Usage:
    rp_name = _get_server_attr("_reasoning_parser_name")
    total_completion = output.completion_tokens
    cached_tokens = getattr(output, "cached_tokens", 0) or 0
    prompt_details = (
        PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens else None
    )
    if reasoning_text and rp_name:
        reasoning_chars = len(reasoning_text)
        content_chars = len(getattr(output, "text", "") or "")
        total_chars = reasoning_chars + content_chars
        if total_chars > 0:
            reasoning_tokens = round(total_completion * reasoning_chars / total_chars)
            if reasoning_chars > 0:
                reasoning_tokens = max(1, reasoning_tokens)
            if content_chars > 0:
                reasoning_tokens = min(reasoning_tokens, max(0, total_completion - 1))
            else:
                reasoning_tokens = min(reasoning_tokens, total_completion)
        else:
            reasoning_tokens = 0
        return Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=total_completion,
            total_tokens=output.prompt_tokens + total_completion,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
            ),
            prompt_tokens_details=prompt_details,
        )
    return Usage(
        prompt_tokens=output.prompt_tokens,
        completion_tokens=total_completion,
        total_tokens=output.prompt_tokens + total_completion,
        prompt_tokens_details=prompt_details,
    )


def get_usage(output: GenerationOutput) -> Usage:
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    cached_tokens = getattr(output, "cached_tokens", 0) or 0
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        prompt_tokens_details=(
            PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens else None
        ),
    )


def _extract_streaming_token_logprobs(
    chunk, tokenizer, top_k: int
) -> list[TokenLogProb]:
    if chunk.logprobs is None or not getattr(chunk, "new_text", None):
        return []
    lps = chunk.logprobs if isinstance(chunk.logprobs, list) else [chunk.logprobs]
    tids = getattr(chunk, "new_token_ids", None) or chunk.tokens or [0]
    return [
        _extract_token_logprob(lp, tid, tokenizer, top_k) for lp, tid in zip(lps, tids)
    ]


def _extract_token_logprob(
    logprobs_array, token_id: int, tokenizer, top_k: int
) -> TokenLogProb:
    import mlx.core as mx
    import numpy as np

    if hasattr(logprobs_array, "astype"):
        logprobs_array = logprobs_array.astype(mx.float32)
    probs = np.array(logprobs_array).flatten()
    # top_k <= 0 means the caller requested no alternative logprobs (OpenAI
    # top_logprobs=0). np.argpartition(probs, -0)[-0:] would otherwise slice
    # [:] and return the entire vocabulary sorted. Return just the sampled
    # token with an empty list instead. (code-review #72)
    if top_k <= 0:
        sampled_text = tokenizer.decode([token_id])
        sampled_bytes = list(sampled_text.encode("utf-8", errors="replace"))
        return TokenLogProb(
            token=sampled_text,
            logprob=float(probs[token_id]) if token_id < len(probs) else 0.0,
            bytes=sampled_bytes,
            top_logprobs=[],
        )
    top_k = min(top_k, len(probs))
    top_indices = np.argpartition(probs, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(probs[top_indices])][::-1]

    top_logprobs = []
    for idx in top_indices:
        idx = int(idx)
        tok_text = tokenizer.decode([idx])
        tok_bytes = list(tok_text.encode("utf-8", errors="replace"))
        top_logprobs.append(
            TopLogProb(
                token=tok_text,
                logprob=float(probs[idx]),
                bytes=tok_bytes,
            )
        )

    sampled_text = tokenizer.decode([token_id])
    sampled_bytes = list(sampled_text.encode("utf-8", errors="replace"))

    return TokenLogProb(
        token=sampled_text,
        logprob=float(probs[token_id]) if token_id < len(probs) else 0.0,
        bytes=sampled_bytes,
        top_logprobs=top_logprobs,
    )


# ── Engine / validation ────────────────────────────────────────────


def get_engine(model_name: str | None = None) -> BaseEngine:
    pool = _get_engine_pool()
    if pool is not None:
        try:
            import asyncio

            loop = asyncio.get_running_loop()
            if loop.is_running():
                try:
                    entry = pool._entries.get(model_name)
                    if entry is not None:
                        return entry.engine
                    for entry in pool._entries.values():
                        return entry.engine
                except Exception:
                    pass
        except RuntimeError:
            pass
        try:
            for entry in pool._entries.values():
                return entry.engine
        except Exception:
            pass
    cfg = _get_server_attr("_config") or _get_server_attr("config")
    if cfg is not None:
        direct = getattr(cfg, "engine", None)
        if direct is not None:
            return direct
    from ..config import get_config

    direct = get_config().engine
    if direct is not None:
        return direct
    raise HTTPException(status_code=503, detail="Model not loaded")


def _resolve_reasoning_enabled(model_name: str | None) -> bool:
    rp = _get_server_attr("_reasoning_parser")
    rp_name = _get_server_attr("_reasoning_parser_name")
    return rp is not None or bool(rp_name)


# ── Unicode validation ─────────────────────────────────────────────


def _find_lone_surrogate(s: str) -> int | None:
    for i, ch in enumerate(s):
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            return i
    return None


def _scan_messages_for_lone_surrogates(messages: list) -> None:
    def _check(value, path: str) -> None:
        if isinstance(value, str):
            offset = _find_lone_surrogate(value)
            if offset is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid unicode in {path}: lone surrogate "
                        f"codepoint U+{ord(value[offset]):04X} at offset "
                        f"{offset} (surrogates must appear as paired "
                        "high/low to encode an astral codepoint)."
                    ),
                )
        elif isinstance(value, dict):
            for k, v in value.items():
                _check(v, f"{path}.{k}" if isinstance(k, str) else path)
        elif isinstance(value, list):
            for j, item in enumerate(value):
                _check(item, f"{path}[{j}]")
        elif hasattr(value, "model_dump"):
            _check(value.model_dump(), path)

    for i, msg in enumerate(messages):
        content = msg.content if hasattr(msg, "content") else msg.get("content")
        if content is not None:
            _check(content, f"messages[{i}].content")

        tcid = (
            msg.tool_call_id
            if hasattr(msg, "tool_call_id")
            else (msg.get("tool_call_id") if isinstance(msg, dict) else None)
        )
        if tcid is not None:
            _check(tcid, f"messages[{i}].tool_call_id")

        name = (
            msg.name
            if hasattr(msg, "name") and getattr(msg, "name", None) is not None
            else (msg.get("name") if isinstance(msg, dict) else None)
        )
        if name is not None:
            _check(name, f"messages[{i}].name")

        tcs = (
            msg.tool_calls
            if hasattr(msg, "tool_calls")
            else (msg.get("tool_calls") if isinstance(msg, dict) else None)
        )
        if tcs:
            _check(tcs, f"messages[{i}].tool_calls")


def _validate_model_name(request_model: str) -> None:
    if request_model is None:
        return
    if request_model == "":
        raise HTTPException(
            status_code=400,
            detail="model must not be empty",
        )
    model_name = _get_server_attr("_model_name")
    model_alias = _get_server_attr("_model_alias")
    model_path = _get_server_attr("_model_path")
    if not model_name:
        return
    accepted = {model_name}
    if model_alias:
        accepted.add(model_alias)
    if model_path:
        accepted.add(model_path)
    if request_model not in accepted:
        raise HTTPException(
            status_code=404,
            detail=f"The model `{request_model}` does not exist. "
            f"Available: {model_name}",
        )


# ── Tool call parsing ──────────────────────────────────────────────


def _parse_tool_calls_with_parser(
    output_text: str,
    request=None,
    *,
    structured_tool_calls: list[dict] | None = None,
) -> tuple[str, list | None]:
    if structured_tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                type="function",
                function=FunctionCall(
                    name=tc["name"],
                    arguments=tc["arguments"],
                ),
            )
            for tc in structured_tool_calls
        ]
        return output_text or "", tool_calls

    enable_auto_tool_choice = _get_server_attr("_enable_auto_tool_choice", False)
    tool_call_parser_name = _get_server_attr("_tool_call_parser")
    rp_name = _get_server_attr("_reasoning_parser_name")

    request_dict = request.model_dump() if request else None

    engine = None
    try:
        engine = get_engine()
    except Exception:
        pass
    tokenizer = None
    if engine is not None and hasattr(engine, "_tokenizer"):
        tokenizer = engine._tokenizer

    if not enable_auto_tool_choice or not tool_call_parser_name:
        if rp_name and request and request.tools:
            _PARSER_MAP = {"minimax": "minimax"}
            inferred = _PARSER_MAP.get(rp_name)
            if inferred:
                try:
                    parser_cls = ToolParserManager.get_tool_parser(inferred)
                    parser = parser_cls(tokenizer)
                    parser.reset()
                    result = parser.extract_tool_calls(output_text, request_dict)
                    if result.tools_called:
                        tool_calls = [
                            ToolCall(
                                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                type="function",
                                function=FunctionCall(
                                    name=tc["name"],
                                    arguments=tc["arguments"],
                                ),
                            )
                            for tc in result.tool_calls
                        ]
                        return result.content or "", tool_calls
                except Exception as e:
                    logger.debug(f"Auto-infer tool parser failed: {e}")
        return parse_tool_calls(output_text, request_dict)

    try:
        parser_cls = ToolParserManager.get_tool_parser(tool_call_parser_name)
        parser = parser_cls(tokenizer)
    except Exception as e:
        logger.warning(f"Failed to create tool parser '{tool_call_parser_name}': {e}")
        return parse_tool_calls(output_text, request_dict)

    try:
        parser.reset()
        result = parser.extract_tool_calls(output_text, request_dict)
        if result.tools_called:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in result.tool_calls
            ]
            return result.content or "", tool_calls
        else:
            return parse_tool_calls(output_text, request_dict)
    except Exception as e:
        logger.warning(f"Tool parser error: {e}")
        return parse_tool_calls(output_text, request_dict)


def _validate_tool_call_params(tool_calls: list, tools: list) -> None:
    from ..api.tool_logits import _extract_param_schemas, validate_param_value

    tool_defs = [t.model_dump() if hasattr(t, "model_dump") else t for t in tools]

    tool_by_name: dict[str, dict] = {}
    for tool in tool_defs:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function", tool)
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        if not name:
            continue
        scoped = _extract_param_schemas([tool])
        tool_by_name[name] = {k.split(".", 1)[1]: v for k, v in scoped.items()}

    for tc in tool_calls:
        func = tc.function if hasattr(tc, "function") else tc.get("function", {})
        func_name = func.name if hasattr(func, "name") else func.get("name", "")
        args_str = (
            func.arguments
            if hasattr(func, "arguments")
            else func.get("arguments", "{}")
        )

        called_tool_schemas = tool_by_name.get(func_name)
        if called_tool_schemas is None:
            continue

        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"Tool call '{func_name}': arguments is not valid JSON: {args_str!r}"
            )
            continue

        if not isinstance(args, dict):
            continue

        for param_name, param_value in args.items():
            schema = called_tool_schemas.get(param_name)
            if not schema:
                continue
            is_valid, error = validate_param_value(json.dumps(param_value), schema)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Tool call '{func_name}' parameter '{param_name}' "
                        f"violates declared schema: {error}. The model "
                        "produced a schema-violating argument value; retry "
                        "with a more constrained prompt or relax the schema."
                    ),
                )


# ── Message helpers ────────────────────────────────────────────────


def _inject_json_instruction(messages: list, instruction: str) -> list:
    messages = list(messages)

    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

    if system_idx is not None:
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{instruction}\n\n{existing}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{instruction}\n\n{existing}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


def _maybe_pin_system_prompt(messages: list) -> None:
    pin_system_prompt = _get_server_attr("_pin_system_prompt", False)
    if not pin_system_prompt:
        return

    engine = None
    try:
        engine = get_engine()
    except Exception:
        return
    if engine is None:
        return

    system_content = None
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", None)
            )
            if isinstance(content, str):
                system_content = content
                break

    if not system_content:
        return

    prompt_hash = hashlib.sha256(system_content.encode()).hexdigest()[:16]
    pinned_hash = _get_server_attr("_pinned_system_prompt_hash")
    if prompt_hash == pinned_hash:
        return

    try:
        tokenizer = None
        if hasattr(engine, "_tokenizer"):
            tokenizer = engine._tokenizer
        elif hasattr(engine, "_model") and hasattr(engine._model, "tokenizer"):
            tokenizer = engine._model.tokenizer

        if tokenizer is None:
            return

        system_tokens = tokenizer.encode(system_content)
        if not system_tokens or len(system_tokens) < 16:
            return

        if hasattr(engine, "_prefix_cache") and engine._prefix_cache is not None:
            cache = engine._prefix_cache
            if hasattr(cache, "pin_prefix"):
                if cache.pin_prefix(system_tokens):
                    try:
                        from .. import server as _srv

                        _srv._pinned_system_prompt_hash = prompt_hash
                    except Exception:
                        pass
                    logger.info(
                        f"Auto-pinned system prompt: {len(system_tokens)} tokens, "
                        f"hash={prompt_hash}"
                    )
                    return

        if hasattr(engine, "_cache_manager") and engine._cache_manager is not None:
            cache = engine._cache_manager
            if hasattr(cache, "pin_prefix"):
                if cache.pin_prefix(system_tokens):
                    try:
                        from .. import server as _srv

                        _srv._pinned_system_prompt_hash = prompt_hash
                    except Exception:
                        pass
                    logger.info(
                        f"Auto-pinned system prompt (trie): {len(system_tokens)} tokens, "
                        f"hash={prompt_hash}"
                    )
                    return

    except Exception as e:
        logger.debug(f"System prompt pinning failed: {e}")


# ── Disconnect guard re-exports ────────────────────────────────────


# ── Context-length pre-check ───────────────────────────────────────

_FALLBACK_MAX_CONTEXT_TOKENS = 4_194_304


def get_model_max_context(engine) -> int:
    def _maybe_int(value) -> int | None:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        if ivalue <= 0:
            return None
        return ivalue

    model = getattr(engine, "_model", None) or getattr(engine, "model", None)

    if model is not None:
        args = getattr(model, "args", None)
        if args is not None:
            direct = _maybe_int(getattr(args, "max_position_embeddings", None))
            if direct is not None:
                return direct
            text_cfg = getattr(args, "text_config", None)
            if text_cfg is not None:
                nested = _maybe_int(getattr(text_cfg, "max_position_embeddings", None))
                if nested is not None:
                    return nested
        config = getattr(model, "config", None)
        if config is not None:
            cfg_direct = _maybe_int(getattr(config, "max_position_embeddings", None))
            if cfg_direct is not None:
                return cfg_direct
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                nested = _maybe_int(getattr(text_cfg, "max_position_embeddings", None))
                if nested is not None:
                    return nested

    tokenizer = getattr(engine, "tokenizer", None) or getattr(
        engine, "_tokenizer", None
    )
    if tokenizer is not None:
        tok_max = getattr(tokenizer, "model_max_length", None)
        if tok_max is not None:
            if isinstance(tok_max, int | float) and 0 < tok_max < 10_000_000:
                return int(tok_max)

    return _FALLBACK_MAX_CONTEXT_TOKENS


def count_prompt_tokens(engine, prompt) -> int:
    if isinstance(prompt, list):
        if not prompt:
            return 0
        first = prompt[0]
        if isinstance(first, int):
            return len(prompt)
        if isinstance(first, list):
            try:
                return max((len(p) for p in prompt if isinstance(p, list)), default=0)
            except TypeError:
                return 0
        if isinstance(first, str) and len(prompt) == 1:
            prompt = first
        else:
            return 0
    if not isinstance(prompt, str):
        return 0

    tokenizer = getattr(engine, "tokenizer", None) or getattr(
        engine, "_tokenizer", None
    )
    if tokenizer is None:
        return 0
    try:
        bos = getattr(tokenizer, "bos_token", None)
        add_special_tokens = bos is None or not prompt.startswith(bos)
        token_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return len(token_ids)
    except Exception:
        logger.debug("count_prompt_tokens: tokenizer.encode failed", exc_info=True)
        return 0


def enforce_context_length(
    engine,
    prompt_tokens: int,
    *,
    max_tokens: int | None = None,
) -> None:
    max_context = get_model_max_context(engine)
    completion = int(max_tokens) if max_tokens else 0
    requested_total = int(prompt_tokens) + max(0, completion)
    if requested_total <= max_context:
        return

    detail = (
        f"This model's maximum context length is {max_context} tokens. "
        f"However, you requested {requested_total} tokens "
        f"({int(prompt_tokens)} prompt + {max(0, completion)} completion). "
        "Please reduce the length of the messages or completion."
    )
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": detail,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "messages",
            }
        },
    )


def _build_prompt_with_thinking_compat(
    build_prompt,
    messages: list,
    *,
    tools: list | None,
    enable_thinking: bool | None,
):
    try:
        return build_prompt(messages, tools=tools, enable_thinking=enable_thinking)
    except TypeError as exc:
        msg = str(exc)
        if "enable_thinking" not in msg or "unexpected keyword" not in msg.lower():
            raise
        return build_prompt(messages, tools=tools)


def enforce_context_length_for_messages(
    engine,
    messages: list,
    *,
    tools: list | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> int | None:
    if getattr(engine, "is_mllm", False):
        return None
    build_prompt = getattr(engine, "build_prompt", None)
    if build_prompt is None:
        return None
    try:
        prompt = _build_prompt_with_thinking_compat(
            build_prompt,
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
    except HTTPException:
        raise
    except Exception as exc:
        err_msg = str(exc)
        err_type = type(exc).__name__
        if (
            "TemplateError" in err_type
            or "template" in err_msg.lower()
            or ("user" in err_msg.lower() and "found" in err_msg.lower())
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Chat template error: {err_msg}",
            )
        return None
    if not prompt:
        return None
    prompt_tokens = count_prompt_tokens(engine, prompt)
    if prompt_tokens <= 0:
        return None
    enforce_context_length(engine, prompt_tokens, max_tokens=max_tokens)
    return prompt_tokens


def repair_messages_fit_context(
    engine,
    repair_messages: list,
    *,
    tools: list | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> bool:
    if getattr(engine, "is_mllm", False):
        return True
    build_prompt = getattr(engine, "build_prompt", None)
    if build_prompt is None:
        return True
    try:
        prompt = _build_prompt_with_thinking_compat(
            build_prompt,
            repair_messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
    except Exception:
        return True
    if not prompt:
        return True
    prompt_tokens = count_prompt_tokens(engine, prompt)
    if prompt_tokens <= 0:
        return True
    max_context = get_model_max_context(engine)
    completion = int(max_tokens) if max_tokens else 0
    requested_total = int(prompt_tokens) + max(0, completion)
    return requested_total <= max_context


def enforce_context_length_for_prompt(
    engine,
    prompt,
    *,
    max_tokens: int | None = None,
) -> None:
    if getattr(engine, "is_mllm", False):
        return
    if not prompt:
        return
    prompt_tokens = count_prompt_tokens(engine, prompt)
    if prompt_tokens <= 0:
        return
    enforce_context_length(engine, prompt_tokens, max_tokens=max_tokens)


from .disconnect_guard import (  # noqa: E402, F401
    _disconnect_guard,
    _wait_with_disconnect,
)
