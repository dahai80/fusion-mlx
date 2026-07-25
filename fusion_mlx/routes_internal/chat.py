"""Compatibility shim: re-exports router + restores deleted helpers.

The original ``routes_internal/chat.py`` was deleted in f4e2c05.
Tests still import from this path. This shim re-exports ``router``
from the current ``api.openai_routes`` and restores the deleted
helper functions that tests depend on.
"""

import json
import logging
import re
import uuid

from ..api.openai_routes import router  # noqa: F401
from ..service.helpers import (
    _parse_tool_calls_with_parser,  # noqa: F401
    _resolve_enable_thinking,  # noqa: F401
    _resolve_temperature,  # noqa: F401
    _resolve_top_p,  # noqa: F401
    get_engine,  # noqa: F401
)

logger = logging.getLogger(__name__)

_SAFE_DEEPSEEK_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _tool_call_name(tc) -> str | None:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return fn.get("name")
        if fn is not None:
            return getattr(fn, "name", None)
        return tc.get("name")
    fn = getattr(tc, "function", None)
    if isinstance(fn, dict):
        return fn.get("name")
    if fn is not None:
        return getattr(fn, "name", None)
    return getattr(tc, "name", None)


def _forced_tool_call_prefix(parser_name: str | None, function_name: str) -> str | None:
    if not function_name:
        return None
    _verified_json_tool_call_parsers = {"hermes"}
    if parser_name in _verified_json_tool_call_parsers:
        return f'\u001b\n{{"name": {json.dumps(function_name)}, "arguments": '
    if parser_name == "deepseek_v31":
        if not _SAFE_DEEPSEEK_TOOL_NAME_RE.fullmatch(function_name):
            return None
        return (
            "<\uff5ctool\u2581calls\u2581begin\uff5c>"
            "<\uff5ctool\u2581call\u2581begin\uff5c>"
            f"{function_name}"
            "<\uff5ctool\u2581sep\uff5c>"
        )
    if parser_name in ("deepseek_v3", "deepseek_r1_0528"):
        if not _SAFE_DEEPSEEK_TOOL_NAME_RE.fullmatch(function_name):
            return None
        return (
            "<\uff5ctool\u2581calls\u2581begin\uff5c>"
            "<\uff5ctool\u2581call\u2581begin\uff5c>"
            f"function<\uff5ctool\u2581sep\uff5c>{function_name}\n```json\n"
        )
    return None


def _synthesize_forced_tool_call(
    name: str, arguments: str = "{}", *, raw_text: str | None = None
):
    from ..api.models import FunctionCall, ToolCall

    recovered = _recover_partial_tool_args(raw_text, expected_name=name)
    final_args = recovered if recovered is not None else arguments

    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=FunctionCall(name=name, arguments=final_args),
    )


def _recover_partial_tool_args(
    raw_text: str | None, expected_name: str | None = None
) -> str | None:
    if not raw_text:
        return None
    text = raw_text

    def _open_wire_span_start(idx: int) -> int | None:
        _WIRE_SPAN_OPENERS = (
            "\u001b",
            "<function=",
            "<function>",
            "[TOOL_CALLS]",
            "<|python_tag|>",
            "<|tool_calls_section_begin|>",
            "<minimax:tool_call>",
            "<invoke",
            "\u001b",
        )
        _WIRE_SPAN_CLOSERS = (
            "\u001b",
            "</function>",
            "[/TOOL_CALLS]",
            "<|tool_calls_section_end|>",
            "</minimax:tool_call>",
            "</invoke>",
            "\u001b",
        )
        prefix = text[:idx]
        op_pos = -1
        for opener in _WIRE_SPAN_OPENERS:
            pos = prefix.rfind(opener)
            if pos > op_pos:
                op_pos = pos
        if op_pos < 0:
            return None
        cl_pos = -1
        for closer in _WIRE_SPAN_CLOSERS:
            pos = prefix.rfind(closer)
            if pos > cl_pos:
                cl_pos = pos
        return op_pos if op_pos >= 0 and op_pos > cl_pos else None

    import re as _re

    pattern = _re.compile(r'"arguments"\s*:\s*\{')
    for m in pattern.finditer(text):
        idx = m.end() - 1
        if expected_name:
            span_start = _open_wire_span_start(idx)
            lookback = span_start if span_start is not None else max(0, idx - 512)
            window = text[lookback:idx]
            if (
                f'"name": "{expected_name}"' not in window
                and f'"name":"{expected_name}"' not in window
            ):
                continue
        depth = 0
        in_string = False
        escape = False
        pos = idx
        scan_end = min(len(text), idx + 8192)
        while pos < scan_end:
            ch = text[pos]
            if escape:
                escape = False
            elif in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx : pos + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            break
            pos += 1
    return None


def _normalize_ui_tars_tcs_for_chat(tool_calls: list | None) -> list | None:
    if not tool_calls:
        return tool_calls
    from ..tool_parsers.ui_tars_tool_parser import (
        normalize_ui_tars_chat_tool_call_arguments,
    )

    out = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            out.append(tc)
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        if not isinstance(args, str):
            out.append(tc)
            continue
        new_args = normalize_ui_tars_chat_tool_call_arguments(args, name)
        if new_args is args:
            out.append(tc)
            continue
        new_tc = dict(tc)
        new_fn = dict(fn)
        new_fn["arguments"] = new_args
        new_tc["function"] = new_fn
        out.append(new_tc)
    return out


def _is_harmony_cut_short_stream(
    reasoning_parser,
    accumulated_reasoning: str,
    accumulated_text: str,
    tool_calls_detected: bool,
) -> bool:
    rp_is_harmony = (
        type(reasoning_parser).__name__ == "HarmonyReasoningParser"
        if reasoning_parser is not None
        else False
    )
    return bool(
        rp_is_harmony
        and accumulated_reasoning
        and not accumulated_text
        and not tool_calls_detected
    )


async def stream_chat_completion(*args, **kwargs):
    """Compatibility stub - tests monkeypatch this at runtime."""
    if False:
        yield ""


async def stream_chat_completion_guided(*args, **kwargs):
    """Compatibility stub - tests monkeypatch this at runtime."""
    if False:
        yield ""


async def stream_chat_completion_strict_postgen(*args, **kwargs):
    """Compatibility stub - tests monkeypatch this at runtime."""
    if False:
        yield ""


async def _create_chat_completion_impl(*args, **kwargs):
    """Compatibility stub - tests monkeypatch this at runtime."""
    raise NotImplementedError(
        "_create_chat_completion_impl was removed; "
        "use fusion_mlx.api.openai_routes directly"
    )
