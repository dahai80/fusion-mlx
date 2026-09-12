# SPDX-License-Identifier: Apache-2.0
"""S1.3: Deterministic tool-call JSON argument repairer.

The tool_parsers/ registry parses well-formed tool-call markup but degrades
gracefully on broken JSON arguments (truncation, trailing commas, unescaped
chars, quantization-induced half-token boundaries) by *dropping* them.
This module repairs such broken JSON deterministically — no LLM regeneration
(Rule 5: decide with code, not tokens).

Strategies (applied in order, first success wins):
  1. Fast path: json.loads succeeds → return as-is.
  2. Trailing comma cleanup.
  3. Unclosed string literals (append closing quote).
  4. Unclosed brackets/braces (track depth, close in order).
  5. Truncated mid-value: rewind to last complete key-value pair, close object.

Mounted as an api/ output-layer middleware. Default ON (env
FUSION_MLX_TOOL_JSON_REPAIR=1); set =0 to disable.
"""

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_ENV_DISABLE = "FUSION_MLX_TOOL_JSON_REPAIR"


def is_repair_enabled() -> bool:
    return os.environ.get(_ENV_DISABLE, "1") != "0"


def repair_tool_call_json(raw: str) -> str:
    if not raw or not raw.strip():
        return "{}"
    s = raw.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False)
        if isinstance(obj, (list, str, int, float, bool)) or obj is None:
            return json.dumps({"value": obj}, ensure_ascii=False)
        return s
    except (json.JSONDecodeError, ValueError):
        pass
    repaired = _repair_truncated_json(s)
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            logger.debug(
                "tool_json_repair: fixed args (%d→%d bytes)", len(s), len(repaired)
            )
            return json.dumps(obj, ensure_ascii=False)
        return json.dumps({"value": obj}, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("tool_json_repair: unrecoverable args (%s): %.120r", exc, s)
        return "{}"


def repair_arguments(arguments: Any) -> str:
    if not is_repair_enabled():
        if isinstance(arguments, dict):
            return json.dumps(arguments, ensure_ascii=False)
        if isinstance(arguments, str):
            return arguments
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    if isinstance(arguments, str):
        return repair_tool_call_json(arguments)
    if arguments is None:
        return "{}"
    try:
        return json.dumps({"value": arguments}, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _repair_truncated_json(s: str) -> str:
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    if not s:
        return "{}"
    first = s[0]
    if first not in "{[":
        candidate = _try_wrap_scalar(s)
        if candidate is not None:
            return candidate
        return "{}"
    repaired = _close_unclosed(s)
    return repaired


def _try_wrap_scalar(s: str) -> str | None:
    s = s.strip().rstrip(",")
    if not s:
        return None
    if s.lower() in ("true", "false", "null"):
        return json.dumps({"value": json.loads(s.lower())})
    try:
        num = int(s)
        return json.dumps({"value": num})
    except ValueError:
        pass
    try:
        num = float(s)
        return json.dumps({"value": num})
    except ValueError:
        pass
    if s.startswith('"'):
        closed = _close_string(s)
        try:
            return json.dumps({"value": json.loads(closed)})
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _close_string(s: str) -> str:
    in_str = False
    escaped = False
    for i, ch in enumerate(s):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
    if in_str:
        s = s + '"'
    return s


def _close_unclosed(s: str) -> str:
    stack: list[str] = []
    in_str = False
    escaped = False
    last_complete_end = -1
    depth_at_complete: list[int] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch in "{[":
            stack.append(ch)
            i += 1
            continue
        if ch in "}]":
            if stack and _matches(stack[-1], ch):
                stack.pop()
                if not stack:
                    last_complete_end = i + 1
                    depth_at_complete.append(i + 1)
            i += 1
            continue
        if ch == "," and not stack:
            i += 1
            continue
        i += 1
    if in_str:
        s = s + '"'
        return _close_unclosed(s)
    if not stack:
        candidate = s[:last_complete_end] if last_complete_end > 0 else s
        candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        return candidate
    for opener in reversed(stack):
        s = s + ("}" if opener == "{" else "]")
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def _matches(opener: str, closer: str) -> bool:
    return (opener == "{" and closer == "}") or (opener == "[" and closer == "]")
