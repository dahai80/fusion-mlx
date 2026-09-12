# SPDX-License-Identifier: Apache-2.0
"""
Stream filtering and tool conversion utilities for tool calling.

Includes the ToolCallStreamFilter that suppresses tool-call markup from
streaming content deltas, plus tool-format conversion helpers.
"""

import logging
import re
from typing import Any

from ..openai_models import ToolCall

logger = logging.getLogger(__name__)


class ToolCallStreamFilter:
    """Streaming filter that suppresses tool-call markup from content deltas.

    Detects known tool-call start envelopes during streaming and suppresses
    control markup from assistant-visible content. Supports tokenizer-defined
    delimiters, namespaced XML envelopes, and high-confidence bracket-format
    envelopes handled by ``parse_tool_calls``.

    Suppression is envelope-bounded: control markup is removed, then visible
    prose after a closed envelope continues streaming normally.

    Args:
        tokenizer: The model's tokenizer. Uses tokenizer-defined
            ``tool_call_start`` when available.
    """

    def __init__(self, tokenizer: Any):
        marker = getattr(tokenizer, "tool_call_start", None)
        marker_end = getattr(tokenizer, "tool_call_end", None)
        # Normalize None-like values but preserve empty strings.
        if marker is None:
            marker = ""
        if marker_end is None:
            marker_end = ""
        self._marker_pairs: list[tuple[str, str]] = [("<tool_call>", "</tool_call>")]
        self._suppress_after_markers: list[str] = []
        if marker:
            if marker_end:
                self._marker_pairs.insert(0, (marker, marker_end))
            else:
                # One-sided markers (e.g. Mistral "[TOOL_CALLS]" with no
                # end marker): suppress everything after the start marker.
                self._suppress_after_markers.append(marker)
        self._namespaced_open_re = re.compile(r"<([A-Za-z_][\w.-]*):tool_call>")
        self._bracket_prefixes = ["[Calling tool:", "[Tool call:"]
        self._bracket_call_re = re.compile(
            r"^\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)(?:\(({.*?})\))?\]",
            re.DOTALL,
        )
        self._buffer = ""
        self._suppressing_until: str | None = None
        self._suppressing = False

    @property
    def active(self) -> bool:
        """Whether this filter should run for tool-enabled streams."""
        return True

    def _find_start_envelope(self, text: str) -> tuple[int, int, str | None] | None:
        """Find earliest complete opening envelope.

        Returns:
            tuple(index, consume_len, close_marker_or_none)
            - close_marker_or_none is a close marker to wait for, or ``None``
                when the whole envelope is already contained in consume_len.
        """
        starts: list[tuple[int, int, str | None]] = []

        for marker, close in self._marker_pairs:
            idx = text.find(marker)
            if idx >= 0:
                starts.append((idx, len(marker), close))

        ns_match = self._namespaced_open_re.search(text)
        if ns_match:
            ns = ns_match.group(1)
            starts.append(
                (ns_match.start(), len(ns_match.group(0)), f"</{ns}:tool_call>")
            )

        for bp in self._bracket_prefixes:
            bracket_idx = text.find(bp)
            while bracket_idx >= 0:
                bracket_candidate = text[bracket_idx:]
                bracket_match = self._bracket_call_re.match(bracket_candidate)
                if bracket_match:
                    starts.append((bracket_idx, bracket_match.end(), None))
                bracket_idx = text.find(bp, bracket_idx + 1)

        # One-sided markers: suppress from start marker to end of buffer.
        for sa_marker in self._suppress_after_markers:
            idx = text.find(sa_marker)
            if idx >= 0:
                starts.append((idx, len(text) - idx, "__suppress_permanently__"))

        if not starts:
            return None
        return min(starts, key=lambda x: x[0])

    @staticmethod
    def _partial_prefix_len(text: str, marker: str) -> int:
        """Longest suffix of text that is a proper prefix of marker."""
        max_len = min(len(text), len(marker) - 1)
        for n in range(max_len, 0, -1):
            if text.endswith(marker[:n]):
                return n
        return 0

    @staticmethod
    def _could_be_partial_namespaced_open(candidate: str) -> bool:
        """Return True if candidate could prefix a namespaced <ns:tool_call> tag."""
        if not candidate.startswith("<"):
            return False
        if ">" in candidate:
            return False

        body = candidate[1:]
        if not body:
            return True
        if body.startswith("/"):
            return False

        if ":" not in body:
            return re.match(r"^[A-Za-z_][\w.-]*$", body) is not None

        ns, suffix = body.split(":", 1)
        if not re.match(r"^[A-Za-z_][\w.-]*$", ns):
            return False
        return "tool_call".startswith(suffix)

    def _partial_suffix_len(self, text: str) -> int:
        """Length of trailing suffix that might be an opening-marker prefix."""
        keep = 0
        for marker, _close in self._marker_pairs:
            keep = max(keep, self._partial_prefix_len(text, marker))

        last_lt = text.rfind("<")
        if last_lt >= 0:
            candidate = text[last_lt:]
            if self._could_be_partial_namespaced_open(candidate):
                keep = max(keep, len(candidate))

        # Partial prefix detection for bracket markers (e.g. "[", "[C",
        # "[Cal" could be start of "[Calling tool:" or "[Tool call:").
        for bp in self._bracket_prefixes:
            keep = max(keep, self._partial_prefix_len(text, bp))
        # Same for suppress-after markers (e.g. "[TOOL" for "[TOOL_CALLS]").
        for sa_marker in self._suppress_after_markers:
            keep = max(keep, self._partial_prefix_len(text, sa_marker))

        bracket_idx = -1
        for bp in self._bracket_prefixes:
            idx = text.rfind(bp)
            if idx > bracket_idx:
                bracket_idx = idx
        if bracket_idx >= 0:
            bracket_candidate = text[bracket_idx:]
            # Hold unresolved bracket prefix until we can classify parseable
            # envelope vs literal prose.
            if "]" not in bracket_candidate:
                keep = max(keep, len(bracket_candidate))
                # Do not cap unresolved bracket candidates: capping can leak
                # raw control markup once the prefix grows past the cap.
                return keep

        # Cap retained suffix window to avoid unbounded buffering on malformed text.
        return min(keep, 128)

    def _should_drop_tail_at_finish(self, tail: str) -> bool:
        """Whether unresolved tail should be suppressed under strict mode."""
        if not tail:
            return False

        for marker, _close in self._marker_pairs:
            if marker.startswith(tail):
                return True

        # Drop unresolved bracket tool-call prefixes
        for bp in self._bracket_prefixes:
            if tail.startswith(bp):
                return True

        # Drop unresolved suppress-after marker prefixes
        for sa_marker in self._suppress_after_markers:
            if sa_marker.startswith(tail) or tail.startswith(sa_marker):
                return True

        if not tail.startswith("<"):
            return False
        if ">" in tail:
            return False

        body = tail[1:]
        if not body:
            return True
        if body.startswith("/"):
            return False

        if ":" not in body:
            # Preserve plain literal tails like "<alpha".
            return False

        ns, suffix = body.split(":", 1)
        if not re.match(r"^[A-Za-z_][\w.-]*$", ns):
            return False
        return "tool_call".startswith(suffix)

    def _sanitize_prefix_before_suppression(self, text: str) -> str:
        """Strip unresolved bracket-control prefixes while preserving prose."""
        if not any(bp in text for bp in self._bracket_prefixes):
            return text

        out: list[str] = []
        cursor = 0
        while cursor < len(text):
            bracket_idx = -1
            bracket_prefix = ""
            for bp in self._bracket_prefixes:
                idx = text.find(bp, cursor)
                if idx >= 0 and (bracket_idx < 0 or idx < bracket_idx):
                    bracket_idx = idx
                    bracket_prefix = bp
            if bracket_idx < 0:
                out.append(text[cursor:])
                break

            out.append(text[cursor:bracket_idx])
            after_prefix = bracket_idx + len(bracket_prefix)
            close_idx = text.find("]", after_prefix)
            if close_idx < 0:
                # Drop only the marker token; keep following prose.
                cursor = after_prefix
                continue

            # Preserve balanced literal bracket text that is not being suppressed.
            out.append(text[bracket_idx : close_idx + 1])
            cursor = close_idx + 1

        return "".join(out)

    def feed(self, text: str) -> str:
        """Feed a content delta, return the portion safe to emit."""
        if self._suppressing or not text:
            return ""
        if not self.active:
            return text

        self._buffer += text
        out: list[str] = []

        while self._buffer:
            if self._suppressing_until == "__suppress_permanently__":
                self._suppressing = True
                self._suppressing_until = None
                self._buffer = ""
                break

            if self._suppressing_until is not None:
                end_idx = self._buffer.find(self._suppressing_until)
                if end_idx < 0:
                    keep = self._partial_prefix_len(
                        self._buffer, self._suppressing_until
                    )
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[end_idx + len(self._suppressing_until) :]
                self._suppressing_until = None
                continue

            start = self._find_start_envelope(self._buffer)
            if start:
                idx, consume_len, close_marker = start
                if idx > 0:
                    out.append(
                        self._sanitize_prefix_before_suppression(self._buffer[:idx])
                    )
                self._buffer = self._buffer[idx + consume_len :]
                if close_marker is not None:
                    self._suppressing_until = close_marker
                continue

            keep = self._partial_suffix_len(self._buffer)
            if keep == 0:
                out.append(self._buffer)
                self._buffer = ""
                break
            if len(self._buffer) > keep:
                out.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            break

        return "".join(out)

    def finish(self) -> str:
        """Flush remaining safe buffer content.

        In clean-output strict mode, unresolved marker-like suffixes are dropped
        so partial control markup does not leak into user-visible text.
        """
        if self._suppressing or self._suppressing_until is not None:
            # EH-5 (#811 audit 0906): stream ended mid-tool-call-envelope.
            # The unterminated marker text is dropped so control markup does
            # not leak, but the client gets plain text with no tool_calls and
            # no error event — an Agent waiting on the tool call will spin.
            logger.warning(
                "Stream ended inside an unterminated tool-call envelope — "
                "dropping partial marker, client receives plain text with "
                "NO tool_calls. suppressed=%s until=%r",
                self._suppressing,
                self._suppressing_until,
            )
            self._buffer = ""
            self._suppressing_until = None
            return ""

        keep = self._partial_suffix_len(self._buffer)
        if keep >= len(self._buffer):
            tail = self._buffer
            self._buffer = ""
            if self._should_drop_tail_at_finish(tail):
                # EH-5 (#811 audit 0906): trailing marker-like suffix dropped
                # at stream finish — same silent-degrade risk as above.
                logger.warning(
                    "Stream finished with a dropped tool-call-like tail "
                    "(%r) — client receives plain text with NO tool_calls.",
                    tail,
                )
                return ""
            return tail

        if keep:
            buf = self._buffer[:-keep]
        else:
            buf = self._buffer
        self._buffer = ""
        return buf


def convert_tools_for_template(tools: list | None) -> list[dict] | None:
    """
    Convert OpenAI tools format to format expected by tokenizer.apply_chat_template.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Template format (commonly used by models):
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Args:
        tools: List of ToolDefinition objects or dicts in OpenAI format

    Returns:
        List of tool definitions in template format, or None if no tools
    """
    if not tools:
        return None

    converted = []
    # FC-6 (#0907 audit): track hosted/non-local tool types that this
    # server cannot execute (web_search, file_search, code_interpreter,
    # image_generation, mcp, local_shell, namespace tools). They were
    # silently dropped; log a WARNING so the operator sees that a request's
    # tools were partially ignored rather than serving "as if no tools".
    dropped_types: list[str] = []
    for tool in tools:
        # Handle both Pydantic models and dicts
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            tool_func = tool.get("function")
            tool_name_direct = tool.get("name")
            tool_desc_direct = tool.get("description")
            tool_schema_direct = tool.get("input_schema")
        else:
            tool_type = getattr(tool, "type", None)
            tool_func = getattr(tool, "function", None)
            tool_name_direct = getattr(tool, "name", None)
            tool_desc_direct = getattr(tool, "description", None)
            tool_schema_direct = getattr(tool, "input_schema", None)

        # OpenAI format: {"type": "function", "function": {...}}
        if tool_type == "function" and tool_func:
            # Handle function as dict or Pydantic model
            if isinstance(tool_func, dict):
                func_name = tool_func.get("name", "")
                func_desc = tool_func.get("description", "")
                func_params = tool_func.get(
                    "parameters", {"type": "object", "properties": {}}
                )
            else:
                func_name = getattr(tool_func, "name", "")
                func_desc = getattr(tool_func, "description", "")
                func_params = getattr(
                    tool_func, "parameters", {"type": "object", "properties": {}}
                )

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func_desc,
                        "parameters": func_params,
                    },
                }
            )
            continue

        # Anthropic format: {"name": "...", "description": "...", "input_schema": {...}}
        if tool_name_direct and tool_schema_direct is not None:
            desc = tool_desc_direct or ""
            if isinstance(tool_schema_direct, dict):
                params = tool_schema_direct
            else:
                params = {"type": "object", "properties": {}}

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool_name_direct),
                        "description": str(desc),
                        "parameters": params,
                    },
                }
            )
            continue

        # FC-6: neither OpenAI function nor Anthropic schema format — a
        # hosted/non-local tool type this server cannot execute. Record
        # it instead of silently skipping.
        dropped_types.append(str(tool_type or tool_name_direct or "unknown"))

    if dropped_types:
        logger.warning(
            "FC-6: dropped %d non-local tool(s) local models cannot execute: %s. "
            "Request served as if those tools were absent.",
            len(dropped_types),
            ", ".join(dropped_types),
        )
    return converted if converted else None


# Parameter names that collide with JSON Schema keywords.
# Gemma 4 confuses these with schema-level fields and drops them from
# tool call output.  We rename them before the chat template and restore
# them after parsing the model's response.
_GEMMA4_COLLIDING_PARAMS = {"description"}
_GEMMA4_RENAME_PREFIX = "param_"


def enrich_tool_params_for_gemma4(tools: list[dict]) -> list[dict]:
    """Fix tool schemas for Gemma 4 models.

    1. Renames parameters whose names collide with JSON Schema keywords
        (e.g. ``description`` -> ``param_description``) so Gemma 4 doesn't
        confuse them with schema-level fields.
    2. Adds explicit descriptions to required parameters that lack them.

    Use :func:`restore_gemma4_param_names` on tool call arguments to
    reverse the renaming before returning them to the caller.
    """
    enriched = []
    for tool in tools:
        tool = dict(tool)
        func = dict(tool.get("function", {}))
        params = func.get("parameters", {})
        if isinstance(params, dict) and "properties" in params:
            params = dict(params)
            old_props = params.get("properties", {})
            required = list(params.get("required", []))
            new_props = {}
            new_required = []
            for pname, pdef in old_props.items():
                pdef = dict(pdef)
                if pname in _GEMMA4_COLLIDING_PARAMS:
                    new_name = _GEMMA4_RENAME_PREFIX + pname
                else:
                    new_name = pname
                if not pdef.get("description"):
                    label = "REQUIRED. " if pname in required else ""
                    pdef["description"] = (
                        f"{label}The '{pname}' value"
                        f" (type: {pdef.get('type', 'string')})"
                    )
                new_props[new_name] = pdef
                new_required.append(new_name if pname in required else None)
            params["properties"] = new_props
            params["required"] = [r for r in new_required if r]
            func["parameters"] = params
        tool["function"] = func
        enriched.append(tool)
    return enriched


def restore_gemma4_param_names(arguments: dict) -> dict:
    """Reverse the parameter renaming done by :func:`enrich_tool_params_for_gemma4`."""
    restored = {}
    for k, v in arguments.items():
        if k.startswith(_GEMMA4_RENAME_PREFIX):
            original = k[len(_GEMMA4_RENAME_PREFIX) :]
            if original in _GEMMA4_COLLIDING_PARAMS:
                restored[original] = v
                continue
        restored[k] = v
    return restored


def format_tool_call_for_message(tool_call: ToolCall) -> dict:
    """
    Format a ToolCall object for inclusion in a message.

    Args:
        tool_call: ToolCall object

    Returns:
        Dict representation suitable for message content
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }
