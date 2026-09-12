# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing utilities.

Uses mlx-lm's modular tool parser system to support multiple model formats:
- json_tools: Pure JSON format
- minimax_m2: MiniMax M2 XML format
- function_gemma: Google Gemma function calling format
- glm47: GLM-4.7 format
- qwen3_coder: Qwen3 Coder XML format

The tool parser is automatically selected based on the model's chat template.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from ..openai_models import FunctionCall, ToolCall
from ..tool_json_repair import repair_tool_call_json
from .stream_filter import ToolCallStreamFilter

logger = logging.getLogger(__name__)


def _decode_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    current: Any = value.strip()
    for _ in range(3):
        if not isinstance(current, str):
            return current
        stripped = current.strip()
        if not stripped or stripped[0] not in '[{"':
            return current
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            return current
        if parsed == current:
            return parsed
        current = parsed
    return current


def _tool_parser_result_to_dict(p: Any) -> dict:
    """Normalize tool parser output to a dict with 'name' and 'arguments'.

    mlx-lm's parsers return different types:
    - dict (json_tools, glm47, etc.)
    - dataclass (AnthropicTool, GemmaTool, etc.)
    - object with arbitrary attributes (minimax_m2, function_gemma)
    """
    if isinstance(p, dict):
        return p
    name = getattr(p, "name", None) or getattr(p, "function_name", "")
    arguments = getattr(p, "arguments", None) or getattr(p, "input", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"name": str(name or ""), "arguments": arguments}


def _serialize_tool_call_arguments(arguments: Any) -> str:
    """Serialize parser output to a JSON-object arguments string.

    Chat templates for models with native tool calling (Qwen 3.5/3.6 XML,
    GLM, MiniMax) iterate `arguments.items()` when the call is echoed back
    in history. Anything that does not represent a JSON object must be
    coerced to "{}" here so we never hand the client a non-JSON value that
    the next turn's template would crash on.
    """
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    # mlx-vlm / mlx-lm gemma4 parser returns a JSON-object string per the
    # OpenAI spec. Accept it when it parses back to a dict.
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        # Only attempt repair when the original was unparseable (parsed is
        # None) AND looks like a broken JSON object (starts with "{"). Bare
        # scalars/arrays that are valid non-dict JSON must NOT be repaired —
        # repair_tool_call_json wraps them into {"value": ...}, which would
        # break the "coerce non-dict to {}" contract below.
        if parsed is None and arguments.lstrip().startswith("{"):
            repaired = repair_tool_call_json(arguments)
            try:
                parsed_repaired = json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                parsed_repaired = None
            if isinstance(parsed_repaired, dict):
                logger.info(
                    "tool_calling: repaired broken JSON args "
                    "(orig=%.120r fixed=%.120r)",
                    arguments,
                    repaired,
                )
                return repaired
    logger.warning(
        "Tool parser returned non-dict arguments (type=%s, repr=%.200r); "
        "coercing to empty object to keep downstream template safe.",
        type(arguments).__name__,
        arguments,
    )
    return "{}"


@dataclass(frozen=True)
class ToolCallExtraction:
    """Parsed tool-call result plus sanitized reasoning text."""

    cleaned_text: str
    tool_calls: list[ToolCall] | None
    cleaned_thinking: str
    tool_calls_from_thinking: bool = False


def _parse_xml_tool_calls(text: str) -> tuple[str, list[ToolCall] | None]:
    """
    Fallback parser for XML-based tool call formats.

    Handles models that use <tool_call>...</tool_call> XML format, including:
    - GLM format: <tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    - Qwen/Llama format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    - Generic JSON: <tool_call>{"name": ..., "arguments": ...}</tool_call>

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    pattern = r"␝(.*?)␞"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        content = match.strip()
        try:
            # Try JSON format first: {"name": "func", "arguments": {...}}
            parsed = json.loads(content)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=_serialize_tool_call_arguments(arguments),
                    ),
                )
            )
            continue
        except (json.JSONDecodeError, AttributeError):
            pass

        # Qwen/Llama format: <function=name><parameter=key>value</parameter></function>
        func_match = re.match(r"<function=(\w+)>(.*?)</function>", content, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            params_text = func_match.group(2)
            arguments = {}
            for pm in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_text, re.DOTALL
            ):
                key = pm.group(1)
                val = pm.group(2).strip()
                try:
                    arguments[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = val
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
            continue

        # GLM XML format: func_name<arg_key>k</arg_key><arg_value>v</arg_value>...
        arg_keys = re.findall(r"<arg_key>(.*?)</arg_key>", content)
        arg_values = re.findall(r"<arg_value>(.*?)</arg_value>", content, re.DOTALL)
        if arg_keys:
            # Function name is the text before the first <arg_key>
            name_match = re.match(r"^(.*?)<arg_key>", content, re.DOTALL)
            func_name = (
                name_match.group(1).strip()
                if name_match
                else content.split("<")[0].strip()
            )
            arguments = {}
            for k, v in zip(arg_keys, arg_values):
                # Try to parse JSON values (arrays, objects, numbers, booleans)
                try:
                    arguments[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[k] = v
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )

    if not tool_calls:
        return text, None

    # Remove tool call tags from text
    cleaned = re.sub(r"␝.*?␞", "", text, flags=re.DOTALL).strip()
    return cleaned, tool_calls


def _parse_namespaced_tool_calls(
    text: str, namespace: str
) -> tuple[str, list[ToolCall] | None]:
    """
    Parse namespaced tool call tags like <minimax:tool_call>...</minimax:tool_call>.

    Handles the <invoke name="func"><parameter name="key">value</parameter></invoke>
    format used by MiniMax and similar models.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    tag_start = f"<{namespace}:tool_call>"
    tag_end = f"</{namespace}:tool_call>"
    pattern = re.escape(tag_start) + r"(.*?)" + re.escape(tag_end)
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        content = match.strip()
        # Parse <invoke name="func_name">...<parameter name="key">value</parameter>...</invoke>
        for invoke_match in re.finditer(
            r'<invoke\s+name="([^"]+)">(.*?)</invoke>', content, re.DOTALL
        ):
            func_name = invoke_match.group(1)
            params_text = invoke_match.group(2)
            arguments = {}
            for pm in re.finditer(
                r'<parameter\s+name="([^"]+)">(.*?)</parameter>', params_text, re.DOTALL
            ):
                key = pm.group(1)
                val = pm.group(2).strip()
                try:
                    arguments[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = val
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )

    if not tool_calls:
        return text, None

    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return cleaned, tool_calls


def _parse_bracket_tool_calls(text: str) -> tuple[str, list[ToolCall] | None]:
    """
    Fallback parser for bracket-style tool call formats.

    Recognizes both ``[Calling tool: name(args)]`` and ``[Tool call: name(args)]``
    prefixes, with or without arguments.  Models may emit the args-less form
    ``[Tool call: name]`` when mimicking conversation history.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    # Match with args first (higher fidelity)
    pattern_with_args = (
        r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)\(({.*?})\)\]"
    )
    matched_spans: list = []
    for match in re.finditer(pattern_with_args, text, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2)
        try:
            arguments = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            arguments = {"raw": args_str}
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )
        matched_spans.append(match.span())

    # Match without args (model-generated simplified form)
    pattern_no_args = r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)\]"
    for match in re.finditer(pattern_no_args, text):
        # Skip if this span overlaps with an already-matched with-args span
        start, end = match.span()
        if any(s <= start < e for s, e in matched_spans):
            continue
        name = match.group(1)
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments="{}",
                ),
            )
        )
        matched_spans.append((start, end))

    if not tool_calls:
        return text, None

    # Remove all matched spans from text
    cleaned = re.sub(pattern_with_args, "", text, flags=re.DOTALL)
    cleaned = re.sub(pattern_no_args, "", cleaned).strip()
    return cleaned, tool_calls


# ---------------------------------------------------------------------------
# Gemma 4 robust fallback parser
# ---------------------------------------------------------------------------


def _gemma4_args_to_json_robust(args_str: str) -> dict:
    """Convert Gemma 4 tool call args to a Python dict.

    Handles the common failure cases that mlx-lm's parser cannot:
    - Bare string values without ``<|"|>`` delimiters (e.g. ``{location: Tokyo}``)
    - Spaces after commas in key-value pairs
    """
    import regex

    # 1. Extract <|"|>-delimited strings and replace with placeholders
    strings: list[str] = []

    def _capture(m):
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = regex.sub(r'<\|"\|>(.*?)<\|"\|>', _capture, args_str, flags=regex.DOTALL)

    # 2. Quote bare keys (allow whitespace after { or ,)
    text = regex.sub(r"(?<=[{,])\s*(\w+)\s*:", r' "\1":', text)

    # 3. Restore captured strings as properly escaped JSON strings
    for i, s in enumerate(strings):
        text = text.replace(f"\x00{i}\x00", json.dumps(s))

    # 4. Try json.loads — works when all values are already valid JSON primitives
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 5. Quote bare string values that are not numbers, booleans, or null
    def _quote_bare(m):
        value = m.group(2).strip()
        suffix = m.group(3)
        if value.lower() in ("true", "false", "null"):
            return f": {value}{suffix}"
        try:
            json.loads(value)
            return f": {value}{suffix}"
        except (json.JSONDecodeError, ValueError):
            return f": {json.dumps(value)}{suffix}"

    text = regex.sub(r"(:\s*)([^\",\[\]{}\s][^,}]*?)(\s*[,}])", _quote_bare, text)
    return json.loads(text)


def _parse_gemma4_tool_call_fallback(text: str) -> dict | list:
    """Robust fallback parser for Gemma 4 ``call:name{args}`` format.

    Activated only for Gemma 4 models (guarded by ``tool_call_start`` check).
    Extends mlx-lm's parser to handle:
    - Bare string values without ``<|"|>`` delimiters
    - Colons / dots / hyphens in function names
    """
    import regex

    pattern = regex.compile(r"call:([\w:.-]+)(\{(?:[^{}]|(?2))*\})", regex.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError("No function call found in Gemma 4 format")

    results = []
    for match in matches:
        func_name = match.group(1)
        args_str = match.group(2)

        # Try standard JSON first (model may emit valid JSON args)
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = _gemma4_args_to_json_robust(args_str)

        results.append({"name": func_name, "arguments": arguments})

    return results[0] if len(results) == 1 else results


def parse_tool_calls(
    text: str,
    tokenizer: Any,
    tools: list | None = None,
) -> tuple[str, list[ToolCall] | None]:
    """
    Parse tool calls from model output.

    Uses mlx-lm's TokenizerWrapper tool parser if available, otherwise
    falls back to generic XML tool call parsing for models like GLM.

    Args:
        text: Raw model output text
        tokenizer: mlx-lm's TokenizerWrapper (required)
        tools: Tool definitions for type conversion (optional)

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags and thinking tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
    cleaned_text = text

    # Remove thinking tags if present (reasoning models)
    cleaned_text = re.sub(
        r"<think>.*?</think>", "", cleaned_text, flags=re.DOTALL
    ).strip()

    # Try mlx-lm's native tool parser first
    if getattr(tokenizer, "has_tool_calling", False):
        tool_call_start = tokenizer.tool_call_start
        tool_call_end = tokenizer.tool_call_end
        tool_parser = tokenizer.tool_parser

        if tool_call_start and tool_parser is not None:
            tool_calls = []
            start_escaped = re.escape(tool_call_start)

            if tool_call_end:
                # Paired markers (e.g. <tool_call>...</tool_call>)
                end_escaped = re.escape(tool_call_end)
                pattern = rf"{start_escaped}(.*?){end_escaped}"
                matches = re.findall(pattern, text, re.DOTALL)
            else:
                # One-sided marker (e.g. Mistral/Devstral "[TOOL_CALLS]"):
                # split on the start marker and parse each segment.
                # The model emits: [TOOL_CALLS]name[ARGS]{...}[TOOL_CALLS]name2[ARGS]{...}
                parts = re.split(start_escaped, text)
                # First part is pre-marker text, rest are tool call segments
                matches = [p for p in parts[1:] if p.strip()]

            for match in matches:
                try:
                    parsed = tool_parser(match.strip(), tools)
                    # MiniMax M2 parser returns a list when a single
                    # <minimax:tool_call> block contains multiple <invoke>s.
                    items = parsed if isinstance(parsed, list) else [parsed]
                    for p in items:
                        d = _tool_parser_result_to_dict(p)
                        name = d.get("name", "")
                        arguments = d.get("arguments", {})
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                type="function",
                                function=FunctionCall(
                                    name=name,
                                    arguments=_serialize_tool_call_arguments(arguments),
                                ),
                            )
                        )
                except (
                    ValueError,
                    json.JSONDecodeError,
                    AttributeError,
                    KeyError,
                    SyntaxError,
                    TypeError,
                ) as primary_err:
                    # Gemma 4 only: try robust fallback that handles bare
                    # string values and colons in function names.
                    gemma4_handled = False
                    if tool_call_start == "<|tool_call>":
                        try:
                            parsed = _parse_gemma4_tool_call_fallback(match.strip())
                            items = parsed if isinstance(parsed, list) else [parsed]
                            for p in items:
                                name = p.get("name", "")
                                arguments = p.get("arguments", {})
                                tool_calls.append(
                                    ToolCall(
                                        id=f"call_{uuid.uuid4().hex[:8]}",
                                        type="function",
                                        function=FunctionCall(
                                            name=name,
                                            arguments=_serialize_tool_call_arguments(
                                                arguments
                                            ),
                                        ),
                                    )
                                )
                            gemma4_handled = True
                        except (
                            ValueError,
                            json.JSONDecodeError,
                            KeyError,
                            SyntaxError,
                            TypeError,
                        ):
                            pass

                    if gemma4_handled:
                        continue

                    # Per-match XML fallback: regex-only, no ast.literal_eval,
                    # recovers Qwen/GLM/Hermes-JSON formats. Prevents silent
                    # drop when the native parser raises (e.g. ast.literal_eval
                    # SyntaxError on non-Python-literal parameter values).
                    fb_wrapped = f"␝{match}␞"
                    _, fb_calls = _parse_xml_tool_calls(fb_wrapped)
                    if fb_calls:
                        tool_calls.extend(fb_calls)
                        logger.warning(
                            "Native tool parser failed (%s: %s), "
                            "recovered via XML fallback. Match: %r",
                            type(primary_err).__name__,
                            primary_err,
                            match[:200],
                        )
                    else:
                        logger.warning(
                            "Native tool parser failed (%s: %s) and XML "
                            "fallback could not recover. Dropping match: %r",
                            type(primary_err).__name__,
                            primary_err,
                            match[:200],
                        )
                    continue

            if tool_calls:
                if tool_call_end:
                    cleaned_text = re.sub(
                        rf"{start_escaped}.*?{re.escape(tool_call_end)}",
                        "",
                        cleaned_text,
                        flags=re.DOTALL,
                    ).strip()
                else:
                    # One-sided: everything from first marker to end is tool calls
                    idx = cleaned_text.find(tool_call_start)
                    if idx >= 0:
                        cleaned_text = cleaned_text[:idx].strip()
                return cleaned_text, tool_calls

    # Fallback: bare <tool_call>...</tool_call> tags. Reached when no
    # tokenizer tool-calling is configured (streaming finalize passes the
    # request dict, not a tokenizer) or the native parser found no calls.
    # Wrap each body in the internal ␝/␞ delimiters so the shared
    # _parse_xml_tool_calls recovers JSON / Qwen <function=> / GLM bodies
    # (test_finalize_cross_format_fallback_recovers_xml_tool_call).
    if "<tool_call>" in cleaned_text and "</tool_call>" in cleaned_text:
        wrapped = re.sub(
            r"<tool_call>(.*?)</tool_call>",
            lambda m: f"␝{m.group(1)}␞",
            cleaned_text,
            flags=re.DOTALL,
        )
        if wrapped != cleaned_text:
            logger.debug(
                "parse_tool_calls: bare <tool_call> fallback engaged, "
                "no tokenizer tool-calling available"
            )
            return _parse_xml_tool_calls(wrapped)

    # Fallback: parse XML <tool_call> tags (GLM, Qwen, generic formats)
    if "␝" in cleaned_text:
        # AtomCode 专题优化: 优先调 mlx-lm 原生 qwen3_coder 解析器 (2026-07-19)
        # 原自研 _parse_xml_tool_calls 行 207/231/236 三轮正则全文本扫描 耗 ~50-100ms
        # mlx-lm 原生 qwen3_coder.parse_tool_call 单轮正则 + 结构化解析, 提速 2-3×
        # 失败时降级到自研三轮正则 fallback 保兼容
        try:
            from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call as _mfa_parse

            mfa_result = _mfa_parse(cleaned_text, tools=None)
            if mfa_result is not None:
                # 兼容 mlx-lm 多版本返回格式 (tuple/dict/object)
                if isinstance(mfa_result, tuple):
                    fn_name, fn_args = mfa_result[0], mfa_result[1]
                elif hasattr(mfa_result, "name"):
                    fn_name, fn_args = (
                        mfa_result.name,
                        getattr(mfa_result, "arguments", "{}"),
                    )
                else:
                    fn_name, fn_args = None, "{}"
                stripped = re.sub(r"␝.*?␞", "", cleaned_text, flags=re.DOTALL).strip()

                tc = [
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function={
                            "name": fn_name,
                            "arguments": (
                                fn_args
                                if isinstance(fn_args, str)
                                else json.dumps(fn_args)
                            ),
                        },
                    )
                ]
                return stripped, tc
        except Exception as e:
            logger.debug(f"mlx-lm qwen3_coder 解析失败, 降级自研正则: {e}")
        return _parse_xml_tool_calls(cleaned_text)

    # Fallback: namespaced tool_call tags (e.g. <minimax:tool_call>)
    ns_match = re.search(r"<([A-Za-z_][\w.-]*):tool_call>", cleaned_text)
    if ns_match:
        ns = ns_match.group(1)
        return _parse_namespaced_tool_calls(cleaned_text, ns)

    # Fallback: bracket tool call formats (from text-formatted history)
    if "[Calling tool:" in cleaned_text or "[Tool call:" in cleaned_text:
        return _parse_bracket_tool_calls(cleaned_text)

    # All parsing attempts exhausted. Strip known tool-call markers so raw
    # control markup never leaks into the API response.  Models whose markers
    # overlap with the generic ``<tool_call>`` tag already returned above via
    # Branch 2 (_parse_xml_tool_calls), so this only affects models with
    # unique markers (Gemma 4, Mistral, Pythonic, Kimi K2, Longcat, etc.).
    if getattr(tokenizer, "has_tool_calling", False):
        _start = getattr(tokenizer, "tool_call_start", None)
        _end = getattr(tokenizer, "tool_call_end", None)
        if _start and _end:
            s_esc = re.escape(_start)
            e_esc = re.escape(_end)
            stripped = re.findall(
                rf"{s_esc}(.*?){e_esc}", cleaned_text, flags=re.DOTALL
            )
            if stripped:
                # EH-5 (#811 audit 0906): tool-call markers were present but no
                # parser produced a valid ToolCall. The client is about to
                # receive plain text with no tool_calls and no error signal,
                # so an Agent waiting on a tool invocation will spin. Warn
                # loudly (tools=%s) so the operator can alert; a bounded
                # client-facing signal needs a return-shape change across 14
                # call sites and is tracked separately.
                logger.warning(
                    "Tool call markers found but parsing failed — "
                    "degrading to plain text with NO tool_calls returned "
                    "to client. tools_requested=%d stripped_markers=%s",
                    len(tools or []),
                    stripped,
                )
            cleaned_text = re.sub(
                rf"{s_esc}.*?{e_esc}", "", cleaned_text, flags=re.DOTALL
            ).strip()
        elif _start:
            idx = cleaned_text.find(_start)
            if idx >= 0:
                logger.warning(
                    "Tool call start marker found but parsing failed — "
                    "degrading to plain text with NO tool_calls returned "
                    "to client. tools_requested=%d raw_content=%s",
                    len(tools or []),
                    cleaned_text[idx:],
                )
                cleaned_text = cleaned_text[:idx].strip()

    return cleaned_text, None


def sanitize_tool_call_markup(text: str, tokenizer: Any) -> str:
    """Remove tool-call control markup while preserving surrounding prose."""
    if not text:
        return ""

    stream_filter = ToolCallStreamFilter(tokenizer)
    cleaned = stream_filter.feed(text)
    cleaned += stream_filter.finish()
    return cleaned.strip()


def _extract_tool_names(tools: list) -> set:
    """Extract function names from OpenAI-format tool definitions."""
    names = set()
    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function", {})
            if isinstance(func, dict):
                name = func.get("name")
                if name:
                    names.add(name)
    return names


def extract_tool_calls_with_thinking(
    thinking_content: str,
    regular_content: str,
    tokenizer: Any,
    tools: list | None = None,
) -> ToolCallExtraction:
    """Extract tool calls while keeping a sanitized reasoning transcript.

    When tool calls are found in thinking content (not regular content),
    the ``tools`` parameter controls validation:

    * ``None`` (default) — no tools list was provided.  Thinking-embedded
        calls are kept only when ``regular_content`` is empty (the model
        produced no competing prose).  Otherwise they are dropped as
        potential hallucinated reasoning.
    * ``[]`` — "no tools allowed".  All thinking-embedded calls are
        dropped regardless of ``regular_content``.
    * Non-empty list — name matching is the sole discriminator.
        Calls whose name matches a provided tool are promoted regardless
        of whether regular text was also produced.
    """
    cleaned_text, tool_calls = parse_tool_calls(regular_content, tokenizer, tools)
    cleaned_thinking = sanitize_tool_call_markup(thinking_content, tokenizer)
    tool_calls_from_thinking = False

    if not tool_calls and thinking_content:
        _, tool_calls = parse_tool_calls(thinking_content, tokenizer, tools)
        tool_calls_from_thinking = bool(tool_calls)

        # Guard: validate thinking-embedded tool calls.
        #
        # Three cases:
        # 1. tools is None (not provided) AND regular text exists → drop.
        #    The call is unvalidated and could be hallucinated reasoning.
        # 2. tools is None AND no regular text → keep.  The model clearly
        #    intended a tool invocation (no competing prose).
        # 3. tools is a list (including empty) → name matching is the sole
        #    discriminator.  An empty list means "no tools allowed" so all
        #    calls are dropped.  A non-empty list filters by name, regardless
        #    of whether regular text was also produced.  The previous "regular
        #    text means just reasoning" heuristic was wrong for models
        #    (Qwen3-Coder) that genuinely place tool calls in thinking.
        # See https://github.com/jundot/fusion-mlx/issues/1392
        if tool_calls:
            if tools is None:
                if regular_content.strip():
                    tool_calls = None
                    tool_calls_from_thinking = False
            else:
                valid_names = _extract_tool_names(tools)
                tool_calls = [
                    tc for tc in tool_calls if tc.function.name in valid_names
                ]
                if not tool_calls:
                    tool_calls = None
                    tool_calls_from_thinking = False

    return ToolCallExtraction(
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        cleaned_thinking=cleaned_thinking,
        tool_calls_from_thinking=tool_calls_from_thinking,
    )


def parse_tool_calls_with_thinking_fallback(
    thinking_content: str,
    regular_content: str,
    tokenizer: Any,
    tools: list | None = None,
) -> tuple[str, list[ToolCall] | None]:
    """Parse tool calls from content, falling back to thinking if none found.

    Small reasoning models sometimes generate tool call XML inside <think>
    blocks instead of after </think>. This function first tries the normal
    content, then falls back to parsing from thinking content.

    Args:
        thinking_content: Text extracted from <think>...</think> blocks.
        regular_content: Text outside thinking blocks.
        tokenizer: mlx-lm's TokenizerWrapper.
        tools: Tool definitions for type conversion (optional).

    Returns:
        Tuple of (cleaned_text, tool_calls or None).
        cleaned_text comes from regular_content only (thinking text is
        never promoted to content).
    """
    result = extract_tool_calls_with_thinking(
        thinking_content,
        regular_content,
        tokenizer,
        tools,
    )
    return result.cleaned_text, result.tool_calls
