# SPDX-License-Identifier: Apache-2.0
"""Anthropic /v1/messages tool_choice enforcement helpers.

Ported from routes/anthropic.py (dead-in-prod) into the LIVE api/ surface
as part of #71 deep-dedup Phase 2. The live api/anthropic_routes.py passed
``tools=`` to the engine but NEVER ``tool_choice`` and ran no post-generation
enforcement, so /v1/messages silently ignored ``tool_choice`` (0 refs vs 95
in the dead routes/anthropic.py). These 9 fns close that live gap.

Deferred (separate concerns, not the tool_choice gap):
  - _resolved_sampling_kwargs: duplicates api/_build_sampling_params.
  - _estimate_anthropic_prompt_tokens: streaming usage estimate.
  - _split_tool_input_json: streaming input_json_delta shaping.
"""

import logging
import uuid

from fastapi import HTTPException

from ..service.helpers import (
    _TOOL_USE_REQUIRED_SUFFIX,
    _tool_use_required_named_suffix,
    _validate_tool_call_params,
)

logger = logging.getLogger(__name__)


def _named_tool_choice_target(tool_choice) -> str | None:
    # Return the target tool name when tool_choice pins a specific tool
    # (OpenAI shape {"type":"function","function":{"name":X}}), else None.
    # "auto"/"required"/"none"/unset have no "wrong tool" case to filter.
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "function":
        return None
    target = (tool_choice.get("function") or {}).get("name")
    return target or None


def _tool_call_name_anthropic(tc) -> str | None:
    # Extract the function name from a tool_call entry regardless of shape
    # (dict / pydantic / flat). Inlined to avoid a cross-route import.
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return fn.get("name")
        if fn is not None:
            return getattr(fn, "name", None)
        return tc.get("name")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return getattr(fn, "name", None)
    return getattr(tc, "name", None)


def _is_required_tool_choice(tool_choice) -> bool:
    # True when tool_choice forces the model to call ANY tool. The Anthropic
    # adapter maps {"type":"any"} -> OpenAI "required", so the post-adapter
    # "required" string IS the Anthropic any contract.
    return tool_choice == "required"


def _synthesize_anthropic_forced_tool_call(name: str):
    # Build a single tool_call dict for a forced tool_choice whose parser
    # surfaced no calls. Adapted from routes/anthropic.py to return a DICT
    # (api/ convention: gen.tool_calls are dicts) not a pydantic ToolCall,
    # so the synthesized call merges cleanly with model-emitted dict calls.
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _filter_tool_calls_by_tool_choice(tool_calls, tool_choice) -> list:
    # H-05: tool_choice={"type":"tool","name":X} pins WHICH tool the model
    # must call. Local inference has no decoder-level constraint, so a small
    # model can defy the pin and emit a call to a different tool. Keep only
    # the pinned tool's calls; drop the rest with a warning. "auto"/
    # "required"/"none"/unset pass through unchanged.
    if not tool_calls or not isinstance(tool_choice, dict):
        return tool_calls or []
    target = _named_tool_choice_target(tool_choice)
    if not target:
        return tool_calls
    filtered = []
    dropped: list[str] = []
    for tc in tool_calls:
        if _tool_call_name_anthropic(tc) == target:
            filtered.append(tc)
        else:
            dropped.append(_tool_call_name_anthropic(tc) or "<unknown>")
    if dropped:
        logger.warning(
            "tool_choice pinned %r but model also emitted calls to %s; "
            "dropping the un-pinned calls so the response carries only the "
            "pinned tool (Anthropic /v1/messages H-05 policy).",
            target,
            dropped,
        )
    return filtered


def _enforce_named_tool_choice_present(
    tool_calls,
    tool_choice,
    *,
    original_call_count: int,
) -> tuple[list, bool, str | None]:
    # Return (tool_calls, synthesized, error). When the named-tool contract
    # is satisfied, tool_calls is unchanged. When the model emitted text only
    # or only wrong-tool calls, synthesize the pinned call so valid forced-
    # tool requests keep the tool_use response shape instead of silently
    # returning text for a pinned tool request.
    target = _named_tool_choice_target(tool_choice)
    if not target or tool_calls:
        return tool_calls, False, None
    if original_call_count == 0:
        logger.warning(
            "tool_choice pinned tool %r but the model returned a text "
            "response with no tool_calls; synthesizing the pinned tool_use.",
            target,
        )
    else:
        logger.warning(
            "tool_choice pinned tool %r but the model emitted %d call(s), "
            "none to %r; synthesizing the pinned tool_use.",
            target,
            original_call_count,
            target,
        )
    return [_synthesize_anthropic_forced_tool_call(target)], True, None


def _synthesized_tool_call_schema_error(tool_calls, tools: list | None) -> str | None:
    # Schema-specific failure check for synthesized empty-argument calls.
    # Returns an error string if the pinned tool requires arguments that
    # cannot be synthesized safely, else None.
    if not tool_calls or not tools:
        return None
    first_call = tool_calls[0]
    func = (
        first_call.function
        if hasattr(first_call, "function")
        else first_call.get("function", {})
    )
    func_name = func.name if hasattr(func, "name") else func.get("name", "")
    for tool in tools:
        tool_data = tool.model_dump() if hasattr(tool, "model_dump") else tool
        if not isinstance(tool_data, dict):
            continue
        function_data = tool_data.get("function", tool_data)
        if not isinstance(function_data, dict):
            continue
        if function_data.get("name") != func_name:
            continue
        schema = (
            function_data.get("parameters") or function_data.get("input_schema") or {}
        )
        required = schema.get("required") if isinstance(schema, dict) else None
        if required:
            return (
                f"tool_choice pinned tool {func_name!r} requires arguments "
                "that cannot be synthesized safely"
            )
        break
    try:
        _validate_tool_call_params(tool_calls, tools)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return f"tool_choice synthesized an invalid empty tool input: {detail}"
    return None


def _enforce_required_tool_choice_present(
    tool_calls,
    tool_choice,
    *,
    tools: list | None,
):
    # OpenAI tool_choice="required" / Anthropic {"type":"any"} post-parse
    # enforcement. Returns (tool_calls, error_detail_or_None). When a
    # forced-any tool_choice produced no tool_calls: single-tool case ->
    # synthesize a call to that tool; multi-tool case -> return an error
    # detail the caller surfaces as 422 (non-stream) or SSE error (stream).
    if not _is_required_tool_choice(tool_choice):
        return tool_calls, None
    if tool_calls:
        return tool_calls, None
    if tools and len(tools) == 1:
        tool = tools[0]
        fn = (
            tool.function
            if hasattr(tool, "function")
            else (tool.get("function") if isinstance(tool, dict) else None)
        )
        if fn is None:
            fn = {}
        solo_name = (
            fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        )
        if solo_name:
            logger.warning(
                "tool_choice={'type':'any'} on Anthropic route produced no "
                "tool_calls; synthesising a call to the sole available tool "
                "%r with empty arguments to honour the forced-tool contract.",
                solo_name,
            )
            return [_synthesize_anthropic_forced_tool_call(solo_name)], None
    detail = (
        'tool_choice={"type":"any"} but the model returned a text response '
        "with no tool_calls. Local inference has no decoder-level constraint; "
        "the system-prompt enforcement was insufficient for this prompt."
    )
    return tool_calls, detail


def _inject_tool_use_required_suffix(
    messages: list,
    tool_choice,
    *,
    tools: list | None,
) -> list:
    # Mutate-in-place: append _TOOL_USE_REQUIRED_SUFFIX (or the named
    # variant) to the system message so a forced tool_choice has the same
    # prompt-level lever the OpenAI route applies. No-op when tool_choice
    # does not force a call OR when tools is empty.
    if not tools:
        return messages
    suffix = None
    if _is_required_tool_choice(tool_choice):
        suffix = _TOOL_USE_REQUIRED_SUFFIX
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        named = (tool_choice.get("function") or {}).get("name")
        if named:
            suffix = _tool_use_required_named_suffix(named)
    if not suffix:
        return messages
    has_system = any(
        (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "system"
        for m in messages
    )
    if has_system:
        for i, m in enumerate(messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "system":
                content = (
                    m.get("content")
                    if isinstance(m, dict)
                    else getattr(m, "content", "")
                )
                if isinstance(content, str):
                    new_content = content + suffix
                elif isinstance(content, list):
                    new_content = list(content)
                    new_content.append({"type": "text", "text": suffix})
                elif content is None:
                    new_content = suffix
                else:
                    continue
                if isinstance(m, dict):
                    messages[i] = {**m, "content": new_content}
                else:
                    m.content = new_content
                break
        else:
            messages.insert(0, {"role": "system", "content": suffix.strip()})
    else:
        messages.insert(0, {"role": "system", "content": suffix.strip()})
    return messages


def enforce_tool_choice(tool_calls, tool_choice, tools: list | None):
    # Run the 4-step post-generation tool_choice enforcement. Returns
    # (tool_calls, error_detail_or_None). Caller raises 422 (non-stream) or
    # emits an SSE error event (stream) when error_detail is set. No-op when
    # tool_choice is falsy (auto/none/unset).
    if not tool_choice:
        return tool_calls, None
    original_call_count = len(tool_calls or [])
    tool_calls = _filter_tool_calls_by_tool_choice(tool_calls or [], tool_choice)
    tool_calls, synthesized, named_err = _enforce_named_tool_choice_present(
        tool_calls,
        tool_choice,
        original_call_count=original_call_count,
    )
    if named_err:
        return tool_calls, named_err
    if synthesized:
        synth_err = _synthesized_tool_call_schema_error(tool_calls, tools)
        if synth_err:
            return tool_calls, synth_err
    tool_calls, required_err = _enforce_required_tool_choice_present(
        tool_calls,
        tool_choice,
        tools=tools,
    )
    if required_err:
        return tool_calls, required_err
    return tool_calls, None
