# SPDX-License-Identifier: Apache-2.0
import copy
import json
import logging
import re

logger = logging.getLogger(__name__)

_CHAT_TEMPLATE_ROLE_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|end|>",
    "<|fim_begin|>",
    "<|fim_hole|>",
    "<|fim_end|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<|start|>",
    "<|message|>",
    "<|channel|>",
    "<|return|>",
)


def _collect_role_markers(template_applicator) -> set[str]:
    markers: set[str] = set(_CHAT_TEMPLATE_ROLE_MARKERS)
    tokenizer = template_applicator
    if hasattr(tokenizer, "tokenizer"):
        markers |= _collect_role_markers(tokenizer.tokenizer)
    candidates: list[str] = []
    for attr in ("all_special_tokens", "additional_special_tokens"):
        vals = getattr(tokenizer, attr, None) or []
        if isinstance(vals, (list, tuple, set)):
            candidates.extend(str(v) for v in vals)
    smap = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(smap, dict):
        for v in smap.values():
            if isinstance(v, str):
                candidates.append(v)
            elif isinstance(v, (list, tuple)):
                candidates.extend(str(x) for x in v)
    for tok in candidates:
        if not tok or not isinstance(tok, str):
            continue
        if (
            tok.startswith("<|")
            and tok.endswith("|>")
            or tok.startswith("<")
            and tok.endswith(">")
            and any(kw in tok for kw in ("turn", "header", "message", "channel"))
        ):
            markers.add(tok)
    return markers


def _build_marker_pattern(markers: set[str]) -> re.Pattern | None:
    if not markers:
        return None
    parts = sorted((re.escape(m) for m in markers), key=len, reverse=True)
    return re.compile("|".join(parts))


def _neutralize_in_string(text: str, pattern: re.Pattern) -> str:
    def _sub(match: re.Match) -> str:
        marker = match.group(0)
        return marker[0] + "​" + marker[1:]

    return pattern.sub(_sub, text)


def _sanitize_message_content(content, pattern: re.Pattern):
    if isinstance(content, str):
        return _neutralize_in_string(content, pattern)
    if isinstance(content, list):
        new_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    new_part = dict(part)
                    new_part["text"] = _neutralize_in_string(part["text"], pattern)
                    new_parts.append(new_part)
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        return new_parts
    return content


def _sanitize_messages_for_template(
    messages: list[dict], template_applicator
) -> list[dict]:
    markers = _collect_role_markers(template_applicator)
    pattern = _build_marker_pattern(markers)
    if pattern is None:
        return messages
    sanitized: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        content = msg.get("content")
        new_content = _sanitize_message_content(content, pattern)
        if new_content is content:
            sanitized.append(msg)
            continue
        new_msg = dict(msg)
        new_msg["content"] = new_content
        sanitized.append(new_msg)
    return sanitized


def _part_type_and_text(part) -> tuple[str | None, str | None]:
    if isinstance(part, dict):
        t = part.get("type")
        x = part.get("text")
    else:
        t = getattr(part, "type", None)
        x = getattr(part, "text", None)
    if isinstance(t, str) or t is None:
        t_norm = t
    else:
        t_norm = None
    x_norm = x if isinstance(x, str) else None
    return t_norm, x_norm


def _is_text_only_content_array(content) -> bool:
    if not isinstance(content, list) or not content:
        return False
    for part in content:
        t, x = _part_type_and_text(part)
        if t != "text" or x is None:
            return False
    return True


def _join_text_parts(content: list) -> str:
    return "".join((_part_type_and_text(part)[1] or "") for part in content)


def _normalize_text_only_content_arrays(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        content = msg.get("content")
        role = msg.get("role")
        if isinstance(content, list) and content:
            if _is_text_only_content_array(content):
                new_msg = dict(msg)
                new_msg["content"] = _join_text_parts(content)
                out.append(new_msg)
                continue
            if role == "tool":
                raise ValueError(
                    "tool-role message content must be a string or a "
                    "text-only array of {type:'text', text:str} parts; "
                    "got a non-text content part"
                )
        out.append(msg)
    return out


def _coerce_arguments_to_dict(arguments):
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"value": arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {"value": arguments}


def _tool_call_arguments_need_mutation(tool_call: dict) -> tuple[bool, bool]:
    function = tool_call.get("function")
    nested_needs = (
        isinstance(function, dict)
        and "arguments" in function
        and not isinstance(function.get("arguments"), dict)
    )
    top_needs = "arguments" in tool_call and not isinstance(
        tool_call.get("arguments"), dict
    )
    return nested_needs, top_needs


def _normalize_assistant_tool_call_arguments(messages: list) -> list:
    if not isinstance(messages, list) or not messages:
        return messages
    needs_mutation = False
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            nested_needs, top_needs = _tool_call_arguments_need_mutation(tc)
            if nested_needs or top_needs:
                needs_mutation = True
                break
        if needs_mutation:
            break
    if not needs_mutation:
        return messages
    normalized: list = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            normalized.append(msg)
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            normalized.append(msg)
            continue
        new_tool_calls: list = []
        touched_any = False
        for tc in tool_calls:
            if not isinstance(tc, dict):
                new_tool_calls.append(tc)
                continue
            nested_needs, top_needs = _tool_call_arguments_need_mutation(tc)
            if not nested_needs and not top_needs:
                new_tool_calls.append(tc)
                continue
            new_tc = dict(tc)
            if nested_needs:
                function = tc["function"]
                new_function = dict(function)
                new_function["arguments"] = _coerce_arguments_to_dict(
                    function["arguments"]
                )
                new_tc["function"] = new_function
            if top_needs:
                new_tc["arguments"] = _coerce_arguments_to_dict(tc["arguments"])
            new_tool_calls.append(new_tc)
            touched_any = True
        if touched_any:
            new_msg = dict(msg)
            new_msg["tool_calls"] = new_tool_calls
            normalized.append(new_msg)
        else:
            normalized.append(msg)
    return normalized


def _baseline_sanitize_messages(messages):
    baseline_pattern = _build_marker_pattern(set(_CHAT_TEMPLATE_ROLE_MARKERS))
    if baseline_pattern is None:
        return messages
    fallback: list = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            new_msg = dict(msg)
            new_msg["content"] = _sanitize_message_content(
                msg["content"], baseline_pattern
            )
            fallback.append(new_msg)
        else:
            fallback.append(msg)
    return fallback


def _walk_tools_iter(tools, transform):
    if isinstance(tools, str):
        return transform(tools)
    if not isinstance(tools, (dict, list, tuple)):
        return tools
    root_holder: list = [None]
    stack: list = [(root_holder, 0, tools, 0)]
    tuple_buffers: list = []
    while stack:
        parent, key, src, depth = stack.pop()
        if isinstance(src, str):
            parent[key] = transform(src)
        elif isinstance(src, dict):
            new_dict: dict = {}
            parent[key] = new_dict
            for k, v in src.items():
                if isinstance(v, str):
                    new_dict[k] = transform(v)
                elif isinstance(v, (dict, list, tuple)):
                    new_dict[k] = None
                    stack.append((new_dict, k, v, depth + 1))
                else:
                    new_dict[k] = v
        elif isinstance(src, list):
            new_list: list = [None] * len(src)
            parent[key] = new_list
            for i, v in enumerate(src):
                if isinstance(v, str):
                    new_list[i] = transform(v)
                elif isinstance(v, (dict, list, tuple)):
                    stack.append((new_list, i, v, depth + 1))
                else:
                    new_list[i] = v
        elif isinstance(src, tuple):
            buf: list = [None] * len(src)
            parent[key] = buf
            tuple_buffers.append((depth, parent, key, buf))
            for i, v in enumerate(src):
                if isinstance(v, str):
                    buf[i] = transform(v)
                elif isinstance(v, (dict, list, tuple)):
                    stack.append((buf, i, v, depth + 1))
                else:
                    buf[i] = v
        else:
            parent[key] = src
    tuple_buffers.sort(key=lambda entry: entry[0], reverse=True)
    for _depth, parent, key, buf in tuple_buffers:
        parent[key] = tuple(buf)
    return root_holder[0]


def _baseline_sanitize_tools(tools):
    if not tools:
        return tools
    baseline_pattern = _build_marker_pattern(set(_CHAT_TEMPLATE_ROLE_MARKERS))
    if baseline_pattern is None:
        return tools
    return _walk_tools_iter(tools, lambda s: _neutralize_in_string(s, baseline_pattern))


def _sanitize_tools_for_template(tools, template_applicator):
    if not tools:
        return tools
    markers = _collect_role_markers(template_applicator)
    pattern = _build_marker_pattern(markers)
    if pattern is None:
        return tools
    return _walk_tools_iter(tools, lambda s: _neutralize_in_string(s, pattern))


def _build_tool_injection_text(tools: list[dict]) -> str:
    lines = ["# Available Tools", ""]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        lines.append(f"## {name}")
        if desc:
            lines.append(f"{desc}")
        if props:
            lines.append(f"Parameters: {json.dumps(props, ensure_ascii=False)}")
        if required:
            lines.append(f"Required: {json.dumps(required)}")
        lines.append("")
    lines.append(
        "When you need to use a tool, respond with a JSON object "
        'containing "name" and "arguments" keys.'
    )
    return "\n".join(lines)


def _inject_tools_into_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    injection = _build_tool_injection_text(tools)
    msgs = copy.copy(messages)
    if msgs and msgs[0].get("role") == "system":
        first = dict(msgs[0])
        existing = first.get("content", "")
        if isinstance(existing, list):
            first["content"] = list(existing) + [
                {"type": "text", "text": "\n\n" + injection}
            ]
        else:
            first["content"] = str(existing) + "\n\n" + injection
        msgs[0] = first
    else:
        msgs.insert(0, {"role": "system", "content": injection})
    return msgs


def apply_chat_template(
    template_applicator,
    messages: list[dict],
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    model_name: str = "",
) -> str:
    messages = _normalize_text_only_content_arrays(messages)
    messages = _normalize_assistant_tool_call_arguments(messages)
    try:
        messages = _sanitize_messages_for_template(messages, template_applicator)
    except Exception as e:
        logger.debug(
            "Chat-template marker sanitisation failed (%s); applying "
            "baseline-marker fallback",
            e,
        )
        messages = _baseline_sanitize_messages(messages)
    try:
        tools = _sanitize_tools_for_template(tools, template_applicator)
    except Exception as e:
        logger.debug(
            "Chat-template tool-marker sanitisation failed (%s); applying "
            "baseline-marker fallback",
            e,
        )
        tools = _baseline_sanitize_tools(tools)
    if not hasattr(template_applicator, "apply_chat_template"):
        if tools:
            messages = _inject_tools_into_messages(messages, tools)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return prompt + "\nassistant:"
    if enable_thinking is None:
        enable_thinking = "coder" not in model_name.lower()
    template_kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        template_kwargs["tools"] = tools
    try:
        return template_applicator.apply_chat_template(messages, **template_kwargs)
    except TypeError as e:
        logger.debug("Chat template TypeError, retrying without enable_thinking: %s", e)
        template_kwargs.pop("enable_thinking", None)
        try:
            return template_applicator.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            pass
        template_kwargs.pop("tools", None)
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if tools:
            logger.info(
                "Chat template doesn't support tools param — "
                "injecting %d tool definitions into system prompt",
                len(tools),
            )
            injected = _inject_tools_into_messages(messages, tools)
            try:
                return template_applicator.apply_chat_template(
                    injected, **template_kwargs
                )
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                return template_applicator.apply_chat_template(
                    injected, **template_kwargs
                )
        return template_applicator.apply_chat_template(messages, **template_kwargs)
