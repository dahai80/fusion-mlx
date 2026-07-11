# SPDX-License-Identifier: Apache-2.0
import json
import logging
import uuid

from fastapi import HTTPException

from .constants import is_rescue_payload
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolDefinition,
)
from .responses_models import (
    ResponsesInputItem,
    ResponsesOutputContent,
    ResponsesOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from .utils import normalize_responses_content_part

logger = logging.getLogger(__name__)

SUPPORTED_RESPONSES_TOOL_TYPES: frozenset[str] = frozenset(
    {
        "function",
        "computer_20251022",
    }
)

_RESPONSES_TOOL_TYPE_ALIASES: dict[str, str] = {
    "computer_use_preview": "computer_20251022",
}


def _canonicalize_tool_type(ttype: str | None) -> str | None:
    if not ttype:
        return ttype
    return _RESPONSES_TOOL_TYPE_ALIASES.get(ttype, ttype)


def _raise_unsupported_tool_type(tool_type: str) -> None:
    supported = sorted(SUPPORTED_RESPONSES_TOOL_TYPES)
    aliases = sorted(_RESPONSES_TOOL_TYPE_ALIASES)
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": (
                    f"Tool type {tool_type!r} is not supported by this "
                    f"server. Supported types: {supported} "
                    f"(aliases: {aliases}). "
                    "``computer_20251022`` requires a UI-TARS model to "
                    "be loaded (other vision+tool-calling models may "
                    "not fulfil the request)."
                ),
                "type": "invalid_request_error",
                "code": "unsupported_tool_type",
                "param": "tools",
            }
        },
    )


def normalize_responses_tool_types(tools: list[dict] | None) -> None:
    if not tools:
        return
    _drop_hosted = {
        "web_search",
        "web_search_preview",
        "file_search",
        "code_interpreter",
        "image_generation",
    }
    codex_fingerprint = any(
        isinstance(t, dict) and t.get("type") == "namespace" for t in tools
    )
    flattened: list = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "namespace":
            sub_tools = t.get("tools")
            if (
                isinstance(sub_tools, list)
                and sub_tools
                and all(
                    isinstance(sub, dict)
                    and _canonicalize_tool_type(sub.get("type")) == "function"
                    for sub in sub_tools
                )
            ):
                flattened.extend(sub_tools)
                continue
            flattened.append(t)
            continue
        flattened.append(t)
    if codex_fingerprint:
        flattened = [
            t
            for t in flattened
            if not (
                isinstance(t, dict)
                and _canonicalize_tool_type(t.get("type")) in _drop_hosted
            )
        ]
    tools[:] = flattened
    for t in tools:
        if not isinstance(t, dict):
            continue
        ttype = t.get("type")
        if not ttype:
            continue
        canonical = _canonicalize_tool_type(ttype)
        if canonical != ttype:
            t["type"] = canonical


def validate_responses_tool_types(tools: list[dict] | None) -> None:
    if not tools:
        return
    for t in tools:
        if not isinstance(t, dict):
            continue
        ttype = t.get("type")
        canonical = _canonicalize_tool_type(ttype)
        if canonical and canonical not in SUPPORTED_RESPONSES_TOOL_TYPES:
            _raise_unsupported_tool_type(ttype)


def _is_computer_use_tool(tool: dict) -> bool:
    if not isinstance(tool, dict):
        return False
    return _canonicalize_tool_type(tool.get("type")) == "computer_20251022"


def request_uses_computer_use(request: ResponsesRequest) -> bool:
    return bool(request.tools) and any(_is_computer_use_tool(t) for t in request.tools)


def validate_responses_tool_choice(
    tool_choice: str | dict | None, tools: list[dict] | None
) -> None:
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        if tool_choice == "required" and not tools:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "tool_choice='required' but the request has "
                            "no 'tools' array — the model has nothing to "
                            "choose from. Either drop tool_choice or "
                            "add at least one tool definition."
                        ),
                        "type": "invalid_request_error",
                        "code": "tool_choice_required_without_tools",
                        "param": "tool_choice",
                    }
                },
            )
        return
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            target = tool_choice.get("name") or (
                (tool_choice.get("function") or {}).get("name")
            )
            if not target:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "tool_choice.type='function' requires a "
                                "non-empty 'name' field."
                            ),
                            "type": "invalid_request_error",
                            "code": "tool_choice_missing_name",
                            "param": "tool_choice.name",
                        }
                    },
                )
            tool_names = _submitted_tool_names(tools)
            if target not in tool_names:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                f"tool_choice references function "
                                f"{target!r} which is not present in the "
                                "'tools' array. Add the tool definition "
                                "or pick one of the submitted names."
                            ),
                            "type": "invalid_request_error",
                            "code": "tool_choice_unknown_function",
                            "param": "tool_choice.name",
                        }
                    },
                )


def _submitted_tool_names(tools: list[dict] | None) -> set[str]:
    names: set[str] = set()
    if not tools:
        return names
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            n = t.get("name") or (t.get("function") or {}).get("name")
            if n:
                names.add(n)
        elif t.get("type") == "computer_20251022":
            names.add("computer")
    return names


def _resolve_reasoning_effort(request: ResponsesRequest) -> str | None:
    if request.reasoning_effort is not None:
        return request.reasoning_effort
    reasoning = request.reasoning
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            return effort
    return None


def responses_to_openai(request: ResponsesRequest) -> ChatCompletionRequest:
    messages: list[Message] = []

    if request.instructions:
        messages.append(Message(role="system", content=request.instructions))

    if isinstance(request.input, str):
        messages.append(Message(role="user", content=request.input))
    else:
        for item in request.input:
            converted = _convert_input_item(item)
            messages.extend(converted)

    messages = _merge_system_messages(messages)

    tools = _convert_tools(request.tools)
    tool_choice = _convert_tool_choice(request.tool_choice)
    response_format = _convert_text_format(request.text)
    if response_format is None and request.response_format is not None:
        response_format = request.response_format

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=response_format,
        reasoning_max_tokens=request.reasoning_max_tokens,
        reasoning_effort=_resolve_reasoning_effort(request),
        seed=request.seed,
        chat_template_kwargs=request.chat_template_kwargs,
        enable_thinking=request.enable_thinking,
    )


def _merge_system_messages(messages: list[Message]) -> list[Message]:
    def _to_text(value):
        if isinstance(value, str):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        if isinstance(value, dict):
            return value.get("text") or ""
        if isinstance(value, list):
            return "\n".join(_to_text(v) for v in value)
        return ""

    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        return messages
    system_texts = [
        t for t in (_to_text(m.content) for m in messages if m.role == "system") if t
    ]
    non_system = [m for m in messages if m.role != "system"]
    if not system_texts:
        return non_system
    merged = Message(role="system", content="\n\n".join(system_texts))
    return [merged] + non_system


def openai_to_responses(
    response: ChatCompletionResponse,
    model: str,
    request: ResponsesRequest,
    created_at: int,
) -> ResponsesResponse:
    output: list[ResponsesOutputItem] = []
    choice = response.choices[0] if response.choices else None

    uses_computer_use = request_uses_computer_use(request)

    if choice:
        reasoning_text = getattr(choice.message, "reasoning_content", None) or ""
        if reasoning_text:
            content_for_downstream_check = choice.message.content or ""
            if is_rescue_payload(content_for_downstream_check):
                content_for_downstream_check = ""
            downstream_output_seen = bool(
                content_for_downstream_check.strip() or choice.message.tool_calls
            )
            reasoning_item_status = (
                "incomplete"
                if (choice.finish_reason == "length" and not downstream_output_seen)
                else "completed"
            )
            output.append(
                _build_reasoning_output_item(
                    reasoning_text, status=reasoning_item_status
                )
            )

        text = choice.message.content or ""
        has_tool_calls = bool(choice.message.tool_calls)
        has_reasoning = bool(reasoning_text)
        emit_message_item = bool(text) or not (has_tool_calls or has_reasoning)
        if emit_message_item:
            output.append(
                ResponsesOutputItem(
                    type="message",
                    id=f"msg_{uuid.uuid4().hex[:24]}",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponsesOutputContent(type="output_text", text=text),
                    ],
                )
            )

        for tc in choice.message.tool_calls or []:
            output.append(_build_tool_call_output_item(tc, uses_computer_use))

    status = _convert_status(choice.finish_reason if choice else None)

    usage = _build_responses_usage(response)

    incomplete_details: dict | None = None
    if status == "incomplete":
        incomplete_details = {"reason": "max_output_tokens"}

    return ResponsesResponse(
        created_at=created_at,
        model=model,
        status=status,
        output=output,
        usage=usage,
        parallel_tool_calls=bool(request.parallel_tool_calls),
        tool_choice=request.tool_choice or "auto",
        tools=request.tools or [],
        metadata=request.metadata,
        instructions=request.instructions,
        truncation=request.truncation,
        service_tier=request.service_tier,
        incomplete_details=incomplete_details,
    )


def _build_reasoning_output_item(
    reasoning_text: str, *, status: str = "completed"
) -> ResponsesOutputItem:
    return ResponsesOutputItem(
        type="reasoning",
        id=f"rs_{uuid.uuid4().hex[:24]}",
        status=status,
        summary=[{"type": "summary_text", "text": reasoning_text}],
    )


def _build_tool_call_output_item(
    tool_call, uses_computer_use: bool
) -> ResponsesOutputItem:
    if uses_computer_use and (tool_call.function.name or "") == "computer":
        action = _parse_computer_action(tool_call.function.arguments or "")
        return ResponsesOutputItem(
            type="computer_call",
            id=f"cu_{uuid.uuid4().hex[:24]}",
            call_id=tool_call.id,
            status="completed",
            action=action,
            pending_safety_checks=[],
        )
    return ResponsesOutputItem(
        type="function_call",
        id=f"fc_{uuid.uuid4().hex[:24]}",
        call_id=tool_call.id,
        name=tool_call.function.name,
        arguments=tool_call.function.arguments or "",
        status="completed",
    )


def _parse_computer_action(arguments: str) -> dict:
    if not arguments:
        return {"type": "unknown", "raw": arguments}
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return {"type": "unknown", "raw": arguments}
    if not isinstance(parsed, dict):
        return {"type": "unknown", "raw": arguments}
    out = dict(parsed)
    if "action" in out and "type" not in out:
        out["type"] = out.pop("action")
    if "type" not in out:
        return {"type": "unknown", "raw": arguments}
    try:
        from ..tool_parsers.ui_tars_tool_parser import (
            translate_to_responses_spec_keys,
        )

        return translate_to_responses_spec_keys(out)
    except ImportError:
        logger.debug("ui_tars_tool_parser not available, returning raw action")
        return out


def _convert_input_item(item: ResponsesInputItem) -> list[Message]:
    if item.type == "message":
        return [_message_item_to_chat(item)]
    if item.type == "function_call":
        return [_function_call_to_chat(item)]
    if item.type == "function_call_output":
        return [_function_call_output_to_chat(item)]
    if item.type == "reasoning":
        return []
    return []


_RESPONSES_TO_CHAT_ROLE = {
    "developer": "system",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


def _message_item_to_chat(item: ResponsesInputItem) -> Message:
    raw_role = item.role or "user"
    role = _RESPONSES_TO_CHAT_ROLE.get(raw_role, raw_role)
    content = item.content

    if isinstance(content, str):
        chat_content = (
            ""
            if content == ""
            else [
                normalize_responses_content_part(
                    {"type": "input_text", "text": content}
                )
            ]
        )
    elif content is None:
        raise ValueError("Responses message content is required")
    else:
        parts = []
        for c in content:
            parts.append(normalize_responses_content_part(c))
        if not parts:
            raise ValueError("Responses message content must not be empty")
        chat_content = parts

    return Message(role=role, content=chat_content)


def _function_call_to_chat(item: ResponsesInputItem) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": item.call_id or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": item.name or "",
                    "arguments": item.arguments or "{}",
                },
            }
        ],
    )


def _function_call_output_to_chat(item: ResponsesInputItem) -> Message:
    out = item.output
    if isinstance(out, (dict, list)):
        text = json.dumps(out)
    elif out is None:
        text = ""
    else:
        text = str(out)
    return Message(
        role="tool",
        content=text,
        tool_call_id=item.call_id or "",
    )


def _convert_tools(tools: list[dict] | None) -> list[ToolDefinition] | None:
    if not tools:
        return None
    converted: list[ToolDefinition] = []
    for t in tools:
        if hasattr(t, "model_dump"):
            t = t.model_dump()
        if not isinstance(t, dict):
            continue
        ttype = t.get("type")
        if ttype == "function":
            name = t.get("name") or t.get("function", {}).get("name", "")
            if not name:
                continue
            converted.append(
                ToolDefinition(
                    type="function",
                    function={
                        "name": name,
                        "description": t.get("description")
                        or t.get("function", {}).get("description", ""),
                        "parameters": t.get("parameters")
                        or t.get("function", {}).get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                )
            )
        elif ttype == "computer_20251022":
            geometry: dict = {
                "type": "object",
                "properties": {
                    "display_width": {"type": "integer"},
                    "display_height": {"type": "integer"},
                    "environment": {"type": "string"},
                },
                "_computer_use": {
                    "display_width": t.get("display_width"),
                    "display_height": t.get("display_height"),
                    "environment": t.get("environment"),
                },
            }
            converted.append(
                ToolDefinition(
                    type="function",
                    function={
                        "name": "computer",
                        "description": "Computer-Use (UI-TARS) GUI action tool",
                        "parameters": geometry,
                    },
                )
            )
        else:
            _raise_unsupported_tool_type(ttype or "<missing>")
    return converted or None


def _convert_tool_choice(tool_choice: str | dict | None) -> str | dict | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "name" in tool_choice:
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
    return None


def _convert_text_format(text: dict | None) -> ResponseFormat | None:
    if not text:
        return None
    # ``text`` may arrive as a ``TextConfig`` pydantic model (the Responses
    # API request field) rather than a dict; normalize via ``model_dump`` so
    # ``schema_`` (aliased "schema") round-trips as "schema" for the dict
    # access below (fixes ``'TextConfig' object has no attribute 'get'``).
    if hasattr(text, "model_dump"):
        text = text.model_dump(by_alias=True, exclude_none=True)
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None
    ftype = fmt.get("type")
    if ftype == "json_object":
        return ResponseFormat(type="json_object")
    if ftype == "json_schema":
        schema = fmt.get("schema") or fmt.get("json_schema")
        name = fmt.get("name") or "response"
        if not isinstance(schema, dict):
            return None
        return ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name=name,
                description=fmt.get("description"),
                schema=schema,
                strict=bool(fmt.get("strict", False)),
            ),
        )
    return None


def _convert_status(openai_finish_reason: str | None) -> str:
    if openai_finish_reason == "length":
        return "incomplete"
    return "completed"


def _build_responses_usage(response: ChatCompletionResponse) -> ResponsesUsage:
    if not response.usage:
        return ResponsesUsage()
    prompt = response.usage.prompt_tokens
    completion = response.usage.completion_tokens
    cached = 0
    if response.usage.prompt_tokens_details is not None:
        cached = response.usage.prompt_tokens_details.cached_tokens or 0
    cached = max(0, min(cached, prompt))
    reasoning = 0
    if response.usage.completion_tokens_details is not None:
        reasoning = response.usage.completion_tokens_details.reasoning_tokens or 0
    return ResponsesUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=prompt + completion,
        input_tokens_details={"cached_tokens": cached},
        output_tokens_details={"reasoning_tokens": reasoning},
    )
