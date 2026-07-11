# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from ..api.responses_adapter import (
    normalize_responses_tool_types,
    openai_to_responses,
    request_uses_computer_use,
    responses_to_openai,
    validate_responses_tool_choice,
    validate_responses_tool_types,
)
from ..api.responses_models import (
    ResponsesRequest,
    ResponsesUsage,
)
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import (
    StreamingToolCallFilter,
    clean_output_text,
    strip_special_tokens,
    strip_thinking_tags,
)
from ..service.helpers import (
    SSE_RESPONSE_HEADERS,
    _resolve_max_tokens,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    get_engine,
    maybe_apply_reasoning_effort,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolved_sampling_kwargs(openai_request: ChatCompletionRequest) -> dict:
    out = {
        "temperature": _resolve_temperature(openai_request.temperature),
        "top_p": _resolve_top_p(openai_request.top_p),
        "stop": getattr(openai_request, "stop", None),
    }
    return out


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/v1/responses")
async def create_response(request: Request):
    body = await request.json()
    responses_request = ResponsesRequest(**body)

    if responses_request.previous_response_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "previous_response_id is not supported by this server — "
                "fusion-mlx is a stateless Responses API shim. Re-send the "
                "full conversation history in the `input` field each turn."
            ),
        )

    normalize_responses_tool_types(responses_request.tools)
    raw_tools = None
    if responses_request.tools:
        raw_tools = [
            t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t
            for t in responses_request.tools
        ]
    validate_responses_tool_types(raw_tools)
    validate_responses_tool_choice(responses_request.tool_choice, raw_tools)

    openai_request = responses_to_openai(responses_request)
    maybe_apply_reasoning_effort(openai_request)

    _validate_model_name(openai_request.model)
    engine = get_engine(openai_request.model)

    if responses_request.stream:
        return await _stream_responses(
            engine, openai_request, responses_request, request
        )
    else:
        return await _non_stream(engine, openai_request, responses_request, request)


async def _non_stream(
    engine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    request: Request,
) -> Response:
    created_at = int(time.time())

    messages = _prepare_messages(openai_request)

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(openai_request.max_tokens),
        **_resolved_sampling_kwargs(openai_request),
    }
    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    resolved_thinking = getattr(openai_request, "enable_thinking", None)
    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

    start_time = time.perf_counter()
    try:
        output = await asyncio.wait_for(
            engine.chat(messages=messages, **chat_kwargs),
            timeout=300.0,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out")

    elapsed = time.perf_counter() - start_time
    logger.info(
        "responses non-stream: %.2fs, %d tokens",
        elapsed,
        output.num_completion_tokens if hasattr(output, "num_completion_tokens") else 0,
    )

    text = output.text if hasattr(output, "text") else str(output)
    text = strip_special_tokens(text)
    text = strip_thinking_tags(text)
    text = clean_output_text(text)

    tool_calls = None
    if hasattr(output, "tool_calls") and output.tool_calls:
        tool_calls = output.tool_calls

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
    max_tokens = chat_kwargs.get("max_tokens", 0)
    if max_tokens and hasattr(output, "num_completion_tokens"):
        if output.num_completion_tokens >= max_tokens:
            finish_reason = "length"

    from ..api.models import AssistantMessage, ChatCompletionChoice, Usage

    assistant_msg = AssistantMessage(
        content=text,
        tool_calls=tool_calls,
    )
    if hasattr(output, "reasoning_content") and output.reasoning_content:
        assistant_msg.reasoning_content = output.reasoning_content

    prompt_tokens = (
        output.num_prompt_tokens if hasattr(output, "num_prompt_tokens") else 0
    )
    completion_tokens = (
        output.num_completion_tokens if hasattr(output, "num_completion_tokens") else 0
    )

    chat_response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=openai_request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=assistant_msg,
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    responses_response = openai_to_responses(
        chat_response, openai_request.model, responses_request, created_at
    )

    return Response(
        content=responses_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


async def _stream_responses(
    engine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    request: Request,
) -> StreamingResponse:
    created_at = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    messages = _prepare_messages(openai_request)

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(openai_request.max_tokens),
        **_resolved_sampling_kwargs(openai_request),
    }
    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    resolved_thinking = getattr(openai_request, "enable_thinking", None)
    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

    uses_computer_use = request_uses_computer_use(responses_request)

    async def _generate() -> AsyncIterator[str]:
        output_index = 0
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"
        tool_calls_collected = []
        text_parts = []
        reasoning_parts = []
        in_thinking = False
        tool_filter = StreamingToolCallFilter()

        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": openai_request.model,
                    "status": "in_progress",
                    "output": [],
                },
            },
        )

        yield _sse(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": {"id": response_id, "status": "in_progress"},
            },
        )

        try:
            stream = await engine.chat(messages=messages, stream=True, **chat_kwargs)
            async for chunk in stream:
                if await request.is_disconnected():
                    logger.info("Client disconnected during responses stream")
                    break

                delta_text = None
                delta_reasoning = None
                chunk_tool_calls = None
                chunk_finish = None

                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        c = choices[0]
                        delta = c.get("delta", {})
                        delta_text = delta.get("content")
                        delta_reasoning = delta.get("reasoning_content")
                        chunk_tool_calls = delta.get("tool_calls")
                        chunk_finish = c.get("finish_reason")
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get(
                            "completion_tokens", completion_tokens
                        )
                elif hasattr(chunk, "choices") and chunk.choices:
                    c = chunk.choices[0]
                    delta = c.delta if hasattr(c, "delta") else {}
                    delta_text = getattr(delta, "content", None)
                    delta_reasoning = getattr(delta, "reasoning_content", None)
                    chunk_tool_calls = getattr(delta, "tool_calls", None)
                    chunk_finish = getattr(c, "finish_reason", None)
                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = getattr(
                            chunk.usage, "prompt_tokens", prompt_tokens
                        )
                        completion_tokens = getattr(
                            chunk.usage, "completion_tokens", completion_tokens
                        )

                if delta_reasoning:
                    if not in_thinking and not reasoning_parts:
                        in_thinking = True
                        reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
                        yield _sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": output_index,
                                "item": {
                                    "type": "reasoning",
                                    "id": reasoning_id,
                                    "status": "in_progress",
                                    "summary": [],
                                },
                            },
                        )
                    reasoning_parts.append(delta_reasoning)
                    yield _sse(
                        "response.reasoning_summary_text.delta",
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": reasoning_id,
                            "output_index": output_index,
                            "delta": delta_reasoning,
                        },
                    )

                if delta_text:
                    filtered = tool_filter.process(delta_text)
                    if filtered:
                        if not text_parts:
                            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                            yield _sse(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": output_index
                                    + (1 if reasoning_parts else 0),
                                    "item": {
                                        "type": "message",
                                        "id": msg_id,
                                        "role": "assistant",
                                        "status": "in_progress",
                                        "content": [],
                                    },
                                },
                            )
                        text_parts.append(filtered)
                        yield _sse(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "output_index": output_index
                                + (1 if reasoning_parts else 0),
                                "content_index": 0,
                                "delta": filtered,
                            },
                        )

                if chunk_tool_calls:
                    for tc in chunk_tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id", "")
                            func = tc.get("function", {})
                            tc_name = func.get("name", "")
                            tc_args = func.get("arguments", "")
                        else:
                            tc_id = getattr(tc, "id", "")
                            func = getattr(tc, "function", None)
                            tc_name = getattr(func, "name", "") if func else ""
                            tc_args = getattr(func, "arguments", "") if func else ""

                        if (
                            tc_name
                            and tc_id
                            and tc_id not in [t.get("id") for t in tool_calls_collected]
                        ):
                            tool_calls_collected.append(
                                {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": tc_name, "arguments": tc_args},
                                }
                            )
                            tc_output_idx = (
                                output_index
                                + (1 if reasoning_parts else 0)
                                + (1 if text_parts else 0)
                                + len(tool_calls_collected)
                                - 1
                            )
                            if uses_computer_use and tc_name == "computer":
                                yield _sse(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": tc_output_idx,
                                        "item": {
                                            "type": "computer_call",
                                            "id": f"cu_{uuid.uuid4().hex[:24]}",
                                            "call_id": tc_id,
                                            "status": "in_progress",
                                        },
                                    },
                                )
                            else:
                                yield _sse(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": tc_output_idx,
                                        "item": {
                                            "type": "function_call",
                                            "id": f"fc_{uuid.uuid4().hex[:24]}",
                                            "call_id": tc_id,
                                            "name": tc_name,
                                            "status": "in_progress",
                                        },
                                    },
                                )
                        elif tc_args:
                            for existing in tool_calls_collected:
                                if existing.get("id") == tc_id:
                                    existing["function"]["arguments"] += tc_args
                                    break
                            yield _sse(
                                "response.function_call_arguments.delta",
                                {
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": tc_id,
                                    "output_index": 0,
                                    "delta": tc_args,
                                },
                            )

                if chunk_finish:
                    finish_reason = chunk_finish

            remaining = tool_filter.flush()
            if remaining and not text_parts:
                msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                yield _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index + (1 if reasoning_parts else 0),
                        "item": {
                            "type": "message",
                            "id": msg_id,
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                )
                text_parts.append(remaining)
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "output_index": output_index + (1 if reasoning_parts else 0),
                        "content_index": 0,
                        "delta": remaining,
                    },
                )
            elif remaining:
                text_parts.append(remaining)
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "output_index": output_index + (1 if reasoning_parts else 0),
                        "content_index": 0,
                        "delta": remaining,
                    },
                )

        except Exception as e:
            logger.error("responses stream error: %s", e, exc_info=True)
            yield _sse(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "status": "failed",
                        "error": {"type": "server_error", "message": str(e)},
                    },
                },
            )
            return

        status = "incomplete" if finish_reason == "length" else "completed"
        incomplete_details = (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        )

        usage = ResponsesUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": openai_request.model,
                    "status": status,
                    "output": [],
                    "usage": usage.model_dump(exclude_none=True),
                    "incomplete_details": incomplete_details,
                    "parallel_tool_calls": bool(responses_request.parallel_tool_calls),
                    "tool_choice": responses_request.tool_choice or "auto",
                },
            },
        )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=SSE_RESPONSE_HEADERS,
    )


def _prepare_messages(openai_request: ChatCompletionRequest) -> list[dict]:
    messages = []
    for msg in openai_request.messages:
        d = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg
        if isinstance(d, dict):
            messages.append(d)
        else:
            messages.append({"role": getattr(msg, "role", "user"), "content": str(msg)})
    return messages
