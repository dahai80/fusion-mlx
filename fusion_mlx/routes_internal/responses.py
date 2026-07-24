# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..middleware.auth import verify_api_key
from ..api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from ..api.response_format_metrics import (
    incr_strict_repair_attempt,
    incr_strict_repair_skipped_context_overflow,
    incr_strict_repair_success,
    incr_strict_request,
    incr_strict_violation,
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
from ..api.strict_json_schema import (
    build_repair_messages,
    build_violation_envelope,
    repair_retry_enabled,
    strict_enforcement_enabled,
    validate_and_envelope,
)
from ..api.tool_calling import (
    check_schema_validity,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    is_strict_json_schema,
    validate_output_against_schema,
)
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
    _wait_with_disconnect,
    get_engine,
    maybe_apply_reasoning_effort,
    repair_messages_fit_context,
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


def _resolve_strict_context(
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    engine,
) -> dict:
    # R12-4 /v1/responses parity: mirror chat.py's strict flow so the two
    # surfaces agree on the OpenAI strict=true contract. Pre-fix this route
    # called engine.chat directly and dropped the strict flag entirely.
    # Gates fire BEFORE any engine call; the dispatch tick happens once a
    # request passes the gates and is admitted to guided / postgen / disabled.
    response_format = getattr(openai_request, "response_format", None)
    strict_mode = is_strict_json_schema(response_format)
    if not strict_mode:
        return {
            "strict_mode": False,
            "strict_enforcement_active": False,
            "json_schema": None,
            "use_guided": False,
            "use_strict_postgen_validation": False,
        }

    strict_enforcement_active = strict_mode and strict_enforcement_enabled()
    schema_check = extract_json_schema_for_guided(response_format)
    if not schema_check:
        incr_strict_request()
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "text.format.json_schema.strict=true requires a "
                        "non-empty text.format.json_schema.schema. The "
                        "request set strict=true but the schema field is "
                        "missing or empty - the strict contract cannot be "
                        "enforced without one."
                    ),
                    "type": "invalid_request_error",
                    "code": "strict_schema_required",
                    "param": "text.format.schema",
                }
            },
        )
    schema_ok, schema_err = check_schema_validity(schema_check)
    if not schema_ok:
        incr_strict_request()
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "text.format.json_schema.schema is not a valid "
                        f"JSON Schema document: {schema_err}. Fix the "
                        "schema and retry."
                    ),
                    "type": "invalid_request_error",
                    "code": "invalid_strict_schema",
                    "param": "text.format.schema",
                }
            },
        )
    if openai_request.tools:
        incr_strict_request()
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "text.format.json_schema.strict=true cannot be "
                        "combined with 'tools' - the constrained-decoding "
                        "grammar is mutually exclusive with the tool-call "
                        "grammar. Drop one or the other and retry."
                    ),
                    "type": "invalid_request_error",
                    "code": "strict_with_tools_unsupported",
                    "param": "text.format.strict",
                }
            },
        )
    if responses_request.stream and strict_enforcement_active:
        # Constrained decoding on the Responses surface is buffered-only:
        # there is no guided-streaming SSE helper for the Responses event
        # shape today. Reject strict+stream with both escape hatches named.
        incr_strict_request()
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "text.format.json_schema.strict=true with "
                        "stream=true is not supported on /v1/responses - "
                        "constrained decoding on the Responses surface is "
                        "buffered-only. Either drop stream=true and retry, "
                        "or use /v1/chat/completions which supports strict "
                        "streaming."
                    ),
                    "type": "invalid_request_error",
                    "code": "strict_stream_unsupported",
                    "param": "text.format.strict",
                }
            },
        )

    incr_strict_request()
    use_guided = bool(engine.supports_guided_generation)
    use_strict_postgen_validation = False
    if use_guided:
        logger.info(
            "Using guided generation for JSON schema enforcement on "
            "/v1/responses (strict=true)"
        )
    elif strict_enforcement_active:
        use_strict_postgen_validation = True
        logger.info(
            "Strict json_schema mode active without [guided] extra on "
            "/v1/responses - engaging R12-4 post-generate validation + "
            "single repair retry path."
        )
    else:
        logger.warning(
            "Strict json_schema mode requested on /v1/responses but "
            "FUSION_MLX_STRICT_JSON_SCHEMA=off - falling through to "
            "unconstrained generation (legacy silent-pass-through)."
        )

    return {
        "strict_mode": True,
        "strict_enforcement_active": strict_enforcement_active,
        "json_schema": schema_check,
        "use_guided": use_guided,
        "use_strict_postgen_validation": use_strict_postgen_validation,
    }


async def _apply_responses_postgen_validation(
    engine,
    messages,
    chat_kwargs,
    output,
    json_schema,
    *,
    timeout: float = 300.0,
):
    # R12-4 non-guided strict enforcement on /v1/responses: the engine ran
    # UNCONSTRAINED; now validate the buffered output and - on failure -
    # attempt ONE repair retry with a system-prompt hint naming the failing
    # path. Mirrors chat.py:2894-3075 so the 422 envelope shape matches.
    ok, failure_details = validate_and_envelope(output.text or "", json_schema)
    attempts = 1
    if not ok and repair_retry_enabled():
        repair_messages = build_repair_messages(
            messages,
            output.text or "",
            json_schema,
            failure_details or {},
        )
        repair_kwargs = dict(chat_kwargs)
        for _k in ("tools", "tool_choice", "logprobs", "top_logprobs"):
            repair_kwargs.pop(_k, None)
        _repair_fits = repair_messages_fit_context(
            engine,
            repair_messages,
            tools=None,
            max_tokens=repair_kwargs.get("max_tokens"),
            enable_thinking=chat_kwargs.get("enable_thinking"),
        )
        repair_output = None
        if not _repair_fits:
            incr_strict_repair_skipped_context_overflow()
            logger.warning(
                "R12-4 strict json_schema repair retry SKIPPED on "
                "/v1/responses: post-build repair prompt would exceed "
                "model context window. Surfacing the ORIGINAL 422 "
                "json_schema_violation envelope instead of attempting a "
                "retry that would either 502 or truncate."
            )
        else:
            incr_strict_repair_attempt()
            attempts = 2
            logger.info(
                "R12-4 strict json_schema first attempt failed validation "
                "(%s) on /v1/responses; attempting single repair retry.",
                failure_details.get("reason") if failure_details else "?",
            )
            try:
                repair_output = await asyncio.wait_for(
                    engine.chat(messages=repair_messages, **repair_kwargs),
                    timeout=timeout,
                )
            except TimeoutError:
                raise HTTPException(status_code=504, detail="Generation timed out")
            except Exception as repair_err:
                logger.warning(
                    "R12-4 strict json_schema repair retry raised %s: %s "
                    "on /v1/responses; surfacing as 502 (server-side "
                    "generation failure, NOT a schema-validation breach).",
                    type(repair_err).__name__,
                    repair_err,
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": {
                            "message": (
                                "Strict json_schema repair retry failed: "
                                "the engine raised "
                                f"{type(repair_err).__name__} during the "
                                "second generation attempt. The initial "
                                "output had also failed schema validation; "
                                "investigate server logs."
                            ),
                            "type": "api_error",
                            "code": "strict_repair_engine_failure",
                            "param": "text.format",
                            "details": {
                                "initial_failure": failure_details,
                                "repair_exception": type(repair_err).__name__,
                            },
                        }
                    },
                ) from repair_err
        if repair_output is not None:
            ok2, failure2 = validate_and_envelope(repair_output.text or "", json_schema)
            if ok2:
                incr_strict_repair_success()
                logger.info(
                    "R12-4 strict json_schema repair retry succeeded on "
                    "/v1/responses."
                )
                from dataclasses import replace as _dc_replace

                initial_prompt_tokens = output.prompt_tokens
                initial_completion_tokens = output.completion_tokens
                output = _dc_replace(
                    repair_output,
                    prompt_tokens=(initial_prompt_tokens + repair_output.prompt_tokens),
                    completion_tokens=(
                        initial_completion_tokens + repair_output.completion_tokens
                    ),
                )
                ok = True
                failure_details = None
            else:
                failure_details = failure2
    if not ok:
        incr_strict_violation()
        envelope = build_violation_envelope(
            failure_details or {"reason": "schema_violation"},
            param="text.format",
            attempts=attempts,
        )
        logger.warning(
            "R12-4 strict json_schema validation failed after %d attempt(s) "
            "on /v1/responses: %s",
            attempts,
            (failure_details or {}).get("message"),
        )
        raise HTTPException(status_code=422, detail=envelope)
    return output


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
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

    # R12-4 parity: resolve the strict-mode dispatch decision (gates +
    # guided/postgen/disabled) once here so both the stream and non-stream
    # arms see it. The strict_stream_unsupported gate raises 400 for
    # strict+stream before we dispatch to the streaming helper.
    strict_ctx = _resolve_strict_context(openai_request, responses_request, engine)

    if responses_request.stream:
        return await _stream_responses(
            engine, openai_request, responses_request, request
        )
    else:
        return await _non_stream(
            engine, openai_request, responses_request, request, strict_ctx
        )


async def _non_stream(
    engine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    request: Request,
    strict_ctx: dict | None = None,
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

    if strict_ctx is None:
        strict_ctx = {
            "strict_mode": False,
            "use_guided": False,
            "use_strict_postgen_validation": False,
            "json_schema": None,
        }
    strict_mode = strict_ctx["strict_mode"]
    use_guided = strict_ctx["use_guided"]
    use_strict_postgen_validation = strict_ctx["use_strict_postgen_validation"]
    json_schema = strict_ctx["json_schema"]

    start_time = time.perf_counter()
    output = None
    if use_guided and json_schema:
        # Constrained dispatch. Strip any colliding ``raise_on_failure``
        # a caller may have injected via sampling kwargs, then set it
        # explicitly so guided failures propagate to the tight try below
        # (instead of the engine silently returning invalid output).
        guided_kwargs = {
            k: v for k, v in chat_kwargs.items() if k != "raise_on_failure"
        }
        guided_kwargs["raise_on_failure"] = True
        try:
            output = await _wait_with_disconnect(
                engine.generate_with_schema(
                    messages=messages,
                    json_schema=json_schema,
                    **guided_kwargs,
                ),
                request,
                timeout=300.0,
            )
        except HTTPException:
            raise
        except Exception as guided_err:
            if strict_mode:
                incr_strict_violation()
                logger.warning(
                    "Strict json_schema guided generation failed on "
                    "/v1/responses; refusing to fall back to unconstrained "
                    "because strict=true: %s",
                    guided_err,
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": {
                            "message": (
                                "strict response_format could not be "
                                "honored: the constrained-decoding path "
                                f"raised {type(guided_err).__name__} "
                                "before producing any output. The server "
                                "refuses to fall back to unconstrained "
                                "generation because the client asked for "
                                "strict=true."
                            ),
                            "type": "api_error",
                            "code": "strict_schema_violation",
                            "param": "text.format",
                        }
                    },
                ) from guided_err
            logger.warning(
                "Guided generation failed on /v1/responses, falling back "
                "to standard: %s",
                guided_err,
            )
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                request,
                timeout=300.0,
            )
        if strict_mode and output is not None:
            ok, err = validate_output_against_schema(output.text or "", json_schema)
            if not ok:
                incr_strict_violation()
                logger.warning(
                    "Strict json_schema response failed post-decode "
                    "validation on /v1/responses: %s",
                    err,
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": {
                            "message": (
                                "strict response_format violated: model "
                                "output did not validate against the "
                                f"supplied schema ({err}). This indicates "
                                "the constrained-decoding path silently "
                                "degraded."
                            ),
                            "type": "api_error",
                            "code": "strict_schema_violation",
                            "param": "text.format",
                        }
                    },
                )
    elif use_strict_postgen_validation and json_schema:
        try:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                request,
                timeout=300.0,
            )
        except HTTPException:
            raise
        if output is not None:
            output = await _apply_responses_postgen_validation(
                engine, messages, chat_kwargs, output, json_schema
            )
    else:
        try:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                request,
                timeout=300.0,
            )
        except HTTPException:
            raise

    if output is None:
        return Response(status_code=499, content="Client disconnected")

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
