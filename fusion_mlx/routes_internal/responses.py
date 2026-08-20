# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

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
from ..api.thinking import extract_thinking
from ..api.tool_calling import (
    check_schema_validity,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    is_strict_json_schema,
)
from ..api.utils import (
    StreamingToolCallFilter,
    clean_output_text,
    resolve_enable_thinking_default,
    strip_special_tokens,
)
from ..middleware.auth import verify_api_key
from ..service.helpers import (
    SSE_RESPONSE_HEADERS,
    _resolve_max_tokens,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _wait_with_disconnect,
    maybe_apply_reasoning_effort,
    repair_messages_fit_context,
)

logger = logging.getLogger(__name__)


def _parse_inline_tool_bytes(raw: str, chunk: Any):
    # #591 item 2: a ``tool_call``-channel chunk may carry the tool
    # argument JSON as raw ``new_text`` bytes (no structured
    # ``tool_calls``). Best-effort parse into the structured shape so the
    # args surface as a function_call item instead of leaking to
    # ``response.output_text.delta``. Unparseable bytes are dropped
    # (never emitted as text) — return None and let the structured
    # branch simply not fire.
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.debug("#591: unparseable tool_call-channel bytes dropped: %r", raw)
        return None
    if isinstance(obj, dict) and "name" in obj:
        return [
            {
                "id": getattr(chunk, "text", "") or f"call_{uuid.uuid4().hex[:24]}",
                "function": {
                    "name": obj.get("name", ""),
                    "arguments": json.dumps(obj.get("arguments", {})),
                },
            }
        ]
    return None


router = APIRouter()

_pool: Any = None


def set_responses_context(pool) -> None:
    global _pool
    _pool = pool


async def _resolve_engine(model_name: str, adapter_path=None):
    if _pool is not None:
        engine = await _pool.get_engine(
            model_name, _lease=True, adapter_path=adapter_path
        )
        return engine
    from ..service.helpers import get_engine

    logger.debug("_pool None, falling back to helpers.get_engine for %s", model_name)
    return get_engine(model_name)


async def _release_engine(model_name: str, adapter_path=None) -> None:
    if _pool is not None:
        await _pool.release_engine(model_name, adapter_path=adapter_path)


def _resolved_sampling_kwargs(openai_request: ChatCompletionRequest) -> dict:
    out = {
        "temperature": _resolve_temperature(openai_request.temperature),
        "top_p": _resolve_top_p(openai_request.top_p),
        "stop": getattr(openai_request, "stop", None),
    }
    # #355 / R6-H8: forward the extended sampling params the chat surface
    # threads to engine.chat so /v1/responses does not silently drop them.
    top_k = getattr(openai_request, "top_k", None)
    if top_k is not None:
        out["top_k"] = top_k
    min_p = getattr(openai_request, "min_p", None)
    if min_p is not None:
        out["min_p"] = min_p
    repetition_penalty = getattr(openai_request, "repetition_penalty", None)
    if repetition_penalty is not None:
        out["repetition_penalty"] = repetition_penalty
    presence_penalty = getattr(openai_request, "presence_penalty", None)
    if presence_penalty is not None:
        out["presence_penalty"] = presence_penalty
    frequency_penalty = getattr(openai_request, "frequency_penalty", None)
    if frequency_penalty is not None:
        out["frequency_penalty"] = frequency_penalty
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
    # #373 — the legacy engine-method guided path
    # (``engine.supports_guided_generation`` /
    # ``engine.generate_with_schema``) was removed when guided decoding
    # moved to the standalone-helper / grammar-compiler architecture, but
    # the route kept reading those symbols. No engine defines them, so
    # ``use_guided`` was always False and the live constrained path on
    # /v1/responses is the R12-4 post-generate validation branch below
    # (buffered-only by design — no guided-streaming SSE helper exists
    # for the Responses event shape). The dead ``engine.generate_with_schema``
    # call site was a latent ``AttributeError``; it is removed here.
    use_strict_postgen_validation = False
    if strict_enforcement_active:
        use_strict_postgen_validation = True
        logger.info(
            "Strict json_schema mode active on /v1/responses - engaging "
            "R12-4 post-generate validation + single repair retry path "
            "(constrained decoding on the Responses surface is buffered-only)."
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
        _repair_ct = repair_kwargs.get("chat_template_kwargs") or {}
        _repair_fits = repair_messages_fit_context(
            engine,
            repair_messages,
            tools=None,
            max_tokens=repair_kwargs.get("max_tokens"),
            enable_thinking=_repair_ct.get("enable_thinking"),
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
                    "R12-4 strict json_schema repair retry succeeded on /v1/responses."
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
    from ..server import resolve_model_with_profile

    resolved_model, _profile_overrides = resolve_model_with_profile(
        openai_request.model
    )
    adapter_path = getattr(openai_request, "adapters", None)
    logger.info(
        "responses lane resolve_model_with_profile: %s -> %s",
        openai_request.model,
        resolved_model,
    )
    engine = await _resolve_engine(resolved_model, adapter_path=adapter_path)
    if engine is None:
        await _release_engine(resolved_model, adapter_path=adapter_path)
        raise HTTPException(404, f"Model {resolved_model} not available")

    async def _release() -> None:
        await _release_engine(resolved_model, adapter_path=adapter_path)

    # R12-4 parity: resolve the strict-mode dispatch decision (gates +
    # guided/postgen/disabled) once here so both the stream and non-stream
    # arms see it. The strict_stream_unsupported gate raises 400 for
    # strict+stream before we dispatch to the streaming helper.
    strict_ctx = _resolve_strict_context(openai_request, responses_request, engine)

    if responses_request.stream:
        return await _stream_responses(
            engine, openai_request, responses_request, request, _release
        )
    else:
        try:
            return await _non_stream(
                engine, openai_request, responses_request, request, strict_ctx
            )
        finally:
            await _release()


async def _non_stream(
    engine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    request: Request,
    strict_ctx: dict | None = None,
) -> Response:
    created_at = int(time.time())

    messages = _prepare_messages(openai_request)
    from ..tool_parsers.ui_tars_tool_parser import inject_ui_tars_sysprompt_for_lane

    messages = inject_ui_tars_sysprompt_for_lane(
        messages,
        model_name=openai_request.model,
        tool_choice=getattr(openai_request, "tool_choice", None),
        tools=getattr(openai_request, "tools", None),
    )

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(openai_request.max_tokens),
        **_resolved_sampling_kwargs(openai_request),
    }
    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    # #364 — enable_thinking must reach the chat template via
    # chat_template_kwargs, NOT the top-level chat_kwargs (engine.chat
    # only forwards chat_template_kwargs to _apply_chat_template; a
    # top-level enable_thinking is silently dropped). Without this,
    # Qwen3 thinking models run in default thinking-on mode and a
    # truncated response (max_tokens) yields content=None — the visible
    # answer is lost while completion_tokens is still billed. Apply the
    # shared disable-by-default so non-stream matches openai/anthropic
    # routes (resolve_enable_thinking_default).
    ct_kwargs = dict(getattr(openai_request, "chat_template_kwargs", {}) or {})
    resolved_thinking = getattr(openai_request, "enable_thinking", None)
    if resolved_thinking is not None:
        ct_kwargs["enable_thinking"] = resolved_thinking
    resolve_enable_thinking_default(ct_kwargs)
    chat_kwargs["chat_template_kwargs"] = ct_kwargs

    if strict_ctx is None:
        strict_ctx = {
            "strict_mode": False,
            "use_strict_postgen_validation": False,
            "json_schema": None,
        }
    strict_mode = strict_ctx["strict_mode"]
    use_strict_postgen_validation = strict_ctx["use_strict_postgen_validation"]
    json_schema = strict_ctx["json_schema"]

    start_time = time.perf_counter()
    output = None
    # #373 — the legacy engine-method guided branch
    # (``engine.generate_with_schema`` / ``supports_guided_generation``) was
    # removed: no engine defines those symbols, so the branch was dead code
    # and a latent ``AttributeError``. The live constrained path on
    # /v1/responses is the R12-4 post-generate validation branch below
    # (buffered-only — the Responses surface has no guided-streaming SSE
    # helper). Live constrained decoding for the chat surface runs through
    # the grammar-compiler (xgrammar/llguidance) path in openai_routes.py.
    if use_strict_postgen_validation and json_schema:
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

    raw_text = output.text if hasattr(output, "text") else str(output)
    raw_text = strip_special_tokens(raw_text)

    # Extract thinking BEFORE stripping tags so reasoning_content survives
    engine_finish = getattr(output, "finish_reason", None)
    reasoning_content, content_text = extract_thinking(
        raw_text, finish_reason=engine_finish
    )
    content_text = clean_output_text(content_text)

    tool_calls = None
    if hasattr(output, "tool_calls") and output.tool_calls:
        tool_calls = output.tool_calls

    # Use engine finish_reason when available; fall back to heuristic
    if engine_finish:
        finish_reason = engine_finish
    elif tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    from ..api.models import AssistantMessage, ChatCompletionChoice, Usage

    assistant_msg = AssistantMessage(
        content=content_text,
        tool_calls=tool_calls,
    )
    if reasoning_content:
        assistant_msg.reasoning_content = reasoning_content
    # Also propagate from GenerationOutput if present
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
    _release=None,
) -> StreamingResponse:
    created_at = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    messages = _prepare_messages(openai_request)
    from ..tool_parsers.ui_tars_tool_parser import inject_ui_tars_sysprompt_for_lane

    messages = inject_ui_tars_sysprompt_for_lane(
        messages,
        model_name=openai_request.model,
        tool_choice=getattr(openai_request, "tool_choice", None),
        tools=getattr(openai_request, "tools", None),
    )

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(openai_request.max_tokens),
        **_resolved_sampling_kwargs(openai_request),
    }
    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    # #364 — route enable_thinking through chat_template_kwargs (see
    # _non_stream for rationale) so the Qwen3 chat template honors it.
    ct_kwargs_stream = dict(getattr(openai_request, "chat_template_kwargs", {}) or {})
    resolved_thinking = getattr(openai_request, "enable_thinking", None)
    if resolved_thinking is not None:
        ct_kwargs_stream["enable_thinking"] = resolved_thinking
    resolve_enable_thinking_default(ct_kwargs_stream)
    chat_kwargs["chat_template_kwargs"] = ct_kwargs_stream

    uses_computer_use = request_uses_computer_use(responses_request)

    async def _generate() -> AsyncIterator[str]:
        # #513/#520: stream handler rewritten to parse the FLAT
        # ``GenerationOutput`` chunks that ``BaseEngine.stream_chat``
        # actually yields (new_text / channel / text / tool_calls /
        # finish_reason / prompt_tokens / completion_tokens), while
        # keeping the legacy OpenAI-choices dict/obj shapes as fallback.
        # Ordering invariant (#513 reasoning-first): a ``reasoning`` item
        # (possibly with empty summary) is emitted BEFORE the message
        # item; ``output_index`` is a monotonic per-item counter and the
        # terminal ``response.output[]`` array is built 1:1 with the
        # wire-added events. ``channel == "tool_call"`` bytes never reach
        # ``response.output_text.delta`` (#591 item 2 leak guard).
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"
        tool_calls_collected = []
        text_parts = []
        reasoning_parts = []
        tool_filter = StreamingToolCallFilter()

        # Per-item wire state. Each added item consumes the next
        # ``output_index``; the same index space orders ``output[]``.
        next_index = 0
        reasoning_id = None
        reasoning_idx = None
        msg_id = None
        msg_idx = None
        reasoning_emitted = False
        msg_emitted = False
        # Terminal ``output[]`` items, appended in wire order.
        output_items = []

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

        def _ensure_reasoning_item():
            # Emit the leading ``reasoning`` item once, before any message
            # / tool item, even when reasoning text is empty (#513).
            nonlocal reasoning_id, reasoning_idx, reasoning_emitted, next_index
            if reasoning_emitted:
                return None
            reasoning_emitted = True
            reasoning_idx = next_index
            next_index += 1
            reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
            return _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": reasoning_idx,
                    "item": {
                        "type": "reasoning",
                        "id": reasoning_id,
                        "status": "in_progress",
                        "summary": [],
                    },
                },
            )

        def _ensure_message_item():
            nonlocal msg_id, msg_idx, msg_emitted, next_index
            if msg_emitted:
                return None
            msg_emitted = True
            msg_idx = next_index
            next_index += 1
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            return _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": msg_idx,
                    "item": {
                        "type": "message",
                        "id": msg_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                },
            )

        try:
            stream = engine.stream_chat(messages=messages, **chat_kwargs)
            async for chunk in stream:
                if await request.is_disconnected():
                    logger.info("Client disconnected during responses stream")
                    break

                delta_text = None
                delta_reasoning = None
                chunk_tool_calls = None
                chunk_finish = None
                chunk_channel = None

                # Flat GenerationOutput path (#513): engine.stream_chat
                # yields dataclass-like objects with new_text/channel.
                if hasattr(chunk, "new_text") and hasattr(chunk, "channel"):
                    raw_new = getattr(chunk, "new_text", None) or ""
                    chunk_channel = getattr(chunk, "channel", None)
                    chunk_finish = getattr(chunk, "finish_reason", None)
                    chunk_tool_calls = getattr(chunk, "tool_calls", None)
                    pt = getattr(chunk, "prompt_tokens", 0) or 0
                    ct = getattr(chunk, "completion_tokens", 0) or 0
                    if pt:
                        prompt_tokens = pt
                    if ct:
                        completion_tokens = ct
                    # Routing by channel: reasoning -> reasoning delta;
                    # tool_call -> structured calls only (NO text leak);
                    # content/None -> text delta.
                    if chunk_channel == "reasoning":
                        delta_reasoning = raw_new
                    elif chunk_channel == "tool_call":
                        # #591 item 2: tool_call-channel bytes are JSON
                        # arguments, never user-facing text. Try to
                        # surface them as structured tool calls.
                        delta_text = None
                        if not chunk_tool_calls:
                            chunk_tool_calls = _parse_inline_tool_bytes(raw_new, chunk)
                    else:
                        delta_text = raw_new
                elif isinstance(chunk, dict):
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
                    evt = _ensure_reasoning_item()
                    if evt is not None:
                        yield evt
                    reasoning_parts.append(delta_reasoning)
                    yield _sse(
                        "response.reasoning_summary_text.delta",
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": reasoning_id,
                            "output_index": reasoning_idx,
                            "delta": delta_reasoning,
                        },
                    )

                if delta_text:
                    filtered = tool_filter.process(delta_text)
                    if filtered:
                        if not reasoning_emitted:
                            evt = _ensure_reasoning_item()
                            if evt is not None:
                                yield evt
                        added_evt = _ensure_message_item()
                        if added_evt is not None:
                            yield added_evt
                        text_parts.append(filtered)
                        yield _sse(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "output_index": msg_idx,
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
                            if not reasoning_emitted:
                                evt = _ensure_reasoning_item()
                                if evt is not None:
                                    yield evt
                            tool_calls_collected.append(
                                {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": tc_name, "arguments": tc_args},
                                }
                            )
                            tc_idx = next_index
                            next_index += 1
                            if uses_computer_use and tc_name == "computer":
                                yield _sse(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": tc_idx,
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
                                        "output_index": tc_idx,
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
                if not reasoning_emitted:
                    evt = _ensure_reasoning_item()
                    if evt is not None:
                        yield evt
                added_evt = _ensure_message_item()
                if added_evt is not None:
                    yield added_evt
                text_parts.append(remaining)
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "output_index": msg_idx,
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
                        "output_index": msg_idx,
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

        # Reasoning-first: if the model produced NO reasoning bytes at
        # all, still emit an empty ``reasoning`` item before the message
        # (#513 M-3 root case). Done after the loop so it precedes the
        # terminal completed event only when nothing was emitted inline.
        if not reasoning_emitted:
            evt = _ensure_reasoning_item()
            if evt is not None:
                yield evt

        # Build the terminal ``output[]`` array in wire-index order and
        # emit ``output_item.done`` per item so SDK clients see finalized
        # shapes (reasoning summary, message content, tool calls).
        if reasoning_emitted:
            summary_text = "".join(reasoning_parts)
            reasoning_item = {
                "type": "reasoning",
                "id": reasoning_id,
                "status": "completed",
                "summary": (
                    [{"type": "summary_text", "text": summary_text}]
                    if summary_text
                    else []
                ),
            }
            output_items.append(reasoning_item)
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": reasoning_idx,
                    "item": reasoning_item,
                },
            )
        if msg_emitted:
            message_item = {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "".join(text_parts),
                    }
                ],
            }
            output_items.append(message_item)
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": msg_idx,
                    "item": message_item,
                },
            )
        for tc in tool_calls_collected:
            tc_item = {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", ""),
                "status": "completed",
            }
            output_items.append(tc_item)
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": next_index,
                    "item": tc_item,
                },
            )

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
                    "output": output_items,
                    "usage": usage.model_dump(exclude_none=True),
                    "incomplete_details": incomplete_details,
                    "parallel_tool_calls": bool(responses_request.parallel_tool_calls),
                    "tool_choice": responses_request.tool_choice or "auto",
                },
            },
        )

    async def _generate_with_release() -> AsyncIterator[str]:
        try:
            async for event in _generate():
                yield event
        finally:
            if _release is not None:
                await _release()

    return StreamingResponse(
        _generate_with_release(),
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
