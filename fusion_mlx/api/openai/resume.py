# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible resumable streaming and resume-completion routes."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...middleware.auth import check_rate_limit, request_principal, verify_api_key
from ..openai_models import ChatCompletionRequest
from ._common import logger, router
from .streaming import _stream_chat, _stream_chat_generator

_MAX_SESSIONS_PER_PRINCIPAL = 8


@router.post("/stream")
async def start_resumable_stream(
    request: ChatCompletionRequest,
    http_request: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
) -> Any:
    # #801 resumable streaming. Starts a generation whose output survives
    # the originating connection so a dropped client can reconnect via
    # GET /v1/streams/lookup and receive the full text. A background task
    # drives _stream_chat_generator (same formatting path as the normal
    # SSE stream) and appends each chunk to a StreamSession; this route
    # either returns the session id as JSON (return_session_only=true) or
    # itself serves the live SSE tail so a first-time caller sees output
    # immediately and may disconnect/reconnect at will.
    from ...server import resolve_model_with_profile
    from ...stream_session import get_store

    principal = request_principal(http_request)
    store = get_store()
    # #801 DoS guard: cap concurrent live sessions per caller.
    if store.count_for_principal(principal) >= _MAX_SESSIONS_PER_PRINCIPAL:
        logger.warning(
            "resumable stream: principal=%s at session cap (%d), rejecting",
            principal,
            _MAX_SESSIONS_PER_PRINCIPAL,
        )
        raise HTTPException(
            429,
            f"too many concurrent resumable streams (max "
            f"{_MAX_SESSIONS_PER_PRINCIPAL})",
        )
    session_id = f"sess-{uuid.uuid4().hex[:16]}"
    session = store.create(session_id, principal=principal)
    request.stream = True

    model_name, profile_overrides = resolve_model_with_profile(request.model)
    adapter_path = getattr(request, "adapters", None)
    # Resolve engine up front so 404/503 surface as HTTP errors, not ASGI 500s.
    from ._common import _release_engine, _resolve_engine

    engine = await _resolve_engine(model_name, adapter_path=adapter_path)
    if engine is None:
        await _release_engine(model_name, adapter_path=adapter_path)
        store.drop(session_id)
        raise HTTPException(404, f"Model {model_name} not available")

    async def _producer():
        # Outer try/finally GUARANTEES the session is marked terminal even if
        # the inner arms raise before reaching mark_complete/mark_error —
        # otherwise a consumer would hang forever on _new_data.wait().
        try:
            try:
                async for chunk in _stream_chat_generator(
                    request,
                    engine,
                    model_name,
                    adapter_path,
                    principal=principal,
                    profile_overrides=profile_overrides,
                ):
                    session.append(chunk)
            except HTTPException as exc:
                session.mark_error(f"http_{exc.status_code}: {exc.detail}")
            except Exception as exc:
                logger.exception("resumable stream producer failed: %s", exc)
                session.mark_error(f"{type(exc).__name__}: {exc}")
            else:
                session.mark_complete(finish_reason="stop")
        finally:
            if not session.complete:
                logger.error(
                    "resumable stream producer ended without terminal mark: "
                    "%s — forcing error to unblock consumers",
                    session_id,
                )
                session.mark_error("producer ended without terminal mark")
            await _release_engine(model_name, adapter_path=adapter_path)

    # Kick the producer off in the background; it outlives this response.
    producer_task = asyncio.create_task(
        _producer(), name=f"stream-producer-{session_id}"
    )
    session.producer_task = producer_task

    if not getattr(request, "return_session_only", False):
        # Serve the live SSE tail right here so a first-time caller streams
        # immediately; disconnecting does NOT abort the producer.
        return StreamingResponse(
            _resume_consumer(session),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return {"stream_session_id": session_id, "status": "started"}


@router.get("/streams/lookup")
async def lookup_resumable_stream(
    session_id: str,
    http_request: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
) -> Any:
    # #801 resume/lookup an active resumable stream across a new connection.
    # Replay the full buffered SSE from index 0 then live-tail the producer
    # until the session completes. The client receives the same SSE sequence
    # it would have on the original connection, regardless of when it
    # reconnects (until the session TTL of 1h elapses).
    from ...stream_session import get_store

    principal = request_principal(http_request)
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            404,
            f"stream session {session_id} not found (expired or never started)",
        )
    # #801 IDOR: only the principal that started the stream may resume it.
    # A valid API key alone is not enough — it must be the SAME caller.
    if session.principal is not None and session.principal != principal:
        logger.warning(
            "resumable stream lookup DENIED: principal mismatch on %s "
            "(owner=%s caller=%s)",
            session_id,
            session.principal,
            principal,
        )
        raise HTTPException(403, "stream session does not belong to this caller")
    return StreamingResponse(
        _resume_consumer(session),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _resume_consumer(session):
    # Replay buffered events then live-tail until the session is complete.
    # Disconnecting the consumer does NOT cancel the producer. Breaks out
    # if the producer task is done AND the buffer is drained, so a producer
    # that died without a terminal mark (shouldn't happen post-fix) cannot
    # hang the consumer forever.
    idx = 0
    while True:
        while idx < len(session.events):
            yield session.events[idx]
            idx += 1
        if session.complete:
            return
        if session.producer_finished() and idx >= len(session.events):
            # Producer is gone and nothing more will arrive — stop rather
            # than block on _new_data.wait() indefinitely.
            logger.warning(
                "resumable consumer: producer finished without complete; "
                "stopping at %d events",
                idx,
            )
            return
        session._new_data.clear()
        await session._new_data.wait()


class ResumeRequest(BaseModel):
    previous_request_id: str
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = True


@router.post("/resume")
async def resume_completion(
    body: ResumeRequest,
    _auth: bool = Depends(verify_api_key),
) -> Any:
    """Resume a chat completion from a prior request's persisted KV.

    Loads the most recent disk KV checkpoint written for
    ``previous_request_id`` (on a client disconnect / scheduler abort),
    seeds a fresh request with that cached tail, and streams the
    continuation. Requires FUSION_MLX_KV_CHECKPOINT_INTERVAL>0; without
    it the persist end never wrote a checkpoint and this endpoint
    returns 409 so the client falls back to a normal completion.
    """
    from ...service.kv_resume import load_resumable_kv

    loaded = load_resumable_kv(body.previous_request_id, model_name=body.model)
    if loaded is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No resumable KV checkpoint for "
                f"{body.previous_request_id}. Either checkpointing is "
                "disabled (FUSION_MLX_KV_CHECKPOINT_INTERVAL=0) or the "
                "prior request never persisted (clean finish / aborted "
                "before first boundary)."
            ),
        )

    logger.info(
        "Resume: loaded KV for %s at %d tokens -> seeding continuation",
        body.previous_request_id,
        loaded.token_offset,
    )

    chat_req = ChatCompletionRequest(
        model=body.model,
        messages=body.messages,
        max_tokens=body.max_tokens or 4096,
        temperature=body.temperature if body.temperature is not None else 0.7,
        top_p=body.top_p if body.top_p is not None else 0.9,
        stream=body.stream,
    )

    if not body.stream:
        # Non-stream resume: seed via the engine add_request resume path
        # through _run_chat is not plumbed today; route through stream and
        # let callers that need a single buffer collect the SSE. Most
        # disconnect-resume callers stream anyway.
        chat_req.stream = True

    return await _stream_chat(
        chat_req,
        _skip_cap_check=False,
        resume_prompt_cache=list(loaded.cache),
        resume_cached_tokens=int(loaded.token_offset),
    )
