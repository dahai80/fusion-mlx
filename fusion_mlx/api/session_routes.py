# SPDX-License-Identifier: Apache-2.0
"""Per-session token usage stats API routes (issue #226).

- GET  /v1/sessions/{session_id}/stats   -> cumulative token usage
- PUT  /v1/sessions/{session_id}/context -> set per-session max context cap
- GET  /v1/context/budget                -> context window budget (issue #327)

Sessions are scoped by authenticated principal (IDOR fix): each caller can
only read/modify its own sessions. The principal id is derived from the
request via ``request_principal`` (reuses the rate-limit bucket = HMAC of
the bearer token, else client subnet). Foreign sessions return 404 to avoid
existence disclosure.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from ..middleware.auth import check_rate_limit, request_principal, verify_api_key
from ..sessions import get_session_tracker, set_session_max_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["sessions"])

_pool: Any = None
_server_state: Any = None


def set_sessions_context(pool: Any, server_state: Any) -> None:
    global _pool, _server_state
    _pool = pool
    _server_state = server_state


class SessionContextConfig(BaseModel):
    max_context_tokens: int | None = None


@router.get(
    "/sessions/{session_id}/stats",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_session_stats(session_id: str, http_request: Request) -> dict:
    principal = request_principal(http_request)
    stats = get_session_tracker().get(session_id, principal=principal)
    if stats is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats.to_dict()


@router.put(
    "/sessions/{session_id}/context",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def set_session_context(
    session_id: str, config: SessionContextConfig, http_request: Request
) -> dict:
    principal = request_principal(http_request)
    ok = set_session_max_context(
        session_id, config.max_context_tokens, principal=principal
    )
    if not ok:
        raise HTTPException(
            status_code=400, detail="Invalid session_id or max_context_tokens"
        )
    logger.info(
        "session context cap set: principal=%s session=%s max_context_tokens=%s",
        principal[:8],
        session_id[:8],
        config.max_context_tokens,
    )
    return {"session_id": session_id, "max_context_tokens": config.max_context_tokens}


@router.get(
    "/context/budget",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_context_budget(
    http_request: Request,
    model_id: str = Query(..., description="Model identifier"),
    session_id: str | None = Query(
        None, description="Optional session id for cumulative usage"
    ),
) -> dict:
    from ..server import get_max_context_window

    ctx_window = get_max_context_window(model_id)
    if ctx_window is None or ctx_window <= 0:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model or no context window for '{model_id}'",
        )

    prompt_used = 0
    completion_used = 0
    if session_id:
        principal = request_principal(http_request)
        stats = get_session_tracker().get(session_id, principal=principal)
        if stats is not None:
            prompt_used = stats.prompt_tokens
            completion_used = stats.completion_tokens

    total_used = prompt_used + completion_used
    remaining = max(0, ctx_window - total_used)
    utilization = total_used / ctx_window if ctx_window > 0 else 0.0

    warning = utilization > 0.7
    recommendation = None
    if utilization > 0.9:
        recommendation = (
            "Context nearly full. Reduce conversation length or use /compact."
        )
    elif utilization > 0.7:
        recommendation = "Context usage high. Consider compacting conversation."

    return {
        "model_id": model_id,
        "context_window": ctx_window,
        "prompt_tokens_used": prompt_used,
        "completion_tokens_used": completion_used,
        "tokens_used": total_used,
        "remaining_tokens": remaining,
        "utilization": round(utilization, 3),
        "utilization_percent": round(utilization * 100, 1),
        "warning": warning,
        "recommendation": recommendation,
    }
