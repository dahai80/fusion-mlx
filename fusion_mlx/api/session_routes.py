# SPDX-License-Identifier: Apache-2.0
"""Per-session token usage stats API routes (issue #226).

- GET  /v1/sessions/{session_id}/stats   -> cumulative token usage
- PUT  /v1/sessions/{session_id}/context -> set per-session max context cap
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..middleware.auth import check_rate_limit, verify_api_key
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
async def get_session_stats(session_id: str) -> dict:
    stats = get_session_tracker().get(session_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats.to_dict()


@router.put(
    "/sessions/{session_id}/context",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def set_session_context(session_id: str, config: SessionContextConfig) -> dict:
    ok = set_session_max_context(session_id, config.max_context_tokens)
    if not ok:
        raise HTTPException(
            status_code=400, detail="Invalid session_id or max_context_tokens"
        )
    logger.info(
        "session context cap set: session=%s max_context_tokens=%s",
        session_id[:8],
        config.max_context_tokens,
    )
    return {"session_id": session_id, "max_context_tokens": config.max_context_tokens}
