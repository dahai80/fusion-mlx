# SPDX-License-Identifier: Apache-2.0
"""Laya typed-decisions API route for fusion-mlx.

POST /v1/decisions — mlx-serve parity. Takes {model, state, questions} and
returns Laya's predict schema: answers keyed by question id, each with
type/confidence/action.act_probability plus choice+probabilities (choice),
score+legend+probabilities (score), or noul (noul).

The Laya checkpoint layout is non-standard (no top-level config.json), so
this route self-manages a lazy LayaDecisionEngine singleton instead of
going through EnginePool discovery.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["decisions"])

_engine: Any = None
_engine_lock = asyncio.Lock()


async def _get_engine(model: str | None):
    global _engine
    if _engine is not None:
        return _engine
    async with _engine_lock:
        if _engine is not None:
            return _engine
        from ..engines.laya_engine import LayaDecisionEngine

        eng = LayaDecisionEngine(model or "aac6fef/laya-multilingual-mlx")
        await eng.start()
        _engine = eng
        logger.info("Laya decision engine lazy-started: %s", eng.model_name)
        return _engine


class DecisionRequest(BaseModel):
    model: str | None = Field(
        None, description="Laya model id (default aac6fef/laya-multilingual-mlx)"
    )
    state: str | dict[str, Any] = Field(
        ..., description="state string or JSON object (serialized like json.dumps)"
    )
    questions: dict[str, dict[str, Any]] = Field(
        ...,
        description="question id -> {type: choice|score|noul, instructions, criteria}",
    )


@router.post("/v1/decisions", dependencies=[Depends(verify_api_key)])
async def create_decision(req: DecisionRequest):
    if not req.questions:
        raise HTTPException(400, "questions must not be empty")
    if len(req.questions) > 64:
        raise HTTPException(400, "too many questions (max 64)")
    try:
        engine = await _get_engine(req.model)
        result = await engine.decide(req.state, req.questions)
        logger.info(
            "decisions served: model=%s questions=%d input_tokens=%d",
            req.model or "laya",
            len(req.questions),
            result.get("usage", {}).get("input_tokens", 0),
        )
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("decisions validation error: %s", exc)
        raise HTTPException(422, str(exc))
    except Exception as exc:
        logger.exception("decisions failed")
        raise HTTPException(500, "Internal server error")
