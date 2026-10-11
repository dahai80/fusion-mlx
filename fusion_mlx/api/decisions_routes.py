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
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["decisions"])

_engine: Any = None
_engine_lock = asyncio.Lock()
_router_instance: Any = None
_router_lock = asyncio.Lock()

_PRESETS = {"router", "guard", "moderation"}


async def _get_router():
    global _router_instance
    if _router_instance is not None:
        return _router_instance
    async with _router_lock:
        if _router_instance is not None:
            return _router_instance

        def _build_router():
            from ..engines.laya import Router

            return Router(default="english", max_loaded=2)

        _router_instance = await asyncio.get_event_loop().run_in_executor(
            None, _build_router
        )
        logger.info("Laya Router initialized (english+multilingual auto-route)")
        return _router_instance


async def _get_engine(model: str | None):
    global _engine
    if _engine is not None:
        return _engine
    async with _engine_lock:
        if _engine is not None:
            return _engine
        from ..engines.laya_engine import LayaDecisionEngine

        eng = LayaDecisionEngine(model or "aac6fef/laya-multilingual-mlx")
        try:
            await eng.start()
        except Exception as exc:
            logger.error("Laya engine load failed: %s", exc)
            raise HTTPException(503, "Laya decision model unavailable")
        _engine = eng
        logger.info(
            "Laya decision engine lazy-started: %s (P0_NEVER_EVICT)", eng.model_name
        )
        try:
            from ..service.helpers import _get_engine_pool

            pool = _get_engine_pool()
            if pool is not None:
                from ..pool.engine_pool import EngineEntry

                entry = EngineEntry(
                    model_id=eng.model_name,
                    model_path="",
                    model_type="llm",
                    engine_type="simple",
                    estimated_size=eng.memory_mb * 1024 * 1024,
                    engine=eng,
                    is_pinned=True,
                    priority_tier="P0_NEVER_EVICT",
                )
                pool._entries[eng.model_name] = entry
                pool.pin_model(eng.model_name, tier="P0_NEVER_EVICT")
                logger.info(
                    "Laya registered in EnginePool: %s (pinned P0)", eng.model_name
                )
        except Exception:
            logger.debug("EnginePool registration for laya skipped", exc_info=True)
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


def _resolve_preset(name: str) -> dict[str, dict[str, Any]]:
    from ..engines.laya import guard_questions, moderation_questions, router_questions

    if name == "router":
        return router_questions()
    if name == "guard":
        return guard_questions()
    if name == "moderation":
        return moderation_questions()
    raise HTTPException(400, f"unknown preset: {name}")


class LayaDecideRequest(BaseModel):
    prompt: str = Field(..., description="input text to classify/score")
    questions: dict[str, dict[str, Any]] | None = Field(
        None, description="custom questions (mutually exclusive with preset)"
    )
    preset: str | None = Field(
        None,
        description="router | guard | moderation (mutually exclusive with questions)",
    )
    model: str | None = Field(None, description="Laya model id override")
    use_router: bool = Field(
        False,
        description="auto-detect language and route to english/multilingual checkpoint",
    )
    lang: str | None = Field(
        None, description="explicit language hint for router (e.g. 'en', 'zh')"
    )


@router.post("/v1/laya/decide", dependencies=[Depends(verify_api_key)])
async def laya_decide(req: LayaDecideRequest):
    if req.preset and req.questions:
        raise HTTPException(400, "preset and questions are mutually exclusive")
    if not req.preset and not req.questions:
        raise HTTPException(400, "either preset or questions must be specified")
    if req.preset and req.preset not in _PRESETS:
        raise HTTPException(400, f"preset must be one of {_PRESETS}")
    if req.questions is not None and len(req.questions) > 64:
        raise HTTPException(400, "too many questions (max 64)")

    questions = req.questions if req.questions else _resolve_preset(req.preset)
    t0 = time.perf_counter()
    try:
        if req.use_router:
            laya_router = await _get_router()

            def _predict_sync():
                return laya_router.predict(req.prompt, questions, lang=req.lang)

            result = await asyncio.get_event_loop().run_in_executor(None, _predict_sync)
            routing = result.get("routing", {})
            model_name = routing.get("model", "router")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            result["model"] = model_name
            result["latency_ms"] = round(latency_ms, 2)
            result["routing"] = routing
            logger.info(
                "laya/decide: router model=%s preset=%s latency=%.1fms",
                model_name,
                req.preset or "custom",
                latency_ms,
            )
            return result
        engine = await _get_engine(req.model)
        result = await engine.decide(req.prompt, questions)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result["model"] = engine.model_name
        result["latency_ms"] = round(latency_ms, 2)
        logger.info(
            "laya/decide: preset=%s questions=%d latency=%.1fms",
            req.preset or "custom",
            len(questions),
            latency_ms,
        )
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("laya/decide validation error: %s", exc)
        raise HTTPException(422, str(exc))
    except Exception:
        logger.exception("laya/decide failed")
        raise HTTPException(500, "Internal server error")


@router.get("/v1/laya/status", dependencies=[Depends(verify_api_key)])
async def laya_status():
    loaded = _engine is not None
    info: dict[str, Any] = {
        "loaded": loaded,
        "model": _engine.model_name if loaded else None,
        "priority_tier": "P0_NEVER_EVICT",
        "pinned": True,
        "router_available": _router_instance is not None,
    }
    if loaded:
        stats = _engine.get_stats()
        info["stats"] = stats
    if _router_instance is not None:
        info["router"] = {
            "loaded_checkpoints": _router_instance.loaded,
            "max_loaded": _router_instance.max_loaded,
            "default": _router_instance.default,
        }
    logger.debug(
        "laya/status: loaded=%s router=%s", loaded, _router_instance is not None
    )
    return info
