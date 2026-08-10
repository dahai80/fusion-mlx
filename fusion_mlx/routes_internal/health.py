# SPDX-License-Identifier: Apache-2.0
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..middleware.auth import verify_api_key_or_x_api_key, verify_management_access

logger = logging.getLogger(__name__)

probe_router = APIRouter()
router = APIRouter()
# admin_router carries the destructive control-plane routes (cancel,
# cache-clear, DELETE aliases). Lost in the routes/->routes_internal/ rename
# (commit 5fd79e0) along with the DELETE aliases; restored here to match the
# rapid-mlx contract. Auth is the dual Bearer/x-api-key shape (Anthropic
# clients use x-api-key) -- NOT the reverted X-Fusion-Internal header gate.
admin_router = APIRouter(dependencies=[Depends(verify_api_key_or_x_api_key)])


@probe_router.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok"}


@probe_router.get("/health")
async def health():
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    preloading = _server_state.get("preloading", False)
    model_loaded = pool is not None and pool.loaded_model_count > 0
    loaded_models = []
    if pool:
        loaded_models = pool.get_loaded_model_ids()
    ready = pool is not None and not preloading
    return {
        "status": "preloading" if preloading else "healthy",
        "ready": ready,
        "model_loaded": model_loaded,
        "loaded_models": loaded_models,
    }


@probe_router.get("/health/ready")
async def health_ready():
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    preloading = _server_state.get("preloading", False)
    if pool is None or pool.loaded_model_count == 0:
        raise HTTPException(status_code=503, detail="model loading")
    if preloading:
        raise HTTPException(status_code=503, detail="preloading models")
    return {"ready": True}


@probe_router.get("/healthz")
async def healthz():
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    draining = _server_state.get("draining", False)
    preloading = _server_state.get("preloading", False)
    if draining:
        return JSONResponse(
            status_code=503,
            content={
                "status": "draining",
                "ready": False,
                "model_loaded": pool is not None and pool.loaded_model_count > 0,
            },
        )
    if preloading:
        return JSONResponse(
            status_code=503,
            content={
                "status": "preloading",
                "ready": False,
                "model_loaded": pool is not None and pool.loaded_model_count > 0,
            },
        )
    return {
        "status": "healthy",
        "ready": pool is not None,
        "model_loaded": pool is not None and pool.loaded_model_count > 0,
    }


@probe_router.get("/readyz")
async def readyz():
    return await health_ready()


@probe_router.get("/livez")
async def livez():
    return {"status": "alive"}


@router.get("/v1/status")
async def status(_auth: bool = Depends(verify_management_access)):
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    if pool is None or pool.loaded_model_count == 0:
        return {"status": "not_loaded", "model": None, "requests": []}
    from ..server_metrics import get_server_metrics

    metrics = get_server_metrics().to_dict()
    return {
        "status": "ok",
        "loaded_models": pool.get_loaded_model_ids(),
        "total_requests": metrics.get("total_requests", 0),
        "total_prompt_tokens": metrics.get("total_prompt_tokens", 0),
        "total_completion_tokens": metrics.get("total_tokens_generated", 0),
    }


@admin_router.post("/v1/requests/{request_id}/cancel")
async def cancel_request(request_id: str):
    # F-151: cancel MUST actually abort the in-flight request (was a no-op
    # stub that always returned 200/cancelled=True after the routes rename).
    # 404 on unknown id (don't confirm a real engine to ID-pokers), 500
    # generic on engine error (don't echo HF path / repo id from exceptions),
    # success envelope omits model_name (no weight fingerprinting).
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    # Find the engine owning this request. Iterate loaded engines; check
    # scheduler.requests for synchronous existence (abort_request itself
    # is deferred and always returns True, so the 404 signal must come
    # from the existence check, not the abort return value).
    owning_engine = None
    for _model_id, entry in getattr(pool, "_entries", {}).items():
        engine = getattr(entry, "engine", None)
        if engine is None:
            continue
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None and request_id in getattr(scheduler, "requests", {}):
            owning_engine = engine
            break

    if owning_engine is None:
        logger.info("cancel_request: unknown request_id=%s -> 404", request_id)
        raise HTTPException(
            status_code=404,
            detail="Request not found or already finished",
        )

    try:
        await owning_engine.abort_request(request_id)
    except Exception:
        # F-151: don't echo the exception (engine messages may carry the
        # HF repo path / snapshot location). Full traceback to server log.
        logger.exception("Failed to cancel request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel request (see server logs)",
        ) from None

    logger.info("cancel_request: accepted request_id=%s", request_id)
    return {
        "object": "request.cancel",
        "id": request_id,
        "cancelled": True,
    }


@admin_router.delete("/v1/requests/{request_id}")
async def delete_request(request_id: str):
    # OpenAI-style alias for cancelling an active or queued request.
    return await cancel_request(request_id)


@admin_router.post("/v1/cache/clear")
async def clear_cache():
    # Clear the prompt KV cache across loaded engines.
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    cleared = 0
    for _model_id, entry in getattr(pool, "_entries", {}).items():
        engine = getattr(entry, "engine", None)
        if engine is None:
            continue
        model = getattr(engine, "_model", None)
        prompt_cache = getattr(model, "_prompt_cache", None)
        if prompt_cache is not None and hasattr(prompt_cache, "clear"):
            try:
                prompt_cache.clear()
                cleared += 1
            except Exception as e:
                logger.warning("cache clear failed for %s: %s", _model_id, e)
    logger.info("cache/clear: cleared %d engine prompt cache(s)", cleared)
    return {"status": "ok", "cleared": cleared}


@admin_router.delete("/v1/cache")
async def delete_cache():
    # Alias for POST /v1/cache/clear (OpenAI-style DELETE).
    return await clear_cache()
