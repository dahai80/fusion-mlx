# SPDX-License-Identifier: Apache-2.0
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..admin.auth import require_admin
from ..middleware.auth import verify_management_access

logger = logging.getLogger(__name__)

probe_router = APIRouter()
router = APIRouter()


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


@router.post("/v1/requests/{request_id}/cancel")
async def cancel_request(request_id: str, is_admin: bool = Depends(require_admin)):
    from ..server import _server_state

    pool = _server_state.get("engine_pool")
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    logger.info("cancel_request: request_id=%s (best-effort)", request_id)
    return {
        "object": "request.cancel",
        "id": request_id,
        "cancelled": True,
    }
