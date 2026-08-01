# SPDX-License-Identifier: Apache-2.0
"""Non-admin model manager API — scoped-key access for model lifecycle ops.

Allows fsb_*/model_mgr_* prefixed API keys to list, load, and unload models
without requiring admin session or full API key.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from .helpers import _get_engine_pool, format_size
from ..middleware.auth import verify_scoped_api_key

logger = logging.getLogger(__name__)

_router = APIRouter(prefix="/api/model-manager", tags=["model-manager"])


@_router.get("/models")
async def list_models(role: str = Depends(verify_scoped_api_key)):
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        return {"models": []}

    status = engine_pool.get_status()
    models_status = status.get("models", [])
    models = []
    for m in models_status:
        models.append({
            "id": m.get("id", ""),
            "loaded": m.get("loaded", False),
            "is_loading": m.get("is_loading", False),
            "estimated_size": m.get("estimated_size", 0),
            "estimated_size_formatted": format_size(m.get("estimated_size", 0)),
            "pinned": m.get("pinned", False),
            "engine_type": m.get("engine_type", "batched"),
            "model_type": m.get("model_type", "llm"),
        })
    logger.info("model-manager: listed %d models (role=%s)", len(models), role)
    return {"models": models}


@_router.post("/models/{model_id}/load")
async def load_model(
    model_id: str,
    role: str = Depends(verify_scoped_api_key),
):
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is not None:
        return {"status": "ok", "model_id": model_id, "message": f"Already loaded: {model_id}"}
    if entry.is_loading:
        raise HTTPException(status_code=409, detail=f"Model is already loading: {model_id}")

    try:
        await engine_pool.get_engine(model_id)
    except Exception as e:
        logger.error("model-manager: load failed for %s: %s", model_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")

    logger.info("model-manager: loaded %s (role=%s)", model_id, role)
    return {"status": "ok", "model_id": model_id, "message": f"Loaded {model_id}"}


@_router.post("/models/{model_id}/unload")
async def unload_model(
    model_id: str,
    role: str = Depends(verify_scoped_api_key),
):
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.engine is None:
        raise HTTPException(status_code=400, detail=f"Model not loaded: {model_id}")

    await engine_pool.unload_engine_async(model_id)
    logger.info("model-manager: unloaded %s (role=%s)", model_id, role)
    return {"status": "ok", "model_id": model_id, "message": f"Unloaded {model_id}"}


@_router.get("/models/{model_id}/status")
async def model_status(
    model_id: str,
    role: str = Depends(verify_scoped_api_key),
):
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return {
        "id": model_id,
        "loaded": entry.engine is not None,
        "is_loading": entry.is_loading,
        "pinned": entry.pinned,
    }


@_router.get("/embedding/status")
async def embedding_status(role: str = Depends(verify_scoped_api_key)):
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    embedding_models = []
    for mid, entry in engine_pool._entries.items():
        if entry.model_type == "embedding":
            embedding_models.append({
                "id": mid,
                "loaded": entry.engine is not None,
                "is_loading": entry.is_loading,
                "pinned": entry.pinned,
            })
    logger.info("model-manager: embedding status %d models (role=%s)", len(embedding_models), role)
    return {"embedding_models": embedding_models}


@_router.post("/embedding/{model_id}/pin")
async def pin_embedding_model(
    model_id: str,
    role: str = Depends(verify_scoped_api_key),
):
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.model_type != "embedding":
        raise HTTPException(status_code=400, detail=f"Model {model_id} is not an embedding model")

    engine_pool.set_pinned(model_id, True)
    logger.info("model-manager: pinned embedding %s (role=%s)", model_id, role)
    return {"status": "ok", "model_id": model_id, "pinned": True}


@_router.post("/embedding/{model_id}/unpin")
async def unpin_embedding_model(
    model_id: str,
    role: str = Depends(verify_scoped_api_key),
):
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    entry = engine_pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if entry.model_type != "embedding":
        raise HTTPException(status_code=400, detail=f"Model {model_id} is not an embedding model")

    engine_pool.set_pinned(model_id, False)
    logger.info("model-manager: unpinned embedding %s (role=%s)", model_id, role)
    return {"status": "ok", "model_id": model_id, "pinned": False}


router = _router
