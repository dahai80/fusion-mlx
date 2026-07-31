# SPDX-License-Identifier: Apache-2.0
"""Admin routes for LoRA/DORA fine-tuning.

Importers/callers:
  - fusion_mlx.server imports router + set_fine_tune_context for startup wiring
  - fusion_mlx.admin.helpers (no direct import, uses _get_engine_pool)

Affected API:
  POST   /admin/api/fine-tune/jobs              — create training job
  GET    /admin/api/fine-tune/jobs              — list all jobs
  GET    /admin/api/fine-tune/jobs/{id}         — get job details
  POST   /admin/api/fine-tune/jobs/{id}/cancel  — cancel job
  DELETE /admin/api/fine-tune/jobs/{id}         — delete job record
  GET    /admin/api/fine-tune/jobs/{id}/stream   — SSE progress stream
  GET    /admin/api/fine-tune/adapters           — list saved adapters
  DELETE /admin/api/fine-tune/adapters           — delete adapter
  POST   /admin/api/fine-tune/adapters/{model_id}/{adapter_name}/serve  — serve adapter via EnginePool
  POST   /admin/api/fine-tune/adapters/{model_id}/{adapter_name}/unload — unload adapter engine
  GET    /admin/api/fine-tune/models             — list fine-tunable models

Data schemas: FineTuneConfig, FineTuneProgress, FineTuneJob (from fusion_mlx.training.service)

User verbatim instruction: "开始做，注意设计方案需要有GUI的设计和落地方案，提交给macos app，可以先提pr，晚点在梳理macos app都还需要哪些GUI落地"
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from .auth import require_admin
from .helpers import _get_engine_pool

logger = logging.getLogger(__name__)

_fine_tune_service = None
_engine_pool_ref = None

_router = APIRouter()


def set_fine_tune_context(pool, service=None):
    global _engine_pool_ref, _fine_tune_service
    _engine_pool_ref = pool
    _fine_tune_service = service
    if service is not None and pool is not None:
        service.set_engine_pool(pool)


def _get_service():
    global _fine_tune_service
    if _fine_tune_service is None:
        from fusion_mlx.training.service import FineTuneService
        _fine_tune_service = FineTuneService()
        if _engine_pool_ref is not None:
            _fine_tune_service.set_engine_pool(_engine_pool_ref)
    return _fine_tune_service


# =============================================================================
# Job CRUD
# =============================================================================


@_router.post("/api/fine-tune/jobs")
async def create_fine_tune_job(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    body = await request.json()

    model_id = body.get("model_id", "")
    dataset = body.get("dataset", "")
    adapter_name = body.get("adapter_name", "")

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not dataset:
        raise HTTPException(status_code=400, detail="dataset is required")

    from fusion_mlx.training.service import FineTuneConfig

    config_body = body.get("config", {})
    try:
        config = FineTuneConfig(**config_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    pool = _get_engine_pool()
    if pool is not None:
        entry = pool.get_entry(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        if entry.model_type not in ("llm", "vlm", None):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} is not a text model (type: {entry.model_type})",
            )

    job = svc.create_job(
        model_id=model_id,
        dataset=dataset,
        config=config,
        adapter_name=adapter_name,
    )

    svc.start_processing()

    return job.to_dict()


@_router.get("/api/fine-tune/jobs")
async def list_fine_tune_jobs(
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    return [job.to_dict() for job in svc.list_jobs()]


@_router.get("/api/fine-tune/jobs/{job_id}")
async def get_fine_tune_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@_router.post("/api/fine-tune/jobs/{job_id}/cancel")
async def cancel_fine_tune_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    if not svc.cancel_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job not found or not cancellable: {job_id}")
    job = svc.get_job(job_id)
    return job.to_dict() if job else {"status": "cancelled"}


@_router.delete("/api/fine-tune/jobs/{job_id}")
async def delete_fine_tune_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    if not svc.delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job not found or currently running: {job_id}")
    return {"status": "deleted"}


# =============================================================================
# SSE Progress Stream
# =============================================================================


@_router.get("/api/fine-tune/jobs/{job_id}/stream")
async def stream_fine_tune_progress(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_generator():
        seen = 0
        try:
            while True:
                async with job.cond:
                    while seen >= len(job.events) and not job.terminal:
                        try:
                            await asyncio.wait_for(job.cond.wait(), timeout=60.0)
                        except TimeoutError:
                            break
                    new = list(job.events[seen:])
                    seen = len(job.events)
                    done = job.terminal

                for ev in new:
                    yield f"data: {json.dumps(ev)}\n\n"
                if not new and not done:
                    yield ": keepalive\n\n"
                if done:
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Adapter Management
# =============================================================================


@_router.get("/api/fine-tune/adapters")
async def list_adapters(
    model_id: str = "",
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    return svc.list_adapters(model_id=model_id or None)


@_router.delete("/api/fine-tune/adapters")
async def delete_adapter(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    body = await request.json()
    model_id = body.get("model_id", "")
    adapter_name = body.get("adapter_name", "")
    if not model_id or not adapter_name:
        raise HTTPException(status_code=400, detail="model_id and adapter_name required")

    svc = _get_service()
    if not svc.delete_adapter(model_id, adapter_name):
        raise HTTPException(status_code=404, detail="Adapter not found")
    return {"status": "deleted"}


# =============================================================================
# Adapter Serving (hot-swap via EnginePool)
# =============================================================================


@_router.post("/api/fine-tune/adapters/{model_id}/{adapter_name}/serve")
async def serve_adapter(
    model_id: str,
    adapter_name: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    try:
        result = await svc.serve_adapter(model_id, adapter_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@_router.post("/api/fine-tune/adapters/{model_id}/{adapter_name}/unload")
async def unload_adapter(
    model_id: str,
    adapter_name: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    ok = await svc.unload_adapter_engine(model_id, adapter_name)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Adapter engine not found or not loaded: {model_id}/{adapter_name}",
        )
    return {"status": "unloaded"}


# =============================================================================
# Fine-Tunable Models
# =============================================================================


@_router.get("/api/fine-tune/models")
async def list_finetunable_models(
    is_admin: bool = Depends(require_admin),
):
    pool = _get_engine_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    models = []
    for model_id, entry in pool._entries.items():
        if entry.model_type in ("llm", "vlm", None):
            models.append({
                "model_id": model_id,
                "model_type": entry.model_type,
                "model_path": getattr(entry, "model_path", ""),
                "loaded": entry.engine is not None,
            })
    return models


router = _router
