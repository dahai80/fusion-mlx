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
_grpo_service = None
_dpo_service = None

_router = APIRouter()


def set_fine_tune_context(pool, service=None):
    global _engine_pool_ref, _fine_tune_service
    _engine_pool_ref = pool
    _fine_tune_service = service
    if service is not None and pool is not None:
        service.set_engine_pool(pool)


def set_grpo_context(pool, service=None):
    global _grpo_service
    _grpo_service = service
    if service is not None and pool is not None:
        service.set_engine_pool(pool)


def set_dpo_context(pool, service=None):
    global _dpo_service
    _dpo_service = service
    if service is not None and pool is not None:
        service.set_engine_pool(pool)


def _get_grpo_service():
    global _grpo_service
    if _grpo_service is None:
        from fusion_mlx.training.grpo_service import GRPOService

        _grpo_service = GRPOService()
        if _engine_pool_ref is not None:
            _grpo_service.set_engine_pool(_engine_pool_ref)
    return _grpo_service


def _get_dpo_service():
    global _dpo_service
    if _dpo_service is None:
        from fusion_mlx.training.dpo_service import DPOService

        _dpo_service = DPOService()
        if _engine_pool_ref is not None:
            _dpo_service.set_engine_pool(_engine_pool_ref)
    return _dpo_service


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


@_router.get("/api/fine-tune/jobs/models")
async def list_finetunable_models_jobs_path(
    is_admin: bool = Depends(require_admin),
):
    # #397: fusion-trainer calls /admin/api/fine-tune/jobs/models to
    # enumerate trainable models. Register this STATIC path before the
    # parameterized /jobs/{job_id} route, else job_id=="models" shadows it.
    return await list_finetunable_models(is_admin=is_admin)


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
        raise HTTPException(
            status_code=404, detail=f"Job not found or not cancellable: {job_id}"
        )
    job = svc.get_job(job_id)
    return job.to_dict() if job else {"status": "cancelled"}


@_router.delete("/api/fine-tune/jobs/{job_id}")
async def delete_fine_tune_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_service()
    if not svc.delete_job(job_id):
        raise HTTPException(
            status_code=404, detail=f"Job not found or currently running: {job_id}"
        )
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
        raise HTTPException(
            status_code=400, detail="model_id and adapter_name required"
        )

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
            models.append(
                {
                    "model_id": model_id,
                    "model_type": entry.model_type,
                    "model_path": getattr(entry, "model_path", ""),
                    "loaded": entry.engine is not None,
                }
            )
    return models


# =============================================================================
# Logprob Scoring Endpoint (#363 Phase 1)
# =============================================================================


@_router.post("/api/fine-tune/logprob")
async def compute_logprob_endpoint(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    # Score sum log p(completion | prompt) under model_id, optionally with a
    # trained adapter. Loads standalone (separate from inference pool), scores,
    # evicts. Used by external RL trainers to get per-sample logprobs.
    body = await request.json()

    model_id = body.get("model_id", "")
    prompt = body.get("prompt", "")
    completion = body.get("completion", "")
    adapter_name = body.get("adapter_name", "")

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not completion:
        raise HTTPException(status_code=400, detail="completion is required")

    svc = _get_service()
    model_path = svc._resolve_model_path(model_id)
    if model_path is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    adapter_path = None
    if adapter_name:
        from fusion_mlx.training.service import ADAPTER_BASE_DIR

        adapter_path = str(ADAPTER_BASE_DIR / model_id / adapter_name)
        import os

        if not os.path.isdir(adapter_path):
            raise HTTPException(
                status_code=404,
                detail=f"Adapter not found: {model_id}/{adapter_name}",
            )

    from fusion_mlx.training.logprob import score_text

    logger.info("logprob endpoint: model=%s adapter=%s", model_path, adapter_path)
    try:
        result = await asyncio.to_thread(
            score_text, model_path, prompt, completion, adapter_path
        )
    except Exception as e:
        logger.exception("logprob scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    return result.to_dict()


# =============================================================================
# GRPO Training Endpoints (#363 Phase 2)
# =============================================================================


@_router.post("/api/fine-tune/grpo/jobs")
async def create_grpo_job(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    # Create a GRPO (reinforcement learning) training job. Body:
    # {model_id, prompts: [str], adapter_name?, config?: GRPOConfig}.
    body = await request.json()

    model_id = body.get("model_id", "")
    prompts = body.get("prompts", [])
    adapter_name = body.get("adapter_name", "")

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not prompts or not isinstance(prompts, list):
        raise HTTPException(status_code=400, detail="prompts (non-empty list) required")

    from fusion_mlx.training.grpo import GRPOConfig

    config_body = body.get("config", {})
    try:
        config = GRPOConfig(**config_body)
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

    svc = _get_grpo_service()
    job = svc.create_job(
        model_id=model_id,
        prompts=prompts,
        config=config,
        adapter_name=adapter_name,
    )
    svc.start_processing()
    return job.to_dict()


@_router.get("/api/fine-tune/grpo/jobs")
async def list_grpo_jobs(
    is_admin: bool = Depends(require_admin),
):
    svc = _get_grpo_service()
    return [job.to_dict() for job in svc.list_jobs()]


@_router.get("/api/fine-tune/grpo/jobs/{job_id}")
async def get_grpo_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_grpo_service()
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@_router.post("/api/fine-tune/grpo/jobs/{job_id}/cancel")
async def cancel_grpo_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_grpo_service()
    if not svc.cancel_job(job_id):
        raise HTTPException(
            status_code=404, detail=f"Job not found or not cancellable: {job_id}"
        )
    job = svc.get_job(job_id)
    return job.to_dict() if job else {"status": "cancelled"}


@_router.delete("/api/fine-tune/grpo/jobs/{job_id}")
async def delete_grpo_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_grpo_service()
    if not svc.delete_job(job_id):
        raise HTTPException(
            status_code=404, detail=f"Job not found or currently running: {job_id}"
        )
    return {"status": "deleted"}


@_router.get("/api/fine-tune/grpo/jobs/{job_id}/stream")
async def stream_grpo_progress(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_grpo_service()
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
# DPO / ORPO preference-alignment training (#399)
# =============================================================================


def _create_pref_job(request_body: dict):
    # Shared create path for /dpo/jobs and /orpo/jobs. Body:
    # {model_id, preference_pairs: [{prompt, chosen, rejected}], adapter_name?, config?}.
    # config.method is forced to match the endpoint.
    model_id = request_body.get("model_id", "")
    pairs = request_body.get("preference_pairs", [])
    adapter_name = request_body.get("adapter_name", "")

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not pairs or not isinstance(pairs, list):
        raise HTTPException(
            status_code=400, detail="preference_pairs (non-empty list) required"
        )
    for idx, p in enumerate(pairs):
        if not isinstance(p, dict) or not all(
            k in p for k in ("prompt", "chosen", "rejected")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"preference_pairs[{idx}] must have prompt/chosen/rejected",
            )

    from fusion_mlx.training.dpo import DPOConfig

    config_body = dict(request_body.get("config", {}))
    try:
        config = DPOConfig(**config_body)
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

    svc = _get_dpo_service()
    job = svc.create_job(
        model_id=model_id,
        preference_pairs=pairs,
        config=config,
        adapter_name=adapter_name,
    )
    svc.start_processing()
    return job.to_dict()


@_router.post("/api/fine-tune/dpo/jobs")
async def create_dpo_job(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    body = await request.json()
    body.setdefault("config", {})["method"] = "dpo"
    return _create_pref_job(body)


@_router.post("/api/fine-tune/orpo/jobs")
async def create_orpo_job(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    body = await request.json()
    body.setdefault("config", {})["method"] = "orpo"
    return _create_pref_job(body)


@_router.get("/api/fine-tune/dpo/jobs")
async def list_dpo_jobs(
    is_admin: bool = Depends(require_admin),
):
    svc = _get_dpo_service()
    return [job.to_dict() for job in svc.list_jobs()]


@_router.get("/api/fine-tune/dpo/jobs/{job_id}")
async def get_dpo_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_dpo_service()
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@_router.post("/api/fine-tune/dpo/jobs/{job_id}/cancel")
async def cancel_dpo_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_dpo_service()
    if not svc.cancel_job(job_id):
        raise HTTPException(
            status_code=404, detail=f"Job not found or not cancellable: {job_id}"
        )
    job = svc.get_job(job_id)
    return job.to_dict() if job else {"status": "cancelled"}


@_router.delete("/api/fine-tune/dpo/jobs/{job_id}")
async def delete_dpo_job(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_dpo_service()
    if not svc.delete_job(job_id):
        raise HTTPException(
            status_code=404, detail=f"Job not found or currently running: {job_id}"
        )
    return {"status": "deleted"}


@_router.get("/api/fine-tune/dpo/jobs/{job_id}/stream")
async def stream_dpo_progress(
    job_id: str,
    is_admin: bool = Depends(require_admin),
):
    svc = _get_dpo_service()
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


router = _router
