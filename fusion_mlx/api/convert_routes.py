# SPDX-License-Identifier: Apache-2.0
# Async job API for HF->MLX conversion + weight quantization. Reuses the
# `fusion-mlx convert` CLI pipeline (fusion_mlx.cli_convert) as the job body.
# Conversion is long-running + memory-heavy (loads a full model, writes a new
# artifact), so a synchronous endpoint is the wrong shape: jobs run on a
# single-worker thread pool - serialized to avoid OOM on one machine - and are
# polled via GET .../jobs/{job_id}. This implements issue #103's 0.4.8 design.

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, HTTPException, Depends

from .convert_models import ConvertRequest, QuantizeRequest
from ..admin.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["convert"])

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Single-worker pool: a conversion loads a full model into memory, so serialize
# jobs to avoid OOM. A queued job waits for the prior one to finish.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="convert-job")

_FP_QUANT_MODES = ("mxfp4", "nvfp4", "mxfp8")


def _now() -> float:
    return time.time()


def _new_job(kind: str, model: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:16]
    now = _now()
    return {
        "job_id": job_id,
        "kind": kind,
        "model": model,
        "status": "queued",
        "progress": 0.0,
        "output_path": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }


def _set(job: dict[str, Any], **fields: Any) -> None:
    with _jobs_lock:
        job.update(fields)
        job["updated_at"] = _now()


def _run_job(job: dict[str, Any], req: ConvertRequest | QuantizeRequest) -> None:
    # Reuse the convert CLI pipeline: alias resolution -> kwargs build -> run.
    from fusion_mlx.cli_convert import _build_convert_kwargs, _run_convert
    from fusion_mlx.model_aliases import resolve_model

    try:
        model = resolve_model(req.model)
        args_ns = SimpleNamespace(
            out=req.output_path,
            quant_bits=req.quant_bits,
            quant_mode=req.quant_mode,
            quant_group_size=req.quant_group_size,
            dtype=req.dtype,
            upload_repo=req.upload_repo,
            dequantize=getattr(req, "dequantize", False),
            trust_remote_code=req.trust_remote_code,
        )
        kwargs = _build_convert_kwargs(args_ns, model)
        _set(job, status="running", progress=0.1)
        logger.info(
            "convert job %s running: model=%s -> %s (quantize=%s bits=%s mode=%s)",
            job["job_id"],
            model,
            kwargs["mlx_path"],
            kwargs["quantize"],
            kwargs["q_bits"],
            kwargs["q_mode"],
        )
        # mlx-lm's convert() exposes no stable progress callback, so progress is
        # coarse: 0.1 (running) -> 1.0 (done|failed). Do not fake intermediate values.
        out = _run_convert(model, **kwargs)
        _set(job, status="done", progress=1.0, output_path=out)
        logger.info("convert job %s done: output=%s", job["job_id"], out)
    except Exception as exc:
        _set(job, status="failed", progress=1.0, error=str(exc))
        logger.exception("convert job %s failed", job["job_id"])


def _submit(kind: str, req: ConvertRequest | QuantizeRequest) -> dict[str, Any]:
    job = _new_job(kind, req.model)
    with _jobs_lock:
        _jobs[job["job_id"]] = job
    logger.info("%s job %s queued: model=%s", kind, job["job_id"], req.model)
    _executor.submit(_run_job, job, req)
    return {"job_id": job["job_id"], "status": "queued"}


def _list_jobs(kind: str) -> list[dict[str, Any]]:
    with _jobs_lock:
        items = [dict(j) for j in _jobs.values() if j["kind"] == kind]
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def _get_job(job_id: str, kind: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["kind"] != kind:
            raise HTTPException(404, detail=f"Job '{job_id}' not found")
        return dict(job)


@router.post("/convert")
async def start_convert(
    request: ConvertRequest,
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    return _submit("convert", request)


@router.post("/quantize")
async def start_quantize(
    request: QuantizeRequest,
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    if request.quant_bits is None and request.quant_mode not in _FP_QUANT_MODES:
        raise HTTPException(
            400,
            detail="/v1/quantize requires quant_bits or a float quant_mode "
            "(mxfp4/nvfp4/mxfp8)",
        )
    return _submit("quantize", request)


@router.get("/convert/jobs")
async def list_convert_jobs(
    _is_admin: bool = Depends(require_admin),
) -> list[dict[str, Any]]:
    return _list_jobs("convert")


@router.get("/convert/jobs/{job_id}")
async def get_convert_job(
    job_id: str,
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    return _get_job(job_id, "convert")


@router.get("/quantize/jobs")
async def list_quantize_jobs(
    _is_admin: bool = Depends(require_admin),
) -> list[dict[str, Any]]:
    return _list_jobs("quantize")


@router.get("/quantize/jobs/{job_id}")
async def get_quantize_job(
    job_id: str,
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    return _get_job(job_id, "quantize")
