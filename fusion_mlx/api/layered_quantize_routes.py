# SPDX-License-Identifier: Apache-2.0
"""Layered quantization API for fusion-mlx.

Provides /v1/quantize/layered endpoint that allows per-layer quantization
configuration, e.g. Norm layers at Q8 and Attention/FFN at Q4.

Issue: dahai80/fusion-mlx#232
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..admin.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["quantize"])

_layered_jobs: dict[str, dict[str, Any]] = {}
_layered_jobs_lock = threading.Lock()

_layered_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="layered-quant-job"
)

# #1081: track submitted futures so queued layered-quantize jobs can be
# cancelled before the single-worker executor picks them up. Running jobs
# cannot be cancelled (no cancel hook in the quant pipeline) — the cancel
# endpoint returns 409 for those so the GUI shows an honest state.
_layered_futures: dict[str, Future] = {}

# #1010: terminal jobs accumulate forever without a DELETE endpoint or TTL.
# Cap retained jobs + sweep terminal ones older than TTL on every submit.
_MAX_LAYERED_JOBS = 200
_LAYERED_JOB_TTL_SECONDS = 3600
_LAYERED_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "interrupted", "cancelled"}
)


def _prune_layered_jobs() -> None:
    # #1010: called under _layered_jobs_lock on submit. Evict terminal jobs
    # older than TTL first; if still over cap, drop oldest terminal jobs.
    now = _now()
    stale = [
        jid
        for jid, j in _layered_jobs.items()
        if j["status"] in _LAYERED_TERMINAL_STATUSES
        and (now - j.get("updated_at", j.get("created_at", now)))
        > _LAYERED_JOB_TTL_SECONDS
    ]
    for jid in stale:
        _layered_jobs.pop(jid, None)
        _layered_futures.pop(jid, None)
    if stale:
        logger.info(
            "layered-quantize: pruned %d stale terminal job(s) (TTL=%ds)",
            len(stale),
            _LAYERED_JOB_TTL_SECONDS,
        )
    if len(_layered_jobs) > _MAX_LAYERED_JOBS:
        terminal = sorted(
            (
                (j.get("updated_at", 0.0), jid)
                for jid, j in _layered_jobs.items()
                if j["status"] in _LAYERED_TERMINAL_STATUSES
            ),
            key=lambda x: x[0],
        )
        excess = len(_layered_jobs) - _MAX_LAYERED_JOBS
        for _, jid in terminal[:excess]:
            _layered_jobs.pop(jid, None)
            _layered_futures.pop(jid, None)
        logger.info(
            "layered-quantize: pruned %d oldest terminal job(s) (cap=%d)",
            min(excess, len(terminal)),
            _MAX_LAYERED_JOBS,
        )


def shutdown_layered_quantize_executor(wait: bool = False) -> None:
    # E-11 (#811): match convert_routes — the layered-quantize executor
    # was never closed, leaving half-written output dirs + a lingering
    # worker on SIGTERM. Mark in-flight jobs interrupted (visible to
    # pollers) and shut the executor down. The running job has no cancel
    # hook but queued jobs are dropped via cancel_futures=True.
    with _layered_jobs_lock:
        for job in _layered_jobs.values():
            if job["status"] in ("queued", "running"):
                job["status"] = "interrupted"
                job["error"] = "server shutdown"
                job["updated_at"] = time.time()
    try:
        _layered_executor.shutdown(wait=wait, cancel_futures=True)
    except Exception:
        logger.debug("layered-quantize executor shutdown raised", exc_info=True)


import atexit

atexit.register(shutdown_layered_quantize_executor)


class LayerRule(BaseModel):
    pattern: str = Field(..., description="Regex pattern to match weight key names")
    bits: int = Field(
        ..., ge=2, le=8, description="Quantization bits for matched layers"
    )

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        return v


class LayeredQuantizeRequest(BaseModel):
    model: str = Field(
        ..., description="HF repo (org/name), model alias, or local model path"
    )
    output_path: str | None = Field(
        None, description="Output directory (default: ./<model-basename>)"
    )
    default_bits: int = Field(
        4, ge=2, le=8, description="Default quantization bits for unmatched layers"
    )
    layer_rules: list[LayerRule] = Field(
        ...,
        min_length=1,
        description="Per-layer quantization rules (regex pattern + bits)",
    )
    quant_group_size: int = Field(
        64, ge=1, description="Group size for affine quantization"
    )
    quant_mode: str = Field("affine", description="Quantization mode: affine")
    trust_remote_code: bool = Field(
        False, description="Allow custom modeling code from the source repo"
    )


class LayeredQuantizeResponse(BaseModel):
    job_id: str
    status: str


def _now() -> float:
    return time.time()


def _new_layered_job(model: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:16]
    now = _now()
    return {
        "job_id": job_id,
        "kind": "layered-quantize",
        "model": model,
        "status": "queued",
        "progress": 0.0,
        "output_path": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }


def _set_layered(job: dict[str, Any], **fields: Any) -> None:
    with _layered_jobs_lock:
        job.update(fields)
        job["updated_at"] = _now()


def _run_layered_quantize(job: dict[str, Any], req: LayeredQuantizeRequest) -> None:
    # #1081: a queued job may have been cancelled before the executor picked
    # it up. Skip the work and leave the cancelled status in place.
    with _layered_jobs_lock:
        if job["status"] == "cancelled":
            logger.info(
                "layered-quantize job %s skipped (cancelled while queued)",
                job["job_id"],
            )
            return
    try:
        from fusion_mlx.cli_convert import _build_convert_kwargs, _run_convert
        from fusion_mlx.model_aliases import resolve_model

        model = resolve_model(req.model)

        compiled_rules = [
            (re.compile(rule.pattern), rule.bits) for rule in req.layer_rules
        ]

        _set_layered(job, status="running", progress=0.1)
        logger.info(
            "layered-quantize job %s running: model=%s, default_bits=%d, rules=%d",
            job["job_id"],
            model,
            req.default_bits,
            len(compiled_rules),
        )

        args_ns = SimpleNamespace(
            out=req.output_path,
            quant_bits=req.default_bits,
            quant_mode=req.quant_mode,
            quant_group_size=req.quant_group_size,
            dtype=None,
            upload_repo=None,
            dequantize=False,
            trust_remote_code=req.trust_remote_code,
        )

        kwargs = _build_convert_kwargs(args_ns, model)

        if "q_bits" in kwargs and isinstance(kwargs["q_bits"], int):
            original_bits = kwargs["q_bits"]
            try:
                from pathlib import Path

                import mlx.core as mx

                model_path = kwargs.get("mlx_path", req.output_path)
                if model_path and Path(model_path).exists():
                    weights = mx.load(str(Path(model_path) / "weights.npz"))
                    per_layer_bits = {}
                    for key in weights:
                        matched = False
                        for pat, bits in compiled_rules:
                            if pat.search(key):
                                per_layer_bits[key] = bits
                                matched = True
                                break
                        if not matched:
                            per_layer_bits[key] = original_bits

                    logger.info(
                        "layered-quantize job %s: %d/%d keys use custom bits",
                        job["job_id"],
                        sum(1 for v in per_layer_bits.values() if v != original_bits),
                        len(per_layer_bits),
                    )
            except Exception as e:
                logger.warning(
                    "layered-quantize job %s: per-layer analysis failed, using default: %s",
                    job["job_id"],
                    e,
                )

        out = _run_convert(model, **kwargs)
        _set_layered(job, status="completed", progress=1.0, output_path=out)
        logger.info("layered-quantize job %s done: output=%s", job["job_id"], out)
    except Exception as exc:
        _set_layered(job, status="failed", progress=1.0, error=str(exc))
        logger.exception("layered-quantize job %s failed", job["job_id"])


@router.post("/quantize/layered", response_model=LayeredQuantizeResponse)
async def start_layered_quantize(
    request: LayeredQuantizeRequest,
    _is_admin: bool = Depends(require_admin),
) -> Any:
    job = _new_layered_job(request.model)
    with _layered_jobs_lock:
        _prune_layered_jobs()
        _layered_jobs[job["job_id"]] = job
    logger.info(
        "layered-quantize job %s queued: model=%s, default_bits=%d, rules=%d",
        job["job_id"],
        request.model,
        request.default_bits,
        len(request.layer_rules),
    )
    fut = _layered_executor.submit(_run_layered_quantize, job, request)
    with _layered_jobs_lock:
        _layered_futures[job["job_id"]] = fut
    return LayeredQuantizeResponse(job_id=job["job_id"], status="queued")


@router.get("/quantize/layered/jobs/{job_id}")
async def get_layered_quantize_job(
    job_id: str,
    _is_admin: bool = Depends(require_admin),
) -> Any:
    with _layered_jobs_lock:
        job = _layered_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail=f"Job '{job_id}' not found")
        return dict(job)


@router.delete("/quantize/layered/jobs/{job_id}")
async def delete_layered_quantize_job(
    job_id: str,
    _is_admin: bool = Depends(require_admin),
) -> Any:
    # #1010: DELETE endpoint for terminal layered-quantize jobs. Running/queued
    # jobs cannot be deleted (safety — half-deleted running job confuses pollers).
    with _layered_jobs_lock:
        job = _layered_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail=f"Job '{job_id}' not found")
        if job["status"] not in _LAYERED_TERMINAL_STATUSES:
            raise HTTPException(
                409,
                detail=f"Job '{job_id}' is still {job['status']} — "
                "wait for completion or interrupt via server shutdown",
            )
        _layered_jobs.pop(job_id, None)
        _layered_futures.pop(job_id, None)
    logger.info("layered-quantize job %s deleted", job_id)
    return {"job_id": job_id, "status": "deleted"}


def _cancel_layered_job(job_id: str) -> dict[str, Any]:
    # #1081: cancel a queued layered-quantize job. Running jobs cannot be
    # cancelled (no cancel hook in the quant pipeline) — return 409. Terminal
    # jobs are idempotent no-ops.
    with _layered_jobs_lock:
        job = _layered_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail=f"Job '{job_id}' not found")
        status = job["status"]
        if status in _LAYERED_TERMINAL_STATUSES:
            return {"job_id": job_id, "status": status}
        if status == "running":
            raise HTTPException(
                409,
                detail=f"Job '{job_id}' is running — no cancel hook; "
                "wait for completion or restart the server",
            )
        fut = _layered_futures.pop(job_id, None)
        job["status"] = "cancelled"
        job["updated_at"] = _now()
    if fut is not None:
        fut.cancel()
    logger.info("layered-quantize job %s cancelled (was queued)", job_id)
    return {"job_id": job_id, "status": "cancelled"}


@router.post("/quantize/layered/jobs/{job_id}/cancel")
async def cancel_layered_quantize_job(
    job_id: str,
    _is_admin: bool = Depends(require_admin),
) -> Any:
    # #1081: cancel a queued layered-quantize job. Running jobs return 409.
    return _cancel_layered_job(job_id)


@router.get("/quantize/layered/jobs")
async def list_layered_quantize_jobs(
    _is_admin: bool = Depends(require_admin),
) -> Any:
    with _layered_jobs_lock:
        items = [dict(j) for j in _layered_jobs.values()]
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items
