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
from concurrent.futures import ThreadPoolExecutor
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


class LayerRule(BaseModel):
    pattern: str = Field(
        ..., description="Regex pattern to match weight key names"
    )
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
        ..., min_length=1, description="Per-layer quantization rules (regex pattern + bits)"
    )
    quant_group_size: int = Field(
        64, ge=1, description="Group size for affine quantization"
    )
    quant_mode: str = Field(
        "affine", description="Quantization mode: affine"
    )
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


def _run_layered_quantize(
    job: dict[str, Any], req: LayeredQuantizeRequest
) -> None:
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
                import mlx.core as mx
                from pathlib import Path

                model_path = kwargs.get("mlx_path", req.output_path)
                if model_path and Path(model_path).exists():
                    weights = mx.load(str(Path(model_path) / "weights.npz"))
                    per_layer_bits = {}
                    for key in weights.keys():
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
        _set_layered(job, status="done", progress=1.0, output_path=out)
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
        _layered_jobs[job["job_id"]] = job
    logger.info(
        "layered-quantize job %s queued: model=%s, default_bits=%d, rules=%d",
        job["job_id"],
        request.model,
        request.default_bits,
        len(request.layer_rules),
    )
    _layered_executor.submit(_run_layered_quantize, job, request)
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


@router.get("/quantize/layered/jobs")
async def list_layered_quantize_jobs(
    _is_admin: bool = Depends(require_admin),
) -> Any:
    with _layered_jobs_lock:
        items = [dict(j) for j in _layered_jobs.values()]
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items
