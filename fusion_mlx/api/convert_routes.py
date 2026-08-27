# SPDX-License-Identifier: Apache-2.0
# Async job API for HF->MLX conversion + weight quantization. Reuses the
# `fusion-mlx convert` CLI pipeline (fusion_mlx.cli_convert) as the job body.
# Conversion is long-running + memory-heavy (loads a full model, writes a new
# artifact), so a synchronous endpoint is the wrong shape: jobs run on a
# single-worker thread pool - serialized to avoid OOM on one machine - and are
# polled via GET .../jobs/{job_id}. This implements issue #103's 0.4.8 design.

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..admin.auth import require_admin
from ..middleware import require_model_hub_source
from .convert_models import ConvertRequest, MergeAdapterRequest, QuantizeRequest

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
        _set(job, status="completed", progress=1.0, output_path=out)
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


# --- LoRA/DoRA adapter merge (#584) -------------------------------------
# Fuses trained adapters into the base model and persists the merged weights,
# mirroring `mlx_lm.fuse`. Unlike /convert, fusion-mlx's hub client
# (fusion-model-hub #22 _run_merge) calls this synchronously and expects the
# output_path in the response body, so we run on the same serialized
# single-worker _executor (OOM-safe alongside convert/quantize) and await via
# asyncio.wrap_future rather than spawning an async job. Small LoRA merges are
# fast; large bases are bounded by the hub's 300s client timeout.


def _run_merge_sync(
    base_model: str,
    adapter_path: str,
    output_path: str,
    dequantize: bool,
    upload_repo: str | None,
) -> str:
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.utils import dequantize_model, load, save, upload_to_hub

    logger.info(
        "merge-adapter: base=%s adapter=%s -> %s (dequantize=%s)",
        base_model,
        adapter_path,
        output_path,
        dequantize,
    )
    model, tokenizer, config = load(
        base_model, adapter_path=adapter_path, return_config=True
    )

    fused_linears = [
        (name, module.fuse(dequantize=dequantize))
        for name, module in model.named_modules()
        if hasattr(module, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
        logger.info("merge-adapter: fused %d LoRA/DoRA layers", len(fused_linears))
    else:
        logger.warning(
            "merge-adapter: no fuse() layers found in %s (adapter_path mismatch?)",
            base_model,
        )

    if dequantize:
        logger.info("merge-adapter: dequantizing merged model")
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)

    save_path = Path(output_path)
    save(save_path, base_model, model, tokenizer, config, donate_model=False)
    logger.info("merge-adapter: wrote fused model to %s", save_path)

    if upload_repo is not None:
        upload_to_hub(str(save_path), upload_repo)
        logger.info("merge-adapter: uploaded fused model to %s", upload_repo)

    _ = tree_flatten
    return str(save_path)


@router.post("/merge-adapter")
async def merge_adapter(
    request: MergeAdapterRequest,
    _is_admin: bool = Depends(require_admin),
    _source: bool = Depends(require_model_hub_source),
) -> dict[str, Any]:
    if _executor._broken:  # executor was shut down (e.g. during shutdown)
        raise HTTPException(503, detail="Convert/merge executor unavailable")
    from ..model_aliases import resolve_model

    base_model = resolve_model(request.model)
    save_path = request.output_path or f"{Path(request.model).name}-fused"
    logger.info(
        "merge-adapter job: model=%s resolved=%s adapter=%s",
        request.model,
        base_model,
        request.adapter_path,
    )
    try:
        future: Future = _executor.submit(
            _run_merge_sync,
            base_model,
            request.adapter_path,
            save_path,
            request.dequantize,
            request.upload_repo,
        )
        output_path = await asyncio.wrap_future(future)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("merge-adapter failed: model=%s", request.model)
        raise HTTPException(500, detail=f"merge-adapter failed: {exc}") from exc
    return {
        "status": "ok",
        "model": request.model,
        "output_path": output_path,
    }
