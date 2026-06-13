# SPDX-License-Identifier: Apache-2.0
"""Admin panel routes for oMLX server configuration.

This module provides HTTP routes for the admin panel including:
- Login/logout with API key authentication
- Dashboard for server monitoring
- Model settings management (per-model sampling parameters, pinning, default)
- Global settings management
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import (
    require_admin,
)

logger = logging.getLogger(__name__)

PRESET_REMOTE_URL = "https://fusion_mlx.ai/assets/omlx_preset.json"



from .helpers import (
    _get_engine_pool,
)

_router = APIRouter()

# =============================================================================
# Benchmark API Routes (Throughput)
# =============================================================================


@_router.get("/api/bench/active")
async def get_active_benchmark(is_admin: bool = Depends(require_admin)):
    """Return the currently-running throughput benchmark, if any.

    Symmetric to `/api/bench/accuracy/queue/status` — lets a fresh page
    load or a second tab discover an in-flight run so it can attach to
    the SSE stream. Combined with the replay-on-subscribe stream this
    is what makes the multi-tab + page-refresh story actually work.
    """
    from .benchmark import get_active_run

    run = get_active_run()
    if run is None:
        return {"running": False, "bench_id": None, "model_id": None}
    return {
        "running": True,
        "bench_id": run.bench_id,
        "model_id": run.request.model_id,
    }


@_router.post("/api/bench/start")
async def start_benchmark(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    """Start a benchmark run.

    Validates the model, creates a benchmark run, and starts it
    as an asyncio background task. Rejects with 409 if another
    throughput bench is already running — two concurrent runs on
    the same engine produce mutually-corrupted measurements.
    """
    from .benchmark import (
        BenchmarkRequest,
        cleanup_old_runs,
        create_run,
        get_active_run,
        run_benchmark,
    )

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    # One throughput bench at a time. The replay-on-subscribe stream lets
    # clients attach to the already-running one if that's what they want.
    active = get_active_run()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A throughput benchmark is already running "
                f"(bench_id={active.bench_id}, model_id={active.request.model_id})."
            ),
        )

    body = await request.json()
    try:
        bench_request = BenchmarkRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate model exists and is an LLM
    entry = engine_pool.get_entry(bench_request.model_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Model not found: {bench_request.model_id}"
        )
    if entry.model_type not in ("llm", "vlm", None):
        raise HTTPException(
            status_code=400,
            detail=f"Model {bench_request.model_id} is not a supported model (type: {entry.model_type})",
        )

    # Cleanup old runs
    cleanup_old_runs()

    # Create and start the benchmark
    run = create_run(bench_request)
    total_tests = len(bench_request.prompt_lengths) + len(bench_request.batch_sizes) * 2

    run.task = asyncio.create_task(run_benchmark(run, engine_pool))

    logger.info(
        f"Benchmark started: {run.bench_id} model={bench_request.model_id} "
        f"tests={total_tests}"
    )

    return {
        "bench_id": run.bench_id,
        "status": "started",
        "total_tests": total_tests,
    }


@_router.get("/api/bench/{bench_id}/stream")
async def stream_benchmark(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Stream benchmark progress via Server-Sent Events."""
    import json

    from fastapi.responses import StreamingResponse

    from .benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {bench_id}")

    async def event_generator():
        # Replay-then-attach: see /api/bench/accuracy/{id}/stream for the
        # full rationale. The bench stream's terminal events are
        # `upload_done` and `error` — `done` only marks the boundary
        # between tests and upload.
        seen = 0
        try:
            while True:
                async with run.cond:
                    while seen >= len(run.events) and not run.terminal:
                        try:
                            await asyncio.wait_for(run.cond.wait(), timeout=60.0)
                        except TimeoutError:
                            break
                    new = list(run.events[seen:])
                    seen = len(run.events)
                    done = run.terminal

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


@_router.post("/api/bench/{bench_id}/cancel")
async def cancel_benchmark(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Cancel a running benchmark."""
    from .benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {bench_id}")

    if run.status != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark is not running (status: {run.status})",
        )

    if run.task and not run.task.done():
        run.task.cancel()

    return {"status": "cancelled", "bench_id": bench_id}


@_router.get("/api/bench/{bench_id}/results")
async def get_benchmark_results(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Get results from a completed benchmark."""
    from .benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {bench_id}")

    return {
        "bench_id": run.bench_id,
        "status": run.status,
        "results": run.results,
        "error": run.error_message if run.error_message else None,
        "upload_state": run.upload_state,
    }


@_router.get("/api/device-info")
async def get_device_info(
    is_admin: bool = Depends(require_admin),
):
    """Get device hardware info and owner_hash for fusion_mlx.ai integration."""
    from ..utils.hardware import (
        compute_owner_hash,
        get_chip_name,
        get_gpu_core_count,
        get_io_platform_uuid,
        get_total_memory_gb,
        parse_chip_info,
    )

    chip_string = get_chip_name()
    chip_name, chip_variant = parse_chip_info(chip_string)
    memory_gb = round(get_total_memory_gb())
    gpu_cores = get_gpu_core_count()

    owner_hash = None
    io_uuid = get_io_platform_uuid()
    if io_uuid:
        full_hash = compute_owner_hash(io_uuid, chip_name, gpu_cores, memory_gb)
        owner_hash = full_hash[:-1]  # Strip verify character for URL

    return {
        "chip_name": chip_name,
        "chip_variant": chip_variant,
        "memory_gb": memory_gb,
        "gpu_cores": gpu_cores,
        "owner_hash": owner_hash,
    }



router = _router
