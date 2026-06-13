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
# Accuracy Benchmark API Routes (MUST be before throughput {bench_id} routes)
# =============================================================================


@_router.post("/api/bench/accuracy/queue/add")
async def add_to_accuracy_queue(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    """Add a model to the accuracy benchmark queue and start if idle."""
    from .accuracy_benchmark import (
        AccuracyBenchmarkRequest,
        add_to_queue,
        get_queue_status,
        start_next_from_queue,
    )

    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")

    body = await request.json()
    try:
        bench_request = AccuracyBenchmarkRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

    add_to_queue(bench_request)

    logger.info(
        f"Accuracy queue: added {bench_request.model_id} "
        f"benchmarks={list(bench_request.benchmarks.keys())}"
    )

    # Start processing if not already running (synchronous — sets bench_id immediately)
    start_next_from_queue(engine_pool)

    return get_queue_status()


@_router.get("/api/bench/accuracy/queue/status")
async def get_accuracy_queue_status(
    is_admin: bool = Depends(require_admin),
):
    """Get accuracy benchmark queue status."""
    from .accuracy_benchmark import get_queue_status

    return get_queue_status()


@_router.delete("/api/bench/accuracy/queue/{idx}")
async def remove_from_accuracy_queue(
    idx: int,
    is_admin: bool = Depends(require_admin),
):
    """Remove an item from the accuracy benchmark queue."""
    from .accuracy_benchmark import get_queue_status, remove_from_queue

    if not remove_from_queue(idx):
        raise HTTPException(status_code=404, detail=f"Queue index {idx} not found")

    return get_queue_status()


@_router.get("/api/bench/accuracy/results")
async def get_accumulated_accuracy_results(
    is_admin: bool = Depends(require_admin),
):
    """Get all accumulated accuracy benchmark results."""
    from .accuracy_benchmark import get_accumulated_results, get_queue_status

    status = get_queue_status()
    return {
        "results": get_accumulated_results(),
        "running": status["running"],
        "current_model": status["current_model"],
        "current_bench_id": status["current_bench_id"],
    }


@_router.post("/api/bench/accuracy/results/reset")
async def reset_accuracy_results(
    is_admin: bool = Depends(require_admin),
):
    """Clear all accumulated accuracy benchmark results."""
    from .accuracy_benchmark import reset_accumulated_results

    reset_accumulated_results()
    return {"status": "reset"}


@_router.post("/api/bench/accuracy/cancel")
async def cancel_accuracy_queue(
    is_admin: bool = Depends(require_admin),
):
    """Cancel the current run and clear the queue."""
    from .accuracy_benchmark import cancel_queue

    await cancel_queue()
    return {"status": "cancelled"}


@_router.get("/api/bench/accuracy/{bench_id}/stream")
async def stream_accuracy_benchmark(
    bench_id: str,
    is_admin: bool = Depends(require_admin),
):
    """Stream accuracy benchmark progress via Server-Sent Events."""
    import json

    from fastapi.responses import StreamingResponse

    from .accuracy_benchmark import get_run

    run = get_run(bench_id)
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Accuracy benchmark not found: {bench_id}"
        )

    async def event_generator():
        # Replay-then-attach: every subscriber starts at offset 0 of the
        # run's event log and follows along live. Lets the HTML dashboard
        # recover its view on page refresh and lets multiple consumers
        # (e.g. browser + Swift app) share the same run.
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



router = _router
