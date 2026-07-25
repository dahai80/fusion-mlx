# SPDX-License-Identifier: Apache-2.0
"""Garbage collection endpoint for post-compact KV cache release.

POST /api/v1/gc triggers Python gc.collect() + mx.metal.clear_cache()
to release stale KV cache after conversation compaction. Returns
memory-before/after/freed statistics.

Importers: fusion_mlx/server.py includes this router via include_router.
Data schemas: GCResult pydantic model (success, mem_before, mem_after,
  freed, cache_before, cache_after, error).
User instruction: "给fusion-mlx提pr" — add /api/v1/gc backend endpoint
  that fusion-code's requestMlxGC() already calls.
"""

import gc as _gc
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..admin.auth import require_admin
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["gc"],
    dependencies=[Depends(verify_api_key)],
)


class GCResult(BaseModel):
    success: bool = Field(description="Whether GC completed without error")
    mem_before: int | None = Field(
        default=None, description="MLX active memory in bytes before GC"
    )
    mem_after: int | None = Field(
        default=None, description="MLX active memory in bytes after GC"
    )
    freed: int | None = Field(
        default=None, description="Bytes freed (mem_before - mem_after)"
    )
    cache_before: int | None = Field(
        default=None, description="MLX cache memory in bytes before GC"
    )
    cache_after: int | None = Field(
        default=None, description="MLX cache memory in bytes after GC"
    )
    error: str | None = Field(default=None, description="Error message if GC failed")


def _get_mlx_stats() -> dict[str, int | None]:
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            return {
                "active": mx.get_active_memory(),
                "cache": mx.get_cache_memory(),
                "peak": mx.get_peak_memory(),
            }
    except Exception as e:
        logger.debug("mlx stats unavailable: %s", e)
    return {"active": None, "cache": None, "peak": None}


@router.post("/gc", response_model=GCResult)
async def run_gc(is_admin: bool = Depends(require_admin)):
    before = _get_mlx_stats()
    mem_before = before["active"]
    cache_before = before["cache"]

    try:
        _gc.collect()

        try:
            import mlx.core as mx

            if mx.metal.is_available():
                mx.metal.clear_cache()
                logger.info("gc: mx.metal.clear_cache() completed")
        except Exception as e:
            logger.warning("gc: mx.metal.clear_cache() failed: %s", e)

        _gc.collect()

        after = _get_mlx_stats()
        mem_after = after["active"]
        cache_after = after["cache"]

        freed = None
        if mem_before is not None and mem_after is not None:
            freed = max(0, mem_before - mem_after)

        logger.info(
            "gc: mem_before=%s mem_after=%s freed=%s cache_before=%s cache_after=%s",
            mem_before,
            mem_after,
            freed,
            cache_before,
            cache_after,
        )

        return GCResult(
            success=True,
            mem_before=mem_before,
            mem_after=mem_after,
            freed=freed,
            cache_before=cache_before,
            cache_after=cache_after,
        )
    except Exception as e:
        logger.error("gc: failed: %s", e, exc_info=True)
        after = _get_mlx_stats()
        return GCResult(
            success=False,
            mem_before=mem_before,
            mem_after=after["active"],
            freed=None,
            cache_before=cache_before,
            cache_after=after["cache"],
            error=str(e),
        )
