# SPDX-License-Identifier: Apache-2.0
"""R-6 (#0910 audit): GET /v1/runtime-config — read-only config snapshot.

Single-maintainer Beta had no way to introspect the *effective* runtime
config (profile-resolved + env-overridden + settings.json-merged) without
reading logs or settings.json and mentally replaying the resolution.
This endpoint returns the live snapshot for self-serve ops triage:
profile, disabled_modules, scheduler, memory tiers, spec_decode, cache
limits, concurrency budget. Read-only — mutation lives in
``/v1/config/reload``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends

from ..middleware.auth import verify_api_key_or_x_api_key

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key_or_x_api_key)])


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


@router.get("/v1/runtime-config")
async def runtime_config() -> dict[str, Any]:
    logger.info("GET /v1/runtime-config snapshot requested")
    try:
        from ..config import get_config

        cfg = get_config()
    except Exception:
        logger.exception("runtime-config: get_config() failed")
        return {"error": "config unavailable"}

    sched = _safe_get(cfg, "scheduler")
    mem = _safe_get(cfg, "memory")
    snapshot: dict[str, Any] = {
        "profile": _safe_get(cfg, "profile"),
        "disabled_modules": _safe_get(cfg, "disabled_modules", []),
        "scheduler": {
            "max_num_seqs": _safe_get(sched, "max_num_seqs"),
            "max_num_batched_tokens": _safe_get(sched, "max_num_batched_tokens"),
            "max_concurrent_requests": _safe_get(sched, "max_concurrent_requests"),
            "max_waiting": _safe_get(sched, "max_waiting"),
            "policy": _safe_get(sched, "policy"),
            "spec_decode": _safe_get(sched, "spec_decode"),
        },
        "memory": {
            "tier": _safe_get(mem, "tier"),
            "gpu_memory_utilization": _safe_get(mem, "gpu_memory_utilization"),
            "max_cpu_memory_mb": _safe_get(mem, "max_cpu_memory_mb"),
        },
        "cache": {
            "prefix_cache_max_blocks": _safe_get(
                _safe_get(cfg, "cache"), "prefix_cache_max_blocks"
            ),
            "response_cache_enabled": _safe_get(
                _safe_get(cfg, "response_cache"), "enabled"
            ),
            # D2.1: tiered cache coordinator (hot->cold demotion). ON by
            # default; FUSION_MLX_TIERED_CACHE=0 disables.
            "tiered_cache_enabled": os.environ.get("FUSION_MLX_TIERED_CACHE", "1")
            .strip()
            .lower()
            not in ("0", "false", "off"),
        },
        "env_overrides": {
            "FUSION_MAX_CONCURRENT_REQUESTS": os.environ.get(
                "FUSION_MAX_CONCURRENT_REQUESTS"
            ),
            "FUSION_MLX_KV_CHECKPOINT_INTERVAL": os.environ.get(
                "FUSION_MLX_KV_CHECKPOINT_INTERVAL"
            ),
            "FUSION_MLX_TIERED_CACHE": os.environ.get("FUSION_MLX_TIERED_CACHE"),
            "HF_MIRROR": os.environ.get("HF_MIRROR"),
        },
    }
    try:
        from ..api._concurrency import _max_concurrent, _request_sem

        snapshot["concurrency_semaphore"] = {
            "max": _max_concurrent,
            "initialized": _request_sem is not None,
        }
    except Exception:
        logger.debug("semaphore state unavailable", exc_info=True)
    return snapshot
