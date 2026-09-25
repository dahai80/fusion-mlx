# SPDX-License-Identifier: Apache-2.0
"""Per-model startup memory plan.

Borrowed from splash's offline ``EngineMemoryBreakdown`` concept: a
precomputed view of the KV token budget available to a loaded model,
used to (a) cap request context length on admission (refuse over-context
with HTTP 413 instead of OOM-crashing mid-decode) and (b) surface the
budget to operators via /status.

Unlike splash (native C++ precomputed once at startup), this is computed
lazily in the scheduler on first preflight (when memory_monitor model
geometry + the propagated hard limit are both available) and cached.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelMemoryPlan:
    weights_bytes: int = 0
    kv_bytes_per_token: int = 0
    hard_limit_bytes: int = 0
    kv_token_ceiling: int = 0
    max_context_per_request: int = 0
    deficit_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _format_bytes(n: int) -> str:
    if n <= 0:
        return "0"
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f}{units[i]}"


def compute_model_memory_plan(scheduler) -> ModelMemoryPlan | None:
    mm = getattr(scheduler, "memory_monitor", None)
    if mm is None or not mm.has_model_info():
        return None
    hard_limit = getattr(scheduler, "_memory_hard_limit_bytes", 0) or 0
    if hard_limit <= 0:
        return None

    kv_bpt = _kv_bytes_per_token(mm)
    if kv_bpt <= 0:
        return None

    current_usage = _current_usage_bytes(scheduler)
    weights_bytes = getattr(scheduler, "_memory_plan_weights_bytes", 0) or 0
    available_for_kv = max(0, hard_limit - current_usage)

    kv_token_ceiling = available_for_kv // kv_bpt if kv_bpt else 0
    model_max_context = _model_max_context(scheduler)
    max_context_per_request = (
        min(model_max_context, kv_token_ceiling)
        if model_max_context
        else kv_token_ceiling
    )

    minimum_required = weights_bytes + kv_bpt
    deficit_bytes = max(0, minimum_required - hard_limit)

    plan = ModelMemoryPlan(
        weights_bytes=int(weights_bytes),
        kv_bytes_per_token=int(kv_bpt),
        hard_limit_bytes=int(hard_limit),
        kv_token_ceiling=int(kv_token_ceiling),
        max_context_per_request=int(max_context_per_request),
        deficit_bytes=int(deficit_bytes),
    )
    logger.info(
        "[memory-plan] kv_bytes_per_token=%s hard_limit=%s current=%s "
        "kv_token_ceiling=%d max_context_per_request=%d deficit=%s",
        _format_bytes(kv_bpt),
        _format_bytes(hard_limit),
        _format_bytes(current_usage),
        kv_token_ceiling,
        max_context_per_request,
        _format_bytes(deficit_bytes),
    )
    return plan


def _kv_bytes_per_token(mm) -> int:
    try:
        return int(mm.estimate_decode_kv_bytes(1))
    except Exception:
        return 0


def _current_usage_bytes(scheduler) -> int:
    try:
        from fusion_mlx.scheduler.sched_query import _current_usage_bytes as _impl

        return int(_impl(scheduler))
    except Exception:
        return 0


def _model_max_context(scheduler) -> int:
    cfg = None
    model = getattr(scheduler, "model", None)
    if model is not None:
        cfg = getattr(model, "config", None) or getattr(model, "args", None)
    if cfg is None:
        return 0
    for sub_attr in ("text_config", "language_config", "llm_config"):
        sub = getattr(cfg, sub_attr, None)
        if sub is not None and getattr(sub, "max_position_embeddings", None):
            cfg = sub
            break
    return int(getattr(cfg, "max_position_embeddings", 0) or 0)
