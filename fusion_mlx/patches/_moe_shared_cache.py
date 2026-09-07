# Copyright © 2026 Apple Inc.
#
# #803 MLA/DSA dedicated path — DSA shared-expert activation cache.
#
# DeepSeek-V3.2/GLM-MoE-DSA and DeepSeek-V4 MoE layers run a shared-expert
# MLP on every token every layer. The shared expert is token-determined (not
# routed), so within a single layer call two identical input rows yield an
# identical activation — recomputing the second is pure waste. This module
# dedups shared-expert activations by exact input-row content and reuses the
# cached activation for repeats, logging a per-layer hit rate.
#
# Scope is deliberately intra-call (one layer, one forward). Cross-layer reuse
# is INVALID: each layer's shared_experts has its own weights, so the same
# input produces a different activation at a different layer. Cross-forward
# reuse is also invalid (hidden state evolves). The win is bounded to batches
# that contain duplicate input rows (e.g. repeated prompt structure, padded
# duplicates); hit rate is logged so the real benefit is observable, not
# assumed.
#
# Correctness: keys are exact row content (tuple of the row's floats), so a
# cache hit returns the bit-identical activation a fresh compute would have
# produced — no output drift. Cost is paid only when opted in (env-gated,
# default OFF); a size guard skips the dedup bookkeeping for huge prefills so
# the instrumentation path stays usable for decode-sized batches.
#
# Headless-testable: the dedup decision and hit/miss accounting are
# deterministic code; no model weights required. Real-model memory/throughput
# delta vs the generic path is deferred (needs DeepSeek weights on disk).

import logging
import os
import threading
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENV_FLAG = "FUSION_MOE_SHARED_CACHE"
# Skip dedup bookkeeping above this many input elements (B*L*H). Huge prefills
# pay key-construction cost on every row with near-zero duplicate hits, so we
# fall through to a plain compute and record no stats for that call. Decode and
# small batches — where duplicates actually occur — stay instrumented.
_SIZE_GUARD = 1_000_000

_lock = threading.Lock()
# layer_id -> {requests: int, hits: int, misses: int}
_stats: dict[int, dict[str, int]] = {}
_enabled: bool | None = None
_next_layer_id = 0


def shared_cache_enabled() -> bool:
    global _enabled
    if _enabled is None:
        val = os.environ.get(_ENV_FLAG, "").strip().lower()
        _enabled = val in ("1", "true", "yes", "on")
        if _enabled:
            logger.info("#803 MoE shared-expert activation cache enabled")
    return _enabled


def reset_stats() -> None:
    with _lock:
        _stats.clear()


def get_stats() -> dict[int, dict[str, int]]:
    with _lock:
        return {k: dict(v) for k, v in _stats.items()}


def get_stats_flat() -> dict[str, int | float]:
    with _lock:
        total_req = sum(s["requests"] for s in _stats.values())
        total_hits = sum(s["hits"] for s in _stats.values())
        total_miss = sum(s["misses"] for s in _stats.values())
        rate = (total_hits / total_req) if total_req else 0.0
        return {
            "moe_shared_cache_requests": total_req,
            "moe_shared_cache_hits": total_hits,
            "moe_shared_cache_misses": total_miss,
            "moe_shared_cache_hit_rate": round(rate, 4),
            "moe_shared_cache_layers_tracked": len(_stats),
        }


def _bump(layer_id: int, hits: int, misses: int) -> None:
    with _lock:
        s = _stats.setdefault(layer_id, {"requests": 0, "hits": 0, "misses": 0})
        s["requests"] += hits + misses
        s["hits"] += hits
        s["misses"] += misses


def _next_id() -> int:
    global _next_layer_id
    with _lock:
        lid = _next_layer_id
        _next_layer_id += 1
        return lid


class SharedExpertActivationCache:
    # Per-layer, per-forward dedup of shared-expert activations keyed by exact
    # input-row content. Built once per MoE module (layer_idx fixed at init);
    # the per-forward state (_keys/_acts/_hit/_miss) is reset at the top of
    # every get_or_compute call so no state leaks across forwards.

    __slots__ = ("layer_id", "_keys", "_acts", "_hit", "_miss")

    def __init__(self, layer_idx: int | None = None):
        self.layer_id = _next_id() if layer_idx is None else layer_idx
        self._keys: dict[Any, int] = {}
        self._acts: list[mx.array] = []
        self._hit = 0
        self._miss = 0

    @staticmethod
    def _row_key(row: mx.array) -> Any:
        # Exact, collision-free key: the row's flat float values as a tuple.
        # Equal rows → equal tuples → reuse; different rows → different tuples
        # → no false hit. tolist() round-trips the array's values to Python
        # floats injectively for bf16/fp16/fp32, so no drift.
        try:
            return tuple(row.reshape(-1).tolist())
        except Exception:
            return None

    def get_or_compute(self, x: mx.array, compute_fn: Any) -> mx.array:
        # x: [B, L, H] shared-expert input. compute_fn(x) -> [B, L, H] act.
        # Reuses cached activations for repeated input rows, records hit/miss.
        # Only batches with B>1 can contain duplicate rows, so B<=1 (decode)
        # is a plain pass-through with no instrumentation — zero overhead and
        # no meaningless 0%-hit noise in the stats.
        B = x.shape[0]
        if B <= 1 or x.size > _SIZE_GUARD:
            return compute_fn(x)

        # Reset per-forward state for this layer.
        self._keys.clear()
        self._acts.clear()
        self._hit = 0
        self._miss = 0

        # Batch path: dedup unique rows, compute misses once, gather back.
        row_keys = [self._row_key(x[b]) for b in range(B)]
        unique: list[Any] = []
        uniq_idx: dict[Any, int] = {}
        first_occurrence: list[int] = []  # row index of each unique key's first
        plan: list[int] = []
        for b, k in enumerate(row_keys):
            if k is None:
                plan.append(-1)
                continue
            if k in uniq_idx:
                self._hit += 1
                plan.append(uniq_idx[k])
            else:
                self._miss += 1
                uniq_idx[k] = len(unique)
                unique.append(k)
                first_occurrence.append(b)
                plan.append(uniq_idx[k])

        acts: list[mx.array] = []
        if unique:
            miss_idx = mx.array(first_occurrence, dtype=mx.int32)
            x_miss = x[miss_idx]
            miss_act = compute_fn(x_miss)
            for i in range(len(unique)):
                acts.append(miss_act[i : i + 1])

        out = mx.stack(
            [
                acts[p] if p >= 0 else compute_fn(x[b : b + 1])
                for b, p in enumerate(plan)
            ]
        )
        self._bump_and_reset()
        return out.reshape(x.shape).astype(x.dtype)

    def _bump_and_reset(self) -> None:
        _bump(self.layer_id, self._hit, self._miss)
        self._keys.clear()
        self._acts.clear()


def make_layer_cache(
    layer_idx: int | None = None,
) -> SharedExpertActivationCache | None:
    if not shared_cache_enabled():
        return None
    return SharedExpertActivationCache(layer_idx)


__all__ = [
    "SharedExpertActivationCache",
    "shared_cache_enabled",
    "make_layer_cache",
    "reset_stats",
    "get_stats",
    "get_stats_flat",
]
