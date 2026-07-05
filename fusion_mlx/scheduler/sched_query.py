# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for oMLX continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import logging

logger = logging.getLogger(__name__)

import mlx.core as mx

from ..request import Request
from ..utils.proc_memory import get_phys_footprint

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .helpers import (
    _sync_and_clear_cache,
)
from .monkeypatches import _unregister_uid_row, _unregister_uid_rows_for_model
from .types import _PreflightRejection


def has_requests(self) -> bool:
    """Check if there are any pending or running requests.

    Also returns True when a deferred Metal cache clear is pending,
    so that the engine loop keeps calling step() until the clear fires.
    Without this, an idle server would never reach the target step and
    stale buffers would accumulate indefinitely.
    """
    return bool(
        self.waiting
        or self.prefilling
        or self.running
        or self._deferred_clear_at is not None
    )


def fail_all_requests(self) -> list[str]:
    """Remove all running and waiting requests after unrecoverable error.

    Used as a safety net by engine_core when step() raises an
    unexpected exception, to prevent infinite loops.

    Only resets batch_generator (not full cache) because this method
    is called for non-corruption errors — corruption is already
    handled inside step().

    Returns:
        List of failed request IDs.
    """
    failed_ids: list[str] = []
    for request_id in list(self.running):
        failed_ids.append(request_id)
        req = self.requests.pop(request_id, None)
        if req is not None:
            req._extracted_cache = None
            req.prompt_cache = None
    self.running.clear()
    for request in list(self.prefilling):
        failed_ids.append(request.request_id)
        req = self.requests.pop(request.request_id, None)
        if req is not None:
            req._extracted_cache = None
            req.prompt_cache = None
    self.prefilling.clear()
    self._prefill_states.clear()
    for request in list(self.waiting):
        failed_ids.append(request.request_id)
        req = self.requests.pop(request.request_id, None)
        if req is not None:
            req._extracted_cache = None
            req.prompt_cache = None
    self.waiting.clear()
    # Catch in-flight orphans: a request popped from self.waiting but
    # not yet added to self.running (or self.prefilling) sits as a
    # local in _schedule_waiting. If _do_external_prefill raises, the
    # request is unreachable through the three queues but still lives
    # in self.requests (and the engine_core collector / finished_event
    # for its id is still waiting). Without this pass, fail_all_requests
    # returns an incomplete list and the HTTP request hangs forever.
    #
    # Exclude finished requests still awaiting async cache-store cleanup
    # (those have an entry in ``_inflight_store_futures`` — see
    # ``_cleanup_finished`` line ~5267). They have already emitted a
    # ``finished=True`` output to their collector; ``_drain_pending_async_removes``
    # pops them from ``self.requests`` after the store future completes.
    # Failing them here would append an error output that wins over the
    # success for non-streaming ``generate()`` callers (engine_core
    # returns the last queued output).
    for request_id in list(self.requests):
        if request_id in self._inflight_store_futures:
            continue
        failed_ids.append(request_id)
        req = self.requests.pop(request_id, None)
        if req is not None:
            req._extracted_cache = None
            req.prompt_cache = None
    # Clear stale uid mappings for every failed id. Running requests hold
    # real uids; the in-flight orphan above holds the temp_uid assigned at
    # _schedule_waiting (id(request)) that its success-path cleanup never
    # reached. batch_generator is reset below, so these mappings are dead
    # either way. failed_ids excludes _inflight_store_futures ids, so the
    # async-cleanup uids that _drain_pending_async_removes still needs are
    # left intact.
    for rid in failed_ids:
        uid = self.request_id_to_uid.pop(rid, None)
        if uid is not None:
            _unregister_uid_row(self.model, uid)
            self.uid_to_request_id.pop(uid, None)
    # Reset batch generator only (cache is not corrupted)
    _unregister_uid_rows_for_model(self.model)
    self.batch_generator = None
    self._current_sampler_params = None
    # Reclaim fragmented Metal buffers after generation failure.
    # Without this, subsequent requests may hit the same resource
    # limit even though Python references have been cleared.
    # Wrapped in try-except because Metal may already be in an error
    # state — mx.synchronize() or mx.clear_cache() can throw a C++
    # exception that causes SIGABRT if uncaught (#435).
    try:
        _sync_and_clear_cache(self._stream)
    except Exception as e:
        logger.warning(f"Metal cache clear failed during error recovery: {e}")
    return failed_ids


def get_num_waiting(self) -> int:
    """Get number of waiting requests."""
    return len(self.waiting)


def get_num_running(self) -> int:
    """Get number of running requests."""
    return len(self.running)


def _hot_cache_cpu_bytes(self) -> int:
    """Return serialized hot-cache bytes safe to exclude from phys guard."""
    config = getattr(self, "config", None)
    budget = getattr(config, "hot_cache_budget", None)
    if budget is not None:
        try:
            return max(0, int(getattr(budget, "total_bytes", 0)))
        except Exception:
            logger.debug("Failed to read shared hot-cache byte budget")
            return 0

    manager = getattr(self, "paged_ssd_cache_manager", None)
    if manager is None:
        return 0

    try:
        stats = manager.get_stats()
        return max(0, int(getattr(stats, "hot_cache_size_bytes", 0)))
    except Exception:
        try:
            return max(0, int(getattr(manager, "_hot_cache_total_bytes", 0)))
        except Exception:
            logger.debug("Failed to read local hot-cache byte counter")
            return 0


def _current_usage_bytes(self, *, refresh_mlx_active: bool = True) -> int:
    """Current memory usage for scheduler-side guard checks.

    Scheduler steps run on the MLX executor thread, so they can refresh
    mx.get_active_memory() safely.  Event-loop callers such as early
    preflight use the cached executor sample and phys_footprint instead.
    """
    active = self._last_mlx_active_memory_bytes
    if refresh_mlx_active:
        active = max(0, int(mx.get_active_memory()))
        self._last_mlx_active_memory_bytes = active
    hot_cache_bytes = _hot_cache_cpu_bytes(self)
    phys = max(0, int(get_phys_footprint()) - hot_cache_bytes)
    return max(active, phys)


def _memory_component_limit_for_rejection(self, component_limit: int) -> int:
    """Apply hot-cache headroom reservation to a component ceiling."""
    if component_limit <= 0:
        return 0
    hot_reserved = max(
        0, int(getattr(self, "_memory_hot_cache_reserved_bytes", 0) or 0)
    )
    if hot_reserved <= 0:
        return component_limit
    return max(1, component_limit - hot_reserved)


def _preflight_abort_description(self) -> tuple[int, int, float]:
    """Return ``(base_cap, effective_cap, margin)`` for the prefill abort path.

    The abort cap is the stable physical limit used to kill an in-flight
    prefill that would overshoot Metal's wired ceiling.  It is derived
    from ``_memory_abort_limit_bytes`` (set by the enforcer) or falls
    back to ``_memory_hard_limit_bytes`` when the enforcer has not yet
    propagated.
    """
    base_cap = getattr(self, "_memory_abort_limit_bytes", 0)
    if base_cap <= 0:
        base_cap = self._memory_hard_limit_bytes
    margin = getattr(self, "_prefill_abort_margin", 0.90)
    effective = int(base_cap * margin)
    return (int(base_cap), effective, margin)


def _format_rejection_message(
    self,
    *,
    estimated: int,
    current: int,
    peak: int,
    hard_limit: int,
) -> str:
    """Build the prefill-rejection diagnostic.

    Identifies which of static / dynamic / metal_cap is binding so the
    message can steer the user to the right remedy (close apps for
    dynamic, raise sysctl for metal_cap, raise tier or reduce context
    for static).
    """
    from ..utils.hardware import format_bytes

    static = getattr(self, "_memory_static_ceiling_bytes", 0)
    dynamic = getattr(self, "_memory_dynamic_ceiling_bytes", 0)
    metal_cap = getattr(self, "_memory_metal_cap_bytes", 0)

    binding: list[str] = []
    if static and _memory_component_limit_for_rejection(self, static) == hard_limit:
        binding.append("static")
    if dynamic and _memory_component_limit_for_rejection(self, dynamic) == hard_limit:
        binding.append("dynamic")
    if (
        metal_cap
        and _memory_component_limit_for_rejection(self, metal_cap) == hard_limit
    ):
        binding.append("metal_cap")
    binding_str = "/".join(binding) if binding else "effective"

    is_custom = getattr(self, "_memory_guard_tier", "") == "custom"
    if "dynamic" in binding and is_custom:
        advice = (
            f"raise custom_ceiling_bytes in admin Memory settings "
            f"(currently pinned at {format_bytes(dynamic)}), "
            f"or reduce context length"
        )
    elif "dynamic" in binding and static and static > dynamic:
        headroom = max(0, dynamic - current)
        advice = (
            f"close other apps to free RAM "
            f"(static cap is {format_bytes(static)} but only "
            f"{format_bytes(headroom)} is reclaimable right now), "
            f"raise memory_guard_tier (safe → balanced → aggressive), "
            f"or reduce context length"
        )
    elif "metal_cap" in binding:
        advice = (
            f"raise kernel iogpu.wired_limit_mb in Terminal "
            f"(currently caps Metal at {format_bytes(metal_cap)}), "
            f"or reduce context length"
        )
    else:
        advice = (
            "reduce context length or raise memory_guard_tier "
            "(safe → balanced → aggressive)"
        )
    advice = advice[:1].upper() + advice[1:]

    return (
        f"Prefill would require ~{format_bytes(estimated)} peak "
        f"(current {format_bytes(current)} + KV+SDPA {format_bytes(peak)}) "
        f"but {binding_str} ceiling is {format_bytes(hard_limit)}. "
        f"{advice}."
    )


def _preflight_safety_rejection(
    self,
    *,
    num_prompt_tokens: int,
    cached_tokens: int = 0,
    current_usage_bytes: int,
) -> _PreflightRejection | None:
    """Predict whether even the safety floor chunk cannot fit.

    This mirrors the mid-prefill ``_guard_prefill_chunk`` rejection, but
    runs before the route returns a ``StreamingResponse``.  It charges the
    resident KV that will be allocated by the prompt plus the minimum
    chunk transient at the full prompt context length.
    """
    if self.memory_monitor is None:
        return None
    base_cap, cap, margin = _preflight_abort_description(self)
    if cap <= 0:
        return None

    new_tokens = max(int(num_prompt_tokens) - max(int(cached_tokens), 0), 0)
    if new_tokens == 0:
        return None

    floor_chunk = min(max(1, self._prefill_min_chunk_tokens), new_tokens)
    kv_len = max(int(num_prompt_tokens) - 1, 1)
    new_kv, _cached_kv = self.memory_monitor.estimate_prompt_kv_bytes(
        new_tokens, cached_tokens
    )
    min_transient = self.memory_monitor._predicted_chunk_transient(floor_chunk, kv_len)
    if new_kv <= 0 and min_transient <= 0:
        return None

    estimated = int(current_usage_bytes + new_kv + min_transient)
    if estimated <= cap:
        return None

    from ..utils.hardware import format_bytes

    message = (
        "Prefill context too large for available memory "
        f"(preflight safety guard, kv_len={kv_len}, "
        f"min_chunk={floor_chunk}): predicted peak would require "
        f"~{format_bytes(estimated)} "
        f"(current {format_bytes(current_usage_bytes)} + "
        f"KV {format_bytes(new_kv)} + "
        f"min-chunk transient {format_bytes(min_transient)}) "
        f"but prefill safety cap is {format_bytes(cap)} "
        f"({round(margin * 100)}% of effective ceiling "
        f"{format_bytes(base_cap)}). Reduce context length, free system "
        "memory, or loosen memory_guard_tier (safe → balanced → aggressive)."
    )
    return _PreflightRejection(
        message=message,
        estimated_bytes=estimated,
        limit_bytes=int(cap),
    )


def _preflight_memory_check(self, request: "Request") -> _PreflightRejection | None:
    """Estimate whether prefill would exceed memory limits.

    Computes worst-case peak memory for the last prefill chunk
    (model weights + KV cache + SDPA activation/scratch) and rejects
    if it would exceed the hard limit.

    Mirrors MLX SDPA dispatch closely enough that unsupported prefill
    head dimensions are charged for the unfused fp32 score matrix.

    Returns:
        ``_PreflightRejection`` carrying the message + numeric
        estimated / limit bytes if the request should be rejected,
        otherwise ``None``.
    """
    if not self._prefill_memory_guard:
        return None
    if self._memory_hard_limit_bytes <= 0:
        return None
    if self.memory_monitor is None:
        return None

    prompt_tokens = request.num_prompt_tokens
    cached_tokens = request.cached_tokens or 0
    new_tokens = max(prompt_tokens - cached_tokens, 0)

    if new_tokens == 0:
        return None

    peak = self.memory_monitor.estimate_prefill_peak_bytes(
        new_tokens, self.config.prefill_step_size, cached_tokens=cached_tokens
    )
    if peak == 0:
        return None

    current = _current_usage_bytes(self)
    estimated = current + peak
    hard_limit = self._memory_hard_limit_bytes

    if estimated > hard_limit:
        message = _format_rejection_message(
            self,
            estimated=estimated,
            current=current,
            peak=peak,
            hard_limit=hard_limit,
        )
        return _PreflightRejection(
            message=message,
            estimated_bytes=int(estimated),
            limit_bytes=int(hard_limit),
        )

    safety_rejection = _preflight_safety_rejection(
        self,
        num_prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        current_usage_bytes=current,
    )
    if safety_rejection is not None:
        logger.warning(
            "Preflight safety-cap rejected request %s: %s",
            request.request_id,
            safety_rejection.message,
        )
        return safety_rejection

    return None


def _estimate_prefill_peak(self, new_tokens: int) -> int:
    """Estimate worst-case peak memory for a prefill chunk.

    Delegates to memory_monitor when available; falls back to inline
    estimation when the monitor has not been initialised.

    Returns 0 if model info is unavailable or new_tokens is 0.
    """
    if new_tokens <= 0:
        return 0

    if self.memory_monitor is not None:
        peak = self.memory_monitor.estimate_prefill_peak_bytes(
            new_tokens, self.config.prefill_step_size
        )
        if peak > 0:
            return peak

    model = getattr(self, "model", None)
    if model is None:
        return 0

    config = getattr(model, "config", None) or getattr(model, "args", None)
    if config is None:
        return 0

    num_layers = getattr(config, "num_hidden_layers", None) or getattr(
        config, "num_layers", None
    )
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    num_query_heads = getattr(config, "num_attention_heads", None) or num_kv_heads
    head_dim = getattr(config, "head_dim", None)

    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )
        if hidden_size and num_query_heads:
            head_dim = hidden_size // num_query_heads

    if not all([num_layers, num_kv_heads, head_dim]):
        return 0

    dtype_bytes = 2
    try:
        first_weight = None
        for child in model.children():
            if hasattr(child, "weight"):
                first_weight = child.weight
                break
        if first_weight is not None:
            dtype_map = {
                mx.float16: 2,
                mx.bfloat16: 2,
                mx.float32: 4,
                mx.int8: 1,
            }
            dtype_bytes = dtype_map.get(first_weight.dtype, 2)
    except Exception:
        pass

    chunk = min(new_tokens, self.config.prefill_step_size)
    kv_bytes = 2 * num_layers * chunk * num_kv_heads * head_dim * dtype_bytes

    sdpa_bytes = 0
    if head_dim > 128:
        sdpa_bytes = num_query_heads * chunk * chunk * 4

    return kv_bytes + sdpa_bytes
