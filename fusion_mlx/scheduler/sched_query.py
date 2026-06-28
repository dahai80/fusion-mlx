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


def has_requests(self) -> bool:
    """Check if there are any pending or running requests.

    Also returns True when a deferred Metal cache clear is pending,
    so that the engine loop keeps calling step() until the clear fires.
    Without this, an idle server would never reach the target step and
    stale buffers would accumulate indefinitely.
    """
    return bool(self.waiting or self.prefilling or self.running or self._deferred_clear_at is not None)

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
            self.uid_to_request_id.pop(uid, None)
    # Reset batch generator only (cache is not corrupted)
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

def _preflight_memory_check(    self, request: "Request") -> str | None:
    """
    Estimate whether prefill would exceed memory limits.

    Computes worst-case peak memory for:
    - KV cache for the last prefill chunk (K + V)
    - SDPA temporary attention matrix (for head_dim > 128)
    - Existing running requests' KV cache growth
    - 5% safety margin for Metal implicit allocations

    Rejects if the sum would exceed the hard limit.

    Returns:
        Error message string if request should be rejected, None if OK.
    """
    if not self._prefill_memory_guard:
        return None
    if self._memory_hard_limit_bytes <= 0:
        return None

    prompt_tokens = request.num_prompt_tokens
    cached_tokens = request.cached_tokens or 0
    new_tokens = max(prompt_tokens - cached_tokens, 0)

    if new_tokens == 0:
        return None

    # Extract model architecture params for memory estimation
    model = getattr(self, "model", None)
    if model is None:
        return None  # no model, can't estimate

    config = getattr(model, "config", None) or getattr(model, "args", None)
    if config is None:
        return None  # no config, can't estimate

    num_layers = getattr(config, "num_hidden_layers", None) or getattr(
        config, "num_layers", None
    )
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    num_query_heads = getattr(config, "num_attention_heads", None) or num_kv_heads
    head_dim = getattr(config, "head_dim", None)

    # Derive head_dim from hidden_size if not explicit
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )
        if hidden_size and num_query_heads:
            head_dim = hidden_size // num_query_heads

    if not all([num_layers, num_kv_heads, head_dim]):
        return None  # can't estimate without key params

    # Determine dtype size from model weights
    dtype_bytes = 2  # float16 default
    try:
        first_weight = None
        for child in model.children():
            if hasattr(child, "weight"):
                first_weight = child.weight
                break
        if first_weight is not None:
            dtype_map = {
                mx.float16: 2, mx.bfloat16: 2,
                mx.float32: 4, mx.int8: 1,
            }
            dtype_bytes = dtype_map.get(first_weight.dtype, 2)
    except Exception:
        pass

    chunk = min(new_tokens, self.config.prefill_step_size)

    # KV cache for the prefill chunk: 2 (K+V) * layers * chunk * kv_heads * head_dim
    kv_bytes = 2 * num_layers * chunk * num_kv_heads * head_dim * dtype_bytes

    # SDPA temp matrix: only for head_dim > 128
    sdpa_bytes = 0
    if head_dim > 128:
        kv_len = cached_tokens + chunk
        sdpa_bytes = num_query_heads * chunk * kv_len * 4  # float32

    # Estimate running requests' KV cache memory (decode growth headroom)
    running_kv_bytes = 0
    for r in self.running:
        req = self.requests.get(r)
        if req is None:
            continue
        # Count tokens already in KV cache for this request
        req_tokens = getattr(req, "_total_tokens_generated", 0) or 0
        if req_tokens == 0:
            # Fallback: estimate from prompt length (conservative)
            req_tokens = request.num_prompt_tokens // 4
        running_kv_bytes += (
            2 * num_layers * req_tokens * num_kv_heads * head_dim * dtype_bytes
        )

    prefill_peak = kv_bytes + sdpa_bytes
    safety_margin = int(prefill_peak * 0.05)  # 5% for Metal implicit allocs
    total_estimated = prefill_peak + safety_margin

    current = max(mx.get_active_memory(), get_phys_footprint())

    if current + total_estimated > self._memory_hard_limit_bytes:
        from ..utils.hardware import format_bytes

        usage_gb = current / (1024**3)
        ceiling_gb = self._memory_hard_limit_bytes / (1024**3)
        return (
            f"Prefill would require ~{format_bytes(current + total_estimated)} peak "
            f"(current {format_bytes(current)} + KV+SDPA {format_bytes(prefill_peak)} "
            f"+ safety {format_bytes(safety_margin)}) "
            f"but ceiling is {format_bytes(self._memory_hard_limit_bytes)} "
            f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
            f"Reduce context length or lower memory_guard_tier."
        )
    return None


def _estimate_prefill_peak(self, new_tokens: int) -> int:
    """Estimate worst-case peak memory for a prefill chunk.

    Returns 0 if model info is unavailable or new_tokens is 0.
    """
    if new_tokens <= 0:
        return 0

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

    # Determine dtype size
    dtype_bytes = 2
    try:
        first_weight = None
        for child in model.children():
            if hasattr(child, "weight"):
                first_weight = child.weight
                break
        if first_weight is not None:
            dtype_map = {
                mx.float16: 2, mx.bfloat16: 2,
                mx.float32: 4, mx.int8: 1,
            }
            dtype_bytes = dtype_map.get(first_weight.dtype, 2)
    except Exception:
        pass

    chunk = min(new_tokens, self.config.prefill_step_size)

    # KV cache: 2 (K+V) * layers * chunk * kv_heads * head_dim * dtype
    kv_bytes = 2 * num_layers * chunk * num_kv_heads * head_dim * dtype_bytes

    # SDPA temp matrix: only for head_dim > 128
    sdpa_bytes = 0
    if head_dim > 128:
        sdpa_bytes = num_query_heads * chunk * chunk * 4  # float32, conservative kv_len ~ chunk

    return kv_bytes + sdpa_bytes
