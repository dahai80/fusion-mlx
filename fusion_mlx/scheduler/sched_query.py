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

    Computes worst-case peak memory for the last prefill chunk
    (model weights + KV cache + SDPA attention matrix) and rejects
    if it would exceed the hard limit.

    For head_dim > 128, MLX SDPA uses a fallback that materializes
    the full attention matrix [B, n_q, chunk, kv_len] in float32.
    For head_dim <= 128, MLX uses a fused kernel with O(n) memory.

    Returns:
        Error message string if request should be rejected, None if OK.
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
        return None  # can't estimate, skip

    current = max(mx.get_active_memory(), get_phys_footprint())

    if current + peak > self._memory_hard_limit_bytes:
        from ..utils.hardware import format_bytes

        usage_gb = current / (1024**3)
        ceiling_gb = self._memory_hard_limit_bytes / (1024**3)
        return (
            f"Prefill would require ~{format_bytes(current + peak)} peak "
            f"(current {format_bytes(current)} + KV+SDPA {format_bytes(peak)}) "
            f"but ceiling is {format_bytes(self._memory_hard_limit_bytes)} "
            f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
            f"Reduce context length or lower memory_guard_tier."
        )
    return None
