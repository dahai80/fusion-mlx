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

import concurrent.futures
import copy
import gc
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator,
    GenerationBatch,
    PromptProcessingBatch,
    SequenceStateMachine,
    generation_stream,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors

from .cache.observability import CacheRateTracker
from .cache.paged_cache import PagedCacheManager
from .cache.prefix_cache import BlockAwarePrefixCache
from .exceptions import is_cache_corruption_error
from .prefill_progress import get_prefill_tracker
from .prefill_transient_tracker import PrefillTransientTracker
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .speculative.vlm_mtp import VLMMTPDrafter, run_vlm_mtp_decode
from .utils.proc_memory import get_phys_footprint
from .utils.sampling import make_sampler as omlx_make_sampler

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.

from .types import (
     _VLMMTPDecodeState, _VLMMTPResponse, _mx_buffer_access_lock,
     _StoreCacheGate, _PrefillAbortedError, _PrefillState,
     _BoundarySnapshotProvider,
)
from .helpers import (
     _sync_and_clear_cache, _safe_sync_stream,
     _prompt_cache_needs_snapshots, _cache_layer_token_count, _cache_base_sizes,
     _vlm_extra_seq_slice, _slice_vlm_extra, _advance_vlm_extra,
     _KNOWN_SLICEABLE_CACHE_TYPES,
)
from .monkeypatches import _default_generation_stream

def has_requests(self) -> bool:
    """Check if there are any pending or running requests.

    Also returns True when a deferred Metal cache clear is pending,
    so that the engine loop keeps calling step() until the clear fires.
    Without this, an idle server would never reach the target step and
    stale buffers would accumulate indefinitely.
    """
    return bool(sched.waiting or sched.prefilling or sched.running or sched._deferred_clear_at is not None)

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
    for request_id in list(sched.running):
        failed_ids.append(request_id)
        req = sched.requests.pop(request_id, None)
        if req is not None:
            req._extracted_cache = None
            req.prompt_cache = None
    sched.running.clear()
    for request in list(sched.prefilling):
        failed_ids.append(request.request_id)
        req = sched.requests.pop(request.request_id, None)
        if req is not None:
            req._extracted_cache = None
            req.prompt_cache = None
    sched.prefilling.clear()
    sched._prefill_states.clear()
    for request in list(sched.waiting):
        failed_ids.append(request.request_id)
        req = sched.requests.pop(request.request_id, None)
        if req is not None:
            req._extracted_cache = None
            req.prompt_cache = None
    sched.waiting.clear()
    # Catch in-flight orphans: a request popped from sched.waiting but
    # not yet added to sched.running (or sched.prefilling) sits as a
    # local in _schedule_waiting. If _do_external_prefill raises, the
    # request is unreachable through the three queues but still lives
    # in sched.requests (and the engine_core collector / finished_event
    # for its id is still waiting). Without this pass, fail_all_requests
    # returns an incomplete list and the HTTP request hangs forever.
    #
    # Exclude finished requests still awaiting async cache-store cleanup
    # (those have an entry in ``_inflight_store_futures`` — see
    # ``_cleanup_finished`` line ~5267). They have already emitted a
    # ``finished=True`` output to their collector; ``_drain_pending_async_removes``
    # pops them from ``sched.requests`` after the store future completes.
    # Failing them here would append an error output that wins over the
    # success for non-streaming ``generate()`` callers (engine_core
    # returns the last queued output).
    for request_id in list(sched.requests):
        if request_id in sched._inflight_store_futures:
            continue
        failed_ids.append(request_id)
        req = sched.requests.pop(request_id, None)
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
        uid = sched.request_id_to_uid.pop(rid, None)
        if uid is not None:
            sched.uid_to_request_id.pop(uid, None)
    # Reset batch generator only (cache is not corrupted)
    sched.batch_generator = None
    sched._current_sampler_params = None
    # Reclaim fragmented Metal buffers after generation failure.
    # Without this, subsequent requests may hit the same resource
    # limit even though Python references have been cleared.
    # Wrapped in try-except because Metal may already be in an error
    # state — mx.synchronize() or mx.clear_cache() can throw a C++
    # exception that causes SIGABRT if uncaught (#435).
    try:
        _sync_and_clear_cache(sched._stream)
    except Exception as e:
        logger.warning(f"Metal cache clear failed during error recovery: {e}")
    return failed_ids

def get_num_waiting(self) -> int:
    """Get number of waiting requests."""
    return len(sched.waiting)

def get_num_running(self) -> int:
    """Get number of running requests."""
    return len(sched.running)

def _preflight_memory_check(sched, request: "Request") -> str | None:
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
    if not sched._prefill_memory_guard:
        return None
    if sched._memory_hard_limit_bytes <= 0:
        return None
    if sched.memory_monitor is None:
        return None

    prompt_tokens = request.num_prompt_tokens
    cached_tokens = request.cached_tokens or 0
    new_tokens = max(prompt_tokens - cached_tokens, 0)

    if new_tokens == 0:
        return None

    peak = sched.memory_monitor.estimate_prefill_peak_bytes(
        new_tokens, sched.config.prefill_step_size, cached_tokens=cached_tokens
    )
    if peak == 0:
        return None  # can't estimate, skip

    current = max(mx.get_active_memory(), get_phys_footprint())

    if current + peak > sched._memory_hard_limit_bytes:
        from ..utils.hardware import format_bytes

        usage_gb = current / (1024**3)
        ceiling_gb = sched._memory_hard_limit_bytes / (1024**3)
        return (
            f"Prefill would require ~{format_bytes(current + peak)} peak "
            f"(current {format_bytes(current)} + KV+SDPA {format_bytes(peak)}) "
            f"but ceiling is {format_bytes(sched._memory_hard_limit_bytes)} "
            f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
            f"Reduce context length or lower memory_guard_tier."
        )
    return None
