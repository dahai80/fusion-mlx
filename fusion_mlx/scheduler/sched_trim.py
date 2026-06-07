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

def _trim_prompt_cache_for_generation(sched, cache_list: list[Any]) -> bool:
    """Trim each cache layer by one token for exact-hit generation kickoff."""
    if not cache_list:
        return False

    for cache_obj in cache_list:
        if not sched._trim_cache_tree_by_one(cache_obj):
            return False
    return True

def _trim_cache_tree_by_one(sched, cache_obj: Any) -> bool:
    """Trim one token from cache object (recursively for CacheList)."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return all(
            sched._trim_cache_tree_by_one(sub_cache) for sub_cache in sub_caches
        )

    trim_fn = getattr(cache_obj, "trim", None)
    if not callable(trim_fn):
        return False

    try:
        trimmed = trim_fn(1)
        if trimmed is None:
            return True
        return int(trimmed) >= 1
    except Exception:
        return False

def _remove_uid_from_active_batch(sched, uid: int) -> None:
    """Remove UID from BatchGenerator safely.

    vlm_mtp uses negative uids that BatchGenerator never sees; the
    per-uid generator state is owned by ``_vlm_mtp_active`` and gets
    dropped when ``_step_vlm_mtp`` marks the entry finished.
    """
    if uid < 0:
        return
    if sched.batch_generator is None:
        return

    sched.batch_generator.remove([uid])

def _check_pending_aborts_for_uids(sched, uids: list[int]) -> list[int]:
    """Return UIDs that have pending aborts.

    Called during prefill to detect aborted
    requests between chunks. GIL guarantees thread-safe reads of
    _pending_abort_ids from the executor thread.
    """
    if not sched._pending_abort_ids:
        return []
    aborted = []
    for uid in uids:
        request_id = sched.uid_to_request_id.get(uid)
        if request_id and request_id in sched._pending_abort_ids:
            aborted.append(uid)
    return aborted

def abort_request(sched, request_id: str) -> bool:
    """
    Enqueue a request for deferred abort.

    The actual abort is processed at the start of the next step() call,
    ensuring thread safety with the hybrid executor pattern. CPython GIL
    guarantees set.add() is atomic.

    Args:
        request_id: The request ID to abort

    Returns:
        True (abort is always enqueued)
    """
    sched._pending_abort_ids.add(request_id)
    logger.debug(f"Enqueued deferred abort for request {request_id}")
    return True

def _process_pending_aborts(self) -> None:
    """Drain and process pending abort requests.

    Called from step() to ensure aborts are processed in the same
    execution context as generation (thread-safe).
    """
    while sched._pending_abort_ids:
        request_id = sched._pending_abort_ids.pop()
        sched._do_abort_request(request_id)

def _do_abort_request(sched, request_id: str) -> bool:
    """
    Actually abort a request. Must be called from the step() context.

    Args:
        request_id: The request ID to abort

    Returns:
        True if request was found and aborted, False otherwise
    """
    request = sched.requests.get(request_id)
    if request is None:
        return False

    # Remove from waiting queue
    if request.status == RequestStatus.WAITING:
        try:
            sched.waiting.remove(request)
        except ValueError:
            pass

    # Remove from chunked-prefill queue (if mid-prefill)
    if request_id in sched._prefill_states:
        sched._prefill_states.pop(request_id, None)
        sched.prefilling = deque(
            r for r in sched.prefilling if r.request_id != request_id
        )

    # Remove from running (BatchGenerator)
    if request.request_id in sched.request_id_to_uid:
        uid = sched.request_id_to_uid[request.request_id]
        # Synchronize in-flight GPU work before modifying batch state.
        # batch_generator.remove() triggers lazy KV cache array slicing
        # that replaces references to arrays still used by in-flight
        # Metal command buffers.  Without this barrier the Metal driver
        # can hit 'completeMemory() prepare count underflow'.
        _safe_sync_stream(sched._stream)
        sched._remove_uid_from_active_batch(uid)
        if hasattr(sched.model, "unregister_rope_delta"):
            sched.model.unregister_rope_delta(uid)
        del sched.uid_to_request_id[uid]
        del sched.request_id_to_uid[request.request_id]

    if request_id in sched.running:
        del sched.running[request_id]

    # Release blocks for eviction (same as _cleanup_finished)
    if sched.paged_cache_manager is not None:
        block_table = sched.paged_cache_manager.get_block_table(request_id)
        if block_table is None and hasattr(request, "block_table"):
            block_table = request.block_table
        if block_table:
            released = sched.paged_cache_manager.release_for_eviction(
                block_table.block_ids
            )
            if released > 0:
                logger.debug(
                    f"Released {released} blocks for eviction on abort "
                    f"(request {request_id})"
                )

    # Clear request entry from block_aware_cache
    if sched.block_aware_cache is not None:
        sched.block_aware_cache.clear_request_entry(request_id)

    # Clean up streaming detokenizer to prevent state contamination
    sched._cleanup_detokenizer(request_id)

    # Clean up protocol-specific output parser session
    sched._cleanup_output_parser_session(request_id)

    # Clean up VLM adapter state to prevent contamination
    if hasattr(sched.model, "clear_vlm_position_state"):
        sched.model.clear_vlm_position_state()
    if hasattr(sched.model, "clear_pending_embeddings"):
        sched.model.clear_pending_embeddings()

    # Drop any boundary snapshot for this request.
    sched._boundary_cache_snapshots.pop(request_id, None)
    if sched._boundary_snapshot_store is not None:
        sched._boundary_snapshot_store.cleanup_request(request_id)

    # Remove from prefill progress tracker.
    get_prefill_tracker().remove(request_id)

    # Mark as aborted
    request.set_finished(RequestStatus.FINISHED_ABORTED)
    sched.finished_req_ids.add(request_id)

    # Remove from requests dict and clear cache references to release
    # MLX arrays promptly (mirrors _cleanup_finished behavior).
    # _cleanup_request (engine_core) no longer calls remove_finished_request,
    # so this is the single cleanup point for aborted requests.
    req_to_remove = sched.requests.pop(request_id, None)
    if req_to_remove is not None:
        req_to_remove._extracted_cache = None
        req_to_remove.prompt_cache = None

    logger.debug(f"Aborted request {request_id}")
    return True

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
