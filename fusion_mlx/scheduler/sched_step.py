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

from ..cache.observability import CacheRateTracker
from ..cache.paged_cache import PagedCacheManager
from ..cache.prefix_cache import BlockAwarePrefixCache
from ..exceptions import is_cache_corruption_error
from ..prefill_progress import get_prefill_tracker
from ..prefill_transient_tracker import PrefillTransientTracker
from ..request import Request, RequestOutput, RequestStatus, SamplingParams
from ..speculative.vlm_mtp import VLMMTPDrafter, run_vlm_mtp_decode
from ..utils.proc_memory import get_phys_footprint
from ..utils.sampling import make_sampler as omlx_make_sampler

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.

from .config import SchedulerOutput
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

def step(self) -> SchedulerOutput:
    """
    Execute one scheduling step with automatic error recovery.

    This method:
    1. Schedules waiting requests into the batch
    2. Runs one generation step via BatchGenerator
    3. Processes outputs and handles finished requests
    4. On cache corruption: clears all cache and reschedules requests
        for re-prefill (no error raised to caller)

    Returns:
        SchedulerOutput with results of this step
    """
    output = SchedulerOutput()

    # Process pending aborts FIRST (thread-safe with hybrid executor)
    sched._process_pending_aborts()

    # Drain async store_cache completions from prior steps. Each completed
    # entry triggers the deferred batch_generator.remove(uid) on the
    # inference thread. Inflight entries are left for a later step.
    sched._drain_pending_async_removes()

    # Check memory pressure and evict if needed (tiered cache)
    if sched.memory_monitor is not None and sched._step_counter % 64 == 0:
        sched._check_memory_pressure()

    try:
        # Advance in-flight chunked prefills (one chunk per request).
        # Must run before _schedule_waiting() so that completing prefills
        # are inserted into BatchGenerator before the decode step.
        chunked_scheduled: list[Request] = []
        chunked_rejected: list[RequestOutput] = []
        if sched.prefilling:
            sched._advance_chunked_prefills(chunked_scheduled, chunked_rejected)

        # Schedule waiting requests
        scheduled, rejected = sched._schedule_waiting()
        # Merge chunked-prefill completions into the scheduled list.
        if chunked_scheduled:
            scheduled = chunked_scheduled + scheduled
        output.scheduled_request_ids = [r.request_id for r in scheduled]
        output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)
        if chunked_rejected:
            output.outputs.extend(chunked_rejected)
            output.has_work = True
        if rejected:
            output.outputs.extend(rejected)
            output.has_work = True

        # Run generation step if we have running requests.
        # Use next_generated() which returns only GenerationBatch.Response
        # objects (prefill is handled externally before insert).
        if (sched.batch_generator is not None or sched._vlm_mtp_active) and sched.running:
            if sched.batch_generator is not None:
                responses = list(sched.batch_generator.next_generated())
            else:
                responses = []
            # Drive vlm_mtp generators alongside BatchGenerator. Order
            # matters only for log determinism; _process_batch_responses
            # is per-uid.
            if sched._vlm_mtp_active:
                responses.extend(sched._step_vlm_mtp())
            output.has_work = True

            if responses:
                outputs, finished_ids = sched._process_batch_responses(responses)
                output.outputs = outputs
                output.finished_request_ids = finished_ids
                sched._cleanup_finished(finished_ids)

                # Periodic Metal allocator cleanup during long decodes.
                # mx.random.categorical inside the sampler allocates a
                # tiny scalar via gumbel → uniform on every call.
                # omlx ships its own non-compiled sampler
                # (omlx/utils/sampling.py) so that RNG state actually
                # advances in the server, but the trade-off is that
                # those scalars accumulate in the IOGPU residency set
                # — macOS aborts at ~4096 entries. Long contexts
                # (50k+) decoding thousands of tokens hit that limit
                # mid-stream. Synchronise the generation stream first
                # so any in-flight Metal command buffer that still
                # references buffers we're about to drop has
                # completed; the allocator only releases pool entries
                # whose ref count is zero, but the sync guarantees
                # there is no race window. Decode-only path —
                # next_generated() returns nothing during prefill, so
                # we never disrupt prefill activation buffers.
                sched._tokens_since_clear_cache = (
                    getattr(self, "_tokens_since_clear_cache", 0)
                    + len(responses)
                )
                if sched._tokens_since_clear_cache >= 1024:
                    _sync_and_clear_cache(sched._stream)
                    sched._tokens_since_clear_cache = 0

    except _PrefillAbortedError:
        # Prefill was interrupted by a pending abort.
        # BatchGenerator is in an inconsistent state (partial
        # prefill), so reset it entirely. Pending aborts will
        # be processed at the start of the next step().
        sched.batch_generator = None
        sched._current_sampler_params = None
        sched._boundary_cache_snapshots.clear()
        if sched._boundary_snapshot_store is not None:
            sched._boundary_snapshot_store.cleanup_all()
        sched._boundary_snapshot_required = None
        # Move any running requests back to waiting so they
        # can be rescheduled with a fresh BatchGenerator.
        sched._reschedule_running_requests()

    except (TypeError, AttributeError, ValueError) as e:
        if sched._is_cache_corruption_error(e):
            import traceback

            logger.warning(
                f"Cache corruption detected: {e}, "
                f"clearing cache and re-prefilling..."
            )
            logger.debug(f"Cache corruption traceback:\n{traceback.format_exc()}")
            # Full reset: clear batch generator, all caches, VLM state
            sched._recover_from_cache_error()
            # Reschedule requests for re-prefill from scratch.
            # Requests exceeding max corruption retries are failed.
            failed_ids = sched._reschedule_running_requests(is_corruption=True)
            for rid in failed_ids:
                output.outputs.append(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        error=(
                            f"Cache corruption not recoverable "
                            f"after retries: {e}"
                        ),
                    )
                )
                output.finished_request_ids.add(rid)
        else:
            raise

    except Exception as e:
        import traceback

        logger.error(
            f"Error in batch generation step: {e}\n" f"{traceback.format_exc()}"
        )
        raise

    # Clear finished tracking for next step
    sched.finished_req_ids = set()

    # Periodic Metal cache cleanup
    sched._step_counter += 1
    should_clear = sched._should_periodic_clear_cache()
    # Deferred post-completion cleanup: fire once the step counter reaches
    # the target set by _cleanup_finished() (#435, #557).
    if (
        sched._deferred_clear_at is not None
        and sched._step_counter >= sched._deferred_clear_at
    ):
        should_clear = True
        sched._deferred_clear_at = None
    if should_clear:
        _sync_and_clear_cache(sched._stream)
    if (
        sched.config.gc_cleanup_interval > 0
        and sched._step_counter % sched.config.gc_cleanup_interval == 0
    ):
        gc.collect()

    if sched._step_counter % 32 == 0:
        sched._publish_admin_snapshot()

    return output

def _publish_admin_snapshot(self) -> None:
    """Atomically publish a fresh admin-visible snapshot.

    Called from step() on the engine thread, where running/waiting are
    not concurrently mutated. The admin endpoint reads the reference via
    snapshot_for_admin() and never iterates the live structures.
    """
    sched._admin_snapshot = {
        "running_by_id": dict(sched.running),
        "waiting": list(sched.waiting),
    }

def snapshot_for_admin(self) -> dict[str, Any]:
    """Return the most recently published admin snapshot.

    Reference read is GIL-atomic; the dict itself is no longer mutated
    after publication. May be one step stale, which is fine for dashboard
    polling.
    """
    return sched._admin_snapshot

def get_request(sched, request_id: str) -> Request | None:
    """Get a request by ID."""
    return sched.requests.get(request_id)

def remove_finished_request(sched, request_id: str) -> Request | None:
    """Remove a finished request from tracking."""
    return sched.requests.pop(request_id, None)

def get_stats(self) -> dict[str, Any]:
    """Get scheduler statistics."""
    stats = {
        "num_waiting": len(sched.waiting),
        "num_prefilling": len(sched.prefilling),
        "num_running": len(sched.running),
        "num_requests_processed": sched.num_requests_processed,
        "total_prompt_tokens": sched.total_prompt_tokens,
        "total_completion_tokens": sched.total_completion_tokens,
    }
    # Include cache stats
    if sched.block_aware_cache is not None:
        stats["ssd_cache"] = sched.block_aware_cache.get_stats()
    return stats

def get_cache_stats(self) -> dict[str, Any] | None:
    """Get cache statistics."""
    if sched.block_aware_cache is not None:
        return sched.block_aware_cache.get_stats()
    return None

def reset(self) -> None:
    """Reset the scheduler state."""
    # Drain any pending deferred aborts
    sched._pending_abort_ids.clear()

    # Abort all requests directly (reset is synchronous)
    for request_id in list(sched.requests.keys()):
        sched._do_abort_request(request_id)

    sched.waiting.clear()
    sched.prefilling.clear()
    sched._prefill_states.clear()
    sched.running.clear()
    sched.requests.clear()
    sched.finished_req_ids.clear()
    sched.request_id_to_uid.clear()
    sched.uid_to_request_id.clear()
    # Async store_cache bookkeeping. shutdown() drains these before us,
    # but clear here too so reset() is safe to call standalone (e.g. tests
    # or recovery paths) without leaking Request refs through stale futures.
    sched._pending_async_removes.clear()
    sched._inflight_store_futures.clear()
    sched.batch_generator = None
    sched._current_sampler_params = None
    sched._boundary_cache_snapshots.clear()
    if sched._boundary_snapshot_store is not None:
        sched._boundary_snapshot_store.cleanup_all()
    sched._boundary_snapshot_required = None

    # Clear caches
    if sched.block_aware_cache is not None:
        sched.block_aware_cache.clear()
    sched._cache_rate_tracker.clear()

    # Clear detokenizers
    sched._request_detokenizers.clear()

    # Clear protocol-specific output parser sessions
    sched._output_parser_sessions.clear()

    # Cancel any pending deferred Metal cache clear
    sched._deferred_clear_at = None

def deep_reset(self) -> None:
    """
    Deep reset that clears ALL cache state including model-level caches.

    This is more aggressive than reset() and should be used when
    switching engines or recovering from errors.
    """
    # Standard reset first
    sched.reset()

    # Clear any model-level cache state
    # MLX models may have internal cache references
    if hasattr(sched.model, "cache"):
        sched.model.cache = None

    # Some MLX models store cache in layers
    if hasattr(sched.model, "layers"):
        for layer in sched.model.layers:
            if hasattr(layer, "cache"):
                layer.cache = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                layer.self_attn.cache = None

    # Release model and tokenizer references for GC
    sched.model = None
    sched.tokenizer = None

    # Release all cache-related references for GC
    sched.paged_cache_manager = None
    sched.block_aware_cache = None
    sched.memory_monitor = None
    sched._boundary_snapshot_store = None

    # Force garbage collection of any lingering cache objects
    import gc

    gc.collect()

    logger.info("Deep reset completed - all caches cleared")

def shutdown(self) -> None:
    """
    Graceful shutdown.

    Flushes hot cache to SSD and closes the background writer.
    paged SSD cache files are NOT cleared to allow reuse on reload.
    """
    logger.info("Scheduler shutdown initiated...")
    # The store-cache gate is a non-blocking counter (#1496), so there is
    # no step-thread caller to wake here. Inflight futures are drained
    # below before the executor is joined.
    # Wait for any inflight async store_cache futures + drain pending
    # batch_generator removes so the writer thread / underlying paged SSD
    # cache see all blocks before close().
    if sched._store_cache_executor is not None:
        try:
            inflight = list(sched._inflight_store_futures.values())
            if inflight:
                logger.info(
                    "Waiting for %d inflight async store_cache future(s)...",
                    len(inflight),
                )
                concurrent.futures.wait(inflight, timeout=30.0)
            sched._drain_pending_async_removes()
            sched._store_cache_executor.shutdown(wait=True)
            # Final drain after executor join. All workers are now done,
            # so any entries still in _pending_async_removes (skipped by
            # the first drain because their future hadn't completed yet)
            # are guaranteed drainable here. Without this, slow worker
            # finishes between the 30s wait timeout and shutdown(wait=True)
            # would leave KV cache references pinned on Request objects.
            sched._drain_pending_async_removes()
        except Exception as e:
            logger.warning(f"Async store_cache shutdown error: {e}")
        sched._store_cache_executor = None
        sched._store_cache_gate = None
    if sched.paged_ssd_cache_manager is not None:
        sched.paged_ssd_cache_manager.close()
        sched.paged_ssd_cache_manager = None
    logger.info("Scheduler shutdown completed")
