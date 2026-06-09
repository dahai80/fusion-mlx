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

logger = logging.getLogger(__name__)
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

@staticmethod
def _collect_arrays_from_extracted_cache(
    extracted_cache: list[Any],
) -> list[Any]:
    """Collect lazy mx.array references from an _extracted_cache payload.

    Used by G2-async to force a single batched mx.eval on the inference
    thread before handing the cache off to the store_cache worker. The
    worker can then call _extract_tensor_bytes safely (no further Metal
    graph evaluation needed for non-bfloat16, no-op for already-evaluated).

    Walks the per-layer dict format produced by _extract_cache_states:
    each layer is {state, meta_state, class_name, cache_type}, where
    state is a tuple of mx.arrays (or nested for CacheList / TurboQuant).
    """
    arrays: list[Any] = []
    for layer in extracted_cache or []:
        if not isinstance(layer, dict):
            continue
        state = layer.get("state", ())
        if isinstance(state, mx.array):
            arrays.append(state)
            continue
        if not isinstance(state, (list, tuple)):
            continue
        for item in state:
            if isinstance(item, mx.array):
                arrays.append(item)
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, mx.array):
                        arrays.append(sub)
                    elif hasattr(sub, "_fields"):
                        # NamedTuple state (TurboQuant). Walk fields.
                        for fname in sub._fields:
                            val = getattr(sub, fname, None)
                            if isinstance(val, mx.array):
                                arrays.append(val)
            elif hasattr(item, "_fields"):
                for fname in item._fields:
                    val = getattr(item, fname, None)
                    if isinstance(val, mx.array):
                        arrays.append(val)
    return arrays

def _async_store_cache_worker(    self,
    request_id: str,
    token_sequence_to_store: list[int],
    cache_to_store: list[Any],
    model_cache_config: Any | None,
    intermediate_snapshots: dict[int, list[Any]] | None,
    extra_keys: tuple[Any, ...] | None,
    extra_key_token_start: int | None,
    extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None,
) -> None:
    """Run store_cache + paged_cache cleanup off the inference thread.

    Pre-conditions enforced by the caller (_cleanup_finished):
    - mx.async_eval() was called on the inference thread for all
    KV cache arrays, dispatching materialization asynchronously
    without blocking the inference thread. async_eval completes
    Metal command enqueueing before returning, so all commands
    are submitted by the time executor.submit() runs.
    - This worker calls mx.synchronize(self._stream) via the
    _safe_sync_stream helper to wait on the same stream where
    mx.async_eval dispatched the arrays. A bare mx.synchronize()
    with no args only blocks on the default stream (gpu:0) and
    would leave the dispatched per-engine stream's work
    unsynchronized, racing the buffer-protocol access below
    (#1437). Stream objects are not thread-local in MLX (Metal
    device is a global singleton), so mx.synchronize(stream) is
    safe cross-thread; it just calls waitUntilCompleted on the
    command buffer.
    - bfloat16 view+eval inside _extract_tensor_bytes runs on this
    worker's default mx stream, isolated from self._stream;
    the underlying buffer is read-only at this point.
    - batch_generator.remove(uid) is deferred until this worker
    completes (handled by _drain_pending_async_removes).

    paged_cache_manager and block_aware_cache rely on
    threading.RLock so concurrent access from main and worker is safe.
    """
    try:
        # Hold _mx_buffer_access_lock across the worker's mx-buffer
        # access. store_cache eventually drives _extract_tensor_bytes,
        # which reads raw bytes via the buffer protocol; serializing
        # against inference-thread mx.clear_cache / mx.synchronize calls
        # prevents a SIGABRT when those reclaim the underlying Metal
        # buffer pool mid-read (#1106).
        with _mx_buffer_access_lock:
            with self._phase_timer("store_cache_worker_sync"):
                _safe_sync_stream(self._stream)
            block_table = self.block_aware_cache.store_cache(
                request_id,
                token_sequence_to_store,
                cache_to_store,
                model_cache_config=model_cache_config,
                boundary_snapshots=intermediate_snapshots,
                extra_keys=extra_keys,
                extra_key_token_start=extra_key_token_start,
                extra_key_ranges=extra_key_ranges,
            )
        if block_table is None and self.paged_cache_manager is not None:
            block_table = self.paged_cache_manager.get_block_table(request_id)
        if block_table and self.paged_cache_manager is not None:
            self.paged_cache_manager.release_for_eviction(block_table.block_ids)
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear_request_entry(request_id)
    except Exception as e:
        logger.warning("Async store_cache failed for %s: %s", request_id, e)

def _drain_pending_async_removes(self) -> None:
    """Process deferred batch_generator.remove() calls from prior steps.

    Called at the start of every step. For each pending entry, if the
    async store_cache future has finished, perform the
    batch_generator.remove() on the inference thread (Metal-safe) and
    finalize cleanup state. Entries whose futures are still in flight
    are left at the head of the deque for a later step.
    """
    if not self._pending_async_removes:
        return
    while self._pending_async_removes:
        uid, request_id, future = self._pending_async_removes[0]
        if future is not None and not future.done():
            # Worker still busy. Stop draining; check again next step.
            # Inflight entry stays at deque head to preserve order.
            break
        self._pending_async_removes.popleft()
        # Surface worker exceptions for visibility (don't crash step loop).
        if future is not None:
            exc = future.exception()
            if exc is not None:
                logger.warning(
                    "Async store_cache for %s raised: %s", request_id, exc
                )
        # Run batch_generator.remove on the inference thread.
        try:
            _safe_sync_stream(self._stream)
            self._remove_uid_from_active_batch(uid)
            if hasattr(self.model, "unregister_rope_delta"):
                self.model.unregister_rope_delta(uid)
        except Exception as e:
            logger.warning(
                "Deferred batch_generator.remove(uid=%s) failed: %s",
                uid,
                e,
            )
        # Cleanup uid maps now that the slot is reclaimable.
        if uid in self.uid_to_request_id:
            del self.uid_to_request_id[uid]
        if request_id in self.request_id_to_uid:
            del self.request_id_to_uid[request_id]
        self._inflight_store_futures.pop(request_id, None)
        # Boundary snapshots were kept on disk for the worker; safe to
        # delete now that the future has completed. Cleanup was
        # deferred from _cleanup_finished to avoid racing the worker's
        # boundary_snapshot_store.load() calls with rmtree.
        if self._boundary_snapshot_store is not None:
            self._boundary_snapshot_store.cleanup_request(request_id)
        # Worker no longer holds extracted_cache — pop request from
        # self.requests and drop the cache buffer references so MLX
        # arrays can be freed.
        req_to_remove = self.requests.pop(request_id, None)
        if req_to_remove is not None:
            req_to_remove._extracted_cache = None
            req_to_remove.prompt_cache = None

def _calculate_max_blocks(self) -> int:
    """
    Calculate maximum cache blocks for paged SSD-only mode.

    In paged SSD-only mode, blocks don't consume GPU memory (data is on paged SSD),
    so we use a large default that can be limited by SSD capacity.

    Returns:
        Maximum number of cache blocks to allocate.
    """
    # In paged SSD-only mode, use a large default since blocks don't consume GPU memory
    # The actual limit is SSD capacity (paged_ssd_cache_max_size)
    max_blocks = 100000  # Large default for paged SSD-only mode

    block_size = self.config.paged_cache_block_size
    logger.info(
        f"paged SSD-only mode: max_blocks={max_blocks}, block_size={block_size} tokens"
    )

    return max_blocks

def _collect_rotating_window_sizes(    self,
    cache_obj: Any,
    window_sizes: set[int],
) -> None:
    """Collect rotating window sizes recursively from cache objects."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        for sub_cache in sub_caches:
            self._collect_rotating_window_sizes(sub_cache, window_sizes)

    class_name = type(cache_obj).__name__
    if class_name in ("RotatingKVCache", "BatchRotatingKVCache"):
        max_size = getattr(cache_obj, "max_size", 0)
        if isinstance(max_size, int) and max_size > 0:
            window_sizes.add(max_size)

def _detect_rotating_window_sizes(self) -> set[int]:
    """Detect rotating window sizes from model.make_cache() if available."""
    if not hasattr(self.model, "make_cache"):
        return set()

    try:
        cache_list = self.model.make_cache()
    except Exception as e:
        logger.debug(f"Failed to inspect model rotating window sizes: {e}")
        return set()

    if cache_list is None:
        return set()

    window_sizes: set[int] = set()
    for cache_obj in cache_list:
        self._collect_rotating_window_sizes(cache_obj, window_sizes)

    return window_sizes

# Target range for RotatingKVCache block size alignment.
# Using a multiple of window_size within this range reduces SSD I/O
# overhead (fewer, larger block files) while keeping cache restore
# reprocessing reasonable.
_ROTATING_BLOCK_SIZE_MIN = 512
_ROTATING_BLOCK_SIZE_MAX = 1024

def _align_block_size_with_rotating_window(self) -> None:
    """
    Align paged cache block size to a multiple of RotatingKVCache
    window size, targeting 512-1024 tokens per block.

    Block size must be a multiple of window_size so that block
    boundaries align with rotation boundaries. When window_size is
    small (e.g. 128), using it directly as block_size creates too
    many small files. Instead we pick the smallest multiple of
    window_size that falls within [_ROTATING_BLOCK_SIZE_MIN,
    _ROTATING_BLOCK_SIZE_MAX].
    """
    if not self.config.paged_ssd_cache_dir:
        return

    window_sizes = self._detect_rotating_window_sizes()
    if not window_sizes:
        return

    if len(window_sizes) > 1:
        raise ValueError(
            "Multiple RotatingKVCache window sizes detected "
            f"({sorted(window_sizes)}). Set a single aligned block size or "
            "disable paged cache for this model."
        )

    window_size = next(iter(window_sizes))

    # Find the smallest multiple of window_size >= _ROTATING_BLOCK_SIZE_MIN.
    # If window_size itself is already >= max, just use window_size.
    lo = self._ROTATING_BLOCK_SIZE_MIN
    hi = self._ROTATING_BLOCK_SIZE_MAX

    if window_size >= hi or window_size >= lo:
        target_block_size = window_size
    else:
        # window_size < lo: pick smallest multiple in [lo, hi]
        multiplier = (lo + window_size - 1) // window_size  # ceil(lo / ws)
        target_block_size = multiplier * window_size
        if target_block_size > hi:
            # Fall back to largest multiple <= hi
            target_block_size = (hi // window_size) * window_size
            if target_block_size < window_size:
                target_block_size = window_size

    if self.config.paged_cache_block_size != target_block_size:
        logger.info(
            "Aligning paged cache block_size=%s to %s "
            "(RotatingKVCache window_size=%s, multiplier=%sx)",
            self.config.paged_cache_block_size,
            target_block_size,
            window_size,
            target_block_size // window_size,
        )
        self.config.paged_cache_block_size = target_block_size

# Default block size for ArraysCache-only hybrid models.
# Match prefill_step_size (2048) so that boundary caching ON/OFF
# produces identical prefill chunk sizes, eliminating float32↔dtype
# roundtrip differences in GatedDeltaNet recurrent state.
_ARRAYS_CACHE_BLOCK_SIZE = 2048

def _enlarge_block_size_for_arrays_cache(self) -> None:
    """Enlarge block size for ArraysCache-only hybrid models.

    When a model uses ArraysCache (GatedDeltaNet) but not RotatingKVCache,
    a larger block size reduces the number of boundary snapshot stops during
    prefill while still storing valid per-block recurrent state.

    This is skipped if RotatingKVCache was already detected (block size was
    aligned to its window size) or if the user explicitly set a block size
    larger than the default.
    """
    if not self.config.paged_ssd_cache_dir:
        return

    # Skip if RotatingKVCache already adjusted block size.
    rotating_sizes = self._detect_rotating_window_sizes()
    if rotating_sizes:
        return

    # Detect ArraysCache from model.make_cache()
    if not hasattr(self.model, "make_cache"):
        return

    try:
        cache_list = self.model.make_cache()
    except Exception:
        return

    if cache_list is None:
        return

    has_arrays_cache = any(
        self._cache_tree_has_arrays_cache(cache_obj) for cache_obj in cache_list
    )
    if not has_arrays_cache:
        return

    target = self._ARRAYS_CACHE_BLOCK_SIZE
    if self.config.paged_cache_block_size >= target:
        return

    logger.info(
        "Enlarging paged cache block_size=%s to %s for "
        "ArraysCache hybrid model (reduces boundary snapshot overhead)",
        self.config.paged_cache_block_size,
        target,
    )
    self.config.paged_cache_block_size = target

@staticmethod
def _cache_tree_has_arrays_cache(cache_obj: Any) -> bool:
    """Return True if cache_obj contains ArraysCache (recursively)."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return any(
            Scheduler._cache_tree_has_arrays_cache(sub) for sub in sub_caches
        )
    return type(cache_obj).__name__ in ("ArraysCache", "SizedArraysCache")
