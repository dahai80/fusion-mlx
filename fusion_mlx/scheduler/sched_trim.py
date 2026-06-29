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
from collections import deque
from typing import Any

from ..prefill_progress import get_prefill_tracker
from ..request import RequestStatus

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .helpers import (
    _safe_sync_stream,
)
from .monkeypatches import _unregister_uid_row


def _trim_prompt_cache_for_generation(    self, cache_list: list[Any]) -> bool:
    """Trim each cache layer by one token for exact-hit generation kickoff."""
    if not cache_list:
        return False

    for cache_obj in cache_list:
        if not self._trim_cache_tree_by_one(cache_obj):
            return False
    return True

def _trim_cache_tree_by_one(    self, cache_obj: Any) -> bool:
    """Trim one token from cache object (recursively for CacheList)."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return all(
            self._trim_cache_tree_by_one(sub_cache) for sub_cache in sub_caches
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

def _remove_uid_from_active_batch(    self, uid: int) -> None:
    """Remove UID from BatchGenerator safely.

    vlm_mtp uses negative uids that BatchGenerator never sees; the
    per-uid generator state is owned by ``_vlm_mtp_active`` and gets
    dropped when ``_step_vlm_mtp`` marks the entry finished.
    """
    if uid < 0:
        return
    if self.batch_generator is None:
        return

    self.batch_generator.remove([uid])

def _check_pending_aborts_for_uids(    self, uids: list[int]) -> list[int]:
    """Return UIDs that have pending aborts.

    Called during prefill to detect aborted
    requests between chunks. GIL guarantees thread-safe reads of
    _pending_abort_ids from the executor thread.
    """
    if not self._pending_abort_ids:
        return []
    aborted = []
    for uid in uids:
        request_id = self.uid_to_request_id.get(uid)
        if request_id and request_id in self._pending_abort_ids:
            aborted.append(uid)
    return aborted

def abort_request(    self, request_id: str) -> bool:
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
    self._pending_abort_ids.add(request_id)
    logger.debug(f"Enqueued deferred abort for request {request_id}")
    return True

def _process_pending_aborts(self) -> None:
    """Drain and process pending abort requests.

    Called from step() to ensure aborts are processed in the same
    execution context as generation (thread-safe).
    """
    while self._pending_abort_ids:
        request_id = self._pending_abort_ids.pop()
        self._do_abort_request(request_id)

def _do_abort_request(    self, request_id: str) -> bool:
    """
    Actually abort a request. Must be called from the step() context.

    Args:
        request_id: The request ID to abort

    Returns:
        True if request was found and aborted, False otherwise
    """
    request = self.requests.get(request_id)
    if request is None:
        return False

    # Remove from waiting queue
    if request.status == RequestStatus.WAITING:
        try:
            self.waiting.remove(request)
        except ValueError:
            pass

    # Remove from chunked-prefill queue (if mid-prefill)
    if request_id in self._prefill_states:
        self._prefill_states.pop(request_id, None)
        self.prefilling = deque(
            r for r in self.prefilling if r.request_id != request_id
        )

    # Remove from running (BatchGenerator)
    if request.request_id in self.request_id_to_uid:
        uid = self.request_id_to_uid[request.request_id]
        # Synchronize in-flight GPU work before modifying batch state.
        # batch_generator.remove() triggers lazy KV cache array slicing
        # that replaces references to arrays still used by in-flight
        # Metal command buffers.  Without this barrier the Metal driver
        # can hit 'completeMemory() prepare count underflow'.
        _safe_sync_stream(self._stream)
        self._remove_uid_from_active_batch(uid)
        if hasattr(self.model, "unregister_rope_delta"):
            self.model.unregister_rope_delta(uid)
        _unregister_uid_row(self.model, uid)
        del self.uid_to_request_id[uid]
        del self.request_id_to_uid[request.request_id]

    if request_id in self.running:
        del self.running[request_id]

    # Release blocks for eviction (same as _cleanup_finished)
    if self.paged_cache_manager is not None:
        block_table = self.paged_cache_manager.get_block_table(request_id)
        if block_table is None and hasattr(request, "block_table"):
            block_table = request.block_table
        if block_table:
            released = self.paged_cache_manager.release_for_eviction(
                block_table.block_ids
            )
            if released > 0:
                logger.debug(
                    f"Released {released} blocks for eviction on abort "
                    f"(request {request_id})"
                )

    # Clear request entry from block_aware_cache
    if self.block_aware_cache is not None:
        self.block_aware_cache.clear_request_entry(request_id)

    # Clean up streaming detokenizer to prevent state contamination
    self._cleanup_detokenizer(request_id)

    # Clean up protocol-specific output parser session
    self._cleanup_output_parser_session(request_id)

    # Clean up VLM adapter state to prevent contamination
    if hasattr(self.model, "clear_vlm_position_state"):
        self.model.clear_vlm_position_state()
    if hasattr(self.model, "clear_pending_embeddings"):
        self.model.clear_pending_embeddings()

    # Drop any boundary snapshot for this request.
    self._boundary_cache_snapshots.pop(request_id, None)
    if self._boundary_snapshot_store is not None:
        self._boundary_snapshot_store.cleanup_request(request_id)

    # Remove from prefill progress tracker.
    get_prefill_tracker().remove(request_id)

    # Mark as aborted
    request.set_finished(RequestStatus.FINISHED_ABORTED)
    self.finished_req_ids.add(request_id)

    # Remove from requests dict and clear cache references to release
    # MLX arrays promptly (mirrors _cleanup_finished behavior).
    # _cleanup_request (engine_core) no longer calls remove_finished_request,
    # so this is the single cleanup point for aborted requests.
    req_to_remove = self.requests.pop(request_id, None)
    if req_to_remove is not None:
        req_to_remove._extracted_cache = None
        req_to_remove.prompt_cache = None

    logger.debug(f"Aborted request {request_id}")
    return True
