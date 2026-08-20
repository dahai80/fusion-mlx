# SPDX-License-Identifier: Apache-2.0
"""
Boundary Snapshot SSD Store for FusionMLX.

Stores non-sliceable cache layer snapshots (e.g. ArraysCache) to SSD during
prefill, freeing GPU memory immediately.  At request completion the snapshots
are loaded back one block at a time for final SSD cache storage.

Uses the same async-write pattern as PagedSSDCacheManager: tensors are
serialized to raw bytes on the inference thread (Metal-safe), buffered in
``_pending_writes`` for instant read-back, and flushed to disk by a
background writer thread via ``_write_safetensors_no_mx``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .paged_ssd_cache import (
    HAS_MLX,
    _encode_shape,
    _extract_tensor_bytes,
    _has_zero_dim,
    _restore_tensor_from_bytes,
    _write_safetensors_no_mx,
)

if HAS_MLX:
    import mlx.core as mx

logger = logging.getLogger(__name__)

# Default queue bounds — overridden dynamically by memory pressure.
_DEFAULT_MAX_PENDING_WRITES = 128
_MIN_PENDING_WRITES = 16
_MAX_PENDING_WRITES_CAP = 512

_SOFT_LIMIT_PCT = 0.80

_PREFIX_DEFAULT_MAX_BYTES = 20 * 1024 * 1024 * 1024


class _PrefixEntry:
    __slots__ = (
        "prefix_hash",
        "token_count",
        "file_path",
        "size_bytes",
        "last_access",
    )

    def __init__(self, prefix_hash, token_count, file_path, size_bytes, last_access):
        self.prefix_hash = prefix_hash
        self.token_count = token_count
        self.file_path = file_path
        self.size_bytes = size_bytes
        self.last_access = last_access


class BoundarySnapshotSSDStore:
    """Temporary SSD storage for boundary cache snapshots.

    Stores ArraysCache/RotatingKVCache boundary snapshots to SSD during
    prefill to avoid GPU memory accumulation.  Files are ephemeral and
    cleaned up when the request completes or aborts.

    Parameters
    ----------
    base_dir : Path
        Parent directory for the SSD cache (typically ``paged_ssd_cache_dir``).
        Snapshots are stored under ``base_dir/_boundary_snapshots/``.
    """

    # Timeouts applied when acquiring _writer_busy from each cleanup
    # path. cleanup_request is called from the scheduler's abort hot
    # path (~3 sites) and must yield faster than cleanup_all, which
    # also runs at startup / reset where blocking longer is tolerable
    # in exchange for a stronger orphan-avoidance guarantee. The
    # worst-case impact on the timeout fallback is identical in both
    # paths — an orphan file in the recreated dir until the next
    # constructor cleanup — so the only knob is per-call latency.
    _CLEANUP_ALL_TIMEOUT_S = 5.0
    _CLEANUP_REQUEST_TIMEOUT_S = 2.0

    def __init__(
        self,
        base_dir: Path,
        prefix_persist: bool = False,
        prefix_max_bytes: int | None = None,
    ) -> None:
        self._snapshot_dir = base_dir / "_boundary_snapshots"
        # Clean up orphaned files from previous crashes.
        if self._snapshot_dir.exists():
            try:
                shutil.rmtree(self._snapshot_dir)
            except Exception as e:
                logger.warning("Failed to clean up orphaned boundary snapshots: %s", e)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # request_id -> {token_count -> file_path}
        self._file_registry: dict[str, dict[int, Path]] = {}
        self._registry_lock = threading.Lock()

        # Pending writes buffer — raw bytes for instant read-back.
        # key: (request_id, token_count)
        self._pending_writes: dict[tuple[str, int], dict] = {}
        self._pending_lock = threading.Lock()

        # Cancelled requests with remaining queue item counts. Writer
        # thread decrements on each skip; entry is deleted when count
        # reaches zero, preventing unbounded growth. All access is
        # guarded by ``_cancelled_lock`` — the dict was previously
        # mutated unlocked from cleanup_request, cleanup_all, and the
        # writer thread, creating lost-cancellation and counter-
        # underflow races.
        self._cancelled_requests: dict[str, int] = {}
        self._cancelled_lock = threading.Lock()

        # Background writer thread.
        self._write_queue: queue.Queue = queue.Queue(
            maxsize=_DEFAULT_MAX_PENDING_WRITES
        )
        self._shutdown = threading.Event()
        # Held by the writer for the duration of each item's processing.
        # cleanup_all() acquires it after draining the queue so the writer
        # can't be mid-item (creating files inside the just-cleaned dir)
        # when rmtree runs.
        self._writer_busy = threading.Lock()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="boundary-snapshot-writer",
            daemon=True,
        )
        self._writer_thread.start()

        # Cross-restart prefix-keyed persistence namespace. Unlike
        # _snapshot_dir (rmtree'd at startup + per-request, ephemeral),
        # _prefix_dir SURVIVES restart so prefix cache states can be reused
        # across server restarts. Opt-in via prefix_persist (default OFF).
        self._prefix_persist = prefix_persist
        self._prefix_dir = base_dir / "_prefix_snapshots"
        self._prefix_max_bytes = (
            prefix_max_bytes
            if prefix_max_bytes and prefix_max_bytes > 0
            else _PREFIX_DEFAULT_MAX_BYTES
        )
        self._prefix_index: dict[bytes, _PrefixEntry] = {}
        self._prefix_lock = threading.Lock()
        self._prefix_total_bytes = 0
        self._prefix_stats = {
            "writes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "load_failures": 0,
        }
        self._prefix_write_queue: queue.Queue | None = None
        self._prefix_shutdown = threading.Event()
        self._prefix_writer_thread = None
        self._prefix_writer_busy = threading.Lock()
        if prefix_persist:
            # NO rmtree here - prefix snapshots persist across restarts.
            self._prefix_dir.mkdir(parents=True, exist_ok=True)
            self._scan_prefix_dir()
            self._prefix_write_queue = queue.Queue(maxsize=_DEFAULT_MAX_PENDING_WRITES)
            self._prefix_writer_thread = threading.Thread(
                target=self._prefix_writer_loop,
                name="prefix-snapshot-writer",
                daemon=True,
            )
            self._prefix_writer_thread.start()
            logger.info(
                "Boundary prefix persistence enabled (dir=%s, max_bytes=%d)",
                self._prefix_dir,
                self._prefix_max_bytes,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        request_id: str,
        token_count: int,
        snapshot_cache: list[Any],
        extract_cache_states_fn: Callable,
    ) -> bool:
        """Serialize snapshot to SSD (non-blocking).

        Must be called from the inference thread (Metal-safe for mx.eval).

        Parameters
        ----------
        request_id : str
            Unique request identifier.
        token_count : int
            Token boundary count.
        snapshot_cache : list
            Per-layer cache objects (None for skipped sliceable layers).
        extract_cache_states_fn : callable
            ``Scheduler._extract_cache_states`` — converts raw cache objects
            to ``List[Dict[str, Any]]``.

        Returns
        -------
        bool
            True if successfully enqueued for writing.
        """
        if not HAS_MLX:
            return False

        try:
            # 1. Extract dict-format states on inference thread.
            extracted, model_cache_config = extract_cache_states_fn(snapshot_cache)
            if not extracted:
                return False

            # 2. Flatten tensors + metadata for safetensors serialization.
            tensors_raw, metadata = self._serialize_extracted(
                extracted, request_id, token_count
            )

            # 3. Buffer in pending writes for instant read-back.
            pw_key = (request_id, token_count)
            with self._pending_lock:
                self._pending_writes[pw_key] = {
                    "tensors_raw": tensors_raw,
                    "metadata": metadata,
                    "extracted": extracted,  # keep for cheap read-back
                }

            # 4. Compute file path and register.
            file_path = self._file_path(request_id, token_count)
            with self._registry_lock:
                self._file_registry.setdefault(request_id, {})[token_count] = file_path

            # Backpressure check: warn at 80% capacity so caller can
            # throttle new prefills before the hard cap drops snapshots.
            qsize = self._write_queue.qsize()
            soft_limit = int(_DEFAULT_MAX_PENDING_WRITES * _SOFT_LIMIT_PCT)
            if qsize >= soft_limit:
                logger.warning(
                    "Boundary snapshot write queue at %d/%d (%.0f%%) — "
                    "backpressure active; consider reducing concurrent prefills",
                    qsize,
                    _DEFAULT_MAX_PENDING_WRITES,
                    qsize / _DEFAULT_MAX_PENDING_WRITES * 100,
                )

            # 5. Enqueue for background write (hard capacity check).
            # TODO: make this bound dynamic on MLX allocator pressure
            # (see _MIN_PENDING_WRITES / _MAX_PENDING_WRITES_CAP). The
            # original 5b8607cf refactor referenced a
            # _get_dynamic_queue_bound method that was never defined,
            # so every save() raised AttributeError and returned False
            # — silently breaking cross-restart prefix cache (#257).
            max_bound = _DEFAULT_MAX_PENDING_WRITES
            if qsize >= max_bound:
                # Queue at dynamic capacity — drop snapshot to protect memory
                logger.warning(
                    "Boundary snapshot queue at dynamic capacity (%d/%d), "
                    "dropping snapshot %s/%d",
                    qsize,
                    max_bound,
                    request_id,
                    token_count,
                )
                with self._pending_lock:
                    self._pending_writes.pop(pw_key, None)
                with self._registry_lock:
                    req_files = self._file_registry.get(request_id)
                    if req_files is not None:
                        req_files.pop(token_count, None)
                        if not req_files:
                            self._file_registry.pop(request_id, None)
                return False
            try:
                self._write_queue.put_nowait((pw_key, tensors_raw, metadata, file_path))
            except queue.Full:
                # Roll back the pending + registry entries: with no
                # queue item the writer can never decrement
                # _cancelled_requests for this entry, so if a later
                # cleanup_request counts it the rid stays pinned in
                # _cancelled_requests forever and every subsequent
                # save under that rid is silently discarded by the
                # _is_cancelled gates. The previous "stays in memory
                # only" promise was already broken because cleanup
                # discards the in-memory copy anyway.
                logger.warning(
                    "Boundary snapshot write queue full, dropping snapshot %s/%d",
                    request_id,
                    token_count,
                )
                with self._pending_lock:
                    self._pending_writes.pop(pw_key, None)
                with self._registry_lock:
                    req_files = self._file_registry.get(request_id)
                    if req_files is not None:
                        req_files.pop(token_count, None)
                        if not req_files:
                            self._file_registry.pop(request_id, None)
                return False

            return True

        except Exception as e:
            logger.debug("Failed to save boundary snapshot: %s", e)
            return False

    def load(
        self,
        request_id: str,
        token_count: int,
    ) -> list[dict[str, Any]] | None:
        """Load a snapshot, returning extracted cache state dicts.

        Checks the in-memory pending-writes buffer first (zero I/O), then
        falls back to reading the safetensors file from disk.

        Returns
        -------
        list or None
            List of per-layer dicts matching ``_extract_cache_states`` output
            format, or None on failure.
        """
        pw_key = (request_id, token_count)

        # Fast path: still in pending writes buffer.
        with self._pending_lock:
            pending = self._pending_writes.get(pw_key)
            if pending is not None:
                extracted = pending.get("extracted")
                if extracted is not None:
                    return extracted

                # Fallback: reconstruct from raw bytes.
                tensors_raw = pending.get("tensors_raw")
                metadata = pending.get("metadata")
                if tensors_raw and metadata:
                    return self._deserialize(tensors_raw, metadata)

        # Slow path: read from disk.
        file_path = self._file_path(request_id, token_count)
        if not file_path.exists():
            return None

        try:
            data = mx.load(str(file_path), return_metadata=True)
            if isinstance(data, tuple) and len(data) == 2:
                arrays, metadata = data
            else:
                return None
            return self._reconstruct_from_safetensors(arrays, metadata)
        except Exception as e:
            logger.debug(
                "Failed to load boundary snapshot %s/%d: %s",
                request_id,
                token_count,
                e,
            )
            return None

    def has(self, request_id: str, token_count: int) -> bool:
        """Check if a snapshot exists (in memory or on disk)."""
        pw_key = (request_id, token_count)
        with self._pending_lock:
            if pw_key in self._pending_writes:
                return True
        with self._registry_lock:
            req_files = self._file_registry.get(request_id)
            if req_files and token_count in req_files:
                return True
        return False

    def cleanup_request(self, request_id: str) -> None:
        """Delete all snapshot files and pending writes for a request.

        Caller must guarantee no async store_cache worker is still reading
        snapshots for this request — concurrent ``rmtree`` here would race
        the worker's :meth:`load` calls and silently strip block storage.
        :class:`fusion_mlx.scheduler.Scheduler` defers this call until the
        ``store_future`` for ``request_id`` is done.

        Acquires ``_writer_busy`` after marking the request cancelled so
        the writer thread can finish any item it is mid-processing first.
        Without this barrier the writer can pull an item, ``mkdir`` the
        request directory, write its temp file, then ``os.rename`` it
        into the final path *after* we have rmtree'd — leaving an
        orphaned file behind. The ``_cancelled_requests`` counter (held
        under ``_cancelled_lock``) catches the late-rename case if
        ``_writer_busy.acquire`` times out.

        Bounded with a timeout so a stuck I/O on the writer thread
        cannot deadlock request abort paths (called from scheduler's
        hot path at ~3 sites).

        The cancelled-counter is bumped additively and only when at
        least one pending item exists for the rid — see the inline
        comment at the bump site for the two distinct bugs that
        rules out (stale ``rid: 0`` after a timeout for an empty
        cleanup, and overwrites racing with re-entrant cleanup_request
        calls for the same rid).
        """
        # Atomically: count pending items for this rid, drop them, mark
        # the rid cancelled. Holding both locks during the snapshot is
        # required to keep the counter consistent with what the writer
        # will see — a save() call from another thread cannot interleave
        # an enqueue between our count and our cancellation mark.
        #
        # The bump is additive (``get + count``) and skipped entirely
        # when ``count == 0``. Both rules close real bugs:
        #   * Skip-on-zero: cleanup_request("X") for an rid with no
        #     pending items previously wrote ``cancelled[X] = 0`` then
        #     popped it on the acquired path. On the timeout fallback
        #     the pop never runs and the ``X: 0`` entry lingers for
        #     the process lifetime — every subsequent save() under
        #     that rid (or any later reuse of the same string) is
        #     discarded by the writer's ``_is_cancelled`` gates,
        #     which check key membership not value > 0. The counter
        #     must only exist when there is at least one in-flight
        #     item to drain it.
        #   * Additive: a re-entrant cleanup_request("X") for an rid
        #     that already has an in-flight cancellation must NOT
        #     overwrite the previous count. The writer's
        #     ``cleared_by_cleanup`` branch + ``_writer_busy`` lock
        #     together close the file-write race today, but the
        #     per-item dec_cancelled bookkeeping still has to balance.
        #     Overwriting drops the remaining decs on the floor; on
        #     the next ``save()`` under the same rid the writer would
        #     see a non-zero counter from the earlier batch and
        #     silently discard the new item.
        with self._pending_lock:
            keys_to_remove = [k for k in self._pending_writes if k[0] == request_id]
            count = len(keys_to_remove)
            for key in keys_to_remove:
                del self._pending_writes[key]
            if count > 0:
                with self._cancelled_lock:
                    self._cancelled_requests[request_id] = (
                        self._cancelled_requests.get(request_id, 0) + count
                    )

        # Remove from registry.
        with self._registry_lock:
            self._file_registry.pop(request_id, None)

        # Wait briefly for the writer to finish any item it had already
        # pulled. If it's genuinely stuck (slow disk, dead thread) fall
        # back to the cancelled-counter rescue rather than blocking the
        # caller.
        acquired = self._writer_busy.acquire(timeout=self._CLEANUP_REQUEST_TIMEOUT_S)
        try:
            # Remove files.
            req_dir = self._snapshot_dir / request_id
            if req_dir.exists():
                try:
                    shutil.rmtree(req_dir)
                except Exception as e:
                    logger.debug(
                        "Failed to clean up snapshots for %s: %s", request_id, e
                    )
        finally:
            if acquired:
                self._writer_busy.release()
                # Counter entry has done its job — we own the lock so all
                # _is_cancelled-gated work has either run or skipped. Drop
                # the counter so a future racing save() can't leave it
                # elevated forever. CRITICAL: only pop on the acquired
                # path. On timeout the writer is still mid-item and may
                # not yet have consulted ``_is_cancelled``; popping here
                # would defeat the late-rename rescue that the docstring
                # advertises as the timeout-fallback safety net.
                with self._cancelled_lock:
                    self._cancelled_requests.pop(request_id, None)
            else:
                logger.warning(
                    "cleanup_request(%s): writer thread did not yield "
                    "within %.1fs; relying on cancelled-counter rescue "
                    "for late-rename safety",
                    request_id,
                    self._CLEANUP_REQUEST_TIMEOUT_S,
                )

    def cleanup_all(self) -> None:
        """Delete all snapshot files (for reset/startup).

        Synchronizes with the background writer: we drain the queue to
        prevent it from starting a new item, then acquire ``_writer_busy``
        to wait until any item it had already pulled finishes. Without
        this barrier the writer can create ``req-X/temp.safetensors``
        and ``os.rename`` it to its final path *after* we've already
        rmtree'd and recreated the snapshot directory, leaving an
        orphaned file behind.

        Threading: concurrent ``save()`` is safe because the writer
        consults ``_pending_writes`` and ``_is_cancelled`` while
        holding ``_writer_busy``, and ``cleanup_all`` clears both
        under the same lock before rmtree. The earlier "must run on
        the save() thread" constraint is therefore no longer required.
        """
        # Drain write queue so the writer thread doesn't process stale
        # items after the directory is deleted. Put_nowait the sentinel
        # back so shutdown still sees it; on Full just drop and let
        # shutdown re-issue.
        while True:
            try:
                item = self._write_queue.get_nowait()
                if item is None:  # Sentinel — put it back for shutdown.
                    try:
                        self._write_queue.put_nowait(item)
                    except queue.Full:
                        # Drop the sentinel; shutdown will re-enqueue.
                        # If cleanup_all is the LAST call before process
                        # exit without an explicit shutdown(), the writer
                        # thread will only be reaped on daemon teardown.
                        logger.debug("cleanup_all: dropped writer-sentinel on Full")
                    break
            except queue.Empty:
                break

        # Wait for the writer to finish any item it had already pulled.
        # When we own _writer_busy the writer is between items, and we
        # just drained the queue so no new item can start. Bounded so a
        # stuck writer (slow disk, dead thread) cannot deadlock callers
        # — scheduler calls cleanup_all() from its abort / reset hot
        # path. After the timeout we proceed anyway: the worst case is
        # an orphaned file in the recreated directory, which next
        # startup's cleanup_all() will clear.
        acquired = self._writer_busy.acquire(timeout=self._CLEANUP_ALL_TIMEOUT_S)
        try:
            if not acquired:
                logger.warning(
                    "cleanup_all: writer thread did not yield within "
                    "%.1fs; proceeding with rmtree — late-rename may "
                    "orphan a file under the recreated snapshot dir "
                    "until next startup.",
                    self._CLEANUP_ALL_TIMEOUT_S,
                )
            with self._pending_lock:
                self._pending_writes.clear()
            with self._registry_lock:
                self._file_registry.clear()
            with self._cancelled_lock:
                # Only safe to clear when we own _writer_busy — otherwise
                # a writer mid-_dec_cancelled would race. On timeout we
                # leave the counter intact so the rescue path stays
                # effective for in-flight items.
                if acquired:
                    self._cancelled_requests.clear()

            if self._snapshot_dir.exists():
                try:
                    shutil.rmtree(self._snapshot_dir)
                except Exception as e:
                    logger.debug("Failed to clean up all boundary snapshots: %s", e)
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        finally:
            if acquired:
                self._writer_busy.release()

    def shutdown(self) -> None:
        """Stop background writer thread."""
        self._shutdown.set()
        try:
            self._write_queue.put_nowait(None)  # Sentinel
        except queue.Full:
            pass
        self._writer_thread.join(timeout=5.0)
        if self._prefix_persist and self._prefix_writer_thread is not None:
            self._prefix_shutdown.set()
            if self._prefix_write_queue is not None:
                try:
                    self._prefix_write_queue.put_nowait(None)  # Sentinel
                except queue.Full:
                    pass
            self._prefix_writer_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Cross-restart prefix persistence
    # ------------------------------------------------------------------

    @staticmethod
    def compute_prefix_chain_hashes(
        token_ids,
        block_size: int,
        model_name: str,
    ) -> list[bytes]:
        """Chain-hash each full-block boundary of token_ids.

        hash_k = sha256(hash_{k-1} || block_k_tokens || model_name).
        Incremental -> O(N) for all boundaries. Stable across restart
        so a re-hashed prompt prefix matches a previously persisted one.
        """
        hashes: list[bytes] = []
        if block_size <= 0 or not token_ids:
            return hashes
        parent = b""
        model_bytes = model_name.encode("utf-8")
        n_full = len(token_ids) // block_size
        for k in range(n_full):
            start = k * block_size
            block = token_ids[start : start + block_size]
            h = hashlib.sha256()
            h.update(parent)
            h.update(struct.pack(f"<{len(block)}i", *block))
            h.update(model_bytes)
            parent = h.digest()
            hashes.append(parent)
        return hashes

    def _prefix_file_path(self, prefix_hash: bytes, token_count: int) -> Path:
        hex_hash = prefix_hash.hex()
        return self._prefix_dir / hex_hash[:2] / f"{hex_hash}_{token_count}.safetensors"

    def save_prefix(
        self,
        prefix_hash: bytes,
        token_count: int,
        snapshot_cache: list[Any],
        extract_cache_states_fn: Callable,
        model_name: str,
    ) -> bool:
        """Persist a prefix-keyed snapshot to SSD (non-blocking, survives restart).

        Called from the inference thread (Metal-safe for mx.eval). Reuses the
        same safetensors format as the in-session save(). Returns False when
        persistence is disabled, inputs are invalid, or the write queue is full.
        """
        if not HAS_MLX or not self._prefix_persist:
            return False
        if not prefix_hash or token_count <= 0:
            return False
        try:
            extracted, _model_cache_config = extract_cache_states_fn(snapshot_cache)
            if not extracted:
                return False
            tensors_raw, metadata = self._serialize_extracted(
                extracted, f"prefix:{prefix_hash.hex()[:8]}", token_count
            )
            metadata["prefix_model_name"] = model_name
            file_path = self._prefix_file_path(prefix_hash, token_count)
            qsize = self._prefix_write_queue.qsize()
            soft_limit = int(_DEFAULT_MAX_PENDING_WRITES * _SOFT_LIMIT_PCT)
            if qsize >= soft_limit:
                logger.warning(
                    "Prefix snapshot write queue at %d/%d - backpressure active",
                    qsize,
                    _DEFAULT_MAX_PENDING_WRITES,
                )
            if qsize >= _DEFAULT_MAX_PENDING_WRITES:
                logger.warning(
                    "Prefix snapshot queue full, dropping %s/%d",
                    prefix_hash.hex()[:8],
                    token_count,
                )
                return False
            try:
                self._prefix_write_queue.put_nowait(
                    (prefix_hash, token_count, tensors_raw, metadata, file_path)
                )
            except queue.Full:
                logger.warning(
                    "Prefix snapshot queue full (race), dropping %s/%d",
                    prefix_hash.hex()[:8],
                    token_count,
                )
                return False
            return True
        except Exception as e:
            logger.debug("Failed to save prefix snapshot: %s", e)
            return False

    def has_prefix(self, prefix_hash: bytes, token_count: int) -> bool:
        if not self._prefix_persist or not prefix_hash:
            return False
        with self._prefix_lock:
            entry = self._prefix_index.get(prefix_hash)
            return entry is not None and entry.token_count == token_count

    def find_prefix_snapshot(
        self,
        prefix_hashes: list[bytes],
    ) -> tuple[bytes, int] | None:
        """Return (prefix_hash, token_count) of the longest cached boundary.

        prefix_hashes must be ordered by increasing boundary (output of
        compute_prefix_chain_hashes). Returns None when none are present.
        """
        if not self._prefix_persist or not prefix_hashes:
            return None
        best: tuple[bytes, int] | None = None
        with self._prefix_lock:
            for prefix_hash in prefix_hashes:
                entry = self._prefix_index.get(prefix_hash)
                if entry is not None:
                    best = (entry.prefix_hash, entry.token_count)
        return best

    def load_prefix(
        self,
        prefix_hash: bytes,
        token_count: int,
    ) -> list[dict[str, Any]] | None:
        """Load a prefix snapshot, returning extracted cache state dicts.

        Returns None when persistence is disabled or the snapshot is absent /
        unreadable. Bumps last_access (LRU) on a hit.
        """
        if not self._prefix_persist or not prefix_hash:
            return None
        file_path = self._prefix_file_path(prefix_hash, token_count)
        with self._prefix_lock:
            entry = self._prefix_index.get(prefix_hash)
            if entry is not None and entry.token_count == token_count:
                entry.last_access = time.time()
        if not file_path.exists():
            logger.debug(
                "Prefix snapshot missing on disk: %s/%d",
                prefix_hash.hex()[:8],
                token_count,
            )
            return None
        try:
            data = mx.load(str(file_path), return_metadata=True)
            if isinstance(data, tuple) and len(data) == 2:
                arrays, metadata = data
            else:
                return None
            return self._reconstruct_from_safetensors(arrays, metadata)
        except Exception as e:
            logger.debug(
                "Failed to load prefix snapshot %s/%d: %s",
                prefix_hash.hex()[:8],
                token_count,
                e,
            )
            return None

    def get_prefix_stats(self) -> dict:
        if not self._prefix_persist:
            return {}
        with self._prefix_lock:
            return {
                **self._prefix_stats,
                "entries": len(self._prefix_index),
                "total_bytes": self._prefix_total_bytes,
                "max_bytes": self._prefix_max_bytes,
            }

    def cleanup_prefix_all(self) -> None:
        """Clear all prefix snapshots (manual reset / tests).

        Does NOT affect the in-session _snapshot_dir. Drains the prefix write
        queue and acquires _prefix_writer_busy so the writer is not mid-write.
        """
        if not self._prefix_persist:
            return
        if self._prefix_write_queue is not None:
            while True:
                try:
                    item = self._prefix_write_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    try:
                        self._prefix_write_queue.put_nowait(None)
                    except queue.Full:
                        pass
                    break
        acquired = self._prefix_writer_busy.acquire(timeout=self._CLEANUP_ALL_TIMEOUT_S)
        try:
            with self._prefix_lock:
                self._prefix_index.clear()
                self._prefix_total_bytes = 0
            if self._prefix_dir.exists():
                try:
                    shutil.rmtree(self._prefix_dir)
                except Exception as e:
                    logger.debug("Failed to clean up prefix snapshots: %s", e)
            self._prefix_dir.mkdir(parents=True, exist_ok=True)
        finally:
            if acquired:
                self._prefix_writer_busy.release()

    def _scan_prefix_dir(self) -> None:
        """Rebuild _prefix_index from on-disk prefix snapshots at startup.

        Cheap: parses filename (hex_hash, token_count) + stat() for size/mtime.
        Does NOT load tensor data. num_layers / model_name validated lazily at
        load_prefix time against the safetensors metadata.
        """
        if not self._prefix_dir.exists():
            return
        scanned = 0
        total = 0
        for path in self._prefix_dir.rglob("*.safetensors"):
            name = path.stem
            if "_" not in name:
                continue
            hex_hash, tc_str = name.rsplit("_", 1)
            try:
                token_count = int(tc_str)
                prefix_hash = bytes.fromhex(hex_hash)
            except ValueError:
                continue
            if len(prefix_hash) != 32:
                continue
            try:
                size_bytes = path.stat().st_size
                mtime = path.stat().st_mtime
            except OSError:
                continue
            entry = _PrefixEntry(
                prefix_hash=prefix_hash,
                token_count=token_count,
                file_path=path,
                size_bytes=size_bytes,
                last_access=mtime,
            )
            with self._prefix_lock:
                self._prefix_index[prefix_hash] = entry
                self._prefix_total_bytes += size_bytes
            scanned += 1
            total += size_bytes
        if scanned:
            logger.info(
                "Recovered %d prefix snapshots (%.2f MiB) from %s",
                scanned,
                total / (1024 * 1024),
                self._prefix_dir,
            )

    def _prefix_writer_loop(self) -> None:
        while not self._prefix_shutdown.is_set():
            try:
                item = self._prefix_write_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            with self._prefix_writer_busy:
                self._process_prefix_write_item(item)

    def _process_prefix_write_item(self, item) -> None:
        prefix_hash, token_count, tensors_raw, metadata, file_path = item
        temp_path = None
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")
            _write_safetensors_no_mx(str(temp_path), tensors_raw, metadata)
            os.rename(str(temp_path), str(file_path))
            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                size_bytes = 0
            now = time.time()
            with self._prefix_lock:
                existing = self._prefix_index.get(prefix_hash)
                if existing is not None and existing.token_count == token_count:
                    self._prefix_total_bytes -= existing.size_bytes
                entry = _PrefixEntry(
                    prefix_hash=prefix_hash,
                    token_count=token_count,
                    file_path=file_path,
                    size_bytes=size_bytes,
                    last_access=now,
                )
                self._prefix_index[prefix_hash] = entry
                self._prefix_total_bytes += size_bytes
                self._prefix_stats["writes"] += 1
            self._enforce_prefix_cap()
        except Exception as e:
            logger.debug("Prefix snapshot background write failed: %s", e)
            for p in (temp_path, file_path):
                try:
                    if p is not None and p.exists():
                        p.unlink()
                except Exception:
                    pass

    def _enforce_prefix_cap(self) -> None:
        """LRU-evict prefix snapshots until under the disk cap.

        O(N) scan per eviction (rare - only when over cap). Files unlinked
        outside the lock to avoid holding it during I/O.
        """
        if self._prefix_max_bytes <= 0:
            return
        victims: list[_PrefixEntry] = []
        with self._prefix_lock:
            while (
                self._prefix_total_bytes > self._prefix_max_bytes and self._prefix_index
            ):
                lru_hash = None
                lru_entry = None
                for h, e in self._prefix_index.items():
                    if lru_entry is None or e.last_access < lru_entry.last_access:
                        lru_hash = h
                        lru_entry = e
                if lru_entry is None:
                    break
                self._prefix_index.pop(lru_hash, None)
                self._prefix_total_bytes -= lru_entry.size_bytes
                self._prefix_stats["evictions"] += 1
                victims.append(lru_entry)
        for victim in victims:
            try:
                if victim.file_path.exists():
                    victim.file_path.unlink()
            except Exception as e:
                logger.debug("Failed to unlink evicted prefix snapshot: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_cancelled(self, request_id: str) -> bool:
        """Thread-safe check for cancellation."""
        with self._cancelled_lock:
            return request_id in self._cancelled_requests

    def _dec_cancelled(self, request_id: str) -> None:
        """Decrement cancelled counter under lock; remove entry when
        exhausted. Atomic read-modify-write closes the underflow race
        between two writer-thread iterations / cleanup_all clears."""
        with self._cancelled_lock:
            remaining = self._cancelled_requests.get(request_id, 0) - 1
            if remaining <= 0:
                self._cancelled_requests.pop(request_id, None)
            else:
                self._cancelled_requests[request_id] = remaining

    def _file_path(self, request_id: str, token_count: int) -> Path:
        return self._snapshot_dir / request_id / f"{token_count}.safetensors"

    def _writer_loop(self) -> None:
        """Background thread that writes safetensors files."""
        while not self._shutdown.is_set():
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # Sentinel
                break

            # Hold _writer_busy for the entire item's lifetime so
            # cleanup_all() can serialize with us — otherwise it can
            # rmtree the snapshot directory while we're mid-write and
            # we'd recreate ``req-X/`` underneath it, leaving an
            # orphaned file after the cleanup returns.
            with self._writer_busy:
                self._process_write_item(item)

    def _process_write_item(self, item) -> None:
        """Process one (pw_key, tensors_raw, metadata, file_path) queue item.

        Extracted from ``_writer_loop`` so the busy-lock can wrap it
        cleanly. Called only on the writer thread.
        """
        pw_key, tensors_raw, metadata, file_path = item

        # If cleanup_all or cleanup_request cleared this key from
        # _pending_writes while the item was in the writer's local hand
        # (i.e. between ``get()`` and entering ``with _writer_busy``),
        # treat the write as cancelled. This closes the late-rename
        # window where cleanup runs entirely between the writer's pull
        # and its busy-lock acquisition.
        with self._pending_lock:
            cleared_by_cleanup = pw_key not in self._pending_writes
        if cleared_by_cleanup:
            # If a timed-out cleanup_request bumped ``_cancelled_requests``
            # before clearing pending_writes, this item is one of the N
            # the counter is waiting on. Without this decrement the
            # counter would never reach zero, leaving the rid pinned in
            # ``_cancelled_requests`` for the process lifetime and
            # causing every subsequent write under that rid (or any
            # later reuse of the same string) to be silently discarded.
            if self._is_cancelled(pw_key[0]):
                self._dec_cancelled(pw_key[0])
            return

        # Skip writes for cancelled/cleaned-up requests.
        if self._is_cancelled(pw_key[0]):
            with self._pending_lock:
                self._pending_writes.pop(pw_key, None)
            try:
                req_dir = file_path.parent
                if req_dir.exists():
                    shutil.rmtree(req_dir)
            except Exception:
                logger.debug(
                    "swallowed exception at fusion_mlx/cache/boundary_snapshot_store.py:555"
                )

                pass
            self._dec_cancelled(pw_key[0])
            return

        temp_path = None
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")
            _write_safetensors_no_mx(str(temp_path), tensors_raw, metadata)

            # Request may have been cleaned up while serializing.
            if self._is_cancelled(pw_key[0]):
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    logger.debug(
                        "swallowed exception at fusion_mlx/cache/boundary_snapshot_store.py:572"
                    )

                    pass
                with self._pending_lock:
                    self._pending_writes.pop(pw_key, None)
                self._dec_cancelled(pw_key[0])
                return

            os.rename(str(temp_path), str(file_path))

            # Cleanup may race with a queued write; remove any late file.
            if self._is_cancelled(pw_key[0]):
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    logger.debug(
                        "swallowed exception at fusion_mlx/cache/boundary_snapshot_store.py:587"
                    )

                    pass
                req_dir = file_path.parent
                try:
                    if req_dir.exists():
                        shutil.rmtree(req_dir)
                except Exception:
                    logger.debug(
                        "swallowed exception at fusion_mlx/cache/boundary_snapshot_store.py:594"
                    )

                    pass
                self._dec_cancelled(pw_key[0])
        except Exception as e:
            logger.debug("Background snapshot write failed: %s", e)
            for p in (temp_path, file_path):
                try:
                    if p is not None and p.exists():
                        p.unlink()
                except Exception:
                    logger.debug(
                        "swallowed exception at fusion_mlx/cache/boundary_snapshot_store.py:604"
                    )

                    pass
            # Same bookkeeping invariant as the early-return path: if
            # cleanup_request bumped the counter and the failure was a
            # side-effect of that cleanup (e.g. its rmtree pulled the
            # parent dir out from under our temp write), we still owe
            # one decrement. The _is_cancelled rescue blocks above all
            # return before this except clause runs, so we cannot
            # double-decrement.
            if self._is_cancelled(pw_key[0]):
                self._dec_cancelled(pw_key[0])
        finally:
            # Remove extracted cache objects from pending writes to free
            # memory, but keep tensors_raw for read-back until file is on
            # disk.
            with self._pending_lock:
                pending = self._pending_writes.get(pw_key)
                if pending is not None:
                    pending.pop("extracted", None)
                # If file was written successfully, remove entirely.
                if file_path.exists():
                    self._pending_writes.pop(pw_key, None)

    def _serialize_extracted(
        self,
        extracted: list[dict[str, Any]],
        request_id: str,
        token_count: int,
    ) -> tuple[dict[str, tuple[bytes, str, list[int]]], dict[str, str]]:
        """Convert extracted cache states to tensors_raw + metadata.

        Must be called on the inference thread (for mx.eval / _extract_tensor_bytes).
        """
        arrays: dict[str, Any] = {}  # name -> mx.array
        layer_info: list[dict[str, str]] = []

        for i, layer_state in enumerate(extracted):
            class_name = layer_state.get("class_name", "KVCache")
            cache_type = layer_state.get("cache_type", "KVCache")
            meta_state = layer_state.get("meta_state", ())
            state = layer_state.get("state", ())

            info: dict[str, str] = {
                "class_name": class_name,
                "cache_type": cache_type,
                "meta_state": json.dumps(list(meta_state) if meta_state else []),
            }

            if (
                isinstance(state, list)
                and len(state) >= 1
                and all(isinstance(s, (list, tuple)) for s in state)
            ):
                # CacheList layer: ``state`` is a list of nested sub-state
                # tuples (one per sub-cache, e.g. RotatingKVCache +
                # PoolingCache for DeepSeek V4). Flatten as
                # ``layer_{i}_sub_{j}_state_{k}`` keys so reconstruction
                # can rebuild the nested shape.
                info["has_state"] = "true"
                info["sub_count"] = str(len(state))
                for j, sub_state in enumerate(state):
                    info[f"sub_{j}_count"] = str(len(sub_state))
                    for k, elem in enumerate(sub_state):
                        if not hasattr(elem, "shape"):
                            info[f"sub_{j}_missing_{k}"] = "1"
                            continue
                        if _has_zero_dim(elem.shape):
                            arrays[f"layer_{i}_sub_{j}_state_{k}"] = mx.zeros((1,))
                            info[f"sub_{j}_zero_dim_{k}"] = _encode_shape(elem.shape)
                        else:
                            arrays[f"layer_{i}_sub_{j}_state_{k}"] = elem
            elif isinstance(state, (list, tuple)) and len(state) >= 1:
                # Flat N-tuple state (KVCache, RotatingKVCache, PoolingCache,
                # BatchKVCache). Store every element under
                # ``layer_{i}_state_{k}`` regardless of tuple length.
                has_tensors = any(hasattr(elem, "shape") for elem in state)
                if has_tensors:
                    info["has_state"] = "true"
                    info["state_count"] = str(len(state))
                    for k, elem in enumerate(state):
                        if not hasattr(elem, "shape"):
                            # Non-tensor element (None, scalar). Mark it so
                            # _deserialize can restore the gap.
                            info[f"missing_{k}"] = "1"
                            continue
                        if _has_zero_dim(elem.shape):
                            arrays[f"layer_{i}_state_{k}"] = mx.zeros((1,))
                            info[f"zero_dim_{k}"] = _encode_shape(elem.shape)
                        else:
                            arrays[f"layer_{i}_state_{k}"] = elem
                else:
                    info["has_state"] = "false"
            else:
                info["has_state"] = "false"

            layer_info.append(info)

        # Materialize lazy tensors on inference thread.
        if arrays:
            mx.eval(*arrays.values())
            # Force Metal queue to finish before extracting bytes.
            # Without this the background writer can read un-initialized
            # or garbage buffers — MLX is lazy and eval() only submits
            # to the command queue, it does not wait for completion.
            mx.synchronize()

        # Extract raw bytes (Metal-safe memoryview copy).
        # CRITICAL: use bytes() to create an independent copy.  A bare
        # memoryview would keep the original MLX tensor alive (ref-count
        # anchor) until the writer thread finishes — causing temporary
        # memory spikes under high concurrency.
        tensors_raw = {}
        for name, arr in arrays.items():
            tensors_raw[name] = _extract_tensor_bytes(arr)

        metadata = {
            "request_id": request_id,
            "token_count": str(token_count),
            "num_layers": str(len(extracted)),
            "layer_info": json.dumps(layer_info),
        }

        return tensors_raw, metadata

    def _deserialize(
        self,
        tensors_raw: dict[str, tuple[bytes, str, list[int]]],
        metadata: dict[str, str],
    ) -> list[dict[str, Any]] | None:
        """Reconstruct extracted cache states from raw bytes + metadata."""
        try:
            num_layers = int(metadata["num_layers"])
            layer_info = json.loads(metadata["layer_info"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

        result: list[dict[str, Any]] = []
        for i in range(num_layers):
            info = layer_info[i] if i < len(layer_info) else {}
            class_name = info.get("class_name", "KVCache")
            cache_type = info.get("cache_type", "KVCache")
            meta_state_json = info.get("meta_state", "[]")
            try:
                meta_state = tuple(json.loads(meta_state_json))
            except (ValueError, json.JSONDecodeError):
                meta_state = ()

            if info.get("has_state") == "true":
                # V3 path: state_count meta + layer_{i}_state_{k} keys.
                # V2 fallback: legacy layer_{i}_0/1 + zero_dim_0/1 keys
                # for snapshots written before the N-tuple migration.
                state = self._read_state_tuple_raw(tensors_raw, info, i)
                result.append(
                    {
                        "state": state,
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )
            else:
                # Placeholder for skipped sliceable layers.
                result.append(
                    {
                        "state": (),
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )

        return result

    def _read_state_tuple_raw(
        self,
        tensors_raw: dict[str, tuple[bytes, str, list[int]]],
        info: dict[str, str],
        layer_idx: int,
    ) -> Any:
        """Read state for one layer from raw tensor bytes.

        Returns:
            - ``list`` of nested sub-state tuples for CacheList layers
                (``sub_count`` in info), or
            - ``tuple`` of N elements for flat layers (``state_count`` in
                info, V3 layout), or
            - 2-tuple from V2 polyfill (``layer_{i}_0`` / ``layer_{i}_1``).

        Missing elements come back as ``None``.
        """
        if "sub_count" in info:
            try:
                sub_count = int(info["sub_count"])
            except (ValueError, TypeError):
                return []
            sub_states: list[tuple[Any, ...]] = []
            for j in range(sub_count):
                count_key = f"sub_{j}_count"
                try:
                    count = int(info.get(count_key, "0"))
                except (ValueError, TypeError):
                    count = 0
                sub_elements: list[Any] = []
                for k in range(count):
                    if info.get(f"sub_{j}_missing_{k}") == "1":
                        sub_elements.append(None)
                        continue
                    key = f"layer_{layer_idx}_sub_{j}_state_{k}"
                    if key not in tensors_raw:
                        sub_elements.append(None)
                        continue
                    raw, dtype_str, shape = tensors_raw[key]
                    zd_marker = f"sub_{j}_zero_dim_{k}"
                    if zd_marker in info:
                        zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                        restored = _restore_tensor_from_bytes(raw, dtype_str, [1])
                        sub_elements.append(mx.zeros(zd_shape, dtype=restored.dtype))
                    else:
                        sub_elements.append(
                            _restore_tensor_from_bytes(raw, dtype_str, shape)
                        )
                sub_states.append(tuple(sub_elements))
            return sub_states

        if "state_count" in info:
            try:
                count = int(info["state_count"])
            except (ValueError, TypeError):
                return ()
            elements: list[Any] = []
            for k in range(count):
                if info.get(f"missing_{k}") == "1":
                    elements.append(None)
                    continue
                key = f"layer_{layer_idx}_state_{k}"
                if key not in tensors_raw:
                    elements.append(None)
                    continue
                raw, dtype_str, shape = tensors_raw[key]
                zd_marker = f"zero_dim_{k}"
                if zd_marker in info:
                    zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                    restored = _restore_tensor_from_bytes(raw, dtype_str, [1])
                    elements.append(mx.zeros(zd_shape, dtype=restored.dtype))
                else:
                    elements.append(_restore_tensor_from_bytes(raw, dtype_str, shape))
            return tuple(elements)

        # V2 polyfill — legacy 2-tuple snapshot.
        first = None
        second = None
        key_0 = f"layer_{layer_idx}_0"
        key_1 = f"layer_{layer_idx}_1"
        if key_0 in tensors_raw:
            raw, dtype_str, shape = tensors_raw[key_0]
            if "zero_dim_0" in info:
                zd_shape = tuple(int(d) for d in info["zero_dim_0"].split(","))
                first = _restore_tensor_from_bytes(raw, dtype_str, [1])
                first = mx.zeros(zd_shape, dtype=first.dtype)
            else:
                first = _restore_tensor_from_bytes(raw, dtype_str, shape)
        if key_1 in tensors_raw:
            raw, dtype_str, shape = tensors_raw[key_1]
            if "zero_dim_1" in info:
                zd_shape = tuple(int(d) for d in info["zero_dim_1"].split(","))
                second = _restore_tensor_from_bytes(raw, dtype_str, [1])
                second = mx.zeros(zd_shape, dtype=second.dtype)
            else:
                second = _restore_tensor_from_bytes(raw, dtype_str, shape)
        return (first, second) if first is not None else ()

    def _reconstruct_from_safetensors(
        self,
        arrays: dict[str, Any],
        metadata: dict[str, str],
    ) -> list[dict[str, Any]] | None:
        """Reconstruct from mx.load() result (arrays dict + metadata)."""
        try:
            num_layers = int(metadata["num_layers"])
            layer_info = json.loads(metadata["layer_info"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

        result: list[dict[str, Any]] = []
        for i in range(num_layers):
            info = layer_info[i] if i < len(layer_info) else {}
            class_name = info.get("class_name", "KVCache")
            cache_type = info.get("cache_type", "KVCache")
            meta_state_json = info.get("meta_state", "[]")
            try:
                meta_state = tuple(json.loads(meta_state_json))
            except (ValueError, json.JSONDecodeError):
                meta_state = ()

            if info.get("has_state") == "true":
                state = self._read_state_tuple_arrays(arrays, info, i)
                result.append(
                    {
                        "state": state,
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )
            else:
                result.append(
                    {
                        "state": (),
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )

        return result

    def _read_state_tuple_arrays(
        self,
        arrays: dict[str, Any],
        info: dict[str, str],
        layer_idx: int,
    ) -> Any:
        """N-tuple aware safetensors-loaded variant of
        ``_read_state_tuple_raw`` — sources tensors from a pre-decoded
        ``mx.array`` dict instead of raw bytes. Returns a list of nested
        tuples for CacheList layers (``sub_count`` in info) or a flat
        tuple otherwise.
        """
        if "sub_count" in info:
            try:
                sub_count = int(info["sub_count"])
            except (ValueError, TypeError):
                return []
            sub_states: list[tuple[Any, ...]] = []
            for j in range(sub_count):
                count_key = f"sub_{j}_count"
                try:
                    count = int(info.get(count_key, "0"))
                except (ValueError, TypeError):
                    count = 0
                sub_elements: list[Any] = []
                for k in range(count):
                    if info.get(f"sub_{j}_missing_{k}") == "1":
                        sub_elements.append(None)
                        continue
                    key = f"layer_{layer_idx}_sub_{j}_state_{k}"
                    tensor = arrays.get(key)
                    if tensor is None:
                        sub_elements.append(None)
                        continue
                    zd_marker = f"sub_{j}_zero_dim_{k}"
                    if zd_marker in info:
                        zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                        sub_elements.append(mx.zeros(zd_shape, dtype=tensor.dtype))
                    else:
                        sub_elements.append(tensor)
                sub_states.append(tuple(sub_elements))
            return sub_states

        if "state_count" in info:
            try:
                count = int(info["state_count"])
            except (ValueError, TypeError):
                return ()
            elements: list[Any] = []
            for k in range(count):
                if info.get(f"missing_{k}") == "1":
                    elements.append(None)
                    continue
                key = f"layer_{layer_idx}_state_{k}"
                tensor = arrays.get(key)
                if tensor is None:
                    elements.append(None)
                    continue
                zd_marker = f"zero_dim_{k}"
                if zd_marker in info:
                    zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                    elements.append(mx.zeros(zd_shape, dtype=tensor.dtype))
                else:
                    elements.append(tensor)
            return tuple(elements)

        # V2 polyfill.
        first = arrays.get(f"layer_{layer_idx}_0")
        second = arrays.get(f"layer_{layer_idx}_1")
        if "zero_dim_0" in info and first is not None:
            zd_shape = tuple(int(d) for d in info["zero_dim_0"].split(","))
            first = mx.zeros(zd_shape, dtype=first.dtype)
        if "zero_dim_1" in info and second is not None:
            zd_shape = tuple(int(d) for d in info["zero_dim_1"].split(","))
            second = mx.zeros(zd_shape, dtype=second.dtype)
        return (first, second) if first is not None else ()
