import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _VLMMTPDecodeState:
    """Per-request state for vlm_mtp decode that bypasses BatchGenerator.

    The wrapper generator yields plain Python ints (single-request mode).
    Scheduler iterates it one token per ``step()`` and feeds each token
    into ``_process_batch_responses`` via a synthesized ``_VLMMTPResponse``.
    """

    generator: Any   # Generator[int, None, None] from run_vlm_mtp_decode
    request: Any
    prompt_cache: list[Any]
    sampler: Callable[[Any], Any]
    state_machine: Any
    max_tokens: int
    # Plain stop-token set (EOS + request-specific) for direct membership
    # check; mlx-lm's SequenceStateMachine doesn't expose a "did the last
    # token finish" helper, so we keep a copy.
    stop_token_ids: set[int] = field(default_factory=set)
    emitted: int = 0
    finished: bool = False


@dataclass
class _VLMMTPResponse:
    """BatchGenerator.Response shim emitted by the vlm_mtp decode loop.

    Same field surface used by ``_process_batch_responses``: ``uid``,
    ``token``, ``finish_reason``, ``logprobs``, and an optional
    ``prompt_cache`` returned on the terminal yield so paged-cache reuse
    keeps working.
    """

    uid: int
    token: int
    finish_reason: str | None = None
    logprobs: Any = None
    prompt_cache: Any = None


# Serializes Metal buffer-protocol access from the async store-cache worker
# against inference-thread mx.clear_cache / mx.synchronize calls that can
# invalidate the underlying buffer pool. Closes a SIGABRT path where
# _async_store_cache_worker reads tensor bytes via memoryview while the
# inference thread concurrently issues a reclaim-triggering mx op.
# See: https://github.com/jundot/omlx/issues/1106
_mx_buffer_access_lock = threading.RLock()


class _StoreCacheGate:
    """Non-blocking counter that bounds in-flight store-cache submissions.

    Tracks how many KV caches are alive in the post-completion store-cache
    pipeline. _cleanup_finished records each submission with note_submitted()
    and the future's done callback clears it with note_done(); neither blocks
    the generation step. Backpressure is applied at admission instead —
    _schedule_waiting declines to admit new prefills while in_flight >= cap
    (see has_capacity), so token generation never stalls waiting for an SSD
    write (#1496).

    cap still bounds the concurrent extracted-KV count, which is the OOM
    guard for the burst-finish RAM growth reported in #1383. It is adjusted
    at runtime from ProcessMemoryEnforcer so the pipeline shrinks under
    memory pressure on smaller systems.
    """

    def __init__(self, cap: int) -> None:
        self._cap = max(1, cap)
        self._in_flight = 0
        self._lock = threading.Lock()

    def note_submitted(self) -> None:
        """Record a store-cache job handed to the executor (never blocks)."""
        with self._lock:
            self._in_flight += 1

    def note_done(self) -> None:
        """Record a store-cache job finished (future done callback)."""
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    def set_cap(self, cap: int) -> None:
        with self._lock:
            self._cap = max(1, cap)

    @property
    def cap(self) -> int:
        with self._lock:
            return self._cap

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def has_capacity(self) -> bool:
        """True when another submission would stay within cap.

        Read by _schedule_waiting to decide whether to admit a new prefill.
        """
        with self._lock:
            return self._in_flight < self._cap


class _PrefillAbortedError(Exception):
    """Raised when prefill is interrupted by a pending abort."""

    def __init__(self, aborted_uids: list[int], processed_tokens: int):
        self.aborted_uids = aborted_uids
        self.processed_tokens = processed_tokens
        super().__init__(
            f"Prefill aborted for UIDs {aborted_uids} " f"at {processed_tokens} tokens"
        )


@dataclass
class _PrefillState:
    """Intermediate state for a request undergoing chunked prefill.

    When chunked_prefill=True, a long prefill is spread across multiple
    step() calls (one prefill_step_size chunk per step). This dataclass
    holds all the state needed to resume prefill between steps.
    """

    request: Any
    cache: list   # Accumulated prompt_cache (mutated in-place by each chunk)
    tokens_remaining: Any   # mx.array shape (1, N) — tokens not yet prefilled
    last_token: list   # tokens[-1:] — passed to batch_generator.insert()
    tokens_processed: int   # Cumulative count for boundary snapshot math
    base_size: int   # Prefix cache offset at prefill start (for alignment)
    emitted_boundaries: dict   # {request_id: int} — last emitted boundary count
    boundary_enabled: bool   # Whether boundary snapshots are active
    block_size: int   # Copied from config.paged_cache_block_size
    total_length: int   # len(original tokens) for completeness
    # Pre-built insert-time params (set by _schedule_waiting before enqueuing)
    sampler: Any = None
    sm: Any = None
    per_row_lps: Any = None


class _BoundarySnapshotProvider:
    """Dict-like lazy loader for boundary snapshots.

    Used by ``store_cache()`` to load snapshots from SSD one block at a time
    instead of extracting all intermediate snapshots into memory at once.
    Implements ``__bool__``, ``__contains__``, and ``__getitem__`` to be a
    drop-in replacement for ``Dict[int, List[Dict[str, Any]]]``.
    """

    def __init__(
        self,
        store: Any,   # Optional[BoundarySnapshotSSDStore]
        request_id: str,
        valid_tcs: list[int],
        in_memory_snapshots: dict[int, Any],
        extract_fn: Any,   # Callable — Scheduler._extract_cache_states
    ) -> None:
        self._store = store
        self._request_id = request_id
        self._valid_tcs = set(valid_tcs)
        self._in_memory = in_memory_snapshots
        self._extract_fn = extract_fn

    def __contains__(self, tc: int) -> bool:
        return tc in self._valid_tcs

    def __getitem__(self, tc: int) -> Any:
        snap = self._in_memory.get(tc)
        if snap is not None:
            # In-memory fallback (SSD write failed).
            extracted, _ = self._extract_fn(snap)
            return extracted
        if self._store is not None:
            return self._store.load(self._request_id, tc)
        return None

    def __len__(self) -> int:
        return len(self._valid_tcs)

    def __bool__(self) -> bool:
        return bool(self._valid_tcs)
