# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for FusionMLX continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import concurrent.futures
import logging
import time

logger = logging.getLogger(__name__)
from typing import Any

from ..prefill_progress import get_prefill_tracker
from ..request import Request, RequestOutput
from ..speculative.vlm_mtp import VLMMTPDrafter
from .types import _CacheFreshnessWait, _InflightStoreInfo

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.


def add_request(self, request: Request) -> None:
    """
    Add a new request to the scheduler.

    Raises SchedulerQueueFullError when the waiting queue is at or above
    the configured cap (max(max_num_seqs * 4, 32)). Server layer maps
    this to HTTP 503 + Retry-After.

    Args:
        request: The request to add
    """
    if request.request_id in self.requests:
        raise ValueError(f"Request {request.request_id} already exists")

    # Cap the waiting queue so client-side polling can't accumulate
    # unbounded work and the scheduler can apply backpressure via 503.
    max_waiting = max(self.config.max_num_seqs * 4, 32)
    if len(self.waiting) >= max_waiting:
        from ..exceptions import SchedulerQueueFullError

        raise SchedulerQueueFullError(
            current_depth=len(self.waiting),
            max_depth=max_waiting,
        )

    # Tokenize if needed
    if request.prompt_token_ids is None:
        if isinstance(request.prompt, str):
            request.prompt_token_ids = self.tokenizer.encode(request.prompt)
        else:
            request.prompt_token_ids = list(request.prompt)
        request.num_prompt_tokens = len(request.prompt_token_ids)

    # Cache freshness: if a store_cache is in flight, defer the prefix
    # lookup to _schedule_waiting (executor thread) so add_request (FastAPI
    # event loop) never races an in-flight store_cache and reads stale KV.
    # _should_defer registers a freshness wait for relevant stores (above
    # thresholds); either way prep is deferred to _schedule_waiting.
    if self._inflight_store_futures:
        self._should_defer_for_cache_freshness(request)
        request.remaining_tokens = request.prompt_token_ids
        logger.debug(
            "Deferring prefix-cache prep for %s to _schedule_waiting "
            "(in-flight store_cache present)",
            request.request_id,
        )
    elif self.block_aware_cache is not None:
        # Use paged cache
        block_table, remaining = self.block_aware_cache.fetch_cache(
            request.request_id,
            request.prompt_token_ids,
            extra_keys=request.vlm_extra_keys_for_cache,
            extra_key_token_start=request.vlm_extra_key_token_start_for_cache,
            extra_key_ranges=request.vlm_extra_key_ranges_for_cache,
        )
        if block_table and block_table.num_tokens > 0:
            bypass_hot_cache = self._bypass_hot_cache_under_pressure()
            if bypass_hot_cache:
                logger.info(
                    "Skipping hot-cache preload for %s under memory pressure",
                    request.request_id,
                )
            else:
                self.block_aware_cache.preload_blocks(block_table)
            # Reconstruct actual KVCache objects from stored tensor data
            # Note: reconstruct_cache may modify block_table in-place if
            # partial reconstruction occurs (some blocks invalid)
            original_tokens = block_table.num_tokens
            if bypass_hot_cache:
                reconstructed = self.block_aware_cache.reconstruct_cache(
                    block_table,
                    promote_to_hot_cache=False,
                )
            else:
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
            if reconstructed:
                request.prompt_cache = reconstructed
                request.block_table = block_table
                request.cached_tokens = block_table.num_tokens
                request.shared_prefix_blocks = len(block_table.block_ids)
                # Recalculate remaining_tokens in case block_table was truncated
                request.remaining_tokens = request.prompt_token_ids[
                    block_table.num_tokens :
                ]
                # For exact prefix hits we need cache state at (N-1) and the
                # last prompt token as input to produce the first decode logit.
                # Reusing cache state at N and feeding the last token again
                # shifts the model state and can change greedy output.
                if len(request.remaining_tokens) == 0 and request.cached_tokens > 0:
                    if self._cache_list_needs_boundary_snapshot(request.prompt_cache):
                        # Stateful non-sliceable caches (Rotating/Arrays)
                        # cannot be safely converted from N to N-1 state
                        # without cache-type-specific logic.
                        if self.paged_cache_manager is not None:
                            self.paged_cache_manager.delete_block_table(
                                request.request_id
                            )
                        request.prompt_cache = None
                        request.block_table = None
                        request.cached_tokens = 0
                        request.shared_prefix_blocks = 0
                        request.remaining_tokens = request.prompt_token_ids
                        logger.debug(
                            f"Request {request.request_id}: exact cache hit with "
                            f"stateful cache type, falling back to full prefill "
                            f"for deterministic kickoff"
                        )
                    elif self._trim_prompt_cache_for_generation(request.prompt_cache):
                        request.cached_tokens = max(0, request.cached_tokens - 1)
                        request.remaining_tokens = request.prompt_token_ids[-1:]
                        logger.debug(
                            f"Request {request.request_id}: exact cache hit adjusted "
                            f"to N-1 state for generation kickoff "
                            f"(cached_tokens={request.cached_tokens}, "
                            f"remaining={len(request.remaining_tokens)})"
                        )
                    else:
                        # Fallback to full recompute when cache layers cannot
                        # be safely trimmed by one token (e.g., non-trimmable
                        # recurrent state caches).
                        if self.paged_cache_manager is not None:
                            self.paged_cache_manager.delete_block_table(
                                request.request_id
                            )
                        request.prompt_cache = None
                        request.block_table = None
                        request.cached_tokens = 0
                        request.shared_prefix_blocks = 0
                        request.remaining_tokens = request.prompt_token_ids
                        logger.debug(
                            f"Request {request.request_id}: exact cache hit could "
                            f"not be trimmed safely, falling back to full prefill"
                        )
                if block_table.num_tokens < original_tokens:
                    logger.debug(
                        f"Request {request.request_id}: partial cache hit, "
                        f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks "
                        f"(originally {original_tokens} tokens), "
                        f"{len(request.remaining_tokens)} tokens remaining"
                    )
                else:
                    logger.debug(
                        f"Request {request.request_id}: paged cache hit, "
                        f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks, "
                        f"{len(request.remaining_tokens)} tokens remaining, cache reconstructed"
                    )
            else:
                # Reconstruction failed, treat as cache miss
                if self.paged_cache_manager is not None:
                    self.paged_cache_manager.delete_block_table(request.request_id)
                request.remaining_tokens = request.prompt_token_ids
                logger.debug(
                    f"Request {request.request_id}: paged cache reconstruction failed, "
                    "released shared blocks"
                )
        else:
            request.remaining_tokens = request.prompt_token_ids
    else:
        # No paged SSD cache configured - process all tokens
        request.remaining_tokens = request.prompt_token_ids

    # SpecPrefill scoring is deferred to _schedule_waiting (executor thread).
    # Scoring is a full draft-model forward pass (10-30s for 128k tokens)
    # and must not block the FastAPI event loop.

    # Add to tracking
    self.requests[request.request_id] = request
    self.waiting.append(request)

    logger.debug(
        f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
    )


def set_specprefill_draft_model(
    self, draft_model: Any, draft_model_name: str | None = None
) -> None:
    """Set the draft model for SpecPrefill scoring.

    Creates a separate BlockAwarePrefixCache for the draft model
    using the existing paged SSD cache infrastructure. The model_name
    in compute_block_hash() naturally isolates draft blocks from target.
    """
    self._specprefill_draft_model = draft_model
    self._draft_prefix_cache: Any | None = None

    if (
        self.paged_cache_manager is not None
        and self.paged_ssd_cache_manager is not None
    ):
        try:
            from ..cache.paged_cache import PagedCacheManager
            from ..cache.prefix_cache import BlockAwarePrefixCache

            name = draft_model_name or "specprefill-draft"
            draft_paged = PagedCacheManager(
                block_size=self.config.paged_cache_block_size,
                max_blocks=self.paged_cache_manager.max_blocks,
                model_name=name,
            )
            self._draft_prefix_cache = BlockAwarePrefixCache(
                model=draft_model,
                paged_cache_manager=draft_paged,
                paged_ssd_cache_manager=self.paged_ssd_cache_manager,
            )
            self._draft_prefix_cache.set_cold_restore_callback(
                self._restore_block_from_cold
            )
            logger.info(
                f"SpecPrefill: draft model set with SSD cache (model_name={name})"
            )
        except Exception as e:
            logger.warning(f"SpecPrefill: draft SSD cache setup failed: {e}")
            logger.info("SpecPrefill: draft model set (no SSD cache)")
    else:
        logger.info("SpecPrefill: draft model set (no SSD cache)")


def set_vlm_mtp_drafter(
    self,
    drafter: VLMMTPDrafter | None,
    draft_block_size: int | None = None,
) -> None:
    """Attach a gemma4_assistant drafter for VLM MTP speculative decode.

    Called by ``VLMBatchedEngine.set_vlm_mtp_drafter`` once the assistant
    artifact is loaded. ``None`` clears the toggle.
    """
    self._vlm_mtp_drafter = drafter
    self._vlm_mtp_draft_block_size = draft_block_size
    if drafter is not None:
        logger.info(
            "VLM MTP drafter attached to scheduler (block_size=%s)",
            draft_block_size,
        )


# ── Admission control constants ──────────────────────────────────────────

_MEMORY_ADMISSION_STALL_TIMEOUT_S: float = 60.0
_STORE_CACHE_ADMISSION_STALL_TIMEOUT_S: float = 60.0
_CACHE_FRESHNESS_WAIT_MIN_PROMPT_TOKENS = 8192
_CACHE_FRESHNESS_WAIT_MIN_COMMON_TOKENS = 8192
_CACHE_FRESHNESS_WAIT_MIN_PROMPT_RATIO = 0.30
_CACHE_FRESHNESS_WAIT_TIMEOUT_S = 4.0


# ── Admission blocker helpers ────────────────────────────────────────────


def _clear_memory_admission_blocker(self, request_id: str | None = None) -> None:
    if (
        request_id is not None
        and request_id != self._memory_admission_blocked_request_id
    ):
        return
    self._memory_admission_blocked_request_id = None
    self._memory_admission_blocked_since = 0.0


def _clear_store_cache_admission_blocker(self, request_id: str | None = None) -> None:
    if (
        request_id is not None
        and request_id != self._store_cache_admission_blocked_request_id
    ):
        return
    self._store_cache_admission_blocked_request_id = None
    self._store_cache_admission_blocked_since = 0.0


def _clear_request_admission_bookkeeping(self, request_id: str) -> None:
    self._cache_freshness_waits.pop(request_id, None)
    self._prefix_cache_prepared.discard(request_id)
    self._clear_memory_admission_blocker(request_id)
    self._clear_store_cache_admission_blocker(request_id)


def _memory_admission_stall_output(self, reason: str) -> RequestOutput | None:
    if not self.waiting:
        self._clear_memory_admission_blocker()
        return None

    request = self.waiting[0]
    request_id = request.request_id
    now = time.monotonic()
    if request_id != self._memory_admission_blocked_request_id:
        self._memory_admission_blocked_request_id = request_id
        self._memory_admission_blocked_since = now
        return None

    timeout = getattr(
        self,
        "_MEMORY_ADMISSION_STALL_TIMEOUT_S",
        _MEMORY_ADMISSION_STALL_TIMEOUT_S,
    )
    if now - self._memory_admission_blocked_since < timeout:
        return None

    stalled_for = now - self._memory_admission_blocked_since
    self.waiting.popleft()
    self._release_paged_cache_for_request(request_id)
    self.requests.pop(request_id, None)
    self._clear_request_admission_bookkeeping(request_id)
    get_prefill_tracker().remove(request_id)
    self._clear_memory_admission_blocker(request_id)

    message = (
        "Request could not be admitted because memory pressure persisted "
        f"for {stalled_for:.1f}s ({reason}). Reduce context length, free "
        "memory, lower hot_cache_max_size, or loosen memory_guard_tier."
    )
    logger.warning("Memory admission stalled for %s: %s", request_id, message)
    return RequestOutput(
        request_id=request_id,
        finished=True,
        finish_reason="error",
        error=message,
        error_code="memory_admission_stalled",
        error_metadata={
            "request_id": request_id,
            "reason": reason,
            "stalled_seconds": int(stalled_for),
        },
    )


def _store_cache_admission_stall_output(
    self,
    reason: str,
    *,
    gate_in_flight: int,
    gate_cap: int,
    pending_cleanups: int,
) -> RequestOutput | None:
    if not self.waiting:
        self._clear_store_cache_admission_blocker()
        return None

    request = self.waiting[0]
    request_id = request.request_id
    now = time.monotonic()
    if request_id != self._store_cache_admission_blocked_request_id:
        self._store_cache_admission_blocked_request_id = request_id
        self._store_cache_admission_blocked_since = now
        return None

    timeout = getattr(
        self,
        "_STORE_CACHE_ADMISSION_STALL_TIMEOUT_S",
        _STORE_CACHE_ADMISSION_STALL_TIMEOUT_S,
    )
    if now - self._store_cache_admission_blocked_since < timeout:
        return None

    stalled_for = now - self._store_cache_admission_blocked_since
    self.waiting.popleft()
    self._release_paged_cache_for_request(request_id)
    self.requests.pop(request_id, None)
    self._clear_request_admission_bookkeeping(request_id)
    get_prefill_tracker().remove(request_id)

    message = (
        "Request could not be admitted because store-cache cleanup stayed "
        f"full for {stalled_for:.1f}s ({reason}). The previous response "
        "cache is still being persisted; retry after the cache writer drains "
        "or reduce cache/write pressure."
    )
    logger.warning(
        "Store-cache admission stalled for %s: %s "
        "(in_flight=%d pending_cleanups=%d cap=%d)",
        request_id,
        message,
        gate_in_flight,
        pending_cleanups,
        gate_cap,
    )
    return RequestOutput(
        request_id=request_id,
        finished=True,
        finish_reason="error",
        error=message,
        error_code="store_cache_admission_stalled",
        error_metadata={
            "request_id": request_id,
            "reason": reason,
            "stalled_seconds": int(stalled_for),
            "store_cache_in_flight": gate_in_flight,
            "pending_store_cleanups": pending_cleanups,
            "store_cache_cap": gate_cap,
        },
    )


# ── Cache freshness helpers ──────────────────────────────────────────────


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _store_extra_keys_match(
    info: _InflightStoreInfo,
    request: Request,
) -> bool:
    return (
        info.extra_keys == request.vlm_extra_keys_for_cache
        and info.extra_key_token_start == request.vlm_extra_key_token_start_for_cache
        and info.extra_key_ranges == request.vlm_extra_key_ranges_for_cache
    )


def _find_relevant_inflight_store(
    self,
    request: Request,
) -> tuple[str, concurrent.futures.Future, int] | None:
    if not self._inflight_store_futures:
        return None

    prompt = request.prompt_token_ids or []
    if len(prompt) < _CACHE_FRESHNESS_WAIT_MIN_PROMPT_TOKENS:
        return None

    best_rid: str | None = None
    best_future: concurrent.futures.Future | None = None
    best_common = 0
    for rid, future in list(self._inflight_store_futures.items()):
        if future.done():
            continue
        info = self._inflight_store_info.get(rid)
        if info is None or not _store_extra_keys_match(info, request):
            continue

        common = _common_prefix_len(prompt, info.tokens)
        if common > best_common:
            best_rid = rid
            best_future = future
            best_common = common

    if best_future is None or best_rid is None:
        return None
    if not (
        best_common >= _CACHE_FRESHNESS_WAIT_MIN_COMMON_TOKENS
        or best_common / len(prompt) >= _CACHE_FRESHNESS_WAIT_MIN_PROMPT_RATIO
    ):
        return None

    return best_rid, best_future, best_common


def _should_defer_for_cache_freshness(self, request: Request) -> bool:
    if self.block_aware_cache is None:
        return False

    now = time.monotonic()
    wait = self._cache_freshness_waits.get(request.request_id)
    if wait is not None:
        if wait.future.done():
            self._cache_freshness_waits.pop(request.request_id, None)
            try:
                exc = wait.future.exception()
            except concurrent.futures.CancelledError:
                logger.debug(
                    "Cache freshness deferral saw cancelled store_cache %s "
                    "before prefix lookup for %s",
                    wait.store_request_id,
                    request.request_id,
                )
            else:
                if exc is None:
                    logger.debug(
                        "Completed cache freshness deferral for store_cache %s "
                        "before prefix lookup for %s",
                        wait.store_request_id,
                        request.request_id,
                    )
                else:
                    logger.debug(
                        "Cache freshness deferral saw failed store_cache %s "
                        "before prefix lookup for %s: %s",
                        wait.store_request_id,
                        request.request_id,
                        exc,
                    )
            return False
        if now >= wait.deadline_s:
            self._cache_freshness_waits.pop(request.request_id, None)
            logger.debug(
                "Timed out cache freshness deferral for store_cache %s before "
                "prefix lookup for %s (common_prefix=%d/%d)",
                wait.store_request_id,
                request.request_id,
                wait.common_prefix,
                wait.prompt_len,
            )
            return False
        return True

    match = self._find_relevant_inflight_store(request)
    if match is None:
        return False

    store_request_id, future, common_prefix = match
    prompt_len = len(request.prompt_token_ids or [])
    timeout = _CACHE_FRESHNESS_WAIT_TIMEOUT_S
    self._cache_freshness_waits[request.request_id] = _CacheFreshnessWait(
        store_request_id=store_request_id,
        future=future,
        common_prefix=common_prefix,
        prompt_len=prompt_len,
        deadline_s=now + timeout,
    )

    logger.debug(
        "Deferring admission up to %.1fs for in-flight store_cache %s before "
        "prefix lookup for %s (common_prefix=%d/%d running=%d prefilling=%d)",
        timeout,
        store_request_id,
        request.request_id,
        common_prefix,
        prompt_len,
        len(self.running),
        len(self.prefilling),
    )
    return True


def _prepare_prefix_cache_for_request(self, request: Request) -> None:
    if request.request_id in self._prefix_cache_prepared:
        return

    if self.block_aware_cache is not None:
        block_table, remaining = self.block_aware_cache.fetch_cache(
            request.request_id,
            request.prompt_token_ids,
            extra_keys=request.vlm_extra_keys_for_cache,
            extra_key_token_start=request.vlm_extra_key_token_start_for_cache,
            extra_key_ranges=request.vlm_extra_key_ranges_for_cache,
        )
        if block_table and block_table.num_tokens > 0:
            bypass_hot_cache = self._bypass_hot_cache_under_pressure()
            if bypass_hot_cache:
                logger.info(
                    "Skipping hot-cache preload for %s under memory pressure",
                    request.request_id,
                )
            else:
                self.block_aware_cache.preload_blocks(block_table)
            original_tokens = block_table.num_tokens
            if bypass_hot_cache:
                reconstructed = self.block_aware_cache.reconstruct_cache(
                    block_table,
                    promote_to_hot_cache=False,
                )
            else:
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
            if reconstructed:
                request.prompt_cache = reconstructed
                request.block_table = block_table
                request.cached_tokens = block_table.num_tokens
                request.shared_prefix_blocks = len(block_table.block_ids)
                request.remaining_tokens = request.prompt_token_ids[
                    block_table.num_tokens :
                ]
                if len(request.remaining_tokens) == 0 and request.cached_tokens > 0:
                    if self._cache_list_needs_boundary_snapshot(request.prompt_cache):
                        if self.paged_cache_manager is not None:
                            self.paged_cache_manager.delete_block_table(
                                request.request_id
                            )
                        request.prompt_cache = None
                        request.block_table = None
                        request.cached_tokens = 0
                        request.shared_prefix_blocks = 0
                        request.remaining_tokens = request.prompt_token_ids
                        logger.debug(
                            f"Request {request.request_id}: exact cache hit with "
                            f"stateful cache type, falling back to full prefill "
                            f"for deterministic kickoff"
                        )
                    elif self._trim_prompt_cache_for_generation(request.prompt_cache):
                        request.cached_tokens = max(0, request.cached_tokens - 1)
                        request.remaining_tokens = request.prompt_token_ids[-1:]
                        logger.debug(
                            f"Request {request.request_id}: exact cache hit adjusted "
                            f"to N-1 state for generation kickoff "
                            f"(cached_tokens={request.cached_tokens}, "
                            f"remaining={len(request.remaining_tokens)})"
                        )
                    else:
                        if self.paged_cache_manager is not None:
                            self.paged_cache_manager.delete_block_table(
                                request.request_id
                            )
                        request.prompt_cache = None
                        request.block_table = None
                        request.cached_tokens = 0
                        request.shared_prefix_blocks = 0
                        request.remaining_tokens = request.prompt_token_ids
                        logger.debug(
                            f"Request {request.request_id}: exact cache hit could "
                            f"not be trimmed safely, falling back to full prefill"
                        )
                if block_table.num_tokens < original_tokens:
                    logger.debug(
                        f"Request {request.request_id}: partial cache hit, "
                        f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks "
                        f"(originally {original_tokens} tokens), "
                        f"{len(request.remaining_tokens)} tokens remaining"
                    )
                else:
                    logger.debug(
                        f"Request {request.request_id}: paged cache hit, "
                        f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks, "
                        f"{len(request.remaining_tokens)} tokens remaining, cache reconstructed"
                    )
            else:
                if self.paged_cache_manager is not None:
                    self.paged_cache_manager.delete_block_table(request.request_id)
                request.remaining_tokens = request.prompt_token_ids
                logger.debug(
                    f"Request {request.request_id}: paged cache reconstruction failed, "
                    "released shared blocks"
                )
        else:
            request.remaining_tokens = request.prompt_token_ids
    else:
        request.remaining_tokens = request.prompt_token_ids

    self._try_specprefill_scoring(request)
    self._prefix_cache_prepared.add(request.request_id)


def _refresh_generation_overflow_recovery_ids(self) -> None:
    if not self._generation_overflow_recovery_ids:
        return
    active_ids = set(self.running)
    active_ids.update(request.request_id for request in self.waiting)
    active_ids.update(request.request_id for request in self.prefilling)
    self._generation_overflow_recovery_ids.intersection_update(active_ids)


def _effective_max_num_seqs(self) -> int:
    self._refresh_generation_overflow_recovery_ids()
    if self._serialize_llama4_requests or self._generation_overflow_recovery_ids:
        return 1
    return max(1, self.config.max_num_seqs)


def _num_admitted_requests(self) -> int:
    return len(self.running) + len(self.prefilling)
