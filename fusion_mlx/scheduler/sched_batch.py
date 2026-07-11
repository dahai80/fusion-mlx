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
import os
import time
from collections import deque
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from ..prefill_progress import get_prefill_tracker
from ..request import Request, RequestOutput, RequestStatus
from ..utils.proc_memory import get_phys_footprint
from .helpers import (
    _advance_vlm_extra,
    _cache_base_sizes,
    _prompt_cache_needs_snapshots,
    _slice_vlm_extra,
    _sync_and_clear_cache,
)
from .monkeypatches import _register_uid_rows

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .types import (
    _PrefillAbortedError,
    _PrefillState,
)


def _do_external_prefill(
    self,
    request: "Request",
    tokens: list[int],
    existing_cache: list[Any] | None,
    vlm_embeds: tuple[mx.array, dict[str, Any], int] | None = None,
) -> tuple[list[Any], list[int]]:
    """Run prefill externally (outside BatchGenerator) for a single request.

    Processes tokens[0:N-1] through the model. The last token tokens[N-1]
    is NOT processed here — it will be passed to BatchGenerator.insert()
    so that the first decode step produces the correct logit.

    Args:
        request: The request being prefilled.
        tokens: Full token list to prefill.
        existing_cache: Restored cache from paged SSD (or None).
        vlm_embeds: Optional (inputs_embeds, extra_kwargs, start_offset)
            tuple for VLM requests.

    Returns:
        (prefilled_cache, last_token_list) where last_token_list contains
        the single last token to pass to insert().

    Raises:
        _PrefillAbortedError: If prefill is interrupted by a pending abort.
        RuntimeError: If memory limit exceeded during prefill.
    """
    n_tokens = len(tokens)
    if n_tokens <= 1:
        # Nothing to prefill, return cache + tokens as-is
        cache = existing_cache or make_prompt_cache(self.model)
        if existing_cache is None and hasattr(self.model, "prealloc_caches"):
            self.model.prealloc_caches(cache)
        # NOTE: Do NOT apply TurboQuant here. TurboQuant conversion must
        # happen at insert() time in sched_schedule.py, after prefill
        # completes but before BatchGenerator takes ownership of the cache.
        # merge() blocker resolved via monkeypatches.py.
        return cache, tokens

    # Create or reuse cache
    if existing_cache is not None:
        prompt_cache = existing_cache
    else:
        prompt_cache = make_prompt_cache(self.model)
        if hasattr(self.model, "prealloc_caches"):
            self.model.prealloc_caches(prompt_cache)

    # NOTE: TurboQuant conversion is NOT applied during external prefill.
    # Prefill runs with standard KVCache; TurboQuant quantization happens
    # at insert() time (sched_schedule.py) after prefill completes.
    # merge() blocker resolved via monkeypatches.py.

    # Clear stale mRoPE position state for text-only requests.
    if vlm_embeds is None and hasattr(self.model, "clear_vlm_position_state"):
        # Cached text-only suffix: the prefill forward only runs on the
        # post-cache tokens, so the language model never recomputes the
        # full-sequence mRoPE delta. Capture the existing delta shape before
        # clear() resets it to None, then seed zeros in that shape so decode
        # resumes at the restored offset (text-only => no image offset).
        _seed_delta = None
        if getattr(request, "cached_tokens", 0) > 0:
            _lm = getattr(self.model, "_language_model", None)
            if _lm is not None and hasattr(_lm, "_rope_deltas"):
                _seed_delta = getattr(_lm, "_rope_deltas", None)
        self.model.clear_vlm_position_state()
        if _seed_delta is not None:
            _lm = getattr(self.model, "_language_model", None)
            if _lm is not None and hasattr(_lm, "_rope_deltas"):
                _lm._rope_deltas = mx.zeros_like(_seed_delta)
                logger.debug(
                    "Seeded zero mRoPE delta for cached text-only request %s",
                    request.request_id,
                )

    # Boundary snapshot setup
    block_size = self.config.paged_cache_block_size
    boundary_enabled = (
        block_size > 0
        and self.block_aware_cache is not None
        and _prompt_cache_needs_snapshots(prompt_cache)
    )
    all_boundaries = boundary_enabled  # always stop at every boundary for hybrid models
    base_size = _cache_base_sizes(prompt_cache) if boundary_enabled else 0
    # Sanity check: base_size from cache offsets should match the number
    # of tokens actually cached. A mismatch indicates stale meta_state
    # in a restored RotatingKVCache (e.g. shared layer_meta_states from
    # an earlier store_cache bug). Use cached_tokens which is always
    # derived from block_table.num_tokens and therefore trustworthy.
    if (
        boundary_enabled
        and hasattr(request, "cached_tokens")
        and request.cached_tokens > 0
    ):
        if base_size != request.cached_tokens:
            logger.debug(
                "Cache base_size mismatch: computed %d, expected %d "
                "(cached_tokens). Using cached_tokens for boundary "
                "alignment.",
                base_size,
                request.cached_tokens,
            )
            base_size = request.cached_tokens

    # Prepare VLM embeddings for prefill
    embeds_array: mx.array | None = None
    extra_kwargs: dict[str, Any] | None = None
    if vlm_embeds is not None:
        embeds_array, extra_kwargs, start_offset = vlm_embeds
        embeds_array = embeds_array[:, start_offset:]  # skip cached portion
        if start_offset > 0 and extra_kwargs:
            extra_kwargs = _advance_vlm_extra(extra_kwargs, start_offset)
        # Force _position_ids path in language model for cached VLM
        # prefill. Without this, the delta approach gives sequential
        # positions to image tokens that need 3D mRoPE positions.
        # Setting _rope_deltas=None makes the language model use
        # _position_ids (set by get_input_embeddings) instead.
        # Saved and restored after prefill for decode rope_deltas capture.
        # Only applies to mRoPE VLMs (Qwen2-VL, Qwen2.5-VL, GLM-4V, etc.);
        # non-mRoPE VLMs like Gemma 4 have no _rope_deltas attribute.
        _saved_rope_deltas = None
        if start_offset > 0:
            lm = getattr(self.model, "_language_model", None)
            if lm is not None and hasattr(lm, "_rope_deltas"):
                _saved_rope_deltas = lm._rope_deltas
                lm._rope_deltas = None

    # Prefill tokens[0:N-1] (leave last token for insert())
    prefill_tokens = tokens[:-1]
    last_token = tokens[-1:]
    total_length = len(tokens)

    input_arr = mx.array(prefill_tokens)[None]  # (1, seq_len)
    processed_tokens = 0
    prefill_step_size = self.config.prefill_step_size
    uid = self.request_id_to_uid.get(request.request_id)

    emitted_boundaries: dict[int, int] = {}

    while input_arr.shape[1] > 0:
        remaining = input_arr.shape[1]
        n_to_process = min(prefill_step_size, remaining)

        # Boundary-limited step size
        if boundary_enabled and block_size > 0:
            current_total = base_size + processed_tokens
            next_boundary = ((current_total // block_size) + 1) * block_size
            target_boundary_prefill = next_boundary - base_size
            delta = target_boundary_prefill - processed_tokens
            if delta > 0:
                n_to_process = min(n_to_process, delta)
            n_to_process = max(1, n_to_process)

        # Adaptive throttle: shrink chunk when entering the caution zone
        # so the hard cap is honored before the chunk-end check. Raises
        # RuntimeError if the min chunk would exceed the cap — the
        # #1405 cleanup path catches it and emits an error to the client.
        n_to_process = self._adaptive_chunk_size(
            n_to_process,
            request_id=request.request_id,
            loop_label="external",
        )

        model_kwargs: dict[str, Any] = {}
        if embeds_array is not None and embeds_array.shape[1] > 0:
            model_kwargs["inputs_embeds"] = embeds_array[:, :n_to_process]
            if extra_kwargs:
                model_kwargs["vlm_extra_kwargs"] = _slice_vlm_extra(
                    extra_kwargs, n_to_process
                )

        _throttle_pre = get_phys_footprint()
        self.model(input_arr[:, :n_to_process], cache=prompt_cache, **model_kwargs)
        mx.eval([c.state for c in prompt_cache])
        _throttle_post = get_phys_footprint()
        self._record_chunk_transient(
            n_to_process,
            _throttle_pre,
            _throttle_post,
            request_id=request.request_id,
            loop_label="external",
        )

        input_arr = input_arr[:, n_to_process:]
        if embeds_array is not None:
            embeds_array = embeds_array[:, n_to_process:]
            if extra_kwargs:
                extra_kwargs = _advance_vlm_extra(extra_kwargs, n_to_process)
        processed_tokens += n_to_process

        # Progress callback
        if uid is not None:
            self._on_prompt_progress([(uid, processed_tokens, total_length)])

        # Boundary snapshot emission
        if boundary_enabled:
            total_tokens = base_size + processed_tokens
            if (
                total_tokens > 0
                and total_tokens % block_size == 0
                and emitted_boundaries.get(request.request_id, -1) < total_tokens
            ):
                self._emit_prefill_boundary_snapshot(
                    request, prompt_cache, total_tokens
                )
                emitted_boundaries[request.request_id] = total_tokens

        # Memory monitoring — use max(active, phys_footprint) so MLX
        # cache pool and IOAccelerator-backed allocations that don't
        # show in mx.get_active_memory() still trigger the guard.
        # See utils/proc_memory.py for why phys_footprint matters.
        if self._memory_limit_bytes > 0:
            current = max(mx.get_active_memory(), get_phys_footprint())
            _hard = self._memory_hard_limit_bytes
            _soft = self._memory_limit_bytes
            # If over the soft watermark, clear MLX buffer cache and
            # re-measure to deflate stale cached-but-freed Metal
            # buffers (same fix as chunked prefill path).
            if current > _soft:
                _sync_and_clear_cache(self._stream)
                current = max(mx.get_active_memory(), get_phys_footprint())
                logger.debug(
                    "[memcheck:external] rid=%s n=%d processed=%d "
                    "current=%.3fGB soft=%.3fGB hard=%.3fGB %s "
                    "(post-clear)",
                    request.request_id,
                    n_to_process,
                    processed_tokens,
                    current / 1024**3,
                    _soft / 1024**3,
                    _hard / 1024**3,
                    "OVER_HARD" if _hard > 0 and current > _hard else "OVER_SOFT",
                )
            if (
                self._memory_hard_limit_bytes > 0
                and current > self._memory_hard_limit_bytes
            ):
                logger.warning(
                    f"Prefill force-stopped at {processed_tokens} "
                    f"tokens: memory {current / 1024**3:.1f}GB "
                    f"exceeds ceiling "
                    f"{self._memory_hard_limit_bytes / 1024**3:.1f}GB"
                )
                raise RuntimeError("Memory limit exceeded during prefill")
            elif current > self._memory_limit_bytes:
                logger.warning(
                    f"Prefill above max_bytes at "
                    f"{processed_tokens} tokens: "
                    f"{current / 1024**3:.1f}GB > "
                    f"{self._memory_limit_bytes / 1024**3:.1f}GB "
                    f"(ceiling: "
                    f"{self._memory_hard_limit_bytes / 1024**3:.1f}GB)"
                )

        # Check for pending aborts between prefill chunks.
        abort_uids = self._check_pending_aborts_for_uids(
            [uid] if uid is not None else []
        )
        if abort_uids:
            logger.info(
                f"Prefill interrupted at {processed_tokens}/"
                f"{total_length} tokens: "
                f"{len(abort_uids)} request(s) aborted"
            )
            raise _PrefillAbortedError(abort_uids, processed_tokens)

        # Reclaim Metal intermediates between prefill chunks.
        _sync_and_clear_cache(self._stream)

    # Emit final boundary snapshot if prompt lands exactly on boundary.
    if boundary_enabled:
        total_tokens = base_size + processed_tokens
        if (
            total_tokens > 0
            and total_tokens % block_size == 0
            and emitted_boundaries.get(request.request_id, -1) < total_tokens
        ):
            self._emit_prefill_boundary_snapshot(request, prompt_cache, total_tokens)

    _sync_and_clear_cache(self._stream)

    # Restore _rope_deltas after cached VLM prefill (for decode capture)
    if vlm_embeds is not None and _saved_rope_deltas is not None:
        self.model._language_model._rope_deltas = _saved_rope_deltas

    return prompt_cache, last_token


# ------------------------------------------------------------------
# Adaptive prefill throttle
# ------------------------------------------------------------------

# Discrete step sizes used by the watermark-based throttle. Each tier
# halves SDPA-fallback transient (∝ query_len × kv_len), so crossing
# one tier under memory pressure roughly doubles the available
# headroom for the next chunk's intermediates.
_PREFILL_STEP_TIERS: tuple[int, ...] = (1024, 512, 256, 128)


def _adaptive_chunk_size(
    self,
    requested: int,
    *,
    request_id: str,
    loop_label: str,
) -> int:
    """Shrink the next prefill chunk by bucketing how far current
    memory has crossed the soft watermark.

    The approach is intentionally measurement-free and model-agnostic.
    Once current memory passes the soft watermark
    (``max_bytes * prefill_safe_zone_ratio``, default 0.80) the chunk
    size drops in discrete tiers as we approach the hard cap. This is
    the auto equivalent of PR #1397's manual ``prefill_step_size``
    override — users do not pick a value, the scheduler picks one
    only when memory pressure shows up.

    Tiers (relative to soft → hard band):
    - current < soft watermark        → full chunk (no throttle)
    - first 25% of band               → 1024
    - 25%–50%                          → 512
    - 50%–75%                          → 256
    - 75%+                             → 128 (floor at min_chunk)

    The chunk-end memory check (``self._memory_hard_limit_bytes``
    comparison in the prefill loops) remains as the safety net: if
    memory still exceeds hard cap after this shrink, RuntimeError is
    raised and the #1405 cleanup path emits ``finish_reason="error"``
    to the client.

    Args:
        requested: The chunk size the caller would have used without
            throttle (already clamped by boundary alignment).
        request_id: For debug log correlation.
        loop_label: "external" or "chunked_step", used only for debug
            log identification.

    Returns:
        The chunk size to actually process (>= 1, <= requested).
    """
    soft_base = self._memory_limit_bytes
    hard_cap = self._memory_hard_limit_bytes
    if soft_base <= 0 or hard_cap <= 0 or requested <= 0:
        return requested

    current = max(mx.get_active_memory(), get_phys_footprint())
    soft_watermark = int(soft_base * self._prefill_safe_zone_ratio)

    if current < soft_watermark:
        return requested

    # Bucket by how far into the soft → hard band we are.
    band = max(hard_cap - soft_watermark, 1)
    over_ratio = max(0.0, min(1.0, (current - soft_watermark) / band))

    if over_ratio < 0.25:
        target = _PREFILL_STEP_TIERS[0]  # 1024
    elif over_ratio < 0.50:
        target = _PREFILL_STEP_TIERS[1]  # 512
    elif over_ratio < 0.75:
        target = _PREFILL_STEP_TIERS[2]  # 256
    else:
        target = _PREFILL_STEP_TIERS[3]  # 128

    target = max(target, self._prefill_min_chunk_tokens)
    if requested <= target:
        return requested

    logger.debug(
        "[throttle:%s] shrink rid=%s chunk %d -> %d "
        "(current=%.2fGB shrink_at=%.2fGB ceiling=%.2fGB band_ratio=%.2f)",
        loop_label,
        request_id,
        requested,
        target,
        current / 1024**3,
        soft_watermark / 1024**3,
        hard_cap / 1024**3,
        over_ratio,
    )
    return target


def _record_chunk_transient(
    self,
    n_tokens: int,
    pre_bytes: int,
    post_bytes: int,
    *,
    request_id: str,
    loop_label: str,
) -> None:
    """Feed one chunk's measured transient into the EWMA tracker."""
    delta = post_bytes - pre_bytes
    if delta <= 0:
        logger.debug(
            "[throttle:%s] measure rid=%s n=%d delta=%dB (skipped: <=0)",
            loop_label,
            request_id,
            n_tokens,
            delta,
        )
        return
    self._prefill_transient_tracker.update(n_tokens, delta)
    logger.debug(
        "[throttle:%s] measure rid=%s n=%d transient=%.2fMB "
        "per_token=%.1fKB ewma=%.1fKB samples=%d",
        loop_label,
        request_id,
        n_tokens,
        delta / 1024**2,
        (delta / max(n_tokens, 1)) / 1024,
        self._prefill_transient_tracker.bytes_per_token / 1024,
        self._prefill_transient_tracker.samples,
    )


# ------------------------------------------------------------------
# Chunked prefill helpers (used when config.chunked_prefill=True)
# ------------------------------------------------------------------


def _begin_prefill(
    self,
    request: "Request",
    tokens: list[int],
    existing_cache: "list[Any] | None",
) -> _PrefillState:
    """Initialise a _PrefillState for a non-VLM request.

    Performs all once-per-request setup (cache creation, boundary config,
    token splitting) without running any model forward passes.
    """
    if hasattr(self.model, "clear_vlm_position_state"):
        self.model.clear_vlm_position_state()

    prompt_cache = (
        existing_cache if existing_cache is not None else make_prompt_cache(self.model)
    )
    if existing_cache is None and hasattr(self.model, "prealloc_caches"):
        self.model.prealloc_caches(prompt_cache)

    block_size = self.config.paged_cache_block_size
    boundary_enabled = (
        block_size > 0
        and self.block_aware_cache is not None
        and _prompt_cache_needs_snapshots(prompt_cache)
    )
    base_size = _cache_base_sizes(prompt_cache) if boundary_enabled else 0
    if (
        boundary_enabled
        and hasattr(request, "cached_tokens")
        and request.cached_tokens > 0
        and base_size != request.cached_tokens
    ):
        logger.debug(
            "Cache base_size mismatch: computed %d, expected %d "
            "(cached_tokens). Using cached_tokens for boundary alignment.",
            base_size,
            request.cached_tokens,
        )
        base_size = request.cached_tokens

    prefill_tokens = tokens[:-1]
    last_token = tokens[-1:]
    input_arr = mx.array(prefill_tokens)[None]  # (1, N-1)

    self._prefill_chunk_count = getattr(self, "_prefill_chunk_count", 0)
    return _PrefillState(
        request=request,
        cache=prompt_cache,
        tokens_remaining=input_arr,
        last_token=last_token,
        tokens_processed=0,
        base_size=base_size,
        emitted_boundaries={},
        boundary_enabled=boundary_enabled,
        block_size=block_size,
        total_length=len(tokens),
    )


def _step_prefill_chunk(self, state: _PrefillState) -> bool:
    """Process one prefill chunk from *state*.

    Runs the model on at most prefill_step_size tokens, evals the cache,
    emits any due boundary snapshot, updates the prefill progress
    tracker, and clears Metal intermediates.

    Returns:
        True when all tokens_remaining have been consumed (prefill done).

    Raises:
        RuntimeError: If the hard memory limit is exceeded.
    """
    if state.tokens_remaining.shape[1] == 0:
        return True

    n = min(self.config.prefill_step_size, state.tokens_remaining.shape[1])

    # Clamp to the next block boundary so boundary snapshots fire exactly.
    if state.boundary_enabled and state.block_size > 0:
        current_total = state.base_size + state.tokens_processed
        next_boundary = ((current_total // state.block_size) + 1) * state.block_size
        delta = (next_boundary - state.base_size) - state.tokens_processed
        if delta > 0:
            n = min(n, delta)
        n = max(1, n)

    # Adaptive throttle — see _adaptive_chunk_size docstring. Raises
    # if even prefill_min_chunk_tokens would exceed the cap; #1405
    # cleanup paths in _schedule_waiting / _advance_chunked_prefills
    # convert that into a finish_reason="error" output for the client.
    n = self._adaptive_chunk_size(
        n,
        request_id=state.request.request_id,
        loop_label="chunked_step",
    )

    chunk = state.tokens_remaining[:, :n]
    state.tokens_remaining = state.tokens_remaining[:, n:]
    _throttle_pre = get_phys_footprint()
    self.model(chunk, cache=state.cache)
    mx.eval([c.state for c in state.cache])
    _throttle_post = get_phys_footprint()
    self._record_chunk_transient(
        n,
        _throttle_pre,
        _throttle_post,
        request_id=state.request.request_id,
        loop_label="chunked_step",
    )
    state.tokens_processed += n

    # Boundary snapshot
    if state.boundary_enabled:
        total_tokens = state.base_size + state.tokens_processed
        rid = state.request.request_id
        if (
            total_tokens > 0
            and total_tokens % state.block_size == 0
            and state.emitted_boundaries.get(rid, -1) < total_tokens
        ):
            self._emit_prefill_boundary_snapshot(
                state.request, state.cache, total_tokens
            )
            state.emitted_boundaries[rid] = total_tokens

    # Progress callback so the admin UI prefilling list advances during
    # chunked prefill. _do_external_prefill calls _on_prompt_progress
    # via the temp_uid mapping; the chunked path has no temp uid so we
    # talk to the tracker directly with the request_id.
    get_prefill_tracker().update(
        state.request.request_id,
        state.tokens_processed,
        state.total_length - 1,
        (
            os.path.basename(self.config.model_name.rstrip("/"))
            if self.config.model_name
            else ""
        ),
    )

    # Memory monitoring — use max(active, phys_footprint) so MLX cache
    # pool and IOAccelerator-backed allocations that don't show up in
    # mx.get_active_memory() still trigger the guard. Matches the
    # _do_external_prefill check; on macOS jetsam watches
    # phys_footprint, so the active-only check could miss the page
    # before the kernel kills us.
    #
    # Unlike the inline (external) prefill path which clears the MLX
    # buffer cache on every chunk, the chunked path defers clears to
    # boundary snapshots to avoid GPU stalls.  This means MLX's buffer
    # cache accumulates freed intermediates between clears, inflating
    # both mx.get_active_memory() and phys_footprint.  When we detect
    # the soft watermark has been crossed, we clear the cache and
    # re-measure so the hard/soft checks compare against actual
    # in-use memory, not stale cached buffers.
    if self._memory_limit_bytes > 0:
        current = max(mx.get_active_memory(), get_phys_footprint())
        _hard = self._memory_hard_limit_bytes
        _soft = self._memory_limit_bytes
        # If over the soft watermark, clear MLX buffer cache and
        # re-measure.  The inflated measurement from cached-but-freed
        # Metal buffers causes false-positive OOM warnings and
        # aborts (observed 103-118GB reported for a 27B model that
        # fits in ~40GB after cache reclaim).
        if current > _soft:
            _sync_and_clear_cache(self._stream)
            current = max(mx.get_active_memory(), get_phys_footprint())
            logger.debug(
                "[memcheck:chunked_step] rid=%s n=%d processed=%d/%d "
                "current=%.3fGB soft=%.3fGB hard=%.3fGB %s "
                "(post-clear)",
                state.request.request_id,
                n,
                state.tokens_processed,
                state.total_length - 1,
                current / 1024**3,
                _soft / 1024**3,
                _hard / 1024**3,
                "OVER_HARD" if _hard > 0 and current > _hard else "OVER_SOFT",
            )
        if (
            self._memory_hard_limit_bytes > 0
            and current > self._memory_hard_limit_bytes
        ):
            raise RuntimeError(
                f"Memory limit exceeded during chunked prefill at "
                f"{state.tokens_processed}/{state.total_length - 1} tokens: "
                f"{current / 1024**3:.1f}GB exceeds ceiling "
                f"{self._memory_hard_limit_bytes / 1024**3:.1f}GB"
            )
        elif current > self._memory_limit_bytes:
            logger.warning(
                f"Chunked prefill above max_bytes at "
                f"{state.tokens_processed} tokens: "
                f"{current / 1024**3:.1f}GB > "
                f"{self._memory_limit_bytes / 1024**3:.1f}GB "
                f"(ceiling: "
                f"{self._memory_hard_limit_bytes / 1024**3:.1f}GB)"
            )

    # Only sync+clear at boundary snapshot points or on the final
    # chunk.  Calling mx.synchronize() + mx.clear_cache() on every
    # chunk stalls the GPU pipeline — a 128k prompt at 2048-step
    # size would produce 64 full syncs.  Intermediate chunks already
    # have mx.eval(c.state) above to ensure states are materialized.
    # Note: the memory-monitoring block above may have already cleared
    # the cache if memory pressure was detected; this clear handles
    # the non-pressure path at boundaries.
    is_final = state.tokens_remaining.shape[1] == 0
    had_boundary_snapshot = (
        state.boundary_enabled
        and state.base_size + state.tokens_processed > 0
        and (state.base_size + state.tokens_processed) % state.block_size == 0
    )
    if is_final or had_boundary_snapshot:
        _sync_and_clear_cache(self._stream)
    return is_final


def _emit_final_boundary_if_needed(self, state: _PrefillState) -> None:
    """Emit a final boundary snapshot if the prefill landed on a boundary."""
    if not state.boundary_enabled:
        return
    total_tokens = state.base_size + state.tokens_processed
    rid = state.request.request_id
    if (
        total_tokens > 0
        and total_tokens % state.block_size == 0
        and state.emitted_boundaries.get(rid, -1) < total_tokens
    ):
        self._emit_prefill_boundary_snapshot(state.request, state.cache, total_tokens)


def _insert_prefilled_request(
    self,
    request: "Request",
    state: _PrefillState,
    scheduled: "list[Request]",
) -> None:
    """Insert a fully-prefilled request into BatchGenerator.

    Handles the batch_generator.insert() call, uid bookkeeping, and moving
    the request to self.running. Called from both the inline chunked path
    (first chunk completed immediately) and _advance_chunked_prefills()
    (last chunk completed across steps).

    Precondition: state.sampler, state.sm, state.per_row_lps are set.
    """
    if request.sampling_params.seed is not None:
        mx.random.seed(request.sampling_params.seed)

    per_row_lps = state.per_row_lps if state.per_row_lps is not None else []
    uids = self.batch_generator.insert(
        [state.last_token],
        max_tokens=[request.sampling_params.max_tokens],
        caches=[state.cache] if state.cache else None,
        samplers=[state.sampler],
        logits_processors=[per_row_lps],
        state_machines=[state.sm],
    )

    if uids:
        _register_uid_rows(self.model, uids, [state.sampler], [per_row_lps])
        uid = uids[0]
        self.request_id_to_uid[request.request_id] = uid
        self.uid_to_request_id[uid] = request.request_id
        now = time.monotonic()
        request.batch_uid = uid
        request.status = RequestStatus.RUNNING
        request.generation_started_at = now
        request.last_activity_at = now
        self.running[request.request_id] = request
        scheduled.append(request)

        if hasattr(self.model, "register_rope_delta"):
            self.model.register_rope_delta(uid, request.rope_deltas)

        self.total_prompt_tokens += request.num_prompt_tokens
        cache_info = (
            f", {request.cached_tokens} cached" if request.cached_tokens > 0 else ""
        )
        logger.debug(
            "Scheduled chunked-prefill request %s (uid=%d) "
            "with %d tokens (%d total)%s",
            request.request_id,
            uid,
            len(state.last_token),
            request.num_prompt_tokens,
            cache_info,
        )


def _advance_chunked_prefills(
    self,
    scheduled: "list[Request]",
    rejected: "list[RequestOutput]",
) -> None:
    """Process one prefill chunk per in-flight chunked-prefill request.

    Called at the start of each step() before _schedule_waiting(). Each
    call advances every request in self.prefilling by one prefill_step_size
    chunk. When a request's prefill completes it is inserted into
    BatchGenerator and moved to self.running.

    Args:
        scheduled: The step's running list of newly-scheduled requests;
            completed chunked-prefill requests are appended here.
        rejected: Per-step rejected outputs. A chunked prefill that hits
            the memory hard limit emits a finish_reason="error" entry
            here so the engine can surface the failure to the client.
    """
    if not self.prefilling:
        return

    still_prefilling: deque[Request] = deque()

    for request in self.prefilling:
        rid = request.request_id
        state = self._prefill_states.get(rid)

        # State missing means the request was aborted and cleaned up by
        # _do_abort_request() between steps — just skip it.
        if state is None:
            continue

        try:
            done = self._step_prefill_chunk(state)
        except _PrefillAbortedError:
            # Request aborted mid-chunk. Discard state; the abort will
            # be fully processed by _process_pending_aborts() next step.
            self._prefill_states.pop(rid, None)
            continue
        except RuntimeError as e:
            logger.error("Chunked prefill failed for %s: %s", rid, e)
            self._prefill_states.pop(rid, None)
            self._release_paged_cache_for_request(rid)
            self.requests.pop(rid, None)
            get_prefill_tracker().remove(rid)
            # Drop Metal cache pool buffers held by the aborted chunk's
            # forward / mx.eval transients. Without this, enforcer keeps
            # seeing the burst footprint until the next mx.clear_cache().
            _sync_and_clear_cache()
            # Surface the failure to the engine. Without this, the
            # request is silently dropped and the client hangs.
            rejected.append(
                RequestOutput(
                    request_id=rid,
                    finished=True,
                    finish_reason="error",
                    error=str(e),
                )
            )
            continue

        if not done:
            still_prefilling.append(request)
            continue

        # Prefill complete — emit final boundary snapshot and insert.
        self._prefill_states.pop(rid, None)
        self._emit_final_boundary_if_needed(state)
        _sync_and_clear_cache(self._stream)

        # Ensure a BatchGenerator exists (may not if all requests were
        # previously in chunked prefill with no running decode).
        self._ensure_batch_generator(request.sampling_params)
        if self.batch_generator is None:
            # Unlikely, but if BG creation fails put request back.
            logger.error(
                "BatchGenerator unavailable at chunked-prefill completion "
                "for %s; requeueing.",
                rid,
            )
            still_prefilling.append(request)
            self._prefill_states[rid] = state
            continue

        # Clean up the prefill-progress tracker entry.
        get_prefill_tracker().remove(rid)

        self._insert_prefilled_request(request, state, scheduled)

    self.prefilling = still_prefilling
