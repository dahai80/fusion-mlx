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

from ..prefill_progress import get_prefill_tracker
from ..request import Request
from .helpers import (
    _sync_and_clear_cache,
)

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.


def _try_specprefill_scoring(    self, request: Request) -> None:
    """Score tokens with draft model if SpecPrefill is applicable.

    Uses paged SSD cache for the draft model: if the prompt prefix
    was already scored in a previous turn, the draft cache is restored
    and only the new suffix is prefilled through the draft model.
    """
    if self._specprefill_draft_model is None:
        return

    specprefill_enabled = getattr(request, "_specprefill_enabled", False)
    if not specprefill_enabled:
        return

    if request.vlm_inputs_embeds is not None:
        return

    remaining = request.remaining_tokens or request.prompt_token_ids
    if remaining is None:
        return

    n_remaining = len(remaining)
    from ..patches.specprefill import DEFAULT_KEEP_RATE, DEFAULT_THRESHOLD

    threshold = (
        getattr(request, "_specprefill_threshold", None) or DEFAULT_THRESHOLD
    )
    keep_pct = getattr(request, "_specprefill_keep_pct", None) or DEFAULT_KEEP_RATE

    # Threshold check on TOTAL remaining (not after system exclusion)
    if n_remaining <= threshold:
        return

    # System prompt protection: exclude system tokens from scoring.
    # If paged cache already covered the system prompt, remaining
    # won't include it (effective_system = 0).
    system_end = request.specprefill_system_end
    effective_system = max(0, system_end - request.cached_tokens)
    tokens_to_score = (
        remaining[effective_system:] if effective_system > 0 else remaining
    )
    n_to_score = len(tokens_to_score)

    # If conversation portion is below threshold after system exclusion,
    # skip SpecPrefill (system will be full-prefilled by normal path)
    if n_to_score <= threshold:
        return

    tracker = get_prefill_tracker()
    model_id = os.path.basename(self.config.model_name.rstrip("/"))

    try:
        from ..patches.specprefill import score_tokens, select_chunks

        # Draft prefix cache lookup
        draft_cache = None
        draft_cached_tokens = 0
        if self._draft_prefix_cache is not None:
            try:
                block_table, draft_remaining = self._draft_prefix_cache.fetch_cache(
                    request.request_id, tokens_to_score
                )
                if block_table and block_table.num_tokens > 0:
                    self._draft_prefix_cache.preload_blocks(block_table)
                    reconstructed = self._draft_prefix_cache.reconstruct_cache(
                        block_table
                    )
                    if reconstructed:
                        draft_cache = reconstructed
                        draft_cached_tokens = block_table.num_tokens
            except Exception as e:
                logger.debug(f"SpecPrefill: draft cache fetch failed: {e}")

        spec_extra = {
            "prompt_tokens": request.num_prompt_tokens,
            "system_tokens": request.specprefill_system_end,
            "conversation_tokens": request.num_prompt_tokens - request.specprefill_system_end,
            "cached_tokens": request.cached_tokens,
        }

        def _score_progress(processed: int, total: int, phase: str) -> None:
            tracker.update(
                request.request_id,
                min(processed, total - 1),
                total,
                model_id,
                phase=f"specprefill_{phase}",
                detail="scoring draft tokens",
                extra=spec_extra,
            )

        # Register tracker entry and stream draft scoring progress so the
        # dashboard shows movement during long SpecPrefill scoring pauses.
        tracker.update(
            request.request_id,
            0,
            n_to_score,
            model_id,
            phase="specprefill_scoring",
            detail="scoring draft tokens",
            extra=spec_extra,
        )

        t0 = time.monotonic()
        importance, used_cache = score_tokens(
            self._specprefill_draft_model,
            tokens_to_score,
            prefill_step_size=self.config.prefill_step_size,
            existing_cache=draft_cache,
            progress_callback=_score_progress,
        )
        selected = select_chunks(importance, keep_pct=keep_pct)
        t_score = time.monotonic() - t0

        n_selected = selected.shape[0]
        request.specprefill_indices = selected
        request.specprefill_total_tokens = n_to_score
        request.specprefill_position_offset = (
            request.cached_tokens + effective_system
        )
        request._specprefill_system_tokens = effective_system

        extras = []
        if draft_cached_tokens > 0:
            extras.append(f"draft cache hit {draft_cached_tokens}")
        total_prompt = request.num_prompt_tokens
        system_total = request.specprefill_system_end
        cached = request.cached_tokens
        extras.append(
            f"prompt {total_prompt} = "
            f"system {system_total} + conv {total_prompt - system_total}, "
            f"cached {cached}"
        )

        tracker.update(
            request.request_id,
            n_to_score - 1,
            n_to_score,
            model_id,
            phase="specprefill_selected",
            detail="selected sparse tokens",
            extra={
                **spec_extra,
                "scored_tokens": n_to_score,
                "selected_tokens": n_selected,
                "keep_percent": round(n_selected / n_to_score * 100),
            },
        )

        logger.info(
            f"SpecPrefill: scored {n_to_score} tokens in {t_score:.1f}s, "
            f"selected {n_selected}/{n_to_score} "
            f"(keep={n_selected/n_to_score*100:.0f}%, {', '.join(extras)})"
        )

        # Save draft cache for next turn
        if self._draft_prefix_cache is not None and used_cache is not None:
            try:
                extracted, mcc = self._extract_cache_states(used_cache)
                if extracted:
                    self._draft_prefix_cache.store_cache(
                        request.request_id,
                        tokens_to_score,
                        extracted,
                        model_cache_config=mcc,
                    )
            except Exception as e:
                logger.debug(f"SpecPrefill: draft cache store failed: {e}")

        # Free draft cache from memory.  Use _sync_and_clear_cache() so
        # the engine stream is drained before Metal buffers are
        # returned to the pool — a bare mx.clear_cache() here can race
        # with in-flight async evals and trigger a kernel panic (#557).
        del used_cache
        _sync_and_clear_cache(self._stream)

        # Mark scoring complete (auto-removes tracker entry).
        tracker.update(request.request_id, n_to_score, n_to_score, model_id)

    except Exception as e:
        logger.error(
            f"SpecPrefill scoring failed, falling back to normal path: {e}"
        )
        request.specprefill_indices = None
        tracker.remove(request.request_id)

def _cleanup_specprefill(    self, request_id: str) -> None:
    """Clean up SpecPrefill RoPE patches when a request finishes."""
    if self._specprefill_active_request_id == request_id:
        from ..patches.specprefill import cleanup_rope

        cleanup_rope(self.model)
        self._specprefill_active_request_id = None
        logger.debug(
            f"SpecPrefill: RoPE restored for finished request {request_id}"
        )
