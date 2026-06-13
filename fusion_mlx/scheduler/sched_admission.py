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
from typing import Any

from ..request import Request
from ..speculative.vlm_mtp import VLMMTPDrafter

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.


def add_request(    self, request: Request) -> None:
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

    # Check prefix cache for cached KV state
    if self.block_aware_cache is not None:
        # Use paged cache
        block_table, remaining = self.block_aware_cache.fetch_cache(
            request.request_id,
            request.prompt_token_ids,
            extra_keys=request.vlm_extra_keys_for_cache,
            extra_key_token_start=request.vlm_extra_key_token_start_for_cache,
            extra_key_ranges=request.vlm_extra_key_ranges_for_cache,
        )
        if block_table and block_table.num_tokens > 0:
            self.block_aware_cache.preload_blocks(block_table)
            # Reconstruct actual KVCache objects from stored tensor data
            # Note: reconstruct_cache may modify block_table in-place if
            # partial reconstruction occurs (some blocks invalid)
            original_tokens = block_table.num_tokens
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
                    if self._cache_list_needs_boundary_snapshot(
                        request.prompt_cache
                    ):
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
                    elif self._trim_prompt_cache_for_generation(
                        request.prompt_cache
                    ):
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

def set_vlm_mtp_drafter(    self,
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
