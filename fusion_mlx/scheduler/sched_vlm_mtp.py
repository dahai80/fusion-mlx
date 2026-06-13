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

def _route_to_vlm_mtp(    self,
    request: Request,
    prefilled_cache: list[Any],
    last_tokens: list[int],
    sampler: Callable[[Any], Any],
    state_machine: Any,
) -> int | None:
    """Bypass BatchGenerator and stand up a vlm_mtp generator instead.

    Runs the final forward on ``last_tokens`` with ``return_hidden=True``
    and ``return_shared_kv=True`` so the drafter has the targets it
    needs, samples the first bonus token from the resulting logits, and
    returns a synthesized uid that ``step()`` will drive.

    Returns ``None`` if the eligibility check fails at the last second
    (drafter missing, language model lacks rollback hook, etc.) so the
    caller can fall back to the normal BatchGenerator path.
    """
    drafter = self._vlm_mtp_drafter
    if drafter is None:
        return None

     # Replaced: single-request gate swapped for batched MTP routing.
     # Eligible requests are accumulated into a batch and run together
     # via _mtp_rounds_batch, eliminating the serialization bottleneck.
    from .sched_vlm_mtp_batched import _vlm_mtp_try_enqueue
    return _vlm_mtp_try_enqueue(
        self, request, prefilled_cache, last_tokens, sampler, state_machine
       )


def _log_vlm_mtp_stats(
    self, state: "_VLMMTPDecodeState", finish_reason: str
) -> None:
    """Emit one INFO line per finished vlm_mtp request with the drafter
    acceptance rate measured for that request.

    Reads ``Gemma4AssistantDraftModel.accept_lens`` — a list of accepted
    draft counts per round, populated inside mlx-vlm's ``_mtp_rounds``.
    The drafter mutates this in place and ``reset()`` (called at the
    start of every new round-loop entry) clears it, so we have to read
    before the next eligible request lands. The serialized routing in
    ``_route_to_vlm_mtp`` guarantees one in-flight vlm_mtp generator
    at a time, so the value we read here belongs to ``state.request``.
    """
    drafter = self._vlm_mtp_drafter
    if drafter is None:
        return
    accept_lens = getattr(drafter.model, "accept_lens", None)
    if not accept_lens:
        return
    try:
        lens = [int(x) for x in accept_lens]
    except Exception:
        return
    rounds = len(lens)
    if rounds == 0:
        return
    total_accepted = sum(lens)
    block_size = self._vlm_mtp_draft_block_size or int(
        getattr(drafter.model.config, "block_size", 4)
    )
    max_per_round = max(1, block_size - 1)
    acceptance_rate = total_accepted / (rounds * max_per_round)
    avg_tokens_per_round = (total_accepted + rounds) / rounds
    logger.info(
        "vlm_mtp stats: request=%s finish=%s rounds=%d "
        "accepted=%d/%d (%.1f%%) tokens_per_round=%.2f "
        "emitted=%d block_size=%d",
        state.request.request_id,
        finish_reason,
        rounds,
        total_accepted,
        rounds * max_per_round,
        acceptance_rate * 100,
        avg_tokens_per_round,
        state.emitted,
        block_size,
    )

def _step_vlm_mtp(self) -> list[_VLMMTPResponse]:
    """Advance all active vlm_mtp batches by one yield."""
    from .sched_vlm_mtp_batched import _step_vlm_mtp_batched
    return _step_vlm_mtp_batched(self)

