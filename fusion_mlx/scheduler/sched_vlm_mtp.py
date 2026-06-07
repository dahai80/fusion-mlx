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

def _route_to_vlm_mtp(
    sched,
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
    drafter = sched._vlm_mtp_drafter
    if drafter is None:
        return None

    # Gemma4AssistantDraftModel keeps ``_shared_kv`` / ``_input_embed`` on
    # the module instance, so multiple in-flight ``_mtp_rounds`` generators
    # share one drafter and effectively serialize on it: each round has
    # to ``set_shared_kv`` for its own request before ``draft_block`` runs.
    # Output stays correct because target-side verify is the source of
    # truth in speculative decoding (a stale-drafter round just rejects
    # everything and falls back to a target-only step), but the
    # per-request tok/s is roughly halved under concurrency. Empirically
    # at 4 concurrent, vlm_mtp gives ~14 tok/s each vs BatchGenerator's
    # ~27 tok/s each — BG's batched matmul beats serialized speculative
    # rounds. So we route only the first eligible request through
    # vlm_mtp and let subsequent concurrent requests fall back. A future
    # commit can swap this gate for true batched MTP via
    # ``_mtp_rounds_batch`` if and when omlx prefill exposes batched
    # hidden/shared_kv outputs.
    if sched._vlm_mtp_active:
        logger.info(
            "vlm_mtp routing skipped for %s: drafter is busy with %d "
            "request(s); falling back to BatchGenerator",
            request.request_id,
            len(sched._vlm_mtp_active),
        )
        return None

    lm = getattr(sched.model, "_language_model", None)
    if lm is None or not hasattr(lm, "rollback_speculative_cache"):
        logger.warning(
            "vlm_mtp toggle on but model lacks _language_model with "
            "rollback_speculative_cache (model=%s); falling back to "
            "standard decode for request %s",
            type(sched.model).__name__,
            request.request_id,
        )
        return None

    if not last_tokens:
        logger.warning(
            "vlm_mtp routing skipped: last_tokens empty for request %s",
            request.request_id,
        )
        return None

    last_arr = mx.array(last_tokens)[None]  # (1, len_last)
    try:
        with mx.stream(sched._stream):
            out = lm(
                last_arr,
                cache=prefilled_cache,
                return_hidden=True,
                return_shared_kv=True,
            )
            mx.eval([c.state for c in prefilled_cache])
    except Exception as e:
        logger.warning(
            "vlm_mtp final-prefill forward failed (%s); falling back "
            "to standard decode for request %s",
            e,
            request.request_id,
        )
        return None

    logits = out.logits[:, -1, :]
    first_bonus_arr = sampler(logits)  # mx.array shape [1]
    mx.eval(first_bonus_arr)

    hidden_states = out.hidden_states
    if isinstance(hidden_states, list):
        hidden = hidden_states[-1]
    else:
        hidden = hidden_states
    # Slice to last position so the drafter sees a [B, 1, H] tensor
    # regardless of how many tokens this forward processed.
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]

    # Combine base stop tokens (EOS, Harmony, generation_config) with
    # request-specific stop_token_ids — same shape as _build_state_machine.
    eos_ids: set[int] = sched._get_stop_tokens()
    if request.sampling_params.stop_token_ids:
        eos_ids.update(request.sampling_params.stop_token_ids)

    try:
        generator = run_vlm_mtp_decode(
            target_language_model=lm,
            drafter=drafter,
            prompt_cache=prefilled_cache,
            hidden=hidden,
            shared_kv_states=out.shared_kv_states,
            first_bonus=int(first_bonus_arr.item()),
            max_tokens=request.sampling_params.max_tokens,
            sampler=sampler,
            draft_block_size=sched._vlm_mtp_draft_block_size,
            token_dtype=mx.int32,
            eos_token_ids=eos_ids or None,
        )
    except Exception as e:
        logger.warning(
            "vlm_mtp generator setup failed (%s); falling back for %s",
            e,
            request.request_id,
        )
        return None

    uid = sched._vlm_mtp_next_uid
    sched._vlm_mtp_next_uid -= 1
    sched._vlm_mtp_active[uid] = _VLMMTPDecodeState(
        generator=generator,
        request=request,
        prompt_cache=prefilled_cache,
        sampler=sampler,
        state_machine=state_machine,
        max_tokens=request.sampling_params.max_tokens,
        stop_token_ids=set(eos_ids),
    )
    logger.info(
        "vlm_mtp decode started: request=%s uid=%d block_size=%s",
        request.request_id,
        uid,
        sched._vlm_mtp_draft_block_size,
    )
    return uid

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
    drafter = sched._vlm_mtp_drafter
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
    block_size = sched._vlm_mtp_draft_block_size or int(
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
    """Advance every active vlm_mtp generator by one yield.

    Returns the synthesized responses for ``_process_batch_responses``.
    Mirrors mlx-lm BatchGenerator's per-step contract: one
    ``GenerationBatch.Response``-shaped object per active uid.
    """
    if not sched._vlm_mtp_active:
        return []

    responses: list[_VLMMTPResponse] = []
    for uid, state in list(sched._vlm_mtp_active.items()):
        try:
            with mx.stream(sched._stream):
                token_val = next(state.generator)
        except StopIteration:
            # Round loop exited naturally — terminate with prompt cache
            # so the prefix-cache layer can keep using it.
            sched._log_vlm_mtp_stats(state, "length")
            responses.append(
                _VLMMTPResponse(
                    uid=uid,
                    token=0,
                    finish_reason="length",
                    prompt_cache=state.prompt_cache,
                )
            )
            state.finished = True
            continue

        # Single-request mode yields ints; batch mode (not yet routed
        # by omlx) would yield a list. Guard so the path stays robust
        # if we widen routing later.
        if isinstance(token_val, list):
            # Take the first row (we only route singles for now).
            tok = next((t for t in token_val if t is not None), None)
            if tok is None:
                responses.append(
                    _VLMMTPResponse(
                        uid=uid,
                        token=0,
                        finish_reason="length",
                        prompt_cache=state.prompt_cache,
                    )
                )
                state.finished = True
                continue
            token = int(tok)
        else:
            token = int(token_val)

        state.emitted += 1
        finish_reason: str | None = None
        if state.stop_token_ids and token in state.stop_token_ids:
            finish_reason = "stop"
        elif state.emitted >= state.max_tokens:
            finish_reason = "length"

        if finish_reason is not None:
            sched._log_vlm_mtp_stats(state, finish_reason)

        responses.append(
            _VLMMTPResponse(
                uid=uid,
                token=token,
                finish_reason=finish_reason,
                prompt_cache=(
                    state.prompt_cache if finish_reason is not None else None
                ),
            )
        )
        if finish_reason is not None:
            state.finished = True

    # Drop finished entries.
    for uid in [u for u, s in sched._vlm_mtp_active.items() if s.finished]:
        del sched._vlm_mtp_active[uid]

    return responses
