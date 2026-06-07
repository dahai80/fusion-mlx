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

from .cache.observability import CacheRateTracker
from .cache.paged_cache import PagedCacheManager
from .cache.prefix_cache import BlockAwarePrefixCache
from .exceptions import is_cache_corruption_error
from .prefill_progress import get_prefill_tracker
from .prefill_transient_tracker import PrefillTransientTracker
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .speculative.vlm_mtp import VLMMTPDrafter, run_vlm_mtp_decode
from .utils.proc_memory import get_phys_footprint
from .utils.sampling import make_sampler as omlx_make_sampler

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

from .sched_admission import *      # noqa: F403
from .sched_batch import *      # noqa: F403
from .sched_boundary import *      # noqa: F403
from .sched_cache import *      # noqa: F403
from .sched_init import *      # noqa: F403
from .sched_misc import *      # noqa: F403
from .sched_query import *      # noqa: F403
from .sched_response import *      # noqa: F403
from .sched_schedule import *      # noqa: F403
from .sched_specprefill import *      # noqa: F403
from .sched_step import *      # noqa: F403
from .sched_thinking import *      # noqa: F403
from .sched_token import *      # noqa: F403
from .sched_trim import *      # noqa: F403
from .sched_vlm_mtp import *      # noqa: F403


class Scheduler:
    """Orchestrator — delegates to split modules."""

    def add_request(self, *args, **kwargs):
        return add_request(self, *args, **kwargs)

    def _step_prefill_chunk(self, *args, **kwargs):
        return _step_prefill_chunk(self, *args, **kwargs)

    def _emit_final_boundary_if_needed(self, *args, **kwargs):
        return _emit_final_boundary_if_needed(self, *args, **kwargs)

    def _cache_list_needs_boundary_snapshot(self, *args, **kwargs):
        return _cache_list_needs_boundary_snapshot(self, *args, **kwargs)

    def _extract_boundary_snapshot(self, *args, **kwargs):
        return _extract_boundary_snapshot(self, *args, **kwargs)

    def _maybe_capture_boundary_snapshot(self, *args, **kwargs):
        return _maybe_capture_boundary_snapshot(self, *args, **kwargs)

    def _validate_cache(self, *args, **kwargs):
        return _validate_cache(self, *args, **kwargs)

    def _phase_timer(self, *args, **kwargs):
        return _phase_timer(self, *args, **kwargs)

    def adjust_store_cache_cap(self, *args, **kwargs):
        return adjust_store_cache_cap(self, *args, **kwargs)

    def _evict_blocks_permanently(self, *args, **kwargs):
        return _evict_blocks_permanently(self, *args, **kwargs)

    def _evict_blocks_to_cold(self, *args, **kwargs):
        return _evict_blocks_to_cold(self, *args, **kwargs)

    def _restore_block_from_cold(self, *args, **kwargs):
        return _restore_block_from_cold(self, *args, **kwargs)

    def restore_cold_blocks_for_request(self, *args, **kwargs):
        return restore_cold_blocks_for_request(self, *args, **kwargs)

    def _preflight_memory_check(self, *args, **kwargs):
        return _preflight_memory_check(self, *args, **kwargs)

    def _release_paged_cache_for_request(self, *args, **kwargs):
        return _release_paged_cache_for_request(self, *args, **kwargs)

    def _cleanup_finished(self, *args, **kwargs):
        return _cleanup_finished(self, *args, **kwargs)

    def _is_cache_corruption_error(self, *args, **kwargs):
        return _is_cache_corruption_error(self, *args, **kwargs)

    def _reschedule_running_requests(self, *args, **kwargs):
        return _reschedule_running_requests(self, *args, **kwargs)

    def _schedule_waiting(self, *args, **kwargs):
        return _schedule_waiting(self, *args, **kwargs)

    def _try_specprefill_scoring(self, *args, **kwargs):
        return _try_specprefill_scoring(self, *args, **kwargs)

    def _cleanup_specprefill(self, *args, **kwargs):
        return _cleanup_specprefill(self, *args, **kwargs)

    def get_request(self, *args, **kwargs):
        return get_request(self, *args, **kwargs)

    def remove_finished_request(self, *args, **kwargs):
        return remove_finished_request(self, *args, **kwargs)

    def _build_state_machine(self, *args, **kwargs):
        return _build_state_machine(self, *args, **kwargs)

    def _get_think_token_id(self, *args, **kwargs):
        return _get_think_token_id(self, *args, **kwargs)

    def _detect_needs_think_prefix(self, *args, **kwargs):
        return _detect_needs_think_prefix(self, *args, **kwargs)

    def _ensure_batch_generator(self, *args, **kwargs):
        return _ensure_batch_generator(self, *args, **kwargs)

    def _cache_tree_has_stateful_non_sliceable(self, *args, **kwargs):
        return _cache_tree_has_stateful_non_sliceable(self, *args, **kwargs)

    def _get_detokenizer(self, *args, **kwargs):
        return _get_detokenizer(self, *args, **kwargs)

    def _cleanup_detokenizer(self, *args, **kwargs):
        return _cleanup_detokenizer(self, *args, **kwargs)

    def _cleanup_output_parser_session(self, *args, **kwargs):
        return _cleanup_output_parser_session(self, *args, **kwargs)

    def _on_prompt_progress(self, *args, **kwargs):
        return _on_prompt_progress(self, *args, **kwargs)

    def _apply_turboquant_kv_empty(self, *args, **kwargs):
        return _apply_turboquant_kv_empty(self, *args, **kwargs)

    def _apply_turboquant_kv_convert(self, *args, **kwargs):
        return _apply_turboquant_kv_convert(self, *args, **kwargs)

    def _trim_prompt_cache_for_generation(self, *args, **kwargs):
        return _trim_prompt_cache_for_generation(self, *args, **kwargs)

    def _trim_cache_tree_by_one(self, *args, **kwargs):
        return _trim_cache_tree_by_one(self, *args, **kwargs)

    def _remove_uid_from_active_batch(self, *args, **kwargs):
        return _remove_uid_from_active_batch(self, *args, **kwargs)

    def _check_pending_aborts_for_uids(self, *args, **kwargs):
        return _check_pending_aborts_for_uids(self, *args, **kwargs)

    def abort_request(self, *args, **kwargs):
        return abort_request(self, *args, **kwargs)

    def _do_abort_request(self, *args, **kwargs):
        return _do_abort_request(self, *args, **kwargs)

