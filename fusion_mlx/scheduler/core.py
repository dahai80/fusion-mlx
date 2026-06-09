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
    _deferred_clear_delay,
    _KNOWN_SLICEABLE_CACHE_TYPES,
)
from .monkeypatches import _default_generation_stream

# Import protocol-specific output parser support
try:
    from .adapter.output_parser import (
        OutputParserFactory,
        OutputParserSession,
        detect_output_parser,
    )
    HAS_OUTPUT_PARSER = True
except ImportError:
    OutputParserFactory = None
    OutputParserSession = None
    detect_output_parser = None
    HAS_OUTPUT_PARSER = False

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
from .sched_batch import _adaptive_chunk_size, _advance_chunked_prefills, _begin_prefill, _do_external_prefill, _emit_final_boundary_if_needed, _insert_prefilled_request, _record_chunk_transient, _step_prefill_chunk
from .sched_boundary import _cache_list_needs_boundary_snapshot, _detect_boundary_snapshot_need, _extract_boundary_snapshot, _extract_cache_states, _get_boundary_store_override, _maybe_capture_boundary_snapshot, _merge_boundary_with_full_cache, _normalize_rotating_snapshot_state, _on_prefill_boundary_snapshot, _validate_cache
from .sched_cache import _align_block_size_with_rotating_window, _async_store_cache_worker, _cache_tree_has_arrays_cache, _calculate_max_blocks, _collect_arrays_from_extracted_cache, _collect_rotating_window_sizes, _detect_rotating_window_sizes, _drain_pending_async_removes, _enlarge_block_size_for_arrays_cache
from .sched_init import __init__, _periodic_clear_threshold_bytes, _phase_timer, _should_periodic_clear_cache
from .sched_misc import _check_memory_pressure, _collect_cache_counters, _evict_blocks_permanently, _evict_blocks_to_cold, _format_bytes, _init_tiered_cache, _restore_block_from_cold, _set_model_info_for_monitor
from .sched_query import _preflight_memory_check
from .sched_response import _cleanup_finished, _is_cache_corruption_error, _process_batch_responses, _recover_from_cache_error, _release_paged_cache_for_request, _reschedule_running_requests
from .sched_schedule import _schedule_waiting
from .sched_specprefill import _cleanup_specprefill, _try_specprefill_scoring
from .sched_step import _publish_admin_snapshot
from .sched_thinking import _build_sampler_and_processors, _build_state_machine, _cache_tree_has_stateful_non_sliceable, _detect_needs_think_prefix, _emit_prefill_boundary_snapshot, _ensure_batch_generator, _get_chat_template_text, _get_model_vocab_size, _get_think_token_id, _resolve_think_close_pattern, _resolve_think_end_token_ids
from .sched_token import _apply_turboquant_kv_convert, _apply_turboquant_kv_empty, _cleanup_detokenizer, _cleanup_output_parser_session, _create_batch_generator, _get_detokenizer, _get_output_parser_session, _get_stop_tokens, _get_xtc_special_tokens, _load_generation_config_eos, _on_prompt_progress
from .sched_trim import _check_pending_aborts_for_uids, _do_abort_request, _process_pending_aborts, _remove_uid_from_active_batch, _trim_cache_tree_by_one, _trim_prompt_cache_for_generation
from .sched_vlm_mtp import _log_vlm_mtp_stats, _route_to_vlm_mtp, _step_vlm_mtp



class Scheduler:
    """Orchestrator — delegates to split modules."""
    _PREFILL_STEP_TIERS: tuple[int, ...] = (1024, 512, 256, 128)
    _ROTATING_BLOCK_SIZE_MIN: int = 512
    _ROTATING_BLOCK_SIZE_MAX: int = 1024
    _ARRAYS_CACHE_BLOCK_SIZE: int = 2048

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

    def _deferred_clear_delay(self, *args, **kwargs):
        return _deferred_clear_delay(self, *args, **kwargs)

# Bind the __init__ from sched_init.py as the Scheduler constructor
Scheduler.__init__ = __init__


# Automatically bind all sched_* functions that take `self` as first param
import inspect as _inspect
import sys as _sys

_sched_mod_names = (
    "sched_admission", "sched_batch", "sched_boundary", "sched_cache",
    "sched_init", "sched_misc", "sched_query", "sched_response",
    "sched_schedule", "sched_specprefill", "sched_step", "sched_thinking",
    "sched_token", "sched_trim", "sched_vlm_mtp",
)
for _mod_name in _sched_mod_names:
    _mod = _sys.modules.get(__name__.rsplit(".", 1)[0] + "." + _mod_name)
    if _mod is None:
        continue
    for _attr_name in dir(_mod):
        if _attr_name.startswith("__"):
            continue
        _fn = getattr(_mod, _attr_name)
        if not callable(_fn):
            continue
        try:
            _sig = _inspect.signature(_fn)
        except (ValueError, TypeError):
            continue
        _params = list(_sig.parameters.keys())
        if _params and _params[0] in ("self", "sched"):
            setattr(Scheduler, _attr_name, _fn)
