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

from .helpers import (
    _deferred_clear_delay,
)

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.

# Import protocol-specific output parser support
try:
    from ..parsers.output_parser import (
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

from .sched_admission import *  # noqa: F403
from .sched_batch import *  # noqa: F403
from .sched_batch import (
    _emit_final_boundary_if_needed,
    _step_prefill_chunk,
)
from .sched_boundary import *  # noqa: F403
from .sched_boundary import (
    _cache_list_needs_boundary_snapshot,
    _extract_boundary_snapshot,
    _maybe_capture_boundary_snapshot,
    _validate_cache,
)
from .sched_cache import *  # noqa: F403
from .sched_handoff import export_kv_state, import_kv_state
from .sched_init import *  # noqa: F403
from .sched_init import (
    __init__,
    _phase_timer,
)
from .sched_misc import *  # noqa: F403
from .sched_misc import (
    _evict_blocks_permanently,
    _evict_blocks_to_cold,
    _restore_block_from_cold,
)
from .sched_query import *  # noqa: F403
from .sched_query import (
    _current_usage_bytes,
    _estimate_prefill_peak,
    _hot_cache_cpu_bytes,
    _preflight_memory_check,
)
from .sched_response import *  # noqa: F403
from .sched_response import (
    _cleanup_finished,
    _is_cache_corruption_error,
    _release_paged_cache_for_request,
    _reschedule_running_requests,
)
from .sched_schedule import *  # noqa: F403
from .sched_schedule import _schedule_waiting
from .sched_specprefill import *  # noqa: F403
from .sched_specprefill import _cleanup_specprefill, _try_specprefill_scoring
from .sched_step import *  # noqa: F403
from .sched_thinking import *  # noqa: F403
from .sched_thinking import (
    _build_state_machine,
    _cache_tree_has_stateful_non_sliceable,
    _detect_needs_think_prefix,
    _ensure_batch_generator,
    _get_think_token_id,
)
from .sched_token import *  # noqa: F403
from .sched_token import (
    _apply_turboquant_kv_convert,
    _apply_turboquant_kv_empty,
    _cleanup_detokenizer,
    _cleanup_output_parser_session,
    _get_detokenizer,
    _on_prompt_progress,
)
from .sched_trim import *  # noqa: F403
from .sched_trim import (
    _check_pending_aborts_for_uids,
    _do_abort_request,
    _remove_uid_from_active_batch,
    _trim_cache_tree_by_one,
    _trim_prompt_cache_for_generation,
)
from .sched_vlm_mtp import *  # noqa: F403


class Scheduler:
    """Orchestrator — delegates to split modules."""

    _PREFILL_STEP_TIERS: tuple[int, ...] = (1024, 512, 256, 128)
    _ROTATING_BLOCK_SIZE_MIN: int = 512
    _ROTATING_BLOCK_SIZE_MAX: int = 1024
    _ARRAYS_CACHE_BLOCK_SIZE: int = 2048
    _DEFERRED_CLEAR_DELAY: int = 4

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

    def _current_usage_bytes(self, *args, **kwargs):
        return _current_usage_bytes(self, *args, **kwargs)

    def _hot_cache_cpu_bytes(self):
        return _hot_cache_cpu_bytes(self)

    def _estimate_prefill_peak(self, *args, **kwargs):
        return _estimate_prefill_peak(self, *args, **kwargs)

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

    def export_kv_state(self, *args, **kwargs):
        return export_kv_state(self, *args, **kwargs)

    def import_kv_state(self, *args, **kwargs):
        return import_kv_state(self, *args, **kwargs)


# Bind the __init__ from sched_init.py as the Scheduler constructor
Scheduler.__init__ = __init__


# Automatically bind all sched_* functions that take `self` as first param
import inspect as _inspect
import sys as _sys

_sched_mod_names = (
    "sched_admission",
    "sched_batch",
    "sched_boundary",
    "sched_cache",
    "sched_init",
    "sched_misc",
    "sched_query",
    "sched_response",
    "sched_schedule",
    "sched_specprefill",
    "sched_step",
    "sched_thinking",
    "sched_token",
    "sched_trim",
    "sched_vlm_mtp",
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

# Bind @staticmethod helpers skipped by the self/sched-first auto-bind loop.
# _cache_tree_has_arrays_cache is called as self._cache_tree_has_arrays_cache
# inside _enlarge_block_size_for_arrays_cache (sched_cache.py:402, reachable
# when paged_ssd_cache_dir is set) and directly by unit tests.
from .sched_cache import _cache_tree_has_arrays_cache as _cache_tree_has_arrays_cache

Scheduler._cache_tree_has_arrays_cache = _cache_tree_has_arrays_cache

# _format_bytes is a pure formatter (no self use) declared @staticmethod in
# sched_misc; the self/sched-first auto-bind loop skips staticmethods, so bind
# it explicitly. Called as self._format_bytes(...) and directly by unit tests
# via Scheduler._format_bytes(...).
from .sched_misc import _format_bytes as _format_bytes

Scheduler._format_bytes = _format_bytes

# _collect_arrays_from_extracted_cache is a @staticmethod pure helper in
# sched_cache (no self use) that walks an _extracted_cache payload for lazy
# mx.array refs to pre-eval on the inference thread. The self/sched-first
# auto-bind loop skips staticmethods, so bind it explicitly. Called as
# self._collect_arrays_from_extracted_cache(...) inside _cleanup_finished
# (sched_response.py store_cache_main_collect phase). Without this binding the
# call raised AttributeError, was swallowed by the surrounding except, and
# silently skipped cache storage for every finished request.
from .sched_cache import (
    _collect_arrays_from_extracted_cache as _collect_arrays_from_extracted_cache,
)

Scheduler._collect_arrays_from_extracted_cache = _collect_arrays_from_extracted_cache

# _merge_boundary_with_full_cache is a @staticmethod pure helper in
# sched_boundary (no self use) that fills placeholder layers in a boundary
# snapshot from the full extracted cache. Skipped by the self/sched-first
# auto-bind loop; called as self._merge_boundary_with_full_cache(...) inside
# _cleanup_finished (sched_response.py boundary_override branch). Without this
# binding the call raised AttributeError and silently skipped cache storage
# for every request ending on a partial trailing block.
from .sched_boundary import (
    _merge_boundary_with_full_cache as _merge_boundary_with_full_cache,
)

Scheduler._merge_boundary_with_full_cache = _merge_boundary_with_full_cache
