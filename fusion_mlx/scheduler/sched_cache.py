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

import mlx.core as mx

from .helpers import (
    _safe_sync_stream,
)
from .monkeypatches import _unregister_uid_row

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .helpers import (
    _get_attr_or_key,
    _model_declares_llama4,
)
from .types import (
    _mx_buffer_access_lock,
)

try:
    from ..cache.hybrid_cache import ModelCacheConfig
    HAS_CACHE_TYPE_HANDLERS = True
except ImportError:
    ModelCacheConfig = None
    HAS_CACHE_TYPE_HANDLERS = False


_TURBOQUANT_KV_CACHE_TYPES = frozenset(
    {
        "TurboQuantKVCache",
        "BatchTurboQuantKVCache",
    }
)


_MINIMAX_M3_KV_CACHE_TYPES = frozenset(
    {
        "MiniMaxM3KVCache",
        "MiniMaxM3BatchKVCache",
    }
)


def _is_turboquant_kv_cache(cache_obj):
    return type(cache_obj).__name__ in _TURBOQUANT_KV_CACHE_TYPES


def _is_turboquant_kv_family_cache(cache_obj):
    from mlx_lm.models.cache import KVCache
    return isinstance(cache_obj, KVCache) or _is_turboquant_kv_cache(cache_obj)


@staticmethod
def _collect_arrays_from_extracted_cache(
    extracted_cache: list[Any],
) -> list[Any]:
    """Collect lazy mx.array references from an _extracted_cache payload.

    Used by G2-async to force a single batched mx.eval on the inference
    thread before handing the cache off to the store_cache worker. The
    worker can then call _extract_tensor_bytes safely (no further Metal
    graph evaluation needed for non-bfloat16, no-op for already-evaluated).

    Walks the per-layer dict format produced by _extract_cache_states:
    each layer is {state, meta_state, class_name, cache_type}, where
    state is a tuple of mx.arrays (or nested for CacheList / TurboQuant).
    """
    arrays: list[Any] = []
    for layer in extracted_cache or []:
        if not isinstance(layer, dict):
            continue
        state = layer.get("state", ())
        if isinstance(state, mx.array):
            arrays.append(state)
            continue
        if not isinstance(state, (list, tuple)):
            continue
        for item in state:
            if isinstance(item, mx.array):
                arrays.append(item)
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, mx.array):
                        arrays.append(sub)
                    elif hasattr(sub, "_fields"):
                        # NamedTuple state (TurboQuant). Walk fields.
                        for fname in sub._fields:
                            val = getattr(sub, fname, None)
                            if isinstance(val, mx.array):
                                arrays.append(val)
            elif hasattr(item, "_fields"):
                for fname in item._fields:
                    val = getattr(item, fname, None)
                    if isinstance(val, mx.array):
                        arrays.append(val)
    return arrays

def _async_store_cache_worker(    self,
    request_id: str,
    token_sequence_to_store: list[int],
    cache_to_store: list[Any],
    model_cache_config: Any | None,
    intermediate_snapshots: dict[int, list[Any]] | None,
    extra_keys: tuple[Any, ...] | None,
    extra_key_token_start: int | None,
    extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None,
) -> None:
    """Run store_cache + paged_cache cleanup off the inference thread.

    Pre-conditions enforced by the caller (_cleanup_finished):
    - mx.async_eval() was called on the inference thread for all
    KV cache arrays, dispatching materialization asynchronously
    without blocking the inference thread. async_eval completes
    Metal command enqueueing before returning, so all commands
    are submitted by the time executor.submit() runs.
    - This worker calls mx.synchronize(self._stream) via the
    _safe_sync_stream helper to wait on the same stream where
    mx.async_eval dispatched the arrays. A bare mx.synchronize()
    with no args only blocks on the default stream (gpu:0) and
    would leave the dispatched per-engine stream's work
    unsynchronized, racing the buffer-protocol access below
    (#1437). Stream objects are not thread-local in MLX (Metal
    device is a global singleton), so mx.synchronize(stream) is
    safe cross-thread; it just calls waitUntilCompleted on the
    command buffer.
    - bfloat16 view+eval inside _extract_tensor_bytes runs on this
    worker's default mx stream, isolated from self._stream;
    the underlying buffer is read-only at this point.
    - batch_generator.remove(uid) is deferred until this worker
    completes (handled by _drain_pending_async_removes).

    paged_cache_manager and block_aware_cache rely on
    threading.RLock so concurrent access from main and worker is safe.
    """
    try:
        # Hold _mx_buffer_access_lock across the worker's mx-buffer
        # access. store_cache eventually drives _extract_tensor_bytes,
        # which reads raw bytes via the buffer protocol; serializing
        # against inference-thread mx.clear_cache / mx.synchronize calls
        # prevents a SIGABRT when those reclaim the underlying Metal
        # buffer pool mid-read (#1106).
        with _mx_buffer_access_lock:
            with self._phase_timer("store_cache_worker_sync"):
                _safe_sync_stream(self._stream)
            block_table = self.block_aware_cache.store_cache(
                request_id,
                token_sequence_to_store,
                cache_to_store,
                model_cache_config=model_cache_config,
                boundary_snapshots=intermediate_snapshots,
                extra_keys=extra_keys,
                extra_key_token_start=extra_key_token_start,
                extra_key_ranges=extra_key_ranges,
            )
        if block_table is None and self.paged_cache_manager is not None:
            block_table = self.paged_cache_manager.get_block_table(request_id)
        if block_table and self.paged_cache_manager is not None:
            self.paged_cache_manager.release_for_eviction(block_table.block_ids)
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear_request_entry(request_id)
    except Exception as e:
        logger.warning("Async store_cache failed for %s: %s", request_id, e)

def _drain_pending_async_removes(self) -> None:
    """Process deferred batch_generator.remove() calls from prior steps.

    Called at the start of every step. For each pending entry, if the
    async store_cache future has finished, perform the
    batch_generator.remove() on the inference thread (Metal-safe) and
    finalize cleanup state. Entries whose futures are still in flight
    are left at the head of the deque for a later step.
    """
    if not self._pending_async_removes:
        return
    while self._pending_async_removes:
        uid, request_id, future = self._pending_async_removes[0]
        if future is not None and not future.done():
            # Worker still busy. Stop draining; check again next step.
            # Inflight entry stays at deque head to preserve order.
            break
        self._pending_async_removes.popleft()
        # Surface worker exceptions for visibility (don't crash step loop).
        if future is not None:
            exc = future.exception()
            if exc is not None:
                logger.warning(
                    "Async store_cache for %s raised: %s", request_id, exc
                )
        # Run batch_generator.remove on the inference thread.
        try:
            _safe_sync_stream(self._stream)
            self._remove_uid_from_active_batch(uid)
            if hasattr(self.model, "unregister_rope_delta"):
                self.model.unregister_rope_delta(uid)
        except Exception as e:
            logger.warning(
                "Deferred batch_generator.remove(uid=%s) failed: %s",
                uid,
                e,
            )
        # Cleanup uid maps now that the slot is reclaimable.
        _unregister_uid_row(self.model, uid)
        if uid in self.uid_to_request_id:
            del self.uid_to_request_id[uid]
        if request_id in self.request_id_to_uid:
            del self.request_id_to_uid[request_id]
        self._inflight_store_futures.pop(request_id, None)
        # Boundary snapshots were kept on disk for the worker; safe to
        # delete now that the future has completed. Cleanup was
        # deferred from _cleanup_finished to avoid racing the worker's
        # boundary_snapshot_store.load() calls with rmtree.
        if self._boundary_snapshot_store is not None:
            self._boundary_snapshot_store.cleanup_request(request_id)
        # Worker no longer holds extracted_cache — pop request from
        # self.requests and drop the cache buffer references so MLX
        # arrays can be freed.
        req_to_remove = self.requests.pop(request_id, None)
        if req_to_remove is not None:
            req_to_remove._extracted_cache = None
            req_to_remove.prompt_cache = None

def _calculate_max_blocks(self) -> int:
    """
    Calculate maximum cache blocks for paged SSD-only mode.

    In paged SSD-only mode, blocks don't consume GPU memory (data is on paged SSD),
    so we use a large default that can be limited by SSD capacity.

    Returns:
        Maximum number of cache blocks to allocate.
    """
    # In paged SSD-only mode, use a large default since blocks don't consume GPU memory
    # The actual limit is SSD capacity (paged_ssd_cache_max_size)
    max_blocks = 100000  # Large default for paged SSD-only mode

    block_size = self.config.paged_cache_block_size
    logger.info(
        f"paged SSD-only mode: max_blocks={max_blocks}, block_size={block_size} tokens"
    )

    return max_blocks

def _collect_rotating_window_sizes(    self,
    cache_obj: Any,
    window_sizes: set[int],
) -> None:
    """Collect rotating window sizes recursively from cache objects."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        for sub_cache in sub_caches:
            self._collect_rotating_window_sizes(sub_cache, window_sizes)

    class_name = type(cache_obj).__name__
    if class_name in ("RotatingKVCache", "BatchRotatingKVCache"):
        max_size = getattr(cache_obj, "max_size", 0)
        if isinstance(max_size, int) and max_size > 0:
            window_sizes.add(max_size)

def _detect_rotating_window_sizes(self) -> set[int]:
    """Detect rotating window sizes from model.make_cache() if available."""
    if not hasattr(self.model, "make_cache"):
        return set()

    try:
        cache_list = self.model.make_cache()
    except Exception as e:
        logger.debug(f"Failed to inspect model rotating window sizes: {e}")
        return set()

    if cache_list is None:
        return set()

    window_sizes: set[int] = set()
    for cache_obj in cache_list:
        self._collect_rotating_window_sizes(cache_obj, window_sizes)

    return window_sizes

# Target range for RotatingKVCache block size alignment.
# Using a multiple of window_size within this range reduces SSD I/O
# overhead (fewer, larger block files) while keeping cache restore
# reprocessing reasonable.
_ROTATING_BLOCK_SIZE_MIN = 512
_ROTATING_BLOCK_SIZE_MAX = 1024

def _align_block_size_with_rotating_window(self) -> None:
    """
    Align paged cache block size to a multiple of RotatingKVCache
    window size, targeting 512-1024 tokens per block.

    Block size must be a multiple of window_size so that block
    boundaries align with rotation boundaries. When window_size is
    small (e.g. 128), using it directly as block_size creates too
    many small files. Instead we pick the smallest multiple of
    window_size that falls within [_ROTATING_BLOCK_SIZE_MIN,
    _ROTATING_BLOCK_SIZE_MAX].
    """
    if not self.config.paged_ssd_cache_dir:
        return

    window_sizes = self._detect_rotating_window_sizes()
    if not window_sizes:
        return

    if len(window_sizes) > 1:
        raise ValueError(
            "Multiple RotatingKVCache window sizes detected "
            f"({sorted(window_sizes)}). Set a single aligned block size or "
            "disable paged cache for this model."
        )

    window_size = next(iter(window_sizes))

    # Find the smallest multiple of window_size >= _ROTATING_BLOCK_SIZE_MIN.
    # If window_size itself is already >= max, just use window_size.
    lo = self._ROTATING_BLOCK_SIZE_MIN
    hi = self._ROTATING_BLOCK_SIZE_MAX

    if window_size >= hi or window_size >= lo:
        target_block_size = window_size
    else:
        # window_size < lo: pick smallest multiple in [lo, hi]
        multiplier = (lo + window_size - 1) // window_size  # ceil(lo / ws)
        target_block_size = multiplier * window_size
        if target_block_size > hi:
            # Fall back to largest multiple <= hi
            target_block_size = (hi // window_size) * window_size
            if target_block_size < window_size:
                target_block_size = window_size

    if self.config.paged_cache_block_size != target_block_size:
        logger.info(
            "Aligning paged cache block_size=%s to %s "
            "(RotatingKVCache window_size=%s, multiplier=%sx)",
            self.config.paged_cache_block_size,
            target_block_size,
            window_size,
            target_block_size // window_size,
        )
        self.config.paged_cache_block_size = target_block_size

# Default block size for ArraysCache-only hybrid models.
# Match prefill_step_size (2048) so that boundary caching ON/OFF
# produces identical prefill chunk sizes, eliminating float32↔dtype
# roundtrip differences in GatedDeltaNet recurrent state.
_ARRAYS_CACHE_BLOCK_SIZE = 2048

def _enlarge_block_size_for_arrays_cache(self) -> None:
    """Enlarge block size for ArraysCache-only hybrid models.

    When a model uses ArraysCache (GatedDeltaNet) but not RotatingKVCache,
    a larger block size reduces the number of boundary snapshot stops during
    prefill while still storing valid per-block recurrent state.

    This is skipped if RotatingKVCache was already detected (block size was
    aligned to its window size) or if the user explicitly set a block size
    larger than the default.
    """
    if not self.config.paged_ssd_cache_dir:
        return

    # Skip if RotatingKVCache already adjusted block size.
    rotating_sizes = self._detect_rotating_window_sizes()
    if rotating_sizes:
        return

    # Detect ArraysCache from model.make_cache()
    if not hasattr(self.model, "make_cache"):
        return

    try:
        cache_list = self.model.make_cache()
    except Exception:
        return

    if cache_list is None:
        return

    has_arrays_cache = any(
        self._cache_tree_has_arrays_cache(cache_obj) for cache_obj in cache_list
    )
    if not has_arrays_cache:
        return

    target = self._ARRAYS_CACHE_BLOCK_SIZE
    if self.config.paged_cache_block_size >= target:
        return

    logger.info(
        "Enlarging paged cache block_size=%s to %s for "
        "ArraysCache hybrid model (reduces boundary snapshot overhead)",
        self.config.paged_cache_block_size,
        target,
    )
    self.config.paged_cache_block_size = target

@staticmethod
def _cache_tree_has_arrays_cache(cache_obj: Any) -> bool:
    """Return True if cache_obj contains ArraysCache (recursively)."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return any(
            Scheduler._cache_tree_has_arrays_cache(sub) for sub in sub_caches
        )
    return type(cache_obj).__name__ in ("ArraysCache", "SizedArraysCache")


def _collect_cache_storage_arrays(self, cache_obj: Any) -> list:
    arrays = []
    if isinstance(cache_obj, mx.array):
        return [cache_obj]
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        for sub_cache in sub_caches:
            arrays.extend(self._collect_cache_storage_arrays(sub_cache))
    array_cache = getattr(cache_obj, "cache", None)
    if isinstance(array_cache, (list, tuple)):
        for item in array_cache:
            arrays.extend(self._collect_cache_storage_arrays(item))
    for attr in ("keys", "values", "left_padding", "lengths"):
        value = getattr(cache_obj, attr, None)
        if isinstance(value, mx.array):
            arrays.append(value)
    return arrays


def _materialize_cache_storage(self, cache_list: list) -> None:
    arrays = []
    for cache_obj in cache_list:
        arrays.extend(self._collect_cache_storage_arrays(cache_obj))
    if arrays:
        with _mx_buffer_access_lock:
            mx.eval(*arrays)


def _trim_prompt_cache_by_tokens(self, cache_list: list, n: int) -> bool:
    if not cache_list:
        return False
    if n <= 0:
        return True
    for cache_obj in cache_list:
        if not self._trim_cache_tree_by_tokens(cache_obj, n):
            return False
    return True


def _trim_cache_tree_by_tokens(self, cache_obj: Any, n: int) -> bool:
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return all(
            self._trim_cache_tree_by_tokens(sub_cache, n)
            for sub_cache in sub_caches
        )
    trim_fn = getattr(cache_obj, "trim", None)
    if not callable(trim_fn):
        return False
    try:
        trimmed = trim_fn(n)
        if trimmed is None:
            return True
        return int(trimmed) >= n
    except Exception:
        return False


def _cache_tree_has_class_name(
    self,
    cache_obj: Any,
    class_names: frozenset,
) -> bool:
    if type(cache_obj).__name__ in class_names:
        return True
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return any(
            self._cache_tree_has_class_name(sub_cache, class_names)
            for sub_cache in sub_caches
        )
    return False


def _align_minimax_m3_partial_cache_to_prefill_step(
    self,
    request,
) -> bool:
    cache_list = request.prompt_cache
    block_table = request.block_table
    prompt_tokens = request.prompt_token_ids or []
    if not cache_list or block_table is None or not block_table.block_ids:
        return False
    if (
        block_table.num_tokens <= 0
        or block_table.num_tokens >= len(prompt_tokens)
    ):
        return False

    has_minimax_m3 = any(
        self._cache_tree_has_class_name(cache_obj, _MINIMAX_M3_KV_CACHE_TYPES)
        for cache_obj in cache_list
    )
    if not has_minimax_m3:
        return False

    block_size = int(getattr(self.config, "paged_cache_block_size", 0) or 0)
    prefill_step = int(getattr(self.config, "prefill_step_size", 0) or 0)
    if block_size <= 0 or prefill_step <= block_size:
        return False

    aligned_tokens = (block_table.num_tokens // prefill_step) * prefill_step
    aligned_tokens = (aligned_tokens // block_size) * block_size
    if aligned_tokens <= 0 or aligned_tokens >= block_table.num_tokens:
        return False

    target_block_count = 0
    target_tokens = 0
    for block_id in block_table.block_ids:
        block = (
            self.paged_cache_manager.allocated_blocks.get(block_id)
            if self.paged_cache_manager is not None
            else None
        )
        token_count = int(getattr(block, "token_count", block_size) or block_size)
        if target_tokens + token_count > aligned_tokens:
            break
        target_tokens += token_count
        target_block_count += 1

    if target_tokens != aligned_tokens:
        logger.debug(
            "MiniMax M3 partial cache alignment skipped for %s: cannot align "
            "block table from %d to %d tokens",
            request.request_id,
            block_table.num_tokens,
            aligned_tokens,
        )
        return False

    trim_tokens = block_table.num_tokens - aligned_tokens
    if not self._trim_prompt_cache_by_tokens(cache_list, trim_tokens):
        logger.debug(
            "MiniMax M3 partial cache alignment skipped for %s: cache trim "
            "by %d tokens failed",
            request.request_id,
            trim_tokens,
        )
        return False

    dropped_block_ids = block_table.block_ids[target_block_count:]
    if self.paged_cache_manager is not None:
        for block_id in dropped_block_ids:
            self.paged_cache_manager.free_block(block_id)
    block_table.block_ids = block_table.block_ids[:target_block_count]
    block_table.num_tokens = aligned_tokens

    logger.debug(
        "MiniMax M3 partial cache aligned for %s: trimmed %d tokens, "
        "released %d blocks, new num_tokens=%d",
        request.request_id,
        trim_tokens,
        len(dropped_block_ids),
        aligned_tokens,
    )
    return True


def _model_uses_mla(self) -> bool:
    cached = getattr(self, "_mla_model", None)
    if cached is not None:
        return cached

    detected = False
    model = getattr(self, "model", None)

    def _cfg_has_kv_lora(cfg, depth=0):
        if cfg is None or depth > 3:
            return False
        if isinstance(getattr(cfg, "kv_lora_rank", None), int):
            return True
        return any(
            _cfg_has_kv_lora(getattr(cfg, sub, None), depth + 1)
            for sub in (
                "text_config",
                "llm_config",
                "language_config",
                "thinker_config",
            )
        )

    for holder in (
        model,
        getattr(model, "_language_model", None),
        getattr(model, "language_model", None),
    ):
        if holder is None:
            continue
        if _cfg_has_kv_lora(getattr(holder, "args", None)) or _cfg_has_kv_lora(
            getattr(holder, "config", None)
        ):
            detected = True
            break

    if not detected and model is not None and hasattr(model, "modules"):
        try:
            for m in model.modules():
                if (
                    hasattr(m, "kv_a_proj_with_mqa")
                    or hasattr(m, "kv_a_layernorm")
                    or isinstance(getattr(m, "kv_lora_rank", None), int)
                ):
                    detected = True
                    break
        except Exception:
            pass

    if detected:
        logger.info(
            "TurboQuant disabled: model uses Multi-head Latent Attention "
            "(MLA), which is incompatible with quantized KV cache states; "
            "keeping fp16 KV cache."
        )
    self._mla_model = detected
    return detected


def _model_uses_attention_sinks(self) -> bool:
    cached = getattr(self, "_attention_sink_model", None)
    if cached is not None:
        return cached

    detected = False
    model = getattr(self, "model", None)

    def _has_real_sink_attr(obj):
        for name in ("sinks", "attention_sink_bias", "attn_sink"):
            value = None
            if isinstance(obj, dict):
                value = obj.get(name)
            if value is None:
                data = getattr(obj, "__dict__", {})
                if isinstance(data, dict):
                    value = data.get(name)
            if isinstance(value, mx.array):
                return True
            if value is not None and isinstance(value, (int, float, list, tuple)):
                return True
        return False

    try:
        modules = getattr(model, "modules", None)
    except Exception:
        modules = None
    if type(modules).__module__.startswith("unittest.mock"):
        modules = None
    if not detected and callable(modules):
        try:
            for m in modules():
                if _has_real_sink_attr(m):
                    detected = True
                    break
        except Exception:
            pass

    if detected:
        logger.info(
            "TurboQuant disabled: model uses attention sinks, which are "
            "not supported by TurboQuant's quantized attention kernels; "
            "keeping fp16 KV cache."
        )
    self._attention_sink_model = detected
    return detected


def _turboquant_eligible(self, prompt_cache: list) -> bool:
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    if self._model_uses_mla():
        return False
    if self._model_uses_attention_sinks():
        return False

    def _ok(c):
        if isinstance(c, KVCache):
            return True
        if isinstance(c, ArraysCache):
            return True
        class_name = type(c).__name__
        if class_name in (
            "SizedArraysCache",
            "RotatingKVCache",
            "BatchRotatingKVCache",
            "PrefillReadyRotatingKVCache",
            "TurboQuantKVCache",
            "BatchTurboQuantKVCache",
        ):
            return True
        if class_name in ("MiniMaxM3KVCache", "MiniMaxM3BatchKVCache"):
            return False
        if isinstance(c, CacheList):
            return all(_ok(inner) for inner in c.caches)
        return False

    return bool(prompt_cache) and all(_ok(c) for c in prompt_cache)


def _infer_live_layer_cache_types(self):
    if not HAS_CACHE_TYPE_HANDLERS or ModelCacheConfig is None:
        return None

    make_cache = getattr(self.model, "make_cache", None)
    if not callable(make_cache):
        return None

    try:
        cache_list = make_cache()
    except Exception as e:
        logger.debug("Failed to build cache list for SSD signature: %s", e)
        return None

    if not isinstance(cache_list, (list, tuple)) or not cache_list:
        return None

    cache_list = list(cache_list)
    try:
        model_cache_config = ModelCacheConfig.from_cache_list(
            cache_list,
            model_name=self.config.model_name or "",
        )
        layer_cache_types = model_cache_config.get_type_names()
    except Exception as e:
        logger.debug("Failed to infer SSD layer cache signature: %s", e)
        return None

    if not layer_cache_types:
        return None

    if self._turboquant_kv_bits is None:
        return layer_cache_types

    try:
        if not self._turboquant_eligible(cache_list):
            return layer_cache_types
    except Exception as e:
        logger.debug("Failed to evaluate TurboQuant SSD signature: %s", e)
        return layer_cache_types

    kv_indices = [
        i for i, c in enumerate(cache_list) if _is_turboquant_kv_family_cache(c)
    ]
    skip_last = self._turboquant_skip_last and len(kv_indices) > 1
    last_kv_idx = kv_indices[-1] if skip_last else -1
    for idx in kv_indices:
        if idx != last_kv_idx and idx < len(layer_cache_types):
            layer_cache_types[idx] = "TurboQuantKVCache"

    return layer_cache_types


def refresh_ssd_layer_signature(self):
    manager = self.paged_ssd_cache_manager
    if manager is None:
        return None

    layer_cache_types = self._infer_live_layer_cache_types()
    if not layer_cache_types:
        return None

    try:
        set_signature = getattr(manager, "set_expected_layer_signature", None)
        if callable(set_signature):
            set_signature(layer_cache_types)
        else:
            manager.adopt_layer_signature_if_unset(layer_cache_types)
        manager.invalidate_stale_layer_signature()
    except Exception as e:
        logger.warning("Failed to refresh SSD layer cache signature: %s", e)
        return None

    return layer_cache_types
