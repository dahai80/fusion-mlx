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
from typing import Any, Optional

import mlx.core as mx

from ..request import Request
from .helpers import (
    _KNOWN_SLICEABLE_CACHE_TYPES,
)

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
from .types import (
    _BoundarySnapshotProvider,
)

try:
    from ..cache.hybrid_cache import ModelCacheConfig
    from ..cache.type_registry import CacheTypeRegistry

    HAS_CACHE_TYPE_HANDLERS = True
except ImportError:
    CacheTypeRegistry = None
    ModelCacheConfig = None
    HAS_CACHE_TYPE_HANDLERS = False


def _cache_list_needs_boundary_snapshot(self, cache_list: list[Any]) -> bool:
    """Return True if any layer cache requires boundary snapshots."""
    if not cache_list:
        return False
    return any(
        self._cache_tree_has_stateful_non_sliceable(layer_cache)
        for layer_cache in cache_list
    )


def _on_prefill_boundary_snapshot(
    self,
    request_id: str,
    snapshot_cache: list[Any],
    token_count: int,
) -> None:
    """Record boundary snapshots captured during prefill processing.

    Called from ``_emit_prefill_boundary_snapshot`` at each block
    boundary crossed during prefill. Keyed by ``request_id`` rather
    than ``uid`` because the request has not been inserted into
    ``BatchGenerator`` yet and the uid mapping does not exist —
    routing through it dropped every snapshot silently (#TBD).
    """
    if self.block_aware_cache is None:
        return

    block_size = self.config.paged_cache_block_size
    if block_size <= 0 or token_count <= 0 or token_count % block_size != 0:
        return

    if not self._cache_list_needs_boundary_snapshot(snapshot_cache):
        return

    if request_id not in self._boundary_cache_snapshots:
        self._boundary_cache_snapshots[request_id] = {}

    # Skip if we already have a snapshot at this token count
    if token_count in self._boundary_cache_snapshots[request_id]:
        return

    # Offload snapshot to SSD if store is available, keeping only a
    # None marker in the dict.  Falls back to in-memory storage when
    # the SSD store is unavailable or the write fails.
    if self._boundary_snapshot_store is not None:
        saved = self._boundary_snapshot_store.save(
            request_id,
            token_count,
            snapshot_cache,
            self._extract_cache_states,
        )
        if saved:
            self._boundary_cache_snapshots[request_id][token_count] = None
        else:
            self._boundary_cache_snapshots[request_id][token_count] = snapshot_cache
    else:
        self._boundary_cache_snapshots[request_id][token_count] = snapshot_cache

    self._boundary_snapshot_required = True
    logger.debug(
        "Captured prefill boundary cache snapshot for %s at %s tokens",
        request_id,
        token_count,
    )


def _detect_boundary_snapshot_need(self) -> bool:
    """
    Determine whether boundary snapshots are needed for the current model.

    Evaluated lazily by inspecting model.make_cache() output instead of
    the active batch (which no longer exists in the new API).
    """
    if self._boundary_snapshot_required is not None:
        return self._boundary_snapshot_required

    if not hasattr(self.model, "make_cache"):
        self._boundary_snapshot_required = False
        return False

    try:
        cache_list = self.model.make_cache()
    except Exception:
        self._boundary_snapshot_required = False
        return False

    if not cache_list:
        self._boundary_snapshot_required = False
        return False

    self._boundary_snapshot_required = any(
        self._cache_tree_has_stateful_non_sliceable(layer_cache)
        for layer_cache in cache_list
    )

    if self._boundary_snapshot_required:
        logger.info(
            "Enabled boundary cache snapshots for stateful non-sliceable "
            "cache layers"
        )
    else:
        logger.debug(
            "Boundary cache snapshots disabled (no stateful non-sliceable "
            "cache layers detected)"
        )

    return self._boundary_snapshot_required


def _extract_boundary_snapshot(self, uid: int) -> list[Any] | None:
    """Extract a per-request prompt cache snapshot via extract_cache().

    Uses BatchGenerator.extract_cache() which returns
    Dict[uid, (cache_list, tokens_list)].
    """
    if self.batch_generator is None:
        return None

    try:
        # Synchronize pending engine stream operations before
        # accessing batch cache tensors.
        with self._phase_timer("boundary_capture_sync"):
            # Direct mx.synchronize on module mx: this path runs on the
            # inference thread that owns self._stream, so the cross-thread
            # "no Stream" guard _safe_sync_stream provides is unnecessary.
            # Using the module-level mx also keeps the call observable to
            # sched_boundary.mx test patches.
            mx.synchronize(self._stream)
        with self._phase_timer("boundary_capture_extract"):
            with mx.stream(self._stream):
                result = self.batch_generator.extract_cache([uid])
                if uid not in result:
                    return None
                cache_list, _tokens = result[uid]
                # Only extract non-sliceable layers to avoid costly
                # deep-copy accumulation (same rationale as prefill path).
                return [
                    (
                        c
                        if type(c).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES
                        else None
                    )
                    for c in cache_list
                ]
    except Exception as e:
        logger.debug(f"Failed to extract boundary cache snapshot for uid={uid}: {e}")
        return None


def _maybe_capture_boundary_snapshot(self, request: Request, uid: int) -> None:
    """Capture cache snapshot exactly at block boundaries for safe reuse."""
    if self.block_aware_cache is None:
        return

    block_size = self.config.paged_cache_block_size
    if block_size <= 0:
        return

    total_tokens = request.num_tokens
    if total_tokens <= 0 or total_tokens % block_size != 0:
        return

    if not self._detect_boundary_snapshot_need():
        return

    snapshot_cache = self._extract_boundary_snapshot(uid)
    if not snapshot_cache:
        return

    if request.request_id not in self._boundary_cache_snapshots:
        self._boundary_cache_snapshots[request.request_id] = {}

    # Offload to SSD with in-memory fallback.
    if self._boundary_snapshot_store is not None:
        with self._phase_timer("boundary_snapshot_save"):
            saved = self._boundary_snapshot_store.save(
                request.request_id,
                total_tokens,
                snapshot_cache,
                self._extract_cache_states,
            )
        if saved:
            self._boundary_cache_snapshots[request.request_id][total_tokens] = None
        else:
            self._boundary_cache_snapshots[request.request_id][
                total_tokens
            ] = snapshot_cache
    else:
        self._boundary_cache_snapshots[request.request_id][
            total_tokens
        ] = snapshot_cache

    logger.debug(
        f"Captured boundary cache snapshot for {request.request_id} at "
        f"{total_tokens} tokens"
    )


def _get_boundary_store_override(
    self,
    request_id: str,
    full_token_sequence: list[int],
) -> (
    tuple[
        list[int],
        list[dict[str, Any]],
        Optional["ModelCacheConfig"],
        dict[int, list[dict[str, Any]]],
    ]
    | None
):
    """
    Return boundary-aligned cache payload when final request ends on partial block.

    Returns:
        Tuple of (truncated_tokens, extracted_cache, model_cache_config,
        intermediate_snapshots) where intermediate_snapshots maps
        token_count -> extracted cache states for per-block storage.
    """
    snapshots = self._boundary_cache_snapshots.get(request_id)
    if not snapshots:
        return None

    total_tokens = len(full_token_sequence)
    block_size = self.config.paged_cache_block_size

    # Find all valid boundary-aligned snapshot token counts
    valid_counts = sorted(
        tc for tc in snapshots if 0 < tc <= total_tokens and tc % block_size == 0
    )
    if not valid_counts:
        return None

    # Find the latest snapshot that leaves trailing partial tokens
    # (or equals total if it's block-aligned).
    latest_tc = valid_counts[-1]
    if latest_tc < total_tokens:
        # Trailing partial tokens exist — use this snapshot for truncation
        pass
    elif latest_tc == total_tokens and total_tokens % block_size == 0:
        # Exactly block-aligned — no truncation needed but we still
        # provide intermediate snapshots for per-block storage.
        latest_tc = total_tokens
    else:
        return None

    # Load latest snapshot — may be on SSD (None marker) or in memory.
    latest_snapshot = snapshots[latest_tc]
    if latest_snapshot is None and self._boundary_snapshot_store is not None:
        # Offloaded to SSD — load back.
        extracted_cache = self._boundary_snapshot_store.load(request_id, latest_tc)
        if not extracted_cache:
            return None
        # Build model_cache_config from the main request cache config
        # since the SSD snapshot doesn't carry it.
        model_cache_config = getattr(
            self.requests.get(request_id), "_model_cache_config", None
        )
    elif latest_snapshot is not None:
        extracted_cache, model_cache_config = self._extract_cache_states(
            latest_snapshot
        )
        if not extracted_cache:
            return None
    else:
        return None

    # Eagerly extract in-memory intermediate snapshots here on the inference
    # thread, so the async store-cache worker never touches MLX extraction on
    # its own thread. Snapshots already offloaded to SSD stay lazy: the worker
    # loads pre-extracted state through the store, not raw snapshots. Passing
    # extract_fn=None also means later provider[tc] access returns the
    # already-extracted value without re-invoking _extract_cache_states.
    intermediate_tcs = [tc for tc in valid_counts if tc != latest_tc]
    pre_extracted: dict[int, list[dict[str, Any]]] = {}
    for tc in intermediate_tcs:
        snap = snapshots.get(tc)
        if snap is None:
            continue
        extracted_tc, _ = self._extract_cache_states(snap)
        if extracted_tc:
            pre_extracted[tc] = extracted_tc
    intermediate_snapshots = _BoundarySnapshotProvider(
        store=self._boundary_snapshot_store,
        request_id=request_id,
        valid_tcs=intermediate_tcs,
        in_memory_snapshots=pre_extracted,
        extract_fn=None,
    )

    token_sequence = (
        full_token_sequence[:latest_tc]
        if latest_tc < total_tokens
        else full_token_sequence
    )

    return (
        token_sequence,
        extracted_cache,
        model_cache_config,
        intermediate_snapshots,
    )


@staticmethod
def _merge_boundary_with_full_cache(
    boundary_cache: list[dict[str, Any]],
    full_cache: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill placeholder layers in boundary cache from full extracted cache.

    Boundary snapshots skip sliceable (KVCache) layers to save memory,
    leaving them as ``{'state': (), ...}`` placeholders.  For block
    storage the KV tensors are needed, so we copy them from the full
    extracted cache (which contains the complete sequence).
    """
    if not full_cache or len(boundary_cache) != len(full_cache):
        return boundary_cache

    merged = []
    for bc, fc in zip(boundary_cache, full_cache):
        state = bc.get("state", ())
        # Placeholder layers have state == () (empty tuple).
        if isinstance(state, tuple) and len(state) == 0:
            # Take full cache layer instead.
            merged.append(fc)
        else:
            merged.append(bc)
    return merged


def _validate_cache(self, cache: Any) -> bool:
    """
    Validate that a cache object is usable.

    This prevents NoneType errors when mlx-lm's BatchKVCache
    contains invalid/stale references.

    Args:
        cache: The cache object to validate

    Returns:
        True if cache is valid and usable
    """
    if cache is None:
        return False

    # Check if it's a list of cache layers
    if isinstance(cache, list):
        if len(cache) == 0:
            return False
        # Check each layer
        for layer_cache in cache:
            if layer_cache is None:
                return False
            # Check if layer has expected structure
            # RotatingKVCache may have keys=None (legacy) or zero-length
            # keys (hybrid window padding). Both are valid empty states
            # that will be filled during padding reprocessing.
            if hasattr(layer_cache, "keys") and layer_cache.keys is None:
                if hasattr(layer_cache, "max_size"):
                    continue  # Valid empty RotatingKVCache (keys=None)
                return False
            if hasattr(layer_cache, "values") and layer_cache.values is None:
                if hasattr(layer_cache, "max_size"):
                    continue  # Valid empty RotatingKVCache (values=None)
                return False

    # Check BatchKVCache structure
    if hasattr(cache, "caches"):
        if cache.caches is None:
            return False
        for c in cache.caches:
            if c is None:
                return False

    return True


def _normalize_rotating_snapshot_state(
    self,
    layer_cache: Any,
    state: tuple[Any, Any],
    meta_state: Any,
    layer_idx: int | None = None,
) -> tuple[tuple[Any, Any], tuple[str, str, str, str]]:
    """
    Normalize RotatingKVCache state into merge-safe canonical form.

    Boundary snapshots captured mid-prefill can expose oversized rotating
    buffers (e.g., max_size + chunk_size - 1). Those states are valid for
    in-flight prefill but break BatchRotatingKVCache.merge() after SSD
    restore because merge expects per-request rotating buffers capped to
    max_size. This method canonicalizes to the latest max_size tokens.
    """
    if not isinstance(state, (list, tuple)) or len(state) < 2:
        return state, (
            tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
        )

    keys = state[0]
    values = state[1]
    if keys is None or values is None or not hasattr(keys, "shape"):
        return state, (
            tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
        )

    try:
        keep = (
            int(meta_state[0])
            if meta_state and len(meta_state) >= 1
            else int(getattr(layer_cache, "keep", 0))
        )
        max_size = (
            int(meta_state[1])
            if meta_state and len(meta_state) >= 2
            else int(getattr(layer_cache, "max_size", keys.shape[2]))
        )
        offset = (
            int(meta_state[2])
            if meta_state and len(meta_state) >= 3
            else int(getattr(layer_cache, "offset", keys.shape[2]))
        )
        idx = (
            int(meta_state[3])
            if meta_state and len(meta_state) >= 4
            else int(getattr(layer_cache, "_idx", keys.shape[2]))
        )
    except Exception:
        return state, (
            tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
        )

    ordered_keys = keys
    ordered_values = values
    temporal_order = getattr(layer_cache, "_temporal_order", None)
    if callable(temporal_order):
        try:
            ordered_keys = temporal_order(keys)
            ordered_values = temporal_order(values)
        except Exception:
            ordered_keys = keys
            ordered_values = values

    original_len = int(ordered_keys.shape[2]) if len(ordered_keys.shape) >= 3 else 0
    normalized_keys = ordered_keys
    normalized_values = ordered_values

    if max_size > 0 and original_len > max_size:
        if keep > 0 and keep < max_size:
            tail_len = max_size - keep
            normalized_keys = mx.concatenate(
                [
                    ordered_keys[..., :keep, :],
                    ordered_keys[..., -tail_len:, :],
                ],
                axis=2,
            )
            normalized_values = mx.concatenate(
                [
                    ordered_values[..., :keep, :],
                    ordered_values[..., -tail_len:, :],
                ],
                axis=2,
            )
        else:
            normalized_keys = ordered_keys[..., -max_size:, :]
            normalized_values = ordered_values[..., -max_size:, :]

        try:
            normalized_keys = mx.contiguous(normalized_keys)
            normalized_values = mx.contiguous(normalized_values)
        except Exception:
            logger.debug(
                "swallowed exception at fusion_mlx/scheduler/sched_boundary.py:519"
            )

            pass

    normalized_len = (
        int(normalized_keys.shape[2]) if len(normalized_keys.shape) >= 3 else 0
    )
    # Force case 1 of _temporal_order: _idx == keys.shape[2] means the
    # buffer is already in temporal order (which is exactly what the
    # oversized trim above produces — the contiguous tail of the most
    # recent tokens). Anything else lets _temporal_order re-slice the
    # buffer in the rotated branch (case 2), which is wasted work and
    # obscures the merge contract. See cache.py:431-447 for the branches.
    normalized_idx = normalized_len

    normalized_meta = (
        str(keep),
        str(max_size),
        str(offset),
        str(normalized_idx),
    )

    if original_len != normalized_len or idx != normalized_idx:
        layer_tag = f"layer {layer_idx}: " if layer_idx is not None else ""
        logger.debug(
            "%sNormalized RotatingKVCache snapshot: len %s->%s, idx %s->%s, "
            "offset=%s, max_size=%s",
            layer_tag,
            original_len,
            normalized_len,
            idx,
            normalized_idx,
            offset,
            max_size,
        )

    return (normalized_keys, normalized_values), normalized_meta


def _extract_cache_states(
    self,
    raw_cache: list[Any],
) -> tuple[list[dict[str, Any]], Optional["ModelCacheConfig"]]:
    """
    Extract actual tensor state from each layer cache.

    This extracts the real KV data using mlx-lm's cache.state property,
    allowing the data to be stored and reconstructed later even after
    the BatchGenerator is recreated.

    Also creates a ModelCacheConfig with per-layer type information to
    support hybrid cache models (e.g., KVCache + ArraysCache).

    Args:
        raw_cache: List of cache objects from mlx-lm (KVCache, ArraysCache, etc.)

    Returns:
        Tuple of:
        - List of dicts with {state, meta_state, class_name, cache_type}
        - ModelCacheConfig with per-layer type information (or None)
    """
    if not raw_cache:
        return [], None

    # Build ModelCacheConfig for type information.
    # Skip if raw_cache contains None entries (boundary snapshots with
    # sliceable layers replaced by None) — from_cache_list expects real
    # cache objects and would log noisy NoneType warnings.
    model_cache_config = None
    has_none_layers = any(c is None for c in raw_cache)
    if HAS_CACHE_TYPE_HANDLERS and ModelCacheConfig is not None and not has_none_layers:
        try:
            model_cache_config = ModelCacheConfig.from_cache_list(
                raw_cache,
                model_name=self.model_name if hasattr(self, "model_name") else "",
            )
        except Exception as e:
            logger.debug(f"Failed to build ModelCacheConfig: {e}")

    extracted = []
    for layer_idx, layer_cache in enumerate(raw_cache):
        # Boundary snapshots may contain None for sliceable layers
        # (KVCache) that were skipped during capture to save memory.
        # Insert a placeholder to preserve layer index alignment.
        if layer_cache is None:
            extracted.append(
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                }
            )
            continue
        try:
            class_name = type(layer_cache).__name__

            # Determine cache type using registry if available
            cache_type_name = class_name
            if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                try:
                    cache_type = CacheTypeRegistry.detect_cache_type(layer_cache)
                    cache_type_name = cache_type.value
                except Exception:
                    logger.debug(
                        "swallowed exception at fusion_mlx/scheduler/sched_boundary.py:625"
                    )

                    pass

            # CacheList: composite cache with multiple sub-caches
            if cache_type_name == "CacheList" or class_name == "CacheList":
                if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                    try:
                        handler = CacheTypeRegistry.get_handler_by_class_name(
                            "CacheList"
                        )
                        state_dict = handler.extract_state(layer_cache)
                        extracted.append(
                            {
                                "state": state_dict.get("sub_states", []),
                                "meta_state": (
                                    state_dict.get("sub_class_names", []),
                                    state_dict.get("sub_meta_states", []),
                                ),
                                "class_name": "CacheList",
                                "cache_type": "CacheList",
                            }
                        )
                    except Exception as e:
                        logger.debug(f"CacheList handler extraction failed: {e}")
                        extracted.append(
                            {
                                "state": [],
                                "meta_state": ([], []),
                                "class_name": "CacheList",
                                "cache_type": "CacheList",
                            }
                        )
                else:
                    # Fallback: extract sub-cache state/meta without handlers
                    # MUST append to extracted to prevent layer count mismatch (Issue #1)
                    sub_caches = getattr(layer_cache, "caches", ())
                    sub_states = []
                    sub_class_names = []
                    sub_meta_states = []
                    for sc in sub_caches:
                        sub_states.append(sc.state if hasattr(sc, "state") else ())
                        sub_class_names.append(type(sc).__name__)
                        sub_meta_states.append(getattr(sc, "meta_state", ()))
                    extracted.append(
                        {
                            "state": sub_states,
                            "meta_state": (sub_class_names, sub_meta_states),
                            "class_name": "CacheList",
                            "cache_type": "CacheList",
                        }
                    )
                continue

            if hasattr(layer_cache, "state"):
                state = layer_cache.state
                meta = getattr(layer_cache, "meta_state", ())

                if class_name in ("RotatingKVCache", "BatchRotatingKVCache"):
                    state, meta = self._normalize_rotating_snapshot_state(
                        layer_cache,
                        state,
                        meta,
                        layer_idx=layer_idx,
                    )

                # Preserve the full state tuple regardless of length.
                # Legacy 2-tuple caches (KVCache, RotatingKVCache, ...)
                # surface as (keys, values); 3-tuple caches like
                # PoolingCache surface as (buf_kv, buf_gate, pooled);
                # 4-tuple caches like BatchKVCache surface with the
                # extra offset/padding metadata. Downstream
                # serialization (paged_ssd_cache, boundary_snapshot)
                # is N-tuple aware after the cache architecture
                # refactor — see Section 6 of the implementation
                # plan.
                if isinstance(state, (list, tuple)) and len(state) >= 1:
                    # Validate non-None for legacy KV-style caches only.
                    # PoolingCache's buf_kv may legitimately be None
                    # (fresh cache before any update), so skip the
                    # null guard for non-KV cache classes.
                    if (
                        class_name in ("KVCache", "RotatingKVCache", "BatchKVCache")
                        and len(state) >= 2
                    ):
                        if state[0] is None or state[1] is None:
                            logger.debug(
                                f"Layer {layer_idx} ({class_name}) has None keys/values, "
                                f"skipping cache extraction"
                            )
                            return [], None  # Return empty - cache is corrupted

                    extracted.append(
                        {
                            "state": tuple(state),
                            "meta_state": meta,
                            "class_name": class_name,
                            "cache_type": cache_type_name,
                        }
                    )
                else:
                    # Unexpected state format (e.g. a non-tuple scalar).
                    logger.debug(
                        f"Layer {layer_idx} ({class_name}) has unexpected state format"
                    )
                    meta = getattr(layer_cache, "meta_state", ())
                    # Wrap the scalar so downstream code still gets a
                    # tuple-shaped state. This path is essentially dead
                    # in practice — kept defensive only.
                    extracted.append(
                        {
                            "state": (state,),
                            "meta_state": meta,
                            "class_name": class_name,
                            "cache_type": cache_type_name,
                        }
                    )
            elif hasattr(layer_cache, "cache"):
                # ArraysCache style: state stored in .cache list
                cache_list = layer_cache.cache
                if isinstance(cache_list, list) and len(cache_list) >= 2:
                    state = (cache_list[0], cache_list[1])
                    meta = getattr(layer_cache, "meta_state", ())
                    extracted.append(
                        {
                            "state": state,
                            "meta_state": meta,
                            "class_name": class_name,
                            "cache_type": cache_type_name,
                        }
                    )
                else:
                    logger.debug(
                        f"Layer {layer_idx} ({class_name}) has invalid cache list"
                    )
                    continue
            else:
                logger.debug(
                    f"Layer {layer_idx} ({class_name}) has no state or cache attribute"
                )
                continue

        except Exception as e:
            logger.debug(f"Failed to extract state from cache layer {layer_idx}: {e}")
            continue

    if len(extracted) != len(raw_cache):
        logger.debug(
            f"Incomplete cache extraction: {len(extracted)}/{len(raw_cache)} layers"
        )
        return [], None

    return extracted, model_cache_config
