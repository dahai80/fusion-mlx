from typing import Any

import mlx.core as mx

from .monkeypatches import _default_generation_stream
from .types import _mx_buffer_access_lock


def _should_clear_on_fragmentation(
    frag_ratio: float, threshold: float = 0.7
) -> bool:
    """Return False when fragmentation is too high to safely clear cache.

    Clearing the Metal buffer cache on a highly fragmented heap forces
    re-allocation from scratch, which amplifies fragmentation and can cause
    I/O thrashing.  Skip the clear when frag_ratio >= threshold.
    """
    return frag_ratio < threshold


def _sync_and_clear_cache(stream=None):
    """Synchronize in-flight GPU work before clearing the Metal buffer cache.

    Without synchronization, mx.clear_cache() can release Metal buffers that
    are still referenced by in-flight command buffers submitted via
    mx.async_eval(). This causes the GPU driver to hit a
    'completeMemory() prepare count underflow' kernel panic on M4 hardware
    (and SIGSEGV/SIGABRT on M3).

    Held under _mx_buffer_access_lock so the async store-cache worker cannot
    observe a half-reclaimed Metal buffer pool while it is in the middle of
    reading tensor bytes via the Python buffer protocol (#1106).

    See: https://github.com/jundot/omlx/issues/300, #888, #1106
    """
    with _mx_buffer_access_lock:
        # The engine stream may not have in-flight work on the current thread
        # (e.g. external prefill submits to the default stream). On some MLX
        # builds mx.synchronize raises "There is no Stream(gpu, 0) in current
        # thread" in that case; swallow it since there is nothing to drain.
        target = stream if stream is not None else _default_generation_stream
        try:
            mx.synchronize(target)
        except RuntimeError:
            pass
        mx.synchronize()   # default stream
        mx.clear_cache()


def _safe_sync_stream(stream=None):
    """mx.synchronize(stream) that tolerates cross-thread calls.

    The per-engine stream is owned by the engine's executor thread. Teardown
    paths that run on the main thread (via EngineCore.close) hit "no Stream in
    current thread" RuntimeError. Swallow that specific case so cleanup can
    proceed; re-raise anything else so real GPU errors stay visible.
    """
    target = stream if stream is not None else _default_generation_stream
    try:
        mx.synchronize(target)
    except RuntimeError as e:
        if "no Stream" not in str(e):
            raise


# Cache class names known to be sliceable (no boundary snapshots needed).
# ChunkedKVCache is included once the batch=1 patch above installs its
# extract/filter/size pass-throughs; without it Llama-4 requests fall
# back to the snapshot path unnecessarily.
_KNOWN_SLICEABLE_CACHE_TYPES = frozenset(
    {
        "KVCache",
        "BatchKVCache",
        "QuantizedKVCache",
        "TurboQuantKVCache",
        "BatchTurboQuantKVCache",
        "ChunkedKVCache",
    }
)


def _prompt_cache_needs_snapshots(prompt_cache: list) -> bool:
    """Return True if any layer cache is non-sliceable (needs snapshots).

    Checks the cache objects created during prefill. If all layers
    are known-sliceable types (e.g. KVCache), boundary snapshots
    are unnecessary and can be skipped entirely.
    """
    for cache_obj in prompt_cache:
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            for sub in sub_caches:
                if type(sub).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES:
                    return True
        elif type(cache_obj).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES:
            return True
    return False


def _cache_layer_token_count(cache_obj: Any) -> int:
    """Return the number of tokens stored in a single cache layer."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)) and sub_caches:
        return max(_cache_layer_token_count(sub_cache) for sub_cache in sub_caches)

    offset = getattr(cache_obj, "offset", None)
    if isinstance(offset, (int, float)):
        return int(offset)

    size_fn = getattr(cache_obj, "size", None)
    if callable(size_fn):
        try:
            return int(size_fn())
        except Exception:
            return 0

    return 0


def _cache_base_sizes(caches: list) -> int:
    """Return the base token count of a single-request cache list."""
    if not caches:
        return 0
    try:
        return max(_cache_layer_token_count(c) for c in caches)
    except Exception:
        return 0


def _vlm_extra_seq_slice(val: mx.array, s: slice) -> mx.array:
    """Slice a VLM extra tensor along its seq dimension.

    Standard layout (batch=1, seq, ...): seq at dim 1.
    Special layout (e.g. mRoPE (3, batch, seq)): seq at last dim.
    """
    if val.ndim >= 3 and val.shape[0] == 1:
        return val[:, s]
    if val.ndim >= 3:
        return val[..., s]
    return val[:, s]


def _slice_vlm_extra(extra: dict, n: int) -> dict:
    """Slice VLM extra kwargs to first n tokens along seq dimension."""
    sliced: dict = {}
    for key, val in extra.items():
        if isinstance(val, mx.array) and val.ndim >= 2:
            sliced[key] = _vlm_extra_seq_slice(val, slice(None, n))
        else:
            sliced[key] = val
    return sliced


def _advance_vlm_extra(extra: dict, n: int) -> dict:
    """Advance VLM extra kwargs past first n tokens along seq dimension."""
    advanced: dict = {}
    for key, val in extra.items():
        if isinstance(val, mx.array) and val.ndim >= 2:
            advanced[key] = _vlm_extra_seq_slice(val, slice(n, None))
        else:
            advanced[key] = val
    return advanced


def _deferred_clear_delay(sched) -> int:
    batch_size = len(getattr(sched, "running", {}))
    delay = getattr(sched, "_DEFERRED_CLEAR_DELAY", 4)
    return max(2, min(16, delay + (batch_size // 4)))


def _get_attr_or_key(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if type(value).__module__.startswith("unittest.mock"):
        return None
    return value


def _model_declares_llama4(model) -> bool:
    seen = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if _get_attr_or_key(obj, "model_type") == "llama4":
            return True
        for attr in ("config", "args", "text_config", "language_config", "llm_config"):
            child = _get_attr_or_key(obj, attr)
            if child is not None and not isinstance(
                child, (str, bytes, int, float, bool)
            ):
                stack.append(child)
    return False


def _model_declares_minimax_m3(model) -> bool:
    seen = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        model_type = _get_attr_or_key(obj, "model_type")
        if isinstance(model_type, str) and "minimax" in model_type.lower():
            if "m3" in model_type.lower() or "M3" in model_type:
                return True
        for attr in ("config", "args", "text_config", "language_config", "llm_config"):
            child = _get_attr_or_key(obj, attr)
            if child is not None and not isinstance(
                child, (str, bytes, int, float, bool)
            ):
                stack.append(child)
    return False


def _seed_text_only_mrope_delta_for_cached_prefill(model, request) -> None:
    if getattr(request, "cached_tokens", 0) <= 0:
        return
    lm = getattr(model, "_language_model", None)
    if lm is None or not hasattr(lm, "_rope_deltas"):
        return
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int64)
