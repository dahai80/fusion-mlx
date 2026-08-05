# SPDX-License-Identifier: Apache-2.0
# Positioned KV-cache writes for MLX-LM cache layers.
#
# Provides ``positioned_update_and_fetch`` — a non-appending write that
# places keys/values at an arbitrary ``position`` in an MLX-LM
# ``KVCache`` / ``QuantizedKVCache`` layer (the buffer grows step-aligned
# when it would overflow). This is what ``runtime/disk_kv_checkpoint``'s
# public contract references for pre-checkpoint writes: the caller writes
# the trailing tokens at a specific offset before handing the cache list
# to ``write_checkpoint``.
#
# NOTE (#360): the disk-KV-checkpoint *consumer* (the scheduler hook
# ``_maybe_disk_checkpoint`` / ``_safe_disk_checkpoint``) has NOT been
# migrated to ``fusion_mlx.scheduler`` yet (see
# ``tests/unit/test_scheduler_disk_kv_hook.py``). So this module is
# currently exercised only by its own unit tests + the disk_kv_checkpoint
# contract tests; there is no live runtime caller until that hook lands.
# The API is pinned here so the checkpoint docstring
# (``disk_kv_checkpoint.py:443``) is accurate and the integration point
# is ready when the scheduler hook is wired.

from __future__ import annotations

import logging

import mlx.core as mx
from mlx.utils import tree_map

logger = logging.getLogger(__name__)


def _grow_kv_cache(cache, needed_end: int) -> None:
    # Step-aligned buffer growth so the loader never has to allocate a
    # non-step-aligned buffer on first reuse (matches MLX-LM's own
    # KVCache.step = 256). Grows both keys and values in place. No-op
    # when the existing buffer already covers ``needed_end``.
    step = getattr(cache, "step", 256)
    if step <= 0:
        step = 256
    keys = cache.keys
    # QuantizedKVCache stores a nested tuple (uint32 + 2 scale arrays);
    # KVCache stores a plain array. tree_map handles both shapes uniformly.
    if isinstance(keys, tuple):
        cur = keys[0].shape[2]
        if needed_end <= cur:
            return
        add = ((needed_end - cur + step - 1) // step) * step

        def _expand(x):
            new_x = mx.zeros(
                (x.shape[0], x.shape[1], x.shape[2] + add, x.shape[3]),
                x.dtype,
            )
            return mx.concatenate([x, new_x], axis=-2)

        cache.keys = tree_map(_expand, cache.keys)
        cache.values = tree_map(_expand, cache.values)
        logger.debug(
            "positioned_kv_cache: grew quantized buffer %d -> %d (step=%d)",
            cur,
            cache.keys[0].shape[2],
            step,
        )
    else:
        cur = keys.shape[2]
        if needed_end <= cur:
            return
        add = ((needed_end - cur + step - 1) // step) * step
        new_k = mx.zeros(
            (keys.shape[0], keys.shape[1], keys.shape[2] + add, keys.shape[3]),
            keys.dtype,
        )
        new_v = mx.zeros(
            (
                cache.values.shape[0],
                cache.values.shape[1],
                cache.values.shape[2] + add,
                cache.values.shape[3],
            ),
            cache.values.dtype,
        )
        cache.keys = mx.concatenate([cache.keys, new_k], axis=2)
        cache.values = mx.concatenate([cache.values, new_v], axis=2)
        logger.debug(
            "positioned_kv_cache: grew dense buffer %d -> %d (step=%d)",
            cur,
            cache.keys.shape[2],
            step,
        )


def _is_quantized(cache) -> bool:
    return isinstance(cache.keys, tuple)


def positioned_update_and_fetch(cache, keys, values, position: int):
    # Write ``keys`` / ``values`` (shape ``[B, n_kv_heads, num_steps, head_dim]``)
    # into ``cache`` starting at ``position`` (NOT appended at ``cache.offset``).
    # The buffer grows step-aligned via ``_grow_kv_cache`` if it would overflow.
    # ``cache.offset`` advances to ``max(cache.offset, position + num_steps)``.
    # Returns the fetched slice up to the new offset (same shape contract as
    # ``KVCache.update_and_fetch``). Works on plain ``KVCache`` and
    # ``QuantizedKVCache``; does NOT subclass, so the layer still round-trips
    # through ``mlx_lm.save_prompt_cache`` / ``load_prompt_cache``.
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    num_steps = keys.shape[2]
    if num_steps <= 0:
        return cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]
    end = position + num_steps
    _grow_kv_cache(cache, end)

    if _is_quantized(cache):
        qk = mx.quantize(keys, group_size=cache.group_size, bits=cache.bits)
        qv = mx.quantize(values, group_size=cache.group_size, bits=cache.bits)
        for i in range(len(cache.keys)):
            cache.keys[i][..., position:end, :] = qk[i]
            cache.values[i][..., position:end, :] = qv[i]
    else:
        cache.keys[..., position:end, :] = keys
        cache.values[..., position:end, :] = values

    prev_offset = cache.offset
    cache.offset = max(cache.offset, end)
    logger.debug(
        "positioned_kv_cache: wrote %d tokens at %d (offset %d -> %d)",
        num_steps,
        position,
        prev_offset,
        cache.offset,
    )
    if _is_quantized(cache):
        return tree_map(lambda x: x[..., : cache.offset, :], (cache.keys, cache.values))
    return cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]
