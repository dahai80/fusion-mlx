# SPDX-License-Identifier: Apache-2.0
# P1-4 (catch-up plan G8): TurboQuant compression for the SSD cold tier.
#
# Blocks written to the paged SSD store keep full fp16 KV tensors (E1:
# restore overhead 1.54 ms vs 0.98 ms for TQ at equal hit-rate). This
# module converts a block's plain KVCache layers to TurboQuant packed
# states right before save_block, so the on-disk payload is the same
# `__turboquant_v2__` marker format the live TQ path already restores
# via rebuild_dequantized_kv (prefix_cache TQ restore branch). Non-KV
# layers (GDN nstate, CacheList, arrays caches) pass through untouched.
#
# E2 lesson applied: the codec is constructed once per (head_dim, bits,
# seed) and reused via lru_cache — codec construction was measured at
# 0.28 s/layer when done per call, 6.5 ms when reused.

import logging
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

TQ_COLD_MARKER = "__turboquant_v2__"
TQ_COLD_TYPE = "TurboQuantKVCache"


@lru_cache(maxsize=8)
def _codec(head_dim: int, bits: int, seed: int):
    from mlx_vlm.turboquant import _TurboQuantMSECodec

    return _TurboQuantMSECodec(head_dim, bits, seed)


def quantize_pair(keys: mx.array, values: mx.array, seed: int = 0):
    """Quantize one (keys, values) tensor pair to a TQ packed-state pair.

    The arrays are passed with their full (batch, heads, tokens, head_dim)
    shape: the codec keeps per-vector norms in the leading dims
    (norms.shape[2] must be the token count — _state_length reads it) and
    packs each head_dim vector into uint32 words. This matches exactly how
    the live TurboQuantKVCache packs its state.
    """
    from ..turboquant_kv import auto_select_bits

    head_dim = int(keys.shape[-1])
    bits = auto_select_bits(head_dim)
    codec = _codec(head_dim, bits, seed)
    k_state = codec.quantize(keys)
    v_state = codec.quantize(values)
    return k_state, v_state, bits


def maybe_quantize_block_kv(
    block_kv_data: list,
    layer_cache_types: list[str] | None,
    layer_meta_states: list | None,
    token_count: int,
    seed: int = 0,
) -> tuple[list, list[str] | None, list | None] | None:
    """Convert plain KV layers of a block to TurboQuant packed states.

    Layers that are plain (keys, values) fp16 mx.array pairs become
    (TQ_COLD_MARKER, (key_state, value_state)) markers with cache type
    TQ_COLD_TYPE and a (token_count, bits, seed) meta_state triple — the
    exact convention the TQ restore branch reads (ms[1]=bits, ms[2]=seed).
    Everything else passes through unchanged.

    Returns (block_kv_data, layer_cache_types, layer_meta_states) when at
    least one layer was converted, else None (caller stores as-is).
    """
    changed = False
    # layer_cache_types is SHARED across all blocks of a store_cache call —
    # never mutate it in place (mutating would flip later blocks' gate from
    # KVCache to TQ and silently skip them). Build a per-block copy.
    new_types = list(layer_cache_types) if layer_cache_types is not None else None
    new_meta = list(layer_meta_states) if layer_meta_states is not None else None
    for i, layer in enumerate(block_kv_data):
        if layer_cache_types is not None:
            if i >= len(layer_cache_types) or layer_cache_types[i] != "KVCache":
                continue
        if not (isinstance(layer, (list, tuple)) and len(layer) == 2):
            continue
        keys, values = layer
        if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
            continue
        if keys.ndim != 4 or values.ndim != 4:
            continue
        if keys.shape[-1] != values.shape[-1]:
            continue
        try:
            k_state, v_state, bits = quantize_pair(keys, values, seed)
        except Exception as exc:
            logger.warning(
                "cold TQ: layer %d quantize failed, keeping fp16: %s", i, exc
            )
            continue
        block_kv_data[i] = (TQ_COLD_MARKER, (k_state, v_state))
        if new_types is not None:
            new_types[i] = TQ_COLD_TYPE
        if new_meta is not None:
            new_meta[i] = (str(token_count), str(bits), str(seed))
        changed = True
    if not changed:
        return None
    return block_kv_data, new_types, new_meta
