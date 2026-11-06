# SPDX-License-Identifier: Apache-2.0
"""LSA attention wrapper for qwen3_5 full-attention layers.

Wraps a loaded ``Qwen3NextAttention`` instance: prefill and batched decode
run the dense path unchanged; single-stream decode (L == 1, B == 1) replaces
the dense SDPA with centroid-scored selection (top-p buckets expanded to
top-k + sink + window) gathered from the cache and attended with the same
scaled dot product. The selection index is rebuilt from the cache at the
first decode step of each request - RoPE only rotates the leading dims, so
the cache's post-RoPE keys are valid index input - which also covers radix
cache hits whose restored rows never pass through prefill. The full KV cache
is still maintained every step - the cache surface the scheduler introspects
stays intact; pruning stored rows is a later, separate change.

Batch decode (B > 1) cannot share one per-layer selection, so it falls back
to the dense path until per-sequence selection lands. Cache state is
re-armed per request and on batch reshape: mlx_lm joins and leaves batches
via in-place extend/filter, so object identity alone cannot detect the
transition.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
from fusion_lsa.core.index_space import LearnedProjection
from mlx_lm.models.base import scaled_dot_product_attention

from .selection import LsaLayerSelector, LsaSelectionConfig

logger = logging.getLogger(__name__)


def build_selector(
    projection: LearnedProjection,
    config: LsaSelectionConfig | None,
    rotary_dim: int,
    kv_heads: int,
) -> LsaLayerSelector:
    return LsaLayerSelector(
        projection, config or LsaSelectionConfig(), rotary_dim, kv_heads
    )


class LsaAttention(nn.Module):
    # Drop-in replacement for one full-attention layer's module: owns the
    # original attention (all weights) and the layer's selection state.
    def __init__(self, attn: nn.Module, selector: LsaLayerSelector):
        super().__init__()
        self.attn = attn
        self.selector = selector
        self.last_picked: mx.array | None = None
        self._state_cache = None
        self._pending_rebuild = False
        self._state_width = 0
        self._state_rows = 0
        self._dequant_rows = None

    def _append_dequant_rows(self, keys, values):
        # Grow the preallocated fp16 row buffer with one step's new rows.
        # Buffer doubling amortizes to O(1) copies per row - the concat
        # version re-copied the whole history every step.
        rows_k, rows_v, filled = self._dequant_rows
        n = keys.shape[2]
        if filled + n > rows_k.shape[2]:
            new_cap = max(rows_k.shape[2] * 2, filled + n)
            buf_k = mx.zeros(
                (1, rows_k.shape[1], new_cap, rows_k.shape[3]), dtype=rows_k.dtype
            )
            buf_v = mx.zeros(
                (1, rows_v.shape[1], new_cap, rows_v.shape[3]), dtype=rows_v.dtype
            )
            buf_k[:, :, :filled, :] = rows_k
            buf_v[:, :, :filled, :] = rows_v
            rows_k, rows_v = buf_k, buf_v
        rows_k[:, :, filled:filled + n, :] = keys.astype(rows_k.dtype)
        rows_v[:, :, filled:filled + n, :] = values.astype(rows_v.dtype)
        return (rows_k, rows_v, filled + n)

    def _observe_cache(self, cache) -> None:
        # Detect a new request or a reshaped batch without relying on object
        # identity alone: mlx_lm joins and leaves batches via in-place
        # extend/filter, which keeps the cache object but changes its batch
        # width or shrinks its rows - either invalidates the selection index.
        # A TurboQuant cache stores packed quantized states with no array
        # shape to read - sentinel -1 values leave identity as the only
        # signal there (single-stream requests never reshape in place).
        dequant = getattr(cache, "dequantize", None)
        cached = getattr(cache, "keys", None)
        if dequant is not None:
            width, rows = -1, -1
        else:
            width = int(cached.shape[0]) if cached is not None else 0
            rows = int(cached.shape[2]) if cached is not None else 0
        if (
            cache is not self._state_cache
            or width != self._state_width
            or rows < self._state_rows
        ):
            self.selector.reset()
            self._state_cache = cache
            self._pending_rebuild = True
        self._state_width = width
        self._state_rows = rows

    def __call__(
        self,
        x: mx.array,
        mask=None,
        cache=None,
    ) -> mx.array:
        attn = self.attn
        B, L, _ = x.shape
        num_heads = attn.num_attention_heads
        num_kv_heads = attn.num_key_value_heads

        q_proj_output = attn.q_proj(x)
        queries, gate = mx.split(q_proj_output.reshape(B, L, num_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = attn.k_proj(x), attn.v_proj(x)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(B, L, num_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            # RoPE accepts per-sequence offset arrays; BatchKVCache.offset is
            # a [B] array, so pass it through untouched like the stock layer
            queries = attn.rope(queries, offset=cache.offset)
            keys = attn.rope(keys, offset=cache.offset)
            self._observe_cache(cache)
        else:
            queries = attn.rope(queries)
            keys = attn.rope(keys)

        if L > 1 or B > 1:
            # prefill (any L) and batched decode: dense path. The selection
            # index is rebuilt from the cache at the first single-stream
            # decode step, so prefill does no selector work here
            keys, values = (
                cache.update_and_fetch(keys, values)
                if cache is not None
                else (keys, values)
            )
            output = scaled_dot_product_attention(
                queries, keys, values, cache, attn.scale, mask
            )
            return attn.o_proj(
                output.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate)
            )

        offset = int(cache.offset) if cache is not None else 0
        dequant = getattr(cache, "dequantize", None) if cache is not None else None
        if self._pending_rebuild or (dequant is not None and self._dequant_rows is None):
            # RoPE touches only the first rotary_dim dims and the index reads
            # the remaining free dims, so the cache's post-rope keys are valid
            # index input; rebuild covers prefix rows a radix cache hit
            # restored without prefill passing through this layer
            self._pending_rebuild = False
            if offset > 0:
                if dequant is not None:
                    # TurboQuant stores packed quantized states; the index
                    # needs dequantized rows (8-bit keys keep top-k agreement
                    # at ~0.99 per the TQ KV probe)
                    keys_fp, values_fp = dequant()
                    tokens = keys_fp.shape[2]
                    self._dequant_rows = (
                        keys_fp.astype(keys.dtype),
                        values_fp.astype(values.dtype),
                        tokens,
                    )
                    self.selector.reset()
                    self.selector.append_keys(keys_fp.astype(keys.dtype)[..., :offset, :])
                else:
                    cached = getattr(cache, "keys", None)
                    # KVCache.keys is a preallocated buffer padded with zero
                    # rows beyond offset - only the first offset rows are real
                    if cached is not None and cached.shape[2] >= offset:
                        self.selector.reset()
                        self.selector.append_keys(cached[..., :offset, :])
            elif dequant is not None:
                self._dequant_rows = (
                    mx.zeros((1, keys.shape[1], 256, keys.shape[3]), dtype=keys.dtype),
                    mx.zeros((1, values.shape[1], 256, values.shape[3]), dtype=values.dtype),
                    0,
                )
        self.selector.append_keys(keys)
        picked = self.selector.select(queries, offset)
        self.last_picked = picked
        if dequant is not None:
            # the quantized state returned by update_and_fetch cannot be
            # gathered along time. Dequantize once at rebuild, then grow a
            # preallocated fp16 row buffer with each step's own
            # pre-quantization keys/values. Concat-growing re-copied the whole
            # history every step (~2 GB/step at 16k over 16 layers); buffer
            # writes only touch the new rows.
            cache.update_and_fetch(keys, values)
            self._dequant_rows = self._append_dequant_rows(keys, values)
            keys, values = self._dequant_rows[0], self._dequant_rows[1]
        else:
            keys, values = cache.update_and_fetch(keys, values)
        # read pads unused slots with -1; clamp them to row 0 and mask them
        # out of the softmax instead of syncing on a kept_count slice
        valid = picked >= 0
        idx = mx.where(valid, picked, 0).astype(mx.int32).reshape(1, 1, -1, 1)
        keys_sel = mx.take_along_axis(keys, idx, axis=2)
        values_sel = mx.take_along_axis(values, idx, axis=2)
        attn_mask = (valid.astype(queries.dtype) - 1.0) * 1e9
        output = scaled_dot_product_attention(
            queries, keys_sel, values_sel, None, attn.scale, attn_mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn.o_proj(output * mx.sigmoid(gate))


def wrap_lsa_layers(
    layers: list,
    lsa_map,
    rotary_dim: int,
    kv_heads: int,
    config: LsaSelectionConfig | None = None,
) -> list[int]:
    # Wrap every full-attention layer named in the map (layer index = position
    # in the layer list); returns the wrapped indices for the armed log.
    wrapped: list[int] = []
    for index, layer in enumerate(layers):
        if index not in lsa_map.layers or not hasattr(layer, "self_attn"):
            continue
        roles = lsa_map.projections[index]
        projection = LearnedProjection(
            roles["query_projection"], roles["key_projection"]
        )
        selector = build_selector(projection, config, rotary_dim, kv_heads)
        layer.self_attn = LsaAttention(layer.self_attn, selector)
        wrapped.append(index)
    if wrapped:
        logger.info("LSA wrapped %d full-attention layers: %s", len(wrapped), wrapped)
    else:
        logger.warning(
            "LSA armed but no layer matched the map - dense serving continues"
        )
    return wrapped
