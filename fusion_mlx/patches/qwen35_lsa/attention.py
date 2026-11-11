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
from .tier import TierConfig, TIER_THRESHOLD

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
    _TQ_REBUILD_CHUNK = 4096

    def __init__(self, attn: nn.Module, selector: LsaLayerSelector, tier: TierConfig | None = None, layer_index: int = -1):
        super().__init__()
        self.attn = attn
        self.selector = selector
        self.tier = tier
        self.layer_index = layer_index
        self.last_picked: mx.array | None = None
        self._state_cache = None
        self._pending_rebuild = False
        self._state_width = 0
        self._state_rows = 0
        self._tq_indexed = False

    def _tier_prefill(self, queries, gate, keys, values, attn) -> mx.array:
        # Three-tier attention over the whole history for this chunk's query
        # rows (HSA Pivot A prefill path). queries (1, Hq, L, D) post-RoPE;
        # keys/values (1, Hkv, T, Dv) full history post-update. Decode keeps
        # using the LSA selector - the cache still holds every row.
        from hsa.attention.three_tier_mlx import three_tier_attention_fast_mlx

        tc = self.tier
        w_q_lat, w_k_lat = tc.weights[self.layer_index]
        out = three_tier_attention_fast_mlx(
            queries[0].transpose(1, 0, 2),
            keys[0].transpose(1, 0, 2),
            values[0].transpose(1, 0, 2),
            tc.window,
            tc.block_size,
            w_q_lat,
            w_k_lat,
            latent_clamp=tc.latent_clamp,
            q_offset=int(keys.shape[2]) - queries.shape[2],
        )
        output = out.astype(queries.dtype).transpose(1, 0, 2).reshape(1, queries.shape[2], -1)
        return attn.o_proj(output * mx.sigmoid(gate))

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
            # prefill (any L) and batched decode: dense path, except
            # single-stream prefill above the tier threshold, which runs the
            # three-tier kernel. The selection index is rebuilt from the cache
            # at the first single-stream decode step, so prefill does no
            # selector work here
            keys, values = (
                cache.update_and_fetch(keys, values)
                if cache is not None
                else (keys, values)
            )
            if (
                B == 1
                and self.tier is not None
                and cache is not None
                and int(cache.offset) > TIER_THRESHOLD
            ):
                return self._tier_prefill(queries, gate, keys, values, attn)
            output = scaled_dot_product_attention(
                queries, keys, values, cache, attn.scale, mask
            )
            return attn.o_proj(
                output.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate)
            )

        offset = int(cache.offset) if cache is not None else 0
        dequant = getattr(cache, "dequantize", None) if cache is not None else None
        if self._pending_rebuild or (dequant is not None and not self._tq_indexed):
            # RoPE touches only the first rotary_dim dims and the index reads
            # the remaining free dims, so the cache's post-rope keys are valid
            # index input; rebuild covers prefix rows a radix cache hit
            # restored without prefill passing through this layer
            self._pending_rebuild = False
            if offset > 0:
                if dequant is not None:
                    # TurboQuant stores packed quantized states; the index
                    # needs dequantized rows (8-bit keys keep top-k agreement
                    # at ~0.99 per the TQ KV probe). Dequantize in chunks -
                    # a whole-history fp16 copy costs ~4 GB per layer at 1M
                    # tokens, so rows feed the index and are discarded.
                    self._tq_indexed = True
                    self.selector.reset()
                    from mlx_vlm.turboquant import _slice_state_range

                    for start in range(0, offset, self._TQ_REBUILD_CHUNK):
                        end = min(start + self._TQ_REBUILD_CHUNK, offset)
                        state = _slice_state_range(cache.state[0], start, end)
                        chunk = cache.key_codec.dequantize(state)
                        self.selector.append_keys(chunk.astype(keys.dtype))
                else:
                    cached = getattr(cache, "keys", None)
                    # KVCache.keys is a preallocated buffer padded with zero
                    # rows beyond offset - only the first offset rows are real
                    if cached is not None and cached.shape[2] >= offset:
                        self.selector.reset()
                        self.selector.append_keys(cached[..., :offset, :])
            elif dequant is not None:
                self._tq_indexed = True
        self.selector.append_keys(keys)
        picked = self.selector.select(queries, offset)
        self.last_picked = picked
        if dequant is not None:
            # the quantized state returned by update_and_fetch cannot be
            # gathered along time, so decode gathers the picked rows straight
            # from the packed quantized state and dequantizes only those
            cache.update_and_fetch(keys, values)
            keys_sel, values_sel = None, None
        else:
            keys, values = cache.update_and_fetch(keys, values)
        if dequant is not None:
            # read pads unused slots with -1; clamp them to row 0 and mask
            # them out of the softmax instead of syncing on a slice
            valid = picked >= 0
            idx = mx.where(valid, picked, 0).astype(mx.int32)
            keys_sel, values_sel = cache.gather_dequantize(idx)
            attn_mask = (valid.astype(queries.dtype) - 1.0) * 1e9
        else:
            # read pads unused slots with -1; clamp them to row 0 and mask
            # them out of the softmax instead of syncing on a kept_count slice
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
    tier_config: TierConfig | None = None,
) -> list[int]:
    # Wrap every full-attention layer named in the map (layer index = position
    # in the layer list); returns the wrapped indices for the armed log.
    wrapped: list[int] = []
    tier_wrapped: list[int] = []
    for index, layer in enumerate(layers):
        if index not in lsa_map.layers or not hasattr(layer, "self_attn"):
            continue
        roles = lsa_map.projections[index]
        projection = LearnedProjection(
            roles["query_projection"], roles["key_projection"]
        )
        selector = build_selector(projection, config, rotary_dim, kv_heads)
        layer_tier = None
        if tier_config is not None and index in tier_config.weights:
            layer_tier = tier_config
            tier_wrapped.append(index)
        layer.self_attn = LsaAttention(layer.self_attn, selector, layer_tier, index)
        wrapped.append(index)
    if wrapped:
        logger.info("LSA wrapped %d full-attention layers: %s", len(wrapped), wrapped)
    else:
        logger.warning(
            "LSA armed but no layer matched the map - dense serving continues"
        )
    if tier_wrapped:
        logger.info("tier prefill wrapped %d layers: %s", len(tier_wrapped), tier_wrapped)
    if wrapped:
        logger.info("LSA wrapped %d full-attention layers: %s", len(wrapped), wrapped)
    else:
        logger.warning(
            "LSA armed but no layer matched the map - dense serving continues"
        )
    return wrapped
