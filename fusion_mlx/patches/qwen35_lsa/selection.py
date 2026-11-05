# SPDX-License-Identifier: Apache-2.0
"""LSA decode selection for one layer: incremental bucket index + per-step read.

Serving-side composition of the fusion_lsa primitives: position-free key/query
heads regroup into the shared index space, the learned map projects them, keys
enter the bucket index one closed page at a time, and every decode step reads
top-p buckets expanded to top-k keys plus the sink and the window. Keys past
the last closed page (the mid-page tail) are always attended - they are the
most recent rows and the window's own territory, and closing their page moves
them into the index wholesale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx
from fusion_lsa.core.index_space import LearnedProjection, rownorm
from fusion_lsa.core.reference import read
from fusion_lsa.index.incremental import IncrementalBucketIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LsaSelectionConfig:
    pool_size: int = 4
    bucket_size: int = 8
    bucket_cap: int = 16
    top_p: int = 1024
    top_k: int = 4096
    sink_size: int = 4
    window_size: int = 512

    def __post_init__(self) -> None:
        if self.bucket_size > self.window_size or self.window_size % self.bucket_size:
            raise ValueError(
                f"bucket_size {self.bucket_size} must divide window_size {self.window_size}"
            )


def _dedupe_sorted(values: mx.array) -> mx.array:
    # sorted int32 values with duplicates dropped; this MLX has no unique/compress,
    # so kept elements scatter into their compacted slots (duplicates overwrite
    # themselves with the same value)
    ordered = mx.sort(values)
    keep = mx.concatenate([mx.array([True]), ordered[1:] != ordered[:-1]])
    kept = int(keep.sum().item())
    slot = mx.cumsum(keep.astype(mx.int32)) - 1
    out = mx.full((kept,), -1, dtype=ordered.dtype)
    return mx.put_along_axis(out[None, :], slot[None, :], ordered[None, :], axis=1)[0]


class LsaLayerSelector:
    # One full-attention layer's selection state: the learned projections, the
    # page-boundary bucket index, and the not-yet-closed tail rows.
    def __init__(
        self,
        projection: LearnedProjection,
        config: LsaSelectionConfig,
        rotary_dim: int,
        kv_heads: int,
    ):
        self.projection = projection
        self.config = config
        self.rotary_dim = rotary_dim
        self.kv_heads = kv_heads
        self.index = IncrementalBucketIndex(
            config.pool_size, config.bucket_size, config.bucket_cap, config.window_size
        )
        self._tail: list[mx.array] = []
        self._tail_rows = 0

    def _position_free(self, heads: mx.array) -> mx.array:
        # heads [B, H, L, dh] pre-RoPE; a prefix rotary layout keeps the tail dims
        return heads[..., self.rotary_dim :]

    def index_keys(self, pre_rope_k: mx.array) -> mx.array:
        # [B, Hkv, T, dh] -> [B, T, index_dim] unit rows through the learned key map
        if pre_rope_k.shape[1] != self.kv_heads:
            raise ValueError(
                f"keys carry {pre_rope_k.shape[1]} kv heads, selector built for {self.kv_heads}"
            )
        k_free = self._position_free(pre_rope_k)
        batch, _, tokens, dim = k_free.shape
        keys = k_free.transpose(0, 2, 1, 3).reshape(batch, tokens, self.kv_heads * dim)
        keys = LearnedProjection.project_rows(
            rownorm(keys), self.projection.key_projection
        )
        return keys

    def index_queries(self, pre_rope_q: mx.array) -> mx.array:
        # [B, Hq, L, dh] -> [B, L, index_dim]: group-sum the free dims, then the query map
        q_free = self._position_free(pre_rope_q)
        batch, q_heads, length, dim = q_free.shape
        if q_heads % self.kv_heads:
            raise ValueError(
                f"{q_heads} query heads do not group by {self.kv_heads} kv heads"
            )
        grouped = (
            q_free.reshape(batch, self.kv_heads, q_heads // self.kv_heads, length, dim)
            .sum(axis=2)
            .transpose(0, 2, 1, 3)
            .reshape(batch, length, self.kv_heads * dim)
        )
        return LearnedProjection.project_rows(
            rownorm(grouped), self.projection.query_projection
        )

    def append_keys(self, pre_rope_k: mx.array) -> None:
        # Feed one prefill chunk's keys [B, Hkv, T, dh]; whole pages close into
        # the bucket index, the remainder waits in the tail
        keys = self.index_keys(pre_rope_k)[0]
        page = self.config.window_size
        available = self._tail_rows + keys.shape[0]
        closable = (available // page) * page - self._tail_rows
        if closable > 0:
            if self._tail:
                merged = mx.concatenate([*self._tail, keys[:closable]], axis=0)
                self._tail = []
                self._tail_rows = 0
            else:
                merged = keys[:closable]
            for start in range(0, closable, page):
                self.index.append(merged[start : start + page])
            rest = keys[closable:]
        else:
            rest = keys
        if rest.shape[0]:
            self._tail.append(rest)
            self._tail_rows += rest.shape[0]
        logger.debug(
            "layer selector: %d closed blocks, %d tail keys",
            self.index.block_count,
            self._tail_rows,
        )

    def select(self, pre_rope_q: mx.array, position: int) -> mx.array:
        # One decode step: [B, Hq, 1, dh] at sequence position -> selected key indices
        if self.index.key_count == 0:
            window_start = max(0, position + 1 - self.config.window_size)
            return mx.arange(window_start, position + 1, dtype=mx.int32)
        query = self.index_queries(pre_rope_q)[:, 0, :]
        buckets = self.index.buckets()
        selected, _ = read(
            query,
            mx.array([position]),
            buckets,
            self.config.top_p,
            self.config.top_k,
            self.config.sink_size,
            self.config.window_size,
        )
        picked = selected[0]
        kept_count = int((picked >= 0).sum().item())
        picked = picked[:kept_count]
        if self._tail_rows and position >= self.index.key_count:
            tail = mx.arange(self.index.key_count, position + 1, dtype=mx.int32)
            picked = _dedupe_sorted(mx.concatenate([picked, tail]))
        return picked
