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
import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from fusion_lsa.core.index_space import LearnedProjection, rownorm
from fusion_lsa.core.reference import read
from fusion_lsa.core.repair import repair_memory_delta, select_repair_pages
from fusion_lsa.index.incremental import IncrementalBucketIndex

logger = logging.getLogger(__name__)

_SELECT_LOG = bool(os.environ.get("LSA_SELECT_LOG"))
_PROBE_LO = int(os.environ.get("LSA_PROBE_LO", -1))
_PROBE_HI = int(os.environ.get("LSA_PROBE_HI", -1))
_REPAIR_LOG = bool(os.environ.get("LSA_REPAIR_LOG"))


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


class LsaLayerSelector:
    # One full-attention layer's selection state: the learned projections, the
    # page-boundary bucket index, and the not-yet-closed tail rows.
    def __init__(
        self,
        projection: LearnedProjection,
        config: LsaSelectionConfig,
        rotary_dim: int,
        kv_heads: int,
        kv_bytes_per_token: int = 0,
    ):
        self.projection = projection
        self.config = config
        self.rotary_dim = rotary_dim
        self.kv_heads = kv_heads
        # Bytes of K+V storage one token occupies in one layer's cache; the
        # attention wrapper fills this from its cache geometry so the repair
        # memory-delta log reflects the real expanded-KV cost.
        self.kv_bytes_per_token = kv_bytes_per_token
        self.index = IncrementalBucketIndex(
            config.pool_size, config.bucket_size, config.bucket_cap, config.window_size
        )
        self._tail: list[mx.array] = []
        self._tail_rows = 0

    def reset(self) -> None:
        # Drop all indexed state: a new request must never score its queries
        # against a previous request's keys (stale rows shift every slot and
        # push the needle's bucket outside the visible range)
        self.index = IncrementalBucketIndex(
            self.config.pool_size,
            self.config.bucket_size,
            self.config.bucket_cap,
            self.config.window_size,
        )
        self._tail = []
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

    def select(
        self, pre_rope_q: mx.array, position: int, repair_count: int = 0
    ) -> mx.array:
        # One decode step: [B, Hq, 1, dh] at sequence position -> selected key indices.
        # repair_count > 0 unions that many extra pages (exact q.k scoring over
        # the full page index, same scoring as the select-probe) into the
        # attention set; 0 keeps the static-tier path bit-identical.
        position = int(position)
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
        if repair_count > 0:
            picked = self._append_repair(
                query, position, picked, repair_count
            )
        if _SELECT_LOG:
            # MLX has no boolean-mask indexing: compute min/max over a
            # where-clamped array instead of picked[picked >= 0]
            valid_mask = picked >= 0
            kept_count = int(valid_mask.sum().item())
            probe = -1
            if _PROBE_LO >= 0 and kept_count:
                probe = int(
                    ((picked >= _PROBE_LO) & (picked <= _PROBE_HI)).sum().item()
                )
            big = mx.array(self.index.key_count + self._tail_rows + 1, mx.int32)
            clamped = mx.where(valid_mask, picked, big)
            logger.info(
                "LSA select pos=%d key_count=%d blocks=%d tail=%d picked=%d min=%d max=%d probe=%d",
                position,
                self.index.key_count,
                self.index.block_count,
                self._tail_rows,
                kept_count,
                int(clamped.min().item()) if kept_count else -1,
                int(clamped.max().item()) if kept_count else -1,
                probe,
            )
        if self._tail_rows and position >= self.index.key_count:
            # read's picks sit inside closed pages and the tail is everything
            # past them - disjoint, so no dedupe needed. Duplicate rows would
            # not change the softmax anyway (weights split evenly). No
            # kept_count .item() slice either: the caller masks -1 padding
            # instead, which keeps decode free of GPU syncs.
            tail = mx.arange(self.index.key_count, position + 1, dtype=mx.int32)
            picked = mx.concatenate([picked, tail])
        return picked

    def _append_repair(
        self, query: mx.array, position: int, picked: mx.array, repair_count: int
    ) -> mx.array:
        # Query-dependent repair quota (fusion-lsa #12): pick repair_count
        # extra closed pages by exact q.k over the full page index and add
        # every row of those pages to the attention set. Disjointness holds
        # by construction: the SSOT selector only returns pages outside the
        # static set, static rows ⊆ closed pages, and repair pages ⊆ closed
        # pages (the tail past the last close is never a candidate).
        buckets = self.index.buckets()
        page_size = buckets.cluster_block_size
        per_page = buckets.block_bucket_count
        visible_pages = min(
            position // page_size, self.index.block_count
        )
        if visible_pages <= 0:
            return picked
        # Same scoring as the select-probe: unit query row dot normalized
        # bucket centroids; a page's score is its best bucket.
        scores = query.astype(mx.float32) @ buckets.bucket_centroids.T
        page_scores = np.asarray(
            scores[0][: visible_pages * per_page]
            .reshape(visible_pages, per_page)
            .max(axis=1)
        )
        # Static tier = pages that already contribute rows to `picked`
        # (closed-page rows only; sink/window/tail rows past the last close
        # map outside the page index and are dropped here).
        picked_np = np.asarray(picked)
        in_index = (picked_np >= 0) & (picked_np < self.index.key_count)
        static = (
            np.unique(picked_np[in_index] // page_size)
            if in_index.any()
            else np.array([], dtype=np.int64)
        )
        union = select_repair_pages(page_scores, static, repair_count)
        static_ids = set(int(p) for p in static.tolist())
        repair_ids = [int(p) for p in union.tolist() if int(p) not in static_ids]
        if not repair_ids:
            return picked
        ends = [
            min(p * page_size + page_size, self.index.key_count) for p in repair_ids
        ]
        extra = np.concatenate(
            [np.arange(p * page_size, end, dtype=np.int32) for p, end in zip(repair_ids, ends)]
        )
        if _REPAIR_LOG:
            logger.info(
                "LSA repair pos=%d pages=%s static=%d pages delta_bytes=%d",
                position,
                repair_ids,
                len(static_ids),
                repair_memory_delta(
                    len(repair_ids), page_size, self.kv_bytes_per_token
                ),
            )
        else:
            logger.debug(
                "LSA repair pos=%d pages=%s static=%d pages",
                position,
                repair_ids,
                len(static_ids),
            )
        return mx.concatenate([picked, mx.array(extra, mx.int32)])
