from __future__ import annotations

import logging
from collections import deque
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


class FusionPagedKVPool:
    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        n_kv_heads: int,
        head_dim: int,
        k_head_dim: int | None = None,
        v_head_dim: int | None = None,
        dtype: Any = mx.float16,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.n_kv_heads = n_kv_heads
        self.k_head_dim = k_head_dim or head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.dtype = dtype
        self.keys_pool = mx.zeros(
            (num_blocks, 1, n_kv_heads, block_size, self.k_head_dim), dtype=dtype
        )
        self.values_pool = mx.zeros(
            (num_blocks, 1, n_kv_heads, block_size, self.v_head_dim), dtype=dtype
        )
        self.free_list: deque[int] = deque(range(num_blocks - 1, -1, -1))
        self.in_use: dict[int, str] = {}
        logger.info(
            "paged_kv pool init: cap=%d block_size=%d n_kv=%d head_dim=%d/%d",
            num_blocks,
            block_size,
            n_kv_heads,
            self.k_head_dim,
            self.v_head_dim,
        )

    def alloc_block(self, request_id: str) -> int:
        if not self.free_list:
            logger.error("paged_kv pool exhausted for request=%s", request_id)
            raise RuntimeError(
                f"paged_kv pool exhausted (cap={self.num_blocks}); "
                f"reject request or raise pool_num_blocks"
            )
        pb = self.free_list.pop()
        self.in_use[pb] = request_id
        logger.debug(
            "paged_kv pool alloc block=%d request=%s available=%d",
            pb,
            request_id,
            len(self.free_list),
        )
        return pb

    def free_request(self, request_id: str) -> int:
        freed = [pb for pb, rid in self.in_use.items() if rid == request_id]
        for pb in freed:
            self.in_use.pop(pb, None)
            self.free_list.append(pb)
        logger.info(
            "paged_kv pool free request=%s blocks=%d available=%d",
            request_id,
            len(freed),
            len(self.free_list),
        )
        return len(freed)

    def available(self) -> int:
        return len(self.free_list)

    def stats(self) -> dict:
        return {
            "cap": self.num_blocks,
            "available": self.available(),
            "in_use": len(self.in_use),
        }


class FusionPagedRequestCache:
    def __init__(self, pool: FusionPagedKVPool, request_id: str):
        self.pool = pool
        self.request_id = request_id
        self.block_table: list[int] = []
        self.offset: int = 0
        self._n_kv_heads: int = pool.n_kv_heads
        self._k_head_dim: int = pool.k_head_dim
        self._v_head_dim: int = pool.v_head_dim
        self._dtype: Any = pool.dtype
        self._B: int = 1
        self._is_merged: bool = False
        self._merged_keys: mx.array | None = None
        self._merged_values: mx.array | None = None
        self._merged_padding: list[int] = []

    def _logical_to_block(self, logical_pos: int) -> int:
        return logical_pos // self.pool.block_size

    def _pos_in_block(self, logical_pos: int) -> int:
        return logical_pos % self.pool.block_size

    def update_and_fetch(self, keys, values):
        if self._is_merged:
            raise RuntimeError("cannot update a merged FusionPagedRequestCache")
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        dtype = keys.dtype
        self._B = B
        self._n_kv_heads = n_kv_heads
        self._k_head_dim = k_head_dim
        self._v_head_dim = v_head_dim
        self._dtype = dtype

        prev = self.offset
        end = prev + num_steps
        first_block = self._logical_to_block(prev)
        last_block = self._logical_to_block(end - 1)

        for lb in range(first_block, last_block + 1):
            while len(self.block_table) <= lb:
                pb = self.pool.alloc_block(self.request_id)
                self.block_table.append(pb)
                logger.debug(
                    "paged_kv request=%s block_table grow lb=%d pb=%d",
                    self.request_id,
                    lb,
                    pb,
                )

        for lb in range(first_block, last_block + 1):
            block_start_logical = lb * self.pool.block_size
            block_end_logical = block_start_logical + self.pool.block_size
            s_start = max(prev, block_start_logical) - prev
            s_end = min(end, block_end_logical) - prev
            n = s_end - s_start
            if n <= 0:
                continue
            pb = self.block_table[lb]
            pos_start = self._pos_in_block(max(prev, block_start_logical))
            self.pool.keys_pool[pb, ..., pos_start : pos_start + n, :] = keys[
                ..., s_start:s_end, :
            ]
            self.pool.values_pool[pb, ..., pos_start : pos_start + n, :] = values[
                ..., s_start:s_end, :
            ]

        self.offset = end
        return self._fetch_logical(end)

    def _fetch_logical(self, length: int):
        if self._is_merged:
            return self._merged_keys, self._merged_values
        num_full = length // self.pool.block_size
        rem = length % self.pool.block_size
        k_parts = []
        v_parts = []
        for lb in range(num_full):
            pb = self.block_table[lb]
            k_parts.append(self.pool.keys_pool[pb])
            v_parts.append(self.pool.values_pool[pb])
        if rem:
            lb = num_full
            if lb < len(self.block_table):
                pb = self.block_table[lb]
                k_parts.append(self.pool.keys_pool[pb, ..., :rem, :])
                v_parts.append(self.pool.values_pool[pb, ..., :rem, :])
        if not k_parts:
            d = self._k_head_dim if self._k_head_dim else 1
            empty = mx.zeros((1, 1, 0, d), dtype=self._dtype)
            return empty, empty
        all_k = mx.concatenate(k_parts, axis=-2) if len(k_parts) > 1 else k_parts[0]
        all_v = mx.concatenate(v_parts, axis=-2) if len(v_parts) > 1 else v_parts[0]
        return all_k, all_v

    @property
    def state(self):
        if self._is_merged:
            return self._merged_keys, self._merged_values
        return self._fetch_logical(self.offset)

    @state.setter
    def state(self, v):
        if self._is_merged:
            raise RuntimeError("cannot set state on a merged FusionPagedRequestCache")
        keys, values = v
        if keys is None:
            return
        B, n_kv_heads, length, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        self._B = B
        self._n_kv_heads = n_kv_heads
        self._k_head_dim = k_head_dim
        self._v_head_dim = v_head_dim
        self._dtype = keys.dtype
        self.block_table = []
        self.offset = 0
        num_blocks_needed = (length + self.pool.block_size - 1) // self.pool.block_size
        for lb in range(num_blocks_needed):
            pb = self.pool.alloc_block(self.request_id)
            self.block_table.append(pb)
        for lb in range(num_blocks_needed):
            block_start_logical = lb * self.pool.block_size
            block_end_logical = block_start_logical + self.pool.block_size
            s_start = max(0, block_start_logical)
            s_end = min(length, block_end_logical)
            n = s_end - s_start
            if n <= 0:
                continue
            pb = self.block_table[lb]
            pos_start = self._pos_in_block(max(0, block_start_logical))
            self.pool.keys_pool[pb, ..., pos_start : pos_start + n, :] = keys[
                ..., s_start:s_end, :
            ]
            self.pool.values_pool[pb, ..., pos_start : pos_start + n, :] = values[
                ..., s_start:s_end, :
            ]
        self.offset = length

    @property
    def meta_state(self):
        return ",".join(
            map(str, (self.offset, self.pool.block_size, self.pool.num_blocks))
        )

    @meta_state.setter
    def meta_state(self, v):
        logger.warning(
            "meta_state setter not supported on shared-pool "
            "FusionPagedRequestCache (request=%s); pool geometry is fixed",
            self.request_id,
        )
        raise NotImplementedError(
            "meta_state setter not supported on shared-pool " "FusionPagedRequestCache"
        )

    def is_trimmable(self):
        return True

    def size(self):
        return self.offset

    def trim(self, n):
        if self._is_merged:
            raise RuntimeError("cannot trim a merged FusionPagedRequestCache")
        n = min(self.offset, n)
        self.offset -= n
        new_num_blocks = (
            self.offset + self.pool.block_size - 1
        ) // self.pool.block_size
        while len(self.block_table) > new_num_blocks:
            pb = self.block_table.pop()
            self.pool.in_use.pop(pb, None)
            self.pool.free_list.append(pb)
        return n

    def empty(self):
        return self.offset == 0 and not self.block_table

    @property
    def nbytes(self):
        if self._is_merged:
            return self._merged_keys.nbytes + self._merged_values.nbytes
        return len(self.block_table) * (
            self.pool.keys_pool[0].nbytes + self.pool.values_pool[0].nbytes
        )

    def free_all(self) -> int:
        if self._is_merged:
            freed = 0
            self._merged_keys = None
            self._merged_values = None
            self._is_merged = False
            self.offset = 0
            return freed
        freed = self.pool.free_request(self.request_id)
        self.block_table = []
        self.offset = 0
        return freed

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return cls(pool=caches[0].pool, request_id="__merged_empty__")
        padding = [max_length - l for l in lengths]
        B = len(caches)
        n_kv_heads = caches[0]._n_kv_heads
        k_head_dim = caches[0]._k_head_dim
        v_head_dim = caches[0]._v_head_dim
        dt = caches[0]._dtype
        keys = mx.zeros((B, n_kv_heads, max_length, k_head_dim), dtype=dt)
        values = mx.zeros((B, n_kv_heads, max_length, v_head_dim), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.offset == 0:
                continue
            ck, cv = c.state
            keys[i : i + 1, :, p : p + c.offset, :] = ck[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset, :] = cv[..., : c.offset, :]
        merged = cls(pool=caches[0].pool, request_id="__merged__")
        merged._merged_keys = keys
        merged._merged_values = values
        merged._merged_padding = padding
        merged.offset = max_length
        merged._is_merged = True
        logger.info(
            "paged_kv merge: B=%d max_length=%d padding=%s",
            B,
            max_length,
            padding,
        )
        return merged

    def stats(self) -> dict:
        return {
            "request_id": self.request_id,
            "offset": self.offset,
            "blocks_used": len(self.block_table),
            "block_size": self.pool.block_size,
            "pool_available": self.pool.available(),
            "is_merged": self._is_merged,
        }


__all__ = ["FusionPagedKVPool", "FusionPagedRequestCache"]
