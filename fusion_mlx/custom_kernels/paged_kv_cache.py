from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_DEFAULT_SLAB_SIZE = 64


class FusionPagedKVCache:
    """Block-paged KV cache with lazy slab growth.

    Physical blocks live in growable slabs (each a single contiguous tensor
    holding slab_size blocks), allocated on demand up to a num_blocks cap.
    A block_table maps logical block index -> physical block index; a
    free-list recycles physical indices on trim/evict. Lazy slabs keep Metal
    buffer count and memory proportional to blocks actually used, not the
    cap.

    Stores raw (unquantized) KV per block so attention math is exact; quantize
    on fetch is a Phase 2 concern (fused GQA decode-attention GEMV).
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 256,
        slab_size: int = _DEFAULT_SLAB_SIZE,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if slab_size < 1:
            raise ValueError("slab_size must be >= 1")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.slab_size = slab_size
        self.offset = 0
        self.keys_slabs: list = []
        self.values_slabs: list = []
        self.block_table: list = []
        self.free_list: list = []
        self._shape: tuple | None = None
        self._v_head_dim: int = 0
        self._dtype: Any = None
        self._evicted = 0
        self._swapped = 0
        self._num_alloc = 0
        self._total_blocks = 0

    def _ensure_pool(
        self,
        B: int,
        n_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        dtype: Any,
    ):
        if self.keys_slabs:
            return
        self._shape = (B, n_kv_heads, self.block_size, k_head_dim)
        self._v_head_dim = v_head_dim
        self._dtype = dtype
        logger.debug(
            "paged_kv pool init: cap=%d block_size=%d slab_size=%d shape=%s dtype=%s",
            self.num_blocks,
            self.block_size,
            self.slab_size,
            self._shape,
            dtype,
        )

    def _add_slab(self) -> int:
        if self._total_blocks >= self.num_blocks:
            return -1
        n = min(self.slab_size, self.num_blocks - self._total_blocks)
        B, n_kv_heads, _bs, k_head_dim = self._shape
        v_head_dim = self._v_head_dim
        k_slab = mx.zeros(
            (n, B, n_kv_heads, self.block_size, k_head_dim), dtype=self._dtype
        )
        v_slab = mx.zeros(
            (n, B, n_kv_heads, self.block_size, v_head_dim), dtype=self._dtype
        )
        self.keys_slabs.append(k_slab)
        self.values_slabs.append(v_slab)
        slab_idx = len(self.keys_slabs) - 1
        base = slab_idx * self.slab_size
        for i in range(n - 1, -1, -1):
            self.free_list.append(base + i)
        self._total_blocks += n
        return slab_idx

    def _alloc_block(self) -> int:
        if not self.free_list:
            slab_idx = self._add_slab()
            if slab_idx < 0:
                raise RuntimeError(
                    f"paged_kv pool exhausted: num_blocks={self.num_blocks} "
                    f"block_size={self.block_size} "
                    "(raise num_blocks or evict other requests via evict_request)"
                )
        idx = self.free_list.pop()
        self._num_alloc += 1
        return idx

    def _slab_loc(self, pb: int):
        slab_idx = pb // self.slab_size
        in_slab = pb % self.slab_size
        return self.keys_slabs[slab_idx], self.values_slabs[slab_idx], in_slab

    def _logical_to_block(self, logical_pos: int) -> int:
        return logical_pos // self.block_size

    def _pos_in_block(self, logical_pos: int) -> int:
        return logical_pos % self.block_size

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        dtype = keys.dtype
        self._ensure_pool(B, n_kv_heads, k_head_dim, v_head_dim, dtype)

        prev = self.offset
        end = prev + num_steps
        first_block = self._logical_to_block(prev)
        last_block = self._logical_to_block(end - 1)

        for lb in range(first_block, last_block + 1):
            while len(self.block_table) <= lb:
                self.block_table.append(self._alloc_block())

        for lb in range(first_block, last_block + 1):
            block_start_logical = lb * self.block_size
            block_end_logical = block_start_logical + self.block_size
            s_start = max(prev, block_start_logical) - prev
            s_end = min(end, block_end_logical) - prev
            n = s_end - s_start
            if n <= 0:
                continue
            pb = self.block_table[lb]
            k_slab, v_slab, in_slab = self._slab_loc(pb)
            pos_start = self._pos_in_block(max(prev, block_start_logical))
            k_slab[in_slab, ..., pos_start : pos_start + n, :] = keys[
                ..., s_start:s_end, :
            ]
            v_slab[in_slab, ..., pos_start : pos_start + n, :] = values[
                ..., s_start:s_end, :
            ]

        self.offset = end
        return self._fetch_logical(end)

    def _fetch_logical(self, length: int):
        num_full = length // self.block_size
        rem = length % self.block_size
        k_parts = []
        v_parts = []
        for lb in range(num_full):
            pb = self.block_table[lb]
            k_slab, v_slab, in_slab = self._slab_loc(pb)
            k_parts.append(k_slab[in_slab])
            v_parts.append(v_slab[in_slab])
        if rem:
            lb = num_full
            if lb < len(self.block_table):
                pb = self.block_table[lb]
                k_slab, v_slab, in_slab = self._slab_loc(pb)
                k_parts.append(k_slab[in_slab, ..., :rem, :])
                v_parts.append(v_slab[in_slab, ..., :rem, :])
        if not k_parts:
            d = self._shape[-1] if self._shape else 1
            empty = mx.zeros((1, 1, 0, d), dtype=mx.float16)
            return empty, empty
        all_k = mx.concatenate(k_parts, axis=-2) if len(k_parts) > 1 else k_parts[0]
        all_v = mx.concatenate(v_parts, axis=-2) if len(v_parts) > 1 else v_parts[0]
        return all_k, all_v

    @property
    def state(self):
        return self._fetch_logical(self.offset)

    @state.setter
    def state(self, v):
        keys, values = v
        if keys is None:
            return
        B, n_kv_heads, length, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        self._ensure_pool(B, n_kv_heads, k_head_dim, v_head_dim, keys.dtype)
        self.block_table = []
        self.free_list = []
        self.keys_slabs = []
        self.values_slabs = []
        self._total_blocks = 0
        self.offset = 0
        num_blocks_needed = (length + self.block_size - 1) // self.block_size
        for lb in range(num_blocks_needed):
            self.block_table.append(self._alloc_block())
        for lb in range(num_blocks_needed):
            block_start_logical = lb * self.block_size
            block_end_logical = block_start_logical + self.block_size
            s_start = max(0, block_start_logical)
            s_end = min(length, block_end_logical)
            n = s_end - s_start
            if n <= 0:
                continue
            pb = self.block_table[lb]
            k_slab, v_slab, in_slab = self._slab_loc(pb)
            pos_start = self._pos_in_block(max(0, block_start_logical))
            k_slab[in_slab, ..., pos_start : pos_start + n, :] = keys[
                ..., s_start:s_end, :
            ]
            v_slab[in_slab, ..., pos_start : pos_start + n, :] = values[
                ..., s_start:s_end, :
            ]
        self.offset = length

    @property
    def meta_state(self):
        return ",".join(map(str, (self.offset, self.block_size, self.num_blocks)))

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self.offset = int(parts[0])
        self.block_size = int(parts[1])
        self.num_blocks = int(parts[2])

    def is_trimmable(self):
        return True

    def size(self):
        return self.offset

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        new_num_blocks = (self.offset + self.block_size - 1) // self.block_size
        while len(self.block_table) > new_num_blocks:
            pb = self.block_table.pop()
            self.free_list.append(pb)
        return n

    def empty(self):
        return self.offset == 0 and not self.block_table

    @property
    def nbytes(self):
        if not self.keys_slabs:
            return 0
        slab_n = self.keys_slabs[0].nbytes + self.values_slabs[0].nbytes
        n_in_slab = self.keys_slabs[0].shape[0]
        per_block = slab_n // n_in_slab
        return per_block * len(self.block_table)

    def free_all(self) -> int:
        freed = len(self.block_table)
        for pb in self.block_table:
            self.free_list.append(pb)
        self.block_table = []
        self.offset = 0
        return freed

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def stats(self) -> dict:
        return {
            "offset": self.offset,
            "blocks_used": len(self.block_table),
            "blocks_free": len(self.free_list),
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "slabs": len(self.keys_slabs),
            "total_blocks": self._total_blocks,
            "evicted": self._evicted,
            "swapped": self._swapped,
            "num_alloc": self._num_alloc,
        }


__all__ = ["FusionPagedKVCache"]
