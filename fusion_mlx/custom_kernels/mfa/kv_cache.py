# SPDX-License-Identifier: Apache-2.0
# KV Cache with MFA-accelerated operations.

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import mlx.core as mx

logger = logging.getLogger(__name__)


@runtime_checkable
class KVCacheProtocol(Protocol):

    def append(self, key: mx.array, value: mx.array) -> None: ...

    def k_for_attention(self) -> mx.array: ...

    def v_for_attention(self) -> mx.array: ...

    @property
    def seqlen(self) -> int: ...

    def reset(self) -> None: ...


class DenseKVCache:

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: mx.Dtype = mx.float16,
        batch_size: int = 1,
    ):
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len
        self._dtype = dtype
        self._batch_size = batch_size
        self._offset = 0
        self._k = mx.zeros((batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)
        self._v = mx.zeros((batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)

    def append(self, key: mx.array, value: mx.array) -> None:
        n_new = key.shape[-2]
        new_offset = self._offset + n_new
        if new_offset > self._max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: offset {self._offset} + {n_new} > max {self._max_seq_len}"
            )
        k_in = key
        v_in = value
        if k_in.ndim == 4 and k_in.shape[1] != self._num_heads:
            k_in = mx.transpose(k_in, (0, 2, 1, 3))
            v_in = mx.transpose(v_in, (0, 2, 1, 3))

        self._k[:, :, self._offset : new_offset, :] = k_in
        self._v[:, :, self._offset : new_offset, :] = v_in
        self._offset = new_offset

    def k_for_attention(self) -> mx.array:
        return self._k[:, :, : self._offset, :]

    def v_for_attention(self) -> mx.array:
        return self._v[:, :, : self._offset, :]

    @property
    def seqlen(self) -> int:
        return self._offset

    def reset(self) -> None:
        self._offset = 0

    def __repr__(self) -> str:
        return (
            f"DenseKVCache(B={self._batch_size}, H={self._num_heads}, "
            f"len={self._offset}/{self._max_seq_len}, D={self._head_dim})"
        )


class PagedKVCache:

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        block_size: int = 64,
        dtype: mx.Dtype = mx.float16,
        num_blocks: int = 1024,
        batch_size: int = 1,
    ):
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._dtype = dtype
        self._batch_size = batch_size
        self._num_blocks = num_blocks

        self._k_pool = mx.zeros(
            (num_blocks, num_heads, block_size, head_dim), dtype=dtype
        )
        self._v_pool = mx.zeros(
            (num_blocks, num_heads, block_size, head_dim), dtype=dtype
        )

        self._block_tables: dict[int, list[int]] = {}
        self._next_free_block = 0
        self._seq_lens: dict[int, int] = {}

    def _alloc_block(self) -> int:
        if self._next_free_block >= self._num_blocks:
            raise RuntimeError("PagedKVCache: out of blocks")
        block = self._next_free_block
        self._next_free_block += 1
        return block

    def append(self, key: mx.array, value: mx.array, seq_id: int = 0) -> None:
        n_new = key.shape[-2]
        k_in = key
        v_in = value
        if k_in.ndim == 4 and k_in.shape[1] != self._num_heads:
            k_in = mx.transpose(k_in, (0, 2, 1, 3))
            v_in = mx.transpose(v_in, (0, 2, 1, 3))

        if seq_id not in self._block_tables:
            self._block_tables[seq_id] = []
            self._seq_lens[seq_id] = 0

        offset = self._seq_lens[seq_id]
        block_table = self._block_tables[seq_id]
        bs = self._block_size

        tokens_written = 0
        while tokens_written < n_new:
            block_idx = offset // bs
            while len(block_table) <= block_idx:
                block_table.append(self._alloc_block())
            block_id = block_table[block_idx]
            in_block_offset = offset % bs
            tokens_this_block = min(bs - in_block_offset, n_new - tokens_written)

            src_start = tokens_written
            src_end = tokens_written + tokens_this_block
            dst_start = in_block_offset
            dst_end = in_block_offset + tokens_this_block

            if k_in.ndim == 4:
                self._k_pool[block_id, :, dst_start:dst_end, :] = k_in[
                    0, :, src_start:src_end, :
                ]
                self._v_pool[block_id, :, dst_start:dst_end, :] = v_in[
                    0, :, src_start:src_end, :
                ]
            else:
                self._k_pool[block_id, :, dst_start:dst_end, :] = k_in[
                    src_start:src_end
                ]
                self._v_pool[block_id, :, dst_start:dst_end, :] = v_in[
                    src_start:src_end
                ]

            tokens_written += tokens_this_block
            offset += tokens_this_block

        self._seq_lens[seq_id] = offset

    def k_for_attention(self, seq_id: int = 0) -> mx.array:
        return self._gather(seq_id, self._k_pool)

    def v_for_attention(self, seq_id: int = 0) -> mx.array:
        return self._gather(seq_id, self._v_pool)

    def _gather(self, seq_id: int, pool: mx.array) -> mx.array:
        seq_len = self._seq_lens.get(seq_id, 0)
        if seq_len == 0:
            return mx.zeros(
                (self._batch_size, self._num_heads, 0, self._head_dim),
                dtype=self._dtype,
            )
        block_table = self._block_tables.get(seq_id, [])
        blocks = []
        bs = self._block_size
        for i, block_id in enumerate(block_table):
            blk = pool[block_id : block_id + 1, :, :, :]
            if i == len(block_table) - 1:
                remaining = seq_len - i * bs
                blk = blk[:, :, :remaining, :]
            blocks.append(blk)
        return mx.concatenate(blocks, axis=-2)

    @property
    def seqlen(self) -> int:
        return self._seq_lens.get(0, 0)

    def reset(self, seq_id: int | None = None) -> None:
        if seq_id is None:
            self._block_tables.clear()
            self._seq_lens.clear()
            self._next_free_block = 0
        else:
            if seq_id in self._block_tables:
                del self._block_tables[seq_id]
                del self._seq_lens[seq_id]

    def get_block_table(self, seq_id: int = 0) -> mx.array:
        bt = self._block_tables.get(seq_id, [])
        return mx.array(bt, dtype=mx.int32)

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(blocks={self._next_free_block}/{self._num_blocks}, "
            f"seqs={len(self._seq_lens)}, "
            f"block_size={self._block_size})"
        )


__all__ = [
    "KVCacheProtocol",
    "DenseKVCache",
    "PagedKVCache",
]
