"""CompiledKVCache — mx.compile-compatible KV cache for decode-only path.

Standard KVCache uses Python-level control flow (if/else, shape checks,
dynamic allocation) that breaks mx.compile tracing. This cache uses
concatenate-based updates that are fully traceable.

Strategy: instead of writing into a pre-allocated buffer (which needs
Python int slicing), we concatenate new KV to the existing arrays.
mx.compile can trace this because concatenate is a pure array op.

Trade-off: each decode step grows the arrays by 1 token, causing
reallocation. But mx.compile fuses the concat with the rest of the
graph, and the Metal allocator reuses the old buffer, so the overhead
is minimal in practice.
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class CompiledKVCache:
    """Concatenate-based KV cache compatible with mx.compile.

    Instead of pre-allocating and slicing (which needs Python ints),
    this cache appends new KV via mx.concatenate — a pure array op
    that mx.compile can trace.

    The offset is tracked as a Python int (updated after eval),
    not as an mx.array, because mx.compile needs to see it as
    a constant for shape inference.
    """

    def __init__(self, keys: mx.array, values: mx.array, offset: int):
        self.keys = keys
        self.values = values
        self.offset = offset

    @classmethod
    def from_kv(cls, kv_cache) -> "CompiledKVCache":
        """Create from a standard KVCache after prefill."""
        if kv_cache.keys is None:
            raise ValueError("Cannot create CompiledKVCache from empty KVCache")
        # Take only the filled portion (not the over-allocated buffer)
        return cls(
            keys=kv_cache.keys[..., :kv_cache.offset, :],
            values=kv_cache.values[..., :kv_cache.offset, :],
            offset=kv_cache.offset,
        )

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append new KV and return the full cache.

        Uses concatenate instead of slice-write so mx.compile can trace.
        """
        self.keys = mx.concatenate([self.keys, keys], axis=2)
        self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    def state(self):
        """Return cache state for mx.compile outputs tracking."""
        return [self.keys, self.values]

    def size(self) -> int:
        return self.offset

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        self.keys = self.keys[..., :-n, :] if n > 0 else self.keys
        self.values = self.values[..., :-n, :] if n > 0 else self.values
        return n

    def is_trimmable(self) -> bool:
        return True
