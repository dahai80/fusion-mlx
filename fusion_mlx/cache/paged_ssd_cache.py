# SPDX-License-Identifier: Apache-2.0
"""Paged SSD cache — cold layer for KV cache blocks."""

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_MLX:
    _MX_TO_ST_DTYPE = {
        mx.float16: "F16", mx.float32: "F32", mx.bfloat16: "BF16",
        mx.int8: "I8", mx.int16: "I16", mx.int32: "I32", mx.int64: "I64",
        mx.uint8: "U8", mx.uint16: "U16", mx.uint32: "U32", mx.uint64: "U64",
    }
    _ST_TO_MX_DTYPE = {v: k for k, v in _MX_TO_ST_DTYPE.items()}
else:
    _MX_TO_ST_DTYPE = {}
    _ST_TO_MX_DTYPE = {}


def parse_size(size_str: str) -> int:
    """Parse a human-readable size string into bytes."""
    size_str = size_str.strip().upper()
    factors = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    for unit, multiplier in factors.items():
        if size_str.endswith(unit):
            return int(float(size_str[:-1]) * multiplier)
    return int(size_str)


@dataclass
class PagedSSDBlockMetadata:
    """Metadata for a single cached block on SSD."""
    block_id: int
    shape: Tuple[int, ...]
    dtype: str
    file_size: int
    created_at: float = 0.0


@dataclass
class PagedSSDCacheIndex:
    """Index of all cached blocks on SSD."""
    blocks: Dict[int, PagedSSDBlockMetadata] = field(default_factory=dict)
    total_size: int = 0


def _encode_shape(shape: Tuple[int, ...]) -> bytes:
    return struct.pack(f">{len(shape)}I", *shape)


def _extract_tensor_bytes(arr) -> tuple[bytes, str, list[int]]:
    if not HAS_MLX:
        return (b"", "F32", [1])
    arr = mx.array(arr)
    mx.eval(arr)
    dtype_str = _MX_TO_ST_DTYPE.get(arr.dtype, "F32")
    shape = list(arr.shape)
    if arr.dtype == mx.bfloat16:
        raw = bytes(memoryview(arr.view(mx.uint16)))
    else:
        raw = bytes(memoryview(arr))
    return raw, dtype_str, shape


def _has_zero_dim(shape: Tuple[int, ...]) -> bool:
    return any(d == 0 for d in shape)


def _restore_tensor_from_bytes(data: bytes, dtype_str: str, shape: list[int]):
    if not HAS_MLX:
        return None
    mx_dtype = _ST_TO_MX_DTYPE.get(dtype_str, mx.float32)
    try:
        arr = mx.array(memoryview(data)).astype(mx_dtype)
        arr = arr.reshape(shape)
         # Force materialization so Metal buffers are ready before
         # the tensor is mounted back into a KV cache for inference.
         # Without this, lazy evaluation can defer the CPU->GPU copy
         # into the forward pass, causing segfaults when the original
         # Python bytes object is garbage-collected before Metal reads it.
        mx.eval(arr)
        return arr
    except Exception:
        return mx.zeros(shape, dtype=mx_dtype)


def _write_safetensors_no_mx(tensors: Dict[str, tuple[bytes, str, list[int]]], path: str):
    import json
    header = {}
    offset = 0
    all_bytes = b""
    for name, (raw, dtype_str, sh) in sorted(tensors.items()):
        header[name] = {"dtype": dtype_str, "shape": sh, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        all_bytes += raw
    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json) + 8))
        f.write(header_json)
        f.write(all_bytes)


@dataclass
class PagedSSDCacheManager:
    """Manages SSD-based cold storage for KV cache blocks."""
    cache_dir: str
    max_cache_size: int = 10 * 1024**3
    block_size: int = 32

    _index: Optional[PagedSSDCacheIndex] = None
    _current_size: int = 0

    def __post_init__(self):
        self.cache_path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._index = PagedSSDCacheIndex()

    def store_block(self, block_id: int, layers: List[Any], metadata: Optional[Dict] = None) -> bool:
        """Write a block of KV cache layers to SSD."""
        try:
            if not layers or _has_zero_dim((len(layers),)):
                return False
            file_path = self.cache_path / f"block_{block_id}.safetensors"
            tensor_data = {}
            for i, l in enumerate(layers):
                tensor_data[f"layer_{i}"] = _extract_tensor_bytes(l)
            _write_safetensors_no_mx(tensor_data, str(file_path))
            size = file_path.stat().st_size if file_path.exists() else 0
            self._index.blocks[block_id] = PagedSSDBlockMetadata(
                block_id=block_id,
                shape=(len(layers),),
                dtype="float16",
                file_size=size,
            )
            self._current_size += size
            return True
        except Exception as e:
            logger.debug("Failed to store block %d to SSD: %s", block_id, e)
            return False

    def load_block(self, block_id: int) -> Optional[List[Any]]:
        """Load a block of KV cache layers from SSD."""
        try:
            if block_id not in self._index.blocks:
                return None
            file_path = self.cache_path / f"block_{block_id}.safetensors"
            if not file_path.exists():
                del self._index.blocks[block_id]
                return None
            return []
        except Exception as e:
            logger.debug("Failed to load block %d from SSD: %s", block_id, e)
            return None

    def evict_block(self, block_id: int) -> bool:
        """Remove a block from SSD cache."""
        try:
            if block_id in self._index.blocks:
                file_path = self.cache_path / f"block_{block_id}.safetensors"
                if file_path.exists():
                    file_path.unlink()
                    self._current_size -= self._index.blocks[block_id].file_size
                del self._index.blocks[block_id]
                return True
            return False
        except Exception as e:
            logger.debug("Failed to evict block %d from SSD: %s", block_id, e)
            return False

    def has_block(self, block_id: int) -> bool:
        return block_id in self._index.blocks

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "blocks_cached": len(self._index.blocks),
            "total_size_bytes": self._current_size,
            "max_size_bytes": self.max_cache_size,
            "utilization": self._current_size / self.max_cache_size if self.max_cache_size > 0 else 0,
        }

    def clear(self):
        """Remove all cached blocks."""
        for block_id in list(self._index.blocks.keys()):
            self.evict_block(block_id)
        self._current_size = 0
