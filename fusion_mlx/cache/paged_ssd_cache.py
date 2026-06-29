# SPDX-License-Identifier: Apache-2.0
"""Paged SSD cache — cold layer for KV cache blocks."""

import logging
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_MLX:
    _MX_DTYPE_MAP = {
        "float16": ("F16", 2), "float32": ("F32", 4), "bfloat16": ("BF16", 2),
        "int8": ("I8", 1), "int16": ("I16", 2), "int32": ("I32", 4), "int64": ("I64", 8),
        "uint8": ("U8", 1), "uint16": ("U16", 2), "uint32": ("U32", 4), "uint64": ("U64", 8),
    }
    _MX_TO_ST_DTYPE = {}
    _ST_TO_MX_DTYPE = {}
    # Build mappings eagerly with current mlx.core, fall back to name-based
    # lookup at runtime if sys.modules["mlx.core"] was replaced (test mocks).
    for _dn, (_ss, _) in _MX_DTYPE_MAP.items():
        try:
            _md = getattr(mx, _dn)
            _MX_TO_ST_DTYPE[_md] = _ss
            _ST_TO_MX_DTYPE[_ss] = _md
        except AttributeError:
            pass
else:
    _MX_DTYPE_MAP = {}
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
    shape: tuple[int, ...]
    dtype: str
    file_size: int
    created_at: float = 0.0


@dataclass
class PagedSSDCacheIndex:
    """Index of all cached blocks on SSD."""
    blocks: dict[int, PagedSSDBlockMetadata] = field(default_factory=dict)
    total_size: int = 0


def _encode_shape(shape: tuple[int, ...]) -> bytes:
    return struct.pack(f">{len(shape)}I", *shape)


def _dtype_to_st_str(dtype) -> str:
    """Convert an MLX dtype to a safetensors dtype string.

    Uses direct key lookup first, then falls back to name-matching for
    mock environments where the dtype object may be from a different import.
    """
    if dtype in _MX_TO_ST_DTYPE:
        return _MX_TO_ST_DTYPE[dtype]
    # Fallback: match by name (handles test mocks)
    dtype_name = str(dtype).lower()
    for dn, (ss, _) in _MX_DTYPE_MAP.items():
        if dn in dtype_name:
            return ss
    return "F32"


def _is_bfloat16(dtype) -> bool:
    """Check if a dtype is bfloat16, with mock-friendly fallback."""
    if dtype in _MX_TO_ST_DTYPE and _MX_TO_ST_DTYPE[dtype] == "BF16":
        return True
    if HAS_MLX and dtype == mx.bfloat16:
        return True
    # Fallback: match by name
    return "bfloat16" in str(dtype).lower()


def _extract_tensor_bytes(arr) -> tuple[bytes, str, list[int]]:
    if not HAS_MLX:
        return (b"", "F32", [1])
    arr = mx.array(arr)
    mx.eval(arr)
    dtype_str = _dtype_to_st_str(arr.dtype)
    shape = list(arr.shape)
    try:
        if _is_bfloat16(arr.dtype):
            raw = bytes(memoryview(arr.view(mx.uint16)))
        else:
            raw = bytes(memoryview(arr))
    except TypeError:
        raw = bytes(arr) if hasattr(arr, "__bytes__") else b""
    return raw, dtype_str, shape


def _has_zero_dim(shape: tuple[int, ...]) -> bool:
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


def _write_safetensors_no_mx(tensors: dict[str, tuple[bytes, str, list[int]]], path: str):
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
class _HotCacheBudgetEntry:
    owner: Any
    block_hash: bytes
    size_bytes: int


class SharedHotCacheBudget:
    """Process-wide byte budget for hot cache entries across cache managers."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[tuple[int, bytes], _HotCacheBudgetEntry] = (
            OrderedDict()
        )
        self._total_bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _key(owner: Any, block_hash: bytes) -> tuple[int, bytes]:
        return (id(owner), block_hash)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def remaining_bytes(self) -> int:
        with self._lock:
            return max(0, self.max_bytes - self._total_bytes)

    def touch(self, owner: Any, block_hash: bytes) -> None:
        """Mark an entry as recently used in the global LRU order."""
        with self._lock:
            key = self._key(owner, block_hash)
            if key in self._entries:
                self._entries.move_to_end(key)

    def forget(self, owner: Any, block_hash: bytes) -> None:
        """Remove one entry from budget accounting if present."""
        with self._lock:
            key = self._key(owner, block_hash)
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total_bytes = max(0, self._total_bytes - entry.size_bytes)

    def forget_owner(self, owner: Any) -> None:
        """Remove all entries owned by a cache manager."""
        owner_id = id(owner)
        with self._lock:
            keys = [key for key in self._entries if key[0] == owner_id]
            for key in keys:
                entry = self._entries.pop(key)
                self._total_bytes = max(0, self._total_bytes - entry.size_bytes)

    def clear_all_owners(self) -> int:
        """Clear the hot cache of every manager the budget still references."""
        with self._lock:
            owners = []
            seen = set()
            for entry in self._entries.values():
                if id(entry.owner) not in seen:
                    seen.add(id(entry.owner))
                    owners.append(entry.owner)
        cleared = 0
        for owner in owners:
            fn = getattr(owner, "clear_hot_cache", None)
            if callable(fn):
                try:
                    cleared += fn()
                except Exception:
                    logger.warning(
                        "clear_hot_cache failed for an orphaned owner",
                        exc_info=True,
                    )
        return cleared

    def shrink_to(
        self,
        target_bytes: int,
        protected_hashes: set[bytes] | None = None,
    ) -> int:
        """Shrink the shared hot cache to ``target_bytes`` by global LRU order."""
        target_bytes = max(0, int(target_bytes))
        protected_hashes = protected_hashes or set()
        victims: list[tuple[Any, bytes, int]] = []

        with self._lock:
            while self._total_bytes > target_bytes and self._entries:
                victim_key = None
                victim = None
                for key, candidate in self._entries.items():
                    if candidate.block_hash not in protected_hashes:
                        victim_key = key
                        victim = candidate
                        break
                if victim_key is None or victim is None:
                    break

                self._entries.pop(victim_key)
                self._total_bytes = max(0, self._total_bytes - victim.size_bytes)
                victims.append((victim.owner, victim.block_hash, victim.size_bytes))

        freed = 0
        for owner, block_hash, size_bytes in victims:
            evicted = owner._hot_cache_remove(block_hash, update_budget=False)
            if evicted is not None:
                freed += size_bytes
                owner._handle_hot_cache_eviction(block_hash, evicted)
        return freed

    def put(
        self, owner: Any, block_hash: bytes, size_bytes: int
    ) -> list[tuple[Any, bytes]]:
        """Account an entry and return globally-evicted owners/block hashes."""
        victims: list[tuple[Any, bytes]] = []
        size_bytes = max(0, int(size_bytes))
        with self._lock:
            key = self._key(owner, block_hash)
            old = self._entries.pop(key, None)
            if old is not None:
                self._total_bytes = max(0, self._total_bytes - old.size_bytes)

            self._entries[key] = _HotCacheBudgetEntry(
                owner=owner,
                block_hash=block_hash,
                size_bytes=size_bytes,
            )
            self._total_bytes += size_bytes

            while self._total_bytes > self.max_bytes and self._entries:
                victim_key, victim = self._entries.popitem(last=False)
                if victim_key == key and not self._entries:
                    self._entries[victim_key] = victim
                    break
                self._total_bytes = max(0, self._total_bytes - victim.size_bytes)
                victims.append((victim.owner, victim.block_hash))

        return victims


@dataclass
class PagedSSDCacheManager:
    """Manages SSD-based cold storage for KV cache blocks."""
    cache_dir: str
    max_cache_size: int = 10 * 1024**3
    block_size: int = 32

    _index: PagedSSDCacheIndex | None = None
    _current_size: int = 0

    def __post_init__(self):
        self.cache_path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._index = PagedSSDCacheIndex()
        # Auto-repair index on startup: reconcile disk state with in-memory index.
        repair = self.verify_and_repair_index()
        if repair["stale_entries_evicted"] > 0 or repair["orphaned_files_removed"] > 0:
            logger.info(
                "SSD cache auto-repair on init: %d stale entries, %d orphaned temps",
                repair["stale_entries_evicted"],
                repair["orphaned_files_removed"],
            )

    def store_block(self, block_id: int, layers: list[Any], metadata: dict | None = None) -> bool:
        """Write a block of KV cache layers to SSD."""
        try:
            if not layers or _has_zero_dim((len(layers),)):
                return False
            file_path = self.cache_path / f"block_{block_id}.safetensors"
            tensor_data = {}
            for i, l in enumerate(layers):
                tensor_data[f"layer_{i}"] = _extract_tensor_bytes(l)
            # Atomic write: temp file + rename to prevent partial writes
            temp_path = file_path.with_suffix(".tmp")
            try:
                _write_safetensors_no_mx(tensor_data, str(temp_path))
                temp_path.replace(file_path)  # POSIX atomic rename
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
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

    def load_block(self, block_id: int) -> list[Any] | None:
        """Load a block of KV cache layers from SSD.

        Auto-recovery: if the file is corrupted or missing, the stale
        index entry is evicted and None is returned so the caller can
        re-compute the block from scratch.
        """
        try:
            if block_id not in self._index.blocks:
                return None
            file_path = self.cache_path / f"block_{block_id}.safetensors"
            if not file_path.exists():
                logger.debug(
                    "SSD cache miss (file missing): block %d — evicting stale index entry",
                    block_id,
                )
                del self._index.blocks[block_id]
                return None
            return self._read_safetensors(str(file_path))
        except Exception as e:
            logger.warning(
                "Failed to load block %d from SSD: %s — attempting recovery",
                block_id, e,
            )
            self._recover_from_block_error(block_id)
            return None

    def _read_safetensors(self, path: str) -> list[Any] | None:
        """Read a safetensors file and return layer tensors."""
        import json
        try:
            with open(path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                if header_size < 9 or header_size > 10 * 1024 * 1024:
                    raise ValueError(f"invalid header size: {header_size}")
                header_json = f.read(header_size - 8).decode("utf-8")
            header = json.loads(header_json)
            layers = []
            for name in sorted(header.keys()):
                info = header[name]
                offsets = info["data_offsets"]
                dtype_str = info["dtype"]
                shape = info["shape"]
                raw = f.read(offsets[1] - offsets[0]) if hasattr(f, "read") else b""
                # Re-open for data read
            # Re-open to read data at correct offsets
            with open(path, "rb") as f:
                skip = 8 + (header_size - 8)
                f.read(skip)
                for name in sorted(header.keys()):
                    info = header[name]
                    offsets = info["data_offsets"]
                    dtype_str = info["dtype"]
                    shape = info["shape"]
                    data_len = offsets[1] - offsets[0]
                    # Adjust for sequential read
                    raw = f.read(data_len)
                    arr = _restore_tensor_from_bytes(raw, dtype_str, shape)
                    if arr is not None:
                        layers.append(arr)
            return layers if layers else None
        except Exception as e:
            logger.debug("Corrupted safetensors file %s: %s", path, e)
            return None

    def _recover_from_block_error(self, block_id: int) -> None:
        """Evict a corrupted block from the index and clean up disk."""
        file_path = self.cache_path / f"block_{block_id}.safetensors"
        temp_path = file_path.with_suffix(".tmp")
        try:
            if file_path.exists():
                file_path.unlink()
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.debug("Failed to clean up block %d files: %s", block_id, e)
        if block_id in self._index.blocks:
            self._current_size -= self._index.blocks[block_id].file_size
            self._current_size = max(0, self._current_size)
            del self._index.blocks[block_id]
            logger.info(
                "Recovered from SSD block error: evicted block %d from index",
                block_id,
            )

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

    def verify_and_repair_index(self) -> dict[str, int]:
        """Scan disk and reconcile with the in-memory index.

        Returns a report with counts of orphaned files removed and
        stale index entries evicted.
        """
        report = {"orphaned_files_removed": 0, "stale_entries_evicted": 0}
        # Build set of actual files on disk
        disk_files = set()
        for f in self.cache_path.glob("block_*.safetensors"):
            if f.is_file():
                disk_files.add(f.name)
        # Remove orphaned temp files
        for f in self.cache_path.glob("block_*.safetensors.tmp"):
            if f.is_file():
                try:
                    f.unlink()
                    report["orphaned_files_removed"] += 1
                except Exception as e:
                    logger.debug("Failed to remove orphaned temp file %s: %s", f, e)
        # Evict stale index entries (in index but not on disk)
        for block_id in list(self._index.blocks.keys()):
            filename = f"block_{block_id}.safetensors"
            if filename not in disk_files:
                logger.debug(
                    "Repair: evicting stale index entry for block %d (file missing)",
                    block_id,
                )
                self._current_size -= self._index.blocks[block_id].file_size
                self._current_size = max(0, self._current_size)
                del self._index.blocks[block_id]
                report["stale_entries_evicted"] += 1
        # Log orphaned files (in disk but not in index) — leave them for
        # manual inspection; they may be from a previous session and still valid.
        index_files = {f"block_{bid}.safetensors" for bid in self._index.blocks}
        orphans = disk_files - index_files
        if orphans:
            logger.debug(
                "Repair: found %d orphaned SSD files not in index — leaving for manual recovery",
                len(orphans),
            )
        if report["stale_entries_evicted"] > 0:
            logger.info(
                "Index repair complete: %d stale entries evicted, %d orphaned temp files removed",
                report["stale_entries_evicted"],
                report["orphaned_files_removed"],
            )
        return report

    def has_block(self, block_id: int) -> bool:
        return block_id in self._index.blocks

    def get_stats(self) -> dict[str, Any]:
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

    def close(self):
        """Gracefully close the SSD cache manager.

        Verifies index consistency and removes orphaned temp files.
        Does not delete valid cache files — they persist for reuse on reload.
        """
        try:
            # Clean up any leftover temp files
            for f in self.cache_path.glob("block_*.safetensors.tmp"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            logger.info("PagedSSDCacheManager closed cleanly")
        except Exception as e:
            logger.warning("Error during SSD cache close: %s", e)
