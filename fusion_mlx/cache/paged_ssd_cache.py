# SPDX-License-Identifier: Apache-2.0
"""Paged SSD cache -- cold layer for KV cache blocks."""

import errno
import json
import logging
import os
import queue
import shutil
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = "3"
_READABLE_CACHE_FORMAT_VERSIONS = frozenset({"2", "3"})
_MAX_INLINE_UNLINKS_PER_SAVE = 32

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_MLX:
    _MX_DTYPE_MAP = {
        "float16": ("F16", 2),
        "float32": ("F32", 4),
        "bfloat16": ("BF16", 2),
        "int8": ("I8", 1),
        "int16": ("I16", 2),
        "int32": ("I32", 4),
        "int64": ("I64", 8),
        "uint8": ("U8", 1),
        "uint16": ("U16", 2),
        "uint32": ("U32", 4),
        "uint64": ("U64", 8),
    }
    _MX_TO_ST_DTYPE = {}
    _ST_TO_MX_DTYPE = {}
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


def _canonicalize_layer_cache_types(
    layer_cache_types: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if layer_cache_types is None:
        return None
    wrapper_to_canonical = {
        "SizedArraysCache": "ArraysCache",
        "PrefillReadyRotatingKVCache": "RotatingKVCache",
    }
    return [wrapper_to_canonical.get(ct, ct) for ct in layer_cache_types]


def _cache_compat_signature(
    *,
    model_name: str = "",
    num_layers: int = 0,
    block_size: int = 0,
    layer_cache_types: list[str] | None = None,
) -> str:
    payload = {
        "model_name": model_name or "",
        "num_layers": int(num_layers or 0),
        "block_size": int(block_size or 0),
        "layer_cache_types": list(
            _canonicalize_layer_cache_types(layer_cache_types) or []
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _compute_max_pending_writes(
    block_size_tokens: int = 256,
    kv_bytes_per_token: int = 200_000,
) -> int:
    _FLOOR = 16
    _CEILING = 256
    _HARD_BUDGET_FRACTION = 0.30
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_ram = page_size * phys_pages
    except (ValueError, OSError, AttributeError):
        total_ram = 64 * 1024**3
    bytes_per_slot = max(1, block_size_tokens * kv_bytes_per_token)
    soft_target = max(_FLOOR, total_ram // bytes_per_slot)
    hard_budget = max(_FLOOR, int(total_ram * _HARD_BUDGET_FRACTION / bytes_per_slot))
    cap = min(soft_target, hard_budget)
    return max(_FLOOR, min(cap, _CEILING))


def parse_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    if not size_str:
        raise ValueError("empty size string")
    factors = {
        "TB": 1024**4,
        "GB": 1024**3,
        "MB": 1024**2,
        "KB": 1024,
        "T": 1024**4,
        "G": 1024**3,
        "M": 1024**2,
        "K": 1024,
    }
    for suffix in sorted(factors.keys(), key=len, reverse=True):
        if size_str.endswith(suffix):
            num_part = size_str[: -len(suffix)]
            if not num_part:
                raise ValueError(f"invalid size format: {size_str!r}")
            try:
                return int(float(num_part) * factors[suffix])
            except ValueError:
                raise ValueError(f"invalid size format: {size_str!r}")
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"invalid size format: {size_str!r}")


@dataclass
class PagedSSDBlockMetadata:
    block_hash: bytes = b""
    file_path: Path = field(default_factory=lambda: Path(""))
    file_size: int = 0
    token_count: int = 0
    created_at: float = 0.0
    last_access: float = 0.0
    num_layers: int = 0
    model_name: str = ""
    block_size: int = 0
    cache_signature: str = ""
    layer_cache_types: list[str] | None = None
    layer_meta_states: list[tuple[int, ...]] | None = None

    def touch(self):
        self.last_access = time.time()

    def to_dict(self) -> dict:
        d = {
            "block_hash": self.block_hash.hex(),
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "num_layers": self.num_layers,
            "model_name": self.model_name,
            "block_size": self.block_size,
            "cache_signature": self.cache_signature,
        }
        if self.layer_cache_types is not None:
            d["layer_cache_types"] = list(self.layer_cache_types)
        if self.layer_meta_states is not None:
            d["layer_meta_states"] = [list(s) for s in self.layer_meta_states]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PagedSSDBlockMetadata":
        return cls(
            block_hash=(
                bytes.fromhex(d["block_hash"])
                if isinstance(d.get("block_hash"), str)
                else d.get("block_hash", b"")
            ),
            file_path=Path(d.get("file_path", "")),
            file_size=d.get("file_size", 0),
            token_count=d.get("token_count", 0),
            created_at=d.get("created_at", 0.0),
            last_access=d.get("last_access", 0.0),
            num_layers=d.get("num_layers", 0),
            model_name=d.get("model_name", ""),
            block_size=d.get("block_size", 0),
            cache_signature=d.get("cache_signature", ""),
            layer_cache_types=d.get("layer_cache_types"),
            layer_meta_states=(
                [tuple(s) for s in d["layer_meta_states"]]
                if d.get("layer_meta_states") is not None
                else None
            ),
        )


@dataclass
class PagedSSDCacheIndex:
    max_size_bytes: int = 0
    blocks: OrderedDict[bytes, PagedSSDBlockMetadata] = field(
        default_factory=OrderedDict
    )
    _total_size: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        if not isinstance(self.blocks, OrderedDict):
            self.blocks = OrderedDict()
        if not isinstance(self._lock, type(threading.RLock())):
            self._lock = threading.RLock()

    @property
    def count(self) -> int:
        return len(self.blocks)

    @property
    def total_size(self) -> int:
        return self._total_size

    @property
    def max_size(self) -> int:
        return self.max_size_bytes

    def add(self, metadata: PagedSSDBlockMetadata):
        with self._lock:
            old = self.blocks.get(metadata.block_hash)
            if old is not None:
                self._total_size = max(0, self._total_size - old.file_size)
            self.blocks[metadata.block_hash] = metadata
            self._total_size += metadata.file_size

    def get(self, block_hash: bytes) -> PagedSSDBlockMetadata | None:
        with self._lock:
            return self.blocks.get(block_hash)

    def remove(self, block_hash: bytes) -> PagedSSDBlockMetadata | None:
        with self._lock:
            meta = self.blocks.pop(block_hash, None)
            if meta is not None:
                self._total_size = max(0, self._total_size - meta.file_size)
            return meta

    def touch(self, block_hash: bytes):
        with self._lock:
            if block_hash in self.blocks:
                self.blocks.move_to_end(block_hash)
                self.blocks[block_hash].touch()

    def get_lru_entries(self, count: int) -> list[PagedSSDBlockMetadata]:
        with self._lock:
            sorted_entries = sorted(self.blocks.values(), key=lambda m: m.last_access)
            return sorted_entries[:count]

    def sort_lru(self) -> None:
        with self._lock:
            sorted_items = sorted(self.blocks.items(), key=lambda kv: kv[1].last_access)
            self.blocks.clear()
            for k, v in sorted_items:
                self.blocks[k] = v

    def evict_until_size(self, target_size: int) -> list[PagedSSDBlockMetadata]:
        evicted = []
        with self._lock:
            while self._total_size > target_size and self.blocks:
                lru_hash = min(
                    self.blocks.keys(), key=lambda h: self.blocks[h].last_access
                )
                meta = self.blocks.pop(lru_hash)
                self._total_size = max(0, self._total_size - meta.file_size)
                evicted.append(meta)
        return evicted

    def contains(self, block_hash: bytes) -> bool:
        with self._lock:
            return block_hash in self.blocks

    def get_all_hashes(self) -> list[bytes]:
        with self._lock:
            return list(self.blocks.keys())

    def update_file_size(self, block_hash: bytes, new_size: int):
        with self._lock:
            meta = self.blocks.get(block_hash)
            if meta is not None:
                old_size = meta.file_size
                meta.file_size = new_size
                self._total_size = max(0, self._total_size - old_size + new_size)


def _encode_shape(shape: tuple[int, ...]) -> bytes:
    return struct.pack(f">{len(shape)}I", *shape)


def _dtype_to_st_str(dtype) -> str:
    if dtype in _MX_TO_ST_DTYPE:
        return _MX_TO_ST_DTYPE[dtype]
    dtype_name = str(dtype).lower()
    for dn, (ss, _) in _MX_DTYPE_MAP.items():
        if dn in dtype_name:
            return ss
    return "F32"


def _is_bfloat16(dtype) -> bool:
    if dtype in _MX_TO_ST_DTYPE and _MX_TO_ST_DTYPE[dtype] == "BF16":
        return True
    if HAS_MLX and dtype == mx.bfloat16:
        return True
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
        if len(shape) > 0 and any(d == 0 for d in shape):
            return mx.zeros(shape, dtype=mx_dtype)
        import numpy as np

        if _is_bfloat16(mx_dtype):
            raw_uint16 = mx.array(np.frombuffer(data, dtype=np.uint16)).reshape(shape)
            arr = raw_uint16.view(mx.bfloat16)
        else:
            np_dtype_map = {
                mx.float32: np.float32,
                mx.float16: np.float16,
                mx.int8: np.int8,
                mx.int16: np.int16,
                mx.int32: np.int32,
                mx.int64: np.int64,
                mx.uint8: np.uint8,
                mx.uint16: np.uint16,
                mx.uint32: np.uint32,
                mx.uint64: np.uint64,
            }
            np_dtype = np_dtype_map.get(mx_dtype)
            if np_dtype is not None:
                arr = mx.array(np.frombuffer(data, dtype=np_dtype)).reshape(shape)
            else:
                arr = mx.array(memoryview(data)).astype(mx_dtype).reshape(shape)
        mx.eval(arr)
        return arr
    except Exception as e:
        logger.warning(
            "Corrupt KV cache tensor (shape=%s, dtype=%s): %s; treating as cache miss",
            shape,
            dtype_str,
            e,
        )
        return None


def _write_safetensors_no_mx(
    path: str,
    tensors: dict[str, tuple[bytes, str, list[int]]],
    metadata: dict[str, str] | None = None,
) -> int:
    header = {}
    offset = 0
    chunks = []
    for name, (raw, dtype_str, sh) in sorted(tensors.items()):
        header[name] = {
            "dtype": dtype_str,
            "shape": sh,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        chunks.append(raw)
    if metadata:
        header["__metadata__"] = metadata
    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for chunk in chunks:
            f.write(chunk)
    try:
        return os.path.getsize(path)
    except OSError:
        return offset + len(header_json) + 8


@dataclass
class _HotCacheBudgetEntry:
    owner: Any
    block_hash: bytes
    size_bytes: int


class SharedHotCacheBudget:
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
        with self._lock:
            key = self._key(owner, block_hash)
            if key in self._entries:
                self._entries.move_to_end(key)

    def forget(self, owner: Any, block_hash: bytes) -> None:
        with self._lock:
            key = self._key(owner, block_hash)
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total_bytes = max(0, self._total_bytes - entry.size_bytes)

    def forget_owner(self, owner: Any) -> None:
        owner_id = id(owner)
        with self._lock:
            keys = [key for key in self._entries if key[0] == owner_id]
            for key in keys:
                entry = self._entries.pop(key)
                self._total_bytes = max(0, self._total_bytes - entry.size_bytes)

    def clear_all_owners(self) -> int:
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
                        "clear_hot_cache failed for an orphaned owner", exc_info=True
                    )
        return cleared

    def shrink_to(
        self, target_bytes: int, protected_hashes: set[bytes] | None = None
    ) -> int:
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
class _SSDCacheStats:
    hits: int = 0
    misses: int = 0
    saves: int = 0
    saves_persisted: int = 0
    loads: int = 0
    errors: int = 0
    hot_cache_hits: int = 0
    hot_cache_promotions: int = 0
    hot_cache_evictions: int = 0
    preload_blocks_loaded: int = 0
    preload_calls: int = 0
    preload_time_ms: float = 0.0
    evict_unlink_failures: int = 0
    ssd_write_drops: int = 0
    ssd_inline_write_fallbacks: int = 0
    configured_max_size_bytes: int = 0
    max_size_bytes: int = 0
    total_size_bytes: int = 0
    num_blocks: int = 0
    utilization: float = 0.0


PagedSSDCacheStats = _SSDCacheStats


class PagedSSDCacheManager:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_size_bytes: int = 100 * 1024**3,
        max_cache_size: int | None = None,
        block_size: int = 0,
        expected_model_name: str = "",
        expected_num_layers: int = 0,
        expected_block_size: int = 0,
        expected_block_size_tokens: int = 0,
        expected_kv_bytes_per_token: int = 200_000,
        expected_layer_cache_types: list[str] | None = None,
        hot_cache_max_bytes: int = 0,
        hot_cache_budget: SharedHotCacheBudget | None = None,
        hot_cache_only: bool = False,
    ):
        if cache_dir is None:
            if hot_cache_only:
                # Pure-memory mode with no SSD dir: keep _cache_dir=None so
                # every disk path is a no-op. (If a caller explicitly passes a
                # cache_dir alongside hot_cache_only, honor it - some unit
                # tests use that hybrid to exercise inline eviction without a
                # writer thread.)
                self._cache_dir = None
            else:
                self._cache_dir = Path("/tmp/fusion_mlx_ssd_cache")
        else:
            self._cache_dir = Path(cache_dir)
        effective_max = max_cache_size if max_cache_size is not None else max_size_bytes
        self._max_size = effective_max
        self._configured_max_size = effective_max
        self._expected_model_name = expected_model_name
        self._expected_num_layers = expected_num_layers
        self._expected_block_size = expected_block_size or expected_block_size_tokens
        self._expected_block_size_tokens = (
            expected_block_size_tokens or expected_block_size
        )
        self._expected_kv_bytes_per_token = expected_kv_bytes_per_token
        self._expected_layer_cache_types = (
            list(expected_layer_cache_types)
            if expected_layer_cache_types is not None
            else None
        )
        self._signature_sweep_completed = False
        self._hot_cache_only = hot_cache_only
        self._hot_cache_max_bytes = hot_cache_max_bytes
        self._hot_cache_budget = hot_cache_budget

        self._max_pending_writes = _compute_max_pending_writes(
            block_size_tokens=self._expected_block_size_tokens or 256,
            kv_bytes_per_token=expected_kv_bytes_per_token,
        )
        self._write_queue: queue.Queue = queue.Queue(maxsize=self._max_pending_writes)
        self._pending_write_hashes: set[bytes] = set()
        self._pending_write_hashes_lock = threading.Lock()
        self._pending_writes: dict[bytes, dict] = {}

        self._index = PagedSSDCacheIndex(max_size_bytes=effective_max)
        self._incompatible_index = PagedSSDCacheIndex(max_size_bytes=effective_max)
        self._state_lock = threading.RLock()

        self._stats: dict[str, int | float] = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "saves_persisted": 0,
            "loads": 0,
            "errors": 0,
            "hot_cache_hits": 0,
            "hot_cache_promotions": 0,
            "hot_cache_evictions": 0,
            "preload_blocks_loaded": 0,
            "preload_calls": 0,
            "preload_time_ms": 0.0,
            "evict_unlink_failures": 0,
            "ssd_write_drops": 0,
            "ssd_inline_write_fallbacks": 0,
        }

        self._hot_cache: OrderedDict[bytes, dict] = OrderedDict()
        self._hot_cache_lock = threading.RLock()
        self._hot_cache_total_bytes = 0

        self._disk_usage_cache_time = 0.0
        self._disk_usage_cache_value: int | None = None
        self._shutting_down = False

        if self._hot_cache_only:
            self._writer_thread = None
            logger.info(
                "paged_ssd_cache hot_cache_only mode: in-memory LRU only, "
                "no disk backing (hot_cache_max_bytes=%s, block_size=%s tokens)",
                self._hot_cache_max_bytes,
                self._expected_block_size_tokens,
            )
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        for char in "0123456789abcdef":
            (self._cache_dir / char).mkdir(parents=True, exist_ok=True)

        self._scan_disk_index()

        self._writer_thread = threading.Thread(
            target=self._background_writer,
            daemon=True,
            name="ssd-cache-writer",
        )
        self._writer_thread.start()

    @property
    def cache_dir(self) -> Path | None:
        return self._cache_dir

    @property
    def cache_path(self) -> Path | None:
        return self._cache_dir

    @property
    def max_size(self) -> int:
        return self._get_effective_max_size()

    @property
    def configured_max_size(self) -> int:
        return self._configured_max_size

    @property
    def size(self) -> int:
        with self._state_lock:
            return self._tracked_ssd_size()

    def _tracked_ssd_size(self) -> int:
        return self._index.total_size + self._incompatible_index.total_size

    def _tracked_ssd_count(self) -> int:
        return self._index.count + self._incompatible_index.count

    def _get_file_path(self, block_hash: bytes) -> Path | None:
        if self._cache_dir is None:
            # Pure-memory mode: no disk path. Callers that touch disk guard
            # on hot_cache_only / file_path is not None.
            return None
        hex_str = block_hash.hex()
        sub_dir = hex_str[0]
        return self._cache_dir / sub_dir / f"{hex_str}.safetensors"

    def has_block(self, block_hash: bytes) -> bool:
        with self._state_lock:
            return self._index.contains(block_hash)

    def get_block_metadata(self, block_hash: bytes) -> PagedSSDBlockMetadata | None:
        with self._state_lock:
            return self._index.get(block_hash)

    def save_block(
        self,
        block_hash: bytes,
        cache_data: list | None = None,
        token_count: int = 0,
        model_name: str = "",
        layer_cache_types: list[str] | None = None,
        layer_meta_states: list[tuple[int, ...]] | None = None,
        **kwargs,
    ) -> bool:
        if block_hash is None:
            return False
        with self._state_lock:
            if self._index.contains(block_hash):
                self._index.touch(block_hash)
                self._stats["hits"] += 1
                return True

        if cache_data is None:
            return False

        tensors_raw: dict[str, tuple[bytes, str, list[int]]] = {}
        file_metadata: dict[str, str] = {
            "omlx_cache_format_version": _CACHE_FORMAT_VERSION,
            "block_hash": block_hash.hex(),
            "token_count": str(token_count),
            "num_layers": str(len(cache_data)),
            "model_name": model_name or self._expected_model_name,
            "block_size": str(
                self._expected_block_size
                or self._expected_block_size_tokens
                or token_count
            ),
            "created_at": str(time.time()),
        }
        if layer_cache_types is not None:
            file_metadata["layer_cache_types"] = json.dumps(layer_cache_types)
        if layer_meta_states is not None:
            file_metadata["layer_meta_states"] = json.dumps(
                [list(s) for s in layer_meta_states]
            )

        sig = _cache_compat_signature(
            model_name=model_name or self._expected_model_name,
            num_layers=len(cache_data),
            block_size=self._expected_block_size
            or self._expected_block_size_tokens
            or token_count,
            layer_cache_types=layer_cache_types,
        )
        file_metadata["cache_signature"] = sig

        for i, layer_data in enumerate(cache_data):
            if isinstance(layer_data, tuple) and len(layer_data) == 2:
                marker, sub_list = layer_data
                if marker == "__cache_list__" and isinstance(sub_list, list):
                    total_states = 0
                    for j, sub_state in enumerate(sub_list):
                        # Unwrap __nstate__ marker if present
                        if (
                            isinstance(sub_state, tuple)
                            and len(sub_state) >= 3
                            and isinstance(sub_state[0], str)
                            and sub_state[0] == "__nstate__"
                        ):
                            elements = list(sub_state[2])
                            sub_class = sub_state[1]
                        elif isinstance(sub_state, (list, tuple)):
                            elements = list(sub_state)
                            sub_class = None
                        else:
                            elements = [sub_state]
                            sub_class = None
                        for k, item in enumerate(elements):
                            tensors_raw[f"layer_{i}_sub_{j}_state_{k}"] = (
                                _extract_tensor_bytes(item)
                            )
                        file_metadata[f"layer_{i}_sub_{j}_state_count"] = str(
                            len(elements)
                        )
                        if sub_class is not None:
                            file_metadata[f"layer_{i}_sub_{j}_class"] = sub_class
                        total_states += len(elements)
                    file_metadata[f"layer_{i}_sub_count"] = str(len(sub_list))
                    file_metadata[f"layer_{i}_state_count"] = str(total_states)
                    continue
            # Handle top-level __nstate__ marker (e.g. MiniMaxM3KVCache)
            if (
                isinstance(layer_data, tuple)
                and len(layer_data) >= 3
                and isinstance(layer_data[0], str)
                and layer_data[0] == "__nstate__"
            ):
                elements = list(layer_data[2])
                for k, item in enumerate(elements):
                    tensors_raw[f"layer_{i}_state_{k}"] = _extract_tensor_bytes(item)
                file_metadata[f"layer_{i}_state_count"] = str(len(elements))
                file_metadata[f"layer_{i}_nstate_class"] = str(layer_data[1])
                continue
            if isinstance(layer_data, (list, tuple)):
                state_items = list(layer_data)
                for k, item in enumerate(state_items):
                    tensors_raw[f"layer_{i}_state_{k}"] = _extract_tensor_bytes(item)
                file_metadata[f"layer_{i}_state_count"] = str(len(state_items))

        estimated_file_size = sum(len(r) for r, _, _ in tensors_raw.values())
        file_path = self._get_file_path(block_hash)
        block_metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=file_path,
            file_size=estimated_file_size,
            token_count=token_count,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=len(cache_data),
            model_name=model_name or self._expected_model_name,
            block_size=self._expected_block_size
            or self._expected_block_size_tokens
            or token_count,
            cache_signature=sig,
            layer_cache_types=layer_cache_types,
            layer_meta_states=layer_meta_states,
        )

        hot_entry = {
            "tensors_raw": tensors_raw,
            "file_metadata": file_metadata,
            "num_layers": len(cache_data),
            "layer_cache_types": layer_cache_types,
            "block_metadata": block_metadata,
            "dirty": False,
        }
        self._hot_cache_put(block_hash, hot_entry)

        with self._state_lock:
            self._index.add(block_metadata)
            self._stats["saves"] += 1

        if self._hot_cache_only:
            # Pure-memory mode: hot_cache (populated above) is the sole store.
            # Skip _pending_writes - the writer thread never runs in this mode,
            # so pending entries would never drain and would leak memory while
            # pinning the tensors even after hot_cache LRU eviction.
            return True

        with self._pending_write_hashes_lock:
            self._pending_write_hashes.add(block_hash)
            self._pending_writes[block_hash] = hot_entry

        try:
            self._write_queue.put(
                (block_hash, tensors_raw, file_metadata, block_metadata),
                timeout=1.0,
            )
        except queue.Full:
            logger.debug(
                "SSD write queue full, writing block inline: %s", block_hash.hex()[:16]
            )
            self._stats["ssd_inline_write_fallbacks"] += 1
            self._write_block_inline(
                block_hash, tensors_raw, file_metadata, block_metadata
            )

        self._enforce_size_limit_for_new_block()
        return True

    def _write_block_inline(
        self, block_hash, tensors_raw, file_metadata, block_metadata
    ):
        file_path = self._get_file_path(block_hash)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = file_path.with_suffix(".tmp")
            actual_size = _write_safetensors_no_mx(
                str(temp_path), tensors_raw, file_metadata
            )
            temp_path.replace(file_path)
            with self._state_lock:
                old_meta = self._index.get(block_hash)
                if old_meta is not None:
                    self._index.update_file_size(block_hash, actual_size)
                else:
                    block_metadata.file_size = actual_size
            self._stats["saves_persisted"] += 1
        except Exception as e:
            logger.error(
                "Inline write failed for block %s: %s", block_hash.hex()[:16], e
            )
            self._stats["errors"] += 1
            with self._state_lock:
                self._index.remove(block_hash)
        finally:
            with self._pending_write_hashes_lock:
                self._pending_write_hashes.discard(block_hash)
                self._pending_writes.pop(block_hash, None)

    def load_block(self, block_hash: bytes) -> list | None:
        with self._pending_write_hashes_lock:
            pending = self._pending_writes.get(block_hash)
        if pending is not None:
            self._stats["loads"] += 1
            self._stats["hits"] += 1
            return self._restore_from_pending(pending)

        hot = self._hot_cache_get(block_hash)
        if hot is not None:
            self._stats["loads"] += 1
            self._stats["hot_cache_hits"] += 1
            return self._restore_from_pending(hot)

        with self._state_lock:
            if not self._index.contains(block_hash):
                self._stats["misses"] += 1
                return None

        file_path = self._get_file_path(block_hash)
        if not file_path.exists():
            logger.debug(
                "SSD cache miss (file missing): block %s", block_hash.hex()[:16]
            )
            with self._state_lock:
                self._index.remove(block_hash)
            self._stats["misses"] += 1
            return None

        try:
            raw_result = self._load_safetensors_raw(str(file_path))
            if raw_result is not None:
                tensors_raw, file_metadata = raw_result
                loaded = self._reconstruct_from_raw(tensors_raw, file_metadata)
                if loaded is not None:
                    meta_obj = self._index.get(block_hash)
                    hot_entry = {
                        "tensors_raw": tensors_raw,
                        "file_metadata": file_metadata,
                        "num_layers": int(file_metadata.get("num_layers", len(loaded))),
                        "layer_cache_types": None,
                        "block_metadata": meta_obj,
                        "dirty": False,
                        "_estimated_bytes": sum(
                            len(r) for r, _, _ in tensors_raw.values()
                        ),
                    }
                    if "layer_cache_types" in file_metadata:
                        try:
                            hot_entry["layer_cache_types"] = json.loads(
                                file_metadata["layer_cache_types"]
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass
                    self._hot_cache_put(block_hash, hot_entry)
                    self._stats["loads"] += 1
                    self._stats["hot_cache_promotions"] += 1
                    self._stats["hits"] += 1
                    self._index.touch(block_hash)
                    return loaded
            loaded = self._load_safetensors_file(str(file_path))
            if loaded is not None:
                self._stats["loads"] += 1
                self._stats["hits"] += 1
                self._index.touch(block_hash)
                return loaded
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.warning("Failed to load block from SSD: %s", e)
            self._recover_from_block_error(block_hash)
            self._stats["errors"] += 1
            self._stats["misses"] += 1
            return None

    def load_block_with_metadata(
        self, block_hash: bytes
    ) -> tuple[list | None, dict | None]:
        with self._pending_write_hashes_lock:
            pending = self._pending_writes.get(block_hash)
        if pending is not None:
            data = self._restore_from_pending(pending)
            meta = self._metadata_from_pending(pending)
            return data, meta

        hot = self._hot_cache_get(block_hash)
        if hot is not None:
            data = self._restore_from_pending(hot)
            meta = self._metadata_from_pending(hot)
            return data, meta

        meta_obj = self.get_block_metadata(block_hash)
        if meta_obj is None:
            return None, None

        if self._hot_cache_only:
            # Pure-memory mode: hot_cache miss with metadata present means the
            # tensor was evicted by LRU and there is no disk backing. Drop the
            # now-stale metadata so the index does not grow unbounded, and
            # report a miss (avoids crashing in _get_file_path on cache_dir=None).
            with self._state_lock:
                self._index.remove(block_hash)
            self._stats["misses"] += 1
            return None, None

        data = self.load_block(block_hash)
        if data is None:
            return None, None

        metadata_dict = {
            "num_layers": meta_obj.num_layers,
            "token_count": meta_obj.token_count,
            "model_name": meta_obj.model_name,
            "block_size": meta_obj.block_size,
            "cache_signature": meta_obj.cache_signature,
            "layer_cache_types": meta_obj.layer_cache_types,
            "layer_meta_states": meta_obj.layer_meta_states,
        }
        return data, metadata_dict

    def _metadata_from_pending(self, entry: dict) -> dict:
        fm = entry.get("file_metadata", {})
        return {
            "num_layers": int(fm.get("num_layers", 0)),
            "token_count": int(fm.get("token_count", 0)),
            "model_name": fm.get("model_name", ""),
            "block_size": int(fm.get("block_size", 0)),
            "cache_signature": fm.get("cache_signature", ""),
            "layer_cache_types": (
                json.loads(fm["layer_cache_types"])
                if "layer_cache_types" in fm
                else None
            ),
            "layer_meta_states": (
                [tuple(s) for s in json.loads(fm["layer_meta_states"])]
                if "layer_meta_states" in fm
                else None
            ),
        }

    def _restore_from_pending(self, entry: dict) -> list | None:
        if not HAS_MLX:
            return None
        tensors_raw = entry.get("tensors_raw", {})
        file_metadata = entry.get("file_metadata", {})
        num_layers = entry.get("num_layers", 0)
        layer_cache_types = None
        if "layer_cache_types" in file_metadata:
            try:
                layer_cache_types = json.loads(file_metadata["layer_cache_types"])
            except (json.JSONDecodeError, TypeError):
                pass
        return self._reconstruct_layers(
            tensors_raw, file_metadata, num_layers, layer_cache_types
        )

    def _reconstruct_layers(
        self, tensors_raw, file_metadata, num_layers, layer_cache_types=None
    ):
        if not HAS_MLX:
            return None
        layers = []
        for i in range(num_layers):
            is_cache_list = False
            if layer_cache_types and i < len(layer_cache_types):
                is_cache_list = layer_cache_types[i] == "CacheList"

            if is_cache_list:
                sub_count = int(file_metadata.get(f"layer_{i}_sub_count", 0))
                if sub_count > 0:
                    sub_list = []
                    for j in range(sub_count):
                        sub_state_count = int(
                            file_metadata.get(f"layer_{i}_sub_{j}_state_count", 2)
                        )
                        sub_class = file_metadata.get(f"layer_{i}_sub_{j}_class")
                        items = []
                        for k in range(sub_state_count):
                            raw = tensors_raw.get(f"layer_{i}_sub_{j}_state_{k}")
                            if raw:
                                arr = _restore_tensor_from_bytes(*raw)
                                if arr is not None:
                                    items.append(arr)
                                else:
                                    return None
                        if sub_state_count >= 3 and sub_class:
                            sub_list.append(("__nstate__", sub_class, items))
                        elif len(items) == 2:
                            sub_list.append((items[0], items[1]))
                        elif len(items) == 1:
                            sub_list.append((items[0],))
                    layers.append(sub_list)
                else:
                    state_count = int(file_metadata.get(f"layer_{i}_state_count", 2))
                    items = []
                    for k in range(state_count):
                        raw = tensors_raw.get(f"layer_{i}_state_{k}")
                        if raw:
                            arr = _restore_tensor_from_bytes(*raw)
                            if arr is not None:
                                items.append(arr)
                            else:
                                return None
                    if len(items) == 2:
                        layers.append((items[0], items[1]))
            else:
                state_count = int(file_metadata.get(f"layer_{i}_state_count", 2))
                nstate_class = file_metadata.get(f"layer_{i}_nstate_class")
                items = []
                for k in range(state_count):
                    raw = tensors_raw.get(f"layer_{i}_state_{k}")
                    if raw:
                        arr = _restore_tensor_from_bytes(*raw)
                        if arr is not None:
                            items.append(arr)
                        else:
                            return None
                if state_count >= 3 and nstate_class:
                    layers.append(("__nstate__", nstate_class, items))
                elif len(items) == 2:
                    layers.append((items[0], items[1]))
                elif len(items) == 1:
                    layers.append((items[0],))
        return layers if layers else None

    def _load_safetensors_file(self, path: str) -> list | None:
        if not HAS_MLX:
            return None
        try:
            loaded_arrays, loaded_meta = mx.load(path, return_metadata=True)
            format_ver = loaded_meta.get("omlx_cache_format_version")
            if format_ver is None:
                logger.debug("Rejecting unversioned block: %s", path)
                return None
            if format_ver not in _READABLE_CACHE_FORMAT_VERSIONS:
                logger.debug(
                    "Rejecting block with unsupported format %s: %s",
                    format_ver,
                    path,
                )
                return None
            num_layers = int(loaded_meta.get("num_layers", 0))
            layer_cache_types = None
            if "layer_cache_types" in loaded_meta:
                try:
                    layer_cache_types = json.loads(loaded_meta["layer_cache_types"])
                except (json.JSONDecodeError, TypeError):
                    pass
            return self._reconstruct_layers_from_arrays(
                loaded_arrays, loaded_meta, num_layers, layer_cache_types
            )
        except Exception as e:
            logger.debug("Failed to load safetensors file %s: %s", path, e)
            return None

    def _reconstruct_layers_from_arrays(
        self, arrays, meta, num_layers, layer_cache_types=None
    ):
        layers = []
        for i in range(num_layers):
            is_cache_list = False
            if layer_cache_types and i < len(layer_cache_types):
                is_cache_list = layer_cache_types[i] == "CacheList"

            if is_cache_list:
                sub_count = int(meta.get(f"layer_{i}_sub_count", 0))
                if sub_count > 0:
                    sub_list = []
                    for j in range(sub_count):
                        sub_state_count = int(
                            meta.get(f"layer_{i}_sub_{j}_state_count", 2)
                        )
                        sub_class = meta.get(f"layer_{i}_sub_{j}_class")
                        items = []
                        for k in range(sub_state_count):
                            key = f"layer_{i}_sub_{j}_state_{k}"
                            if key in arrays:
                                items.append(arrays[key])
                        if sub_state_count >= 3 and sub_class:
                            sub_list.append(("__nstate__", sub_class, items))
                        elif len(items) == 2:
                            sub_list.append((items[0], items[1]))
                        elif len(items) == 1:
                            sub_list.append((items[0],))
                    layers.append(sub_list)
                else:
                    state_count = int(meta.get(f"layer_{i}_state_count", 2))
                    items = []
                    for k in range(state_count):
                        key = f"layer_{i}_state_{k}"
                        if key in arrays:
                            items.append(arrays[key])
                    if len(items) == 2:
                        layers.append((items[0], items[1]))
            else:
                state_count = int(meta.get(f"layer_{i}_state_count", 2))
                nstate_class = meta.get(f"layer_{i}_nstate_class")
                items = []
                for k in range(state_count):
                    key = f"layer_{i}_state_{k}"
                    if key in arrays:
                        items.append(arrays[key])
                if state_count >= 3 and nstate_class:
                    layers.append(("__nstate__", nstate_class, items))
                elif len(items) == 2:
                    layers.append((items[0], items[1]))
                elif len(items) == 1:
                    layers.append((items[0],))
        return layers if layers else None

    def _load_safetensors_raw(self, path: str) -> tuple[dict, dict] | None:
        try:
            with open(path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                if header_size < 1 or header_size > 100 * 1024 * 1024:
                    return None
                header_json = f.read(header_size).decode("utf-8")
            header = json.loads(header_json)
            file_metadata = header.pop("__metadata__", {})
            tensors_raw = {}
            for name, info in sorted(header.items()):
                dtype_str = info["dtype"]
                shape = info["shape"]
                data_offsets = info["data_offsets"]
                data_len = data_offsets[1] - data_offsets[0]
                with open(path, "rb") as f:
                    f.seek(8 + header_size + data_offsets[0])
                    raw = f.read(data_len)
                tensors_raw[name] = (raw, dtype_str, shape)
            return tensors_raw, file_metadata
        except Exception as e:
            logger.debug("Failed to read raw safetensors %s: %s", path, e)
            return None

    def _reconstruct_from_raw(
        self, tensors_raw: dict, file_metadata: dict
    ) -> list | None:
        if not HAS_MLX:
            return None
        arrays = {}
        for name, (raw, dtype_str, shape) in tensors_raw.items():
            arr = _restore_tensor_from_bytes(raw, dtype_str, shape)
            if arr is not None:
                arrays[name] = arr
            else:
                return None
        if not arrays:
            return None
        num_layers = int(file_metadata.get("num_layers", 0))
        if num_layers == 0:
            return None
        layer_cache_types = None
        if "layer_cache_types" in file_metadata:
            try:
                layer_cache_types = json.loads(file_metadata["layer_cache_types"])
            except (json.JSONDecodeError, TypeError):
                pass
        return self._reconstruct_layers_from_arrays(
            arrays, file_metadata, num_layers, layer_cache_types
        )

    def delete_block(self, block_hash: bytes) -> bool:
        with self._state_lock:
            meta = self._index.remove(block_hash)
            if meta is None:
                return False
            if not self._hot_cache_only:
                file_path = self._get_file_path(block_hash)
                try:
                    if file_path.exists():
                        file_path.unlink()
                except OSError as e:
                    logger.debug("Failed to unlink %s: %s", file_path, e)
            self._hot_cache_remove(block_hash)
            return True

    def evict_block(self, block_hash: bytes) -> bool:
        return self.delete_block(block_hash)

    def forget_block(self, block_hash: bytes) -> bool:
        with self._state_lock:
            meta = self._index.remove(block_hash)
            if meta is None:
                return False
            self._incompatible_index.add(meta)
            return True

    def evict(self, key: bytes) -> bool:
        return self.delete_block(key)

    def fetch(self, key: bytes) -> tuple[Any, bool]:
        data = self.load_block(key)
        if data is not None:
            return data, True
        return None, False

    def clear(self) -> int:
        count = 0
        with self._state_lock:
            if self._hot_cache_only:
                count = len(self._index.blocks) + len(self._incompatible_index.blocks)
            else:
                for block_hash in list(self._index.blocks.keys()):
                    file_path = self._get_file_path(block_hash)
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            count += 1
                    except OSError as e:
                        logger.debug("Failed to unlink during clear: %s", e)
                for block_hash in list(self._incompatible_index.blocks.keys()):
                    file_path = self._get_file_path(block_hash)
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            count += 1
                    except OSError as e:
                        logger.debug("Failed to unlink incompatible during clear: %s", e)
            self._index = PagedSSDCacheIndex(max_size_bytes=self._max_size)
            self._incompatible_index = PagedSSDCacheIndex(max_size_bytes=self._max_size)
        with self._hot_cache_lock:
            self._hot_cache.clear()
            self._hot_cache_total_bytes = 0
        return count

    def clear_hot_cache(self) -> int:
        with self._hot_cache_lock:
            count = len(self._hot_cache)
            self._hot_cache.clear()
            self._hot_cache_total_bytes = 0
        return count

    def close(self):
        self._shutting_down = True
        if self._writer_thread is not None:
            try:
                self._write_queue.put(None, timeout=5.0)
            except queue.Full:
                pass
            if self._writer_thread.is_alive():
                self._writer_thread.join(timeout=10.0)
        logger.info("PagedSSDCacheManager closed")

    def get_stats(self) -> _SSDCacheStats:
        effective = self._get_effective_max_size()
        total = self._tracked_ssd_size()
        num = self._tracked_ssd_count()
        util = total / effective if effective > 0 else 0.0
        return _SSDCacheStats(
            hits=self._stats["hits"],
            misses=self._stats["misses"],
            saves=self._stats["saves"],
            saves_persisted=self._stats["saves_persisted"],
            loads=self._stats["loads"],
            errors=self._stats["errors"],
            hot_cache_hits=self._stats["hot_cache_hits"],
            hot_cache_promotions=self._stats["hot_cache_promotions"],
            hot_cache_evictions=self._stats["hot_cache_evictions"],
            preload_blocks_loaded=self._stats["preload_blocks_loaded"],
            preload_calls=self._stats["preload_calls"],
            preload_time_ms=self._stats["preload_time_ms"],
            evict_unlink_failures=self._stats["evict_unlink_failures"],
            ssd_write_drops=self._stats["ssd_write_drops"],
            ssd_inline_write_fallbacks=self._stats["ssd_inline_write_fallbacks"],
            configured_max_size_bytes=self._configured_max_size,
            max_size_bytes=effective,
            total_size_bytes=total,
            num_blocks=num,
            utilization=util,
        )

    def get_stats_dict(self) -> dict:
        effective = self._get_effective_max_size()
        total = self._tracked_ssd_size()
        return {
            "cache_dir": str(self._cache_dir),
            "configured_max_size": self._configured_max_size,
            "max_size": effective,
            "total_size": total,
            "num_files": self._tracked_ssd_count(),
            "utilization": total / effective if effective > 0 else 0.0,
        }

    def get_all_metadata(self) -> list[PagedSSDBlockMetadata]:
        with self._state_lock:
            return list(self._index.blocks.values())

    def get_stats_for_model(self, model_name: str) -> _SSDCacheStats:
        normalized_name = model_name.rstrip("/")
        basename = os.path.basename(normalized_name) if normalized_name else ""

        def _matches(candidate: str) -> bool:
            candidate = candidate.rstrip("/")
            if not candidate:
                return False
            if candidate == normalized_name:
                return True
            if basename and os.path.basename(candidate) == basename:
                return True
            return False

        with self._state_lock:
            indexed_entries = [
                meta
                for meta in self._index.blocks.values()
                if _matches(meta.model_name)
            ]
            indexed_size = sum(meta.file_size for meta in indexed_entries)
            indexed_count = len(indexed_entries)

            with self._hot_cache_lock:
                hot_count = 0
                hot_size = 0
                for entry in self._hot_cache.values():
                    blk_meta = entry.get("block_metadata")
                    if blk_meta is None or not _matches(blk_meta.model_name):
                        continue
                    hot_count += 1
                    hot_size += entry.get("_estimated_bytes", 0)

            effective = self._get_effective_max_size()
            util = indexed_size / effective if effective > 0 else 0.0
            return _SSDCacheStats(
                hits=self._stats["hits"],
                misses=self._stats["misses"],
                saves=self._stats["saves"],
                saves_persisted=self._stats["saves_persisted"],
                loads=self._stats["loads"],
                errors=self._stats["errors"],
                hot_cache_hits=self._stats["hot_cache_hits"],
                hot_cache_promotions=self._stats["hot_cache_promotions"],
                hot_cache_evictions=self._stats["hot_cache_evictions"],
                preload_blocks_loaded=self._stats["preload_blocks_loaded"],
                preload_calls=self._stats["preload_calls"],
                preload_time_ms=self._stats["preload_time_ms"],
                evict_unlink_failures=self._stats["evict_unlink_failures"],
                ssd_write_drops=self._stats["ssd_write_drops"],
                ssd_inline_write_fallbacks=self._stats["ssd_inline_write_fallbacks"],
                configured_max_size_bytes=self._configured_max_size,
                max_size_bytes=effective,
                total_size_bytes=indexed_size,
                num_blocks=indexed_count,
                utilization=util,
            )

    def _effective_hot_cache_max_bytes(self) -> int:
        if self._hot_cache_budget is not None:
            return self._hot_cache_budget.max_bytes
        return self._hot_cache_max_bytes

    def _hot_cache_available_bytes(self) -> int:
        if self._hot_cache_budget is not None:
            return self._hot_cache_budget.remaining_bytes
        return max(0, self._hot_cache_max_bytes - self._hot_cache_total_bytes)

    def _hot_cache_entry_size(self, entry: dict) -> int:
        return entry.get("_estimated_bytes", 0)

    def shrink_hot_cache_to(
        self,
        target_bytes: int,
        protected_hashes: set[bytes] | None = None,
    ) -> int:
        target_bytes = max(0, int(target_bytes))
        protected_hashes = protected_hashes or set()

        if self._hot_cache_budget is not None:
            return self._hot_cache_budget.shrink_to(
                target_bytes, protected_hashes=protected_hashes
            )

        evicted_entries: list[tuple[bytes, dict, int]] = []
        with self._hot_cache_lock:
            while self._hot_cache_total_bytes > target_bytes and self._hot_cache:
                victim_hash = None
                for block_hash in self._hot_cache:
                    if block_hash not in protected_hashes:
                        victim_hash = block_hash
                        break
                if victim_hash is None:
                    break

                evicted = self._hot_cache.pop(victim_hash)
                size = self._hot_cache_entry_size(evicted)
                self._hot_cache_total_bytes = max(0, self._hot_cache_total_bytes - size)
                evicted_entries.append((victim_hash, evicted, size))

        freed = 0
        for block_hash, evicted, size in evicted_entries:
            freed += size
            self._handle_hot_cache_eviction(block_hash, evicted)

        if freed:
            logger.info("Shrank hot cache by %d bytes", freed)
        return freed

    def sort_lru_by_last_access(self) -> None:
        with self._state_lock:
            self._index.sort_lru()

    def __repr__(self) -> str:
        return f"PagedSSDCacheManager(cache_dir={self._cache_dir}, blocks={self._tracked_ssd_count()})"

    def _hot_cache_get(self, block_hash: bytes) -> dict | None:
        with self._hot_cache_lock:
            entry = self._hot_cache.get(block_hash)
            if entry is not None:
                self._hot_cache.move_to_end(block_hash)
            return entry

    def _hot_cache_put(self, block_hash: bytes, entry: dict):
        with self._hot_cache_lock:
            old = self._hot_cache.pop(block_hash, None)
            if old is not None:
                est = old.get("_estimated_bytes", 0)
                self._hot_cache_total_bytes = max(0, self._hot_cache_total_bytes - est)
            tensors_raw = entry.get("tensors_raw", {})
            est_bytes = sum(len(r) for r, _, _ in tensors_raw.values())
            entry["_estimated_bytes"] = est_bytes
            self._hot_cache[block_hash] = entry
            self._hot_cache_total_bytes += est_bytes

            while (
                self._hot_cache_max_bytes > 0
                and self._hot_cache_total_bytes > self._hot_cache_max_bytes
                and len(self._hot_cache) > 1
            ):
                evict_hash, evict_entry = self._hot_cache.popitem(last=False)
                self._hot_cache_total_bytes = max(
                    0,
                    self._hot_cache_total_bytes
                    - evict_entry.get("_estimated_bytes", 0),
                )
                if self._hot_cache_budget is not None:
                    self._hot_cache_budget.forget(self, evict_hash)

    def _hot_cache_remove(
        self, block_hash: bytes, update_budget: bool = True
    ) -> dict | None:
        with self._hot_cache_lock:
            entry = self._hot_cache.pop(block_hash, None)
            if entry is not None:
                self._hot_cache_total_bytes = max(
                    0, self._hot_cache_total_bytes - entry.get("_estimated_bytes", 0)
                )
        if update_budget and entry is not None and self._hot_cache_budget is not None:
            self._hot_cache_budget.forget(self, block_hash)
        return entry

    def _handle_hot_cache_eviction(self, block_hash: bytes, entry: dict):
        pass

    def _enqueue_ssd_write(
        self, block_hash, tensors_raw, file_metadata, block_metadata
    ):
        try:
            self._write_queue.put(
                (block_hash, tensors_raw, file_metadata, block_metadata),
                timeout=1.0,
            )
        except queue.Full:
            logger.debug(
                "SSD write queue full on enqueue, dropping: %s", block_hash.hex()[:16]
            )
            self._stats["ssd_write_drops"] += 1

    def _background_writer(self):
        while True:
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down:
                    break
                continue
            if item is None:
                break
            block_hash, tensors_raw, file_metadata, block_metadata = item
            try:
                file_path = self._get_file_path(block_hash)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = file_path.with_suffix(".tmp")
                actual_size = _write_safetensors_no_mx(
                    str(temp_path), tensors_raw, file_metadata
                )
                temp_path.replace(file_path)
                with self._state_lock:
                    old_meta = self._index.get(block_hash)
                    if old_meta is not None:
                        self._index.update_file_size(block_hash, actual_size)
                    else:
                        block_metadata.file_size = actual_size
                self._stats["saves_persisted"] += 1
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    logger.warning(
                        "SSD cache disk full, dropping block write: %s",
                        block_hash.hex()[:16],
                    )
                else:
                    logger.error(
                        "Background write failed for block %s: %s",
                        block_hash.hex()[:16],
                        e,
                    )
                with self._state_lock:
                    self._index.remove(block_hash)
                self._stats["errors"] += 1
            except Exception as e:
                logger.error(
                    "Background write error for block %s: %s", block_hash.hex()[:16], e
                )
                with self._state_lock:
                    self._index.remove(block_hash)
                self._stats["errors"] += 1
            finally:
                with self._pending_write_hashes_lock:
                    self._pending_write_hashes.discard(block_hash)
                    self._pending_writes.pop(block_hash, None)
                self._write_queue.task_done()

    def _scan_disk_index(self):
        scanned = 0
        skipped_incompatible = 0
        for hex_dir in "0123456789abcdef":
            dir_path = self._cache_dir / hex_dir
            if not dir_path.is_dir():
                continue
            for f in dir_path.glob("*.safetensors"):
                if not f.is_file():
                    continue
                scanned += 1
                try:
                    result = self._load_safetensors_raw(str(f))
                    if result is None:
                        continue
                    tensors_raw, file_metadata = result
                    fmt_ver = file_metadata.get("omlx_cache_format_version")
                    if fmt_ver is None:
                        self._add_to_incompatible_index(f, file_metadata)
                        skipped_incompatible += 1
                        continue
                    block_hash_hex = file_metadata.get("block_hash", "")
                    if not block_hash_hex:
                        continue
                    try:
                        block_hash = bytes.fromhex(block_hash_hex)
                    except ValueError:
                        continue
                    try:
                        st = f.stat()
                        scan_file_size = st.st_size
                        scan_last_access = max(
                            st.st_mtime,
                            float(file_metadata.get("last_access", 0)),
                        )
                    except OSError:
                        scan_file_size = 0
                        scan_last_access = float(
                            file_metadata.get("last_access", time.time())
                        )
                    meta = PagedSSDBlockMetadata(
                        block_hash=block_hash,
                        file_path=f,
                        file_size=scan_file_size,
                        token_count=int(file_metadata.get("token_count", 0)),
                        created_at=float(file_metadata.get("created_at", 0)),
                        last_access=scan_last_access,
                        num_layers=int(file_metadata.get("num_layers", 0)),
                        model_name=file_metadata.get("model_name", ""),
                        block_size=int(file_metadata.get("block_size", 0)),
                        cache_signature=file_metadata.get("cache_signature", ""),
                        layer_cache_types=(
                            json.loads(file_metadata["layer_cache_types"])
                            if "layer_cache_types" in file_metadata
                            else None
                        ),
                        layer_meta_states=(
                            [
                                tuple(s)
                                for s in json.loads(file_metadata["layer_meta_states"])
                            ]
                            if "layer_meta_states" in file_metadata
                            else None
                        ),
                    )
                    if not self._is_compatible_block(meta):
                        self._add_to_incompatible_index(f, file_metadata)
                        skipped_incompatible += 1
                        continue
                    self._index.add(meta)
                except Exception as e:
                    logger.debug("Failed to scan SSD cache file %s: %s", f, e)

        self._enforce_incompatible_at_startup()

        logger.info(
            "SSD cache scan complete: %d compatible, skipped_incompatible=%d blocks",
            self._index.count,
            skipped_incompatible,
        )

    def _add_to_incompatible_index(self, file_path: Path, file_metadata: dict):
        block_hash_hex = file_metadata.get("block_hash", "")
        if not block_hash_hex:
            return
        try:
            block_hash = bytes.fromhex(block_hash_hex)
        except ValueError:
            return
        try:
            st = file_path.stat()
            file_size = st.st_size
            last_access = max(
                st.st_mtime,
                float(file_metadata.get("last_access", 0)),
            )
        except OSError:
            file_size = 0
            last_access = float(file_metadata.get("last_access", time.time()))
        meta = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=file_path,
            file_size=file_size,
            token_count=int(file_metadata.get("token_count", 0)),
            created_at=float(file_metadata.get("created_at", 0)),
            last_access=last_access,
            num_layers=int(file_metadata.get("num_layers", 0)),
            model_name=file_metadata.get("model_name", ""),
        )
        self._incompatible_index.add(meta)

    def _enforce_incompatible_at_startup(self):
        effective = self._get_effective_max_size()
        while (
            self._tracked_ssd_size() > effective and self._incompatible_index.count > 0
        ):
            lru = self._incompatible_index.get_lru_entries(1)
            if not lru:
                break
            victim = lru[0]
            file_path = self._get_file_path(victim.block_hash)
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError as e:
                logger.debug("Startup incompatible unlink failed: %s", e)
            self._incompatible_index.remove(victim.block_hash)

    def _is_compatible_block(self, meta: PagedSSDBlockMetadata) -> bool:
        if not self._expected_model_name:
            return True
        if (
            self._expected_num_layers > 0
            and meta.num_layers != self._expected_num_layers
        ):
            return False
        if (
            self._expected_model_name
            and meta.model_name
            and meta.model_name != self._expected_model_name
        ):
            return False
        expected_block_size = (
            self._expected_block_size or self._expected_block_size_tokens
        )
        if (
            expected_block_size > 0
            and meta.block_size > 0
            and meta.block_size != expected_block_size
        ):
            return False
        if self._expected_layer_cache_types and meta.cache_signature:
            expected_sig = _cache_compat_signature(
                model_name=self._expected_model_name,
                num_layers=self._expected_num_layers or meta.num_layers,
                block_size=expected_block_size or meta.block_size,
                layer_cache_types=self._expected_layer_cache_types,
            )
            return meta.cache_signature == expected_sig
        return True

    def _get_effective_max_size(self) -> int:
        if self._cache_dir is None:
            return self._max_size
        now = time.time()
        if (
            self._disk_usage_cache_value is not None
            and (now - self._disk_usage_cache_time) < 30.0
        ):
            cached = self._disk_usage_cache_value
        else:
            try:
                usage = shutil.disk_usage(self._cache_dir)
                disk_available = self._tracked_ssd_size() + usage.free
                disk_limit = int(disk_available * 0.99)
                cached = min(self._max_size, disk_limit)
            except OSError as e:
                logger.warning("Failed to check disk usage: %s", e)
                cached = self._max_size
            self._disk_usage_cache_value = cached
            self._disk_usage_cache_time = now
        return cached

    def _enforce_size_limit_for_new_block(self):
        effective = self._get_effective_max_size()
        if effective < self._configured_max_size * 0.10:
            logger.warning(
                "SSD cache disk pressure: disk nearly full, effective max %d < 10%% of configured %d",
                effective,
                self._configured_max_size,
            )
        with self._state_lock:
            if self._tracked_ssd_size() <= effective:
                return
            unlinks_done = 0
            while self._tracked_ssd_size() > effective:
                if self._incompatible_index.count > 0:
                    lru = self._incompatible_index.get_lru_entries(1)
                    if lru:
                        victim = lru[0]
                        file_path = self._get_file_path(victim.block_hash)
                        try:
                            if file_path.exists():
                                file_path.unlink()
                        except OSError as e:
                            logger.debug("Incompatible unlink failed: %s", e)
                        self._incompatible_index.remove(victim.block_hash)
                        continue
                if self._index.count == 0:
                    break
                if unlinks_done >= _MAX_INLINE_UNLINKS_PER_SAVE:
                    break
                lru = self._index.get_lru_entries(1)
                if not lru:
                    break
                victim = lru[0]
                file_path = self._get_file_path(victim.block_hash)
                try:
                    if file_path.exists():
                        file_path.unlink()
                except OSError as e:
                    logger.debug("Inline unlink failed: %s", e)
                    self._stats["evict_unlink_failures"] += 1
                self._index.remove(victim.block_hash)
                unlinks_done += 1

    def enforce_size_limit(self) -> int:
        effective = self._get_effective_max_size()
        freed = 0
        with self._state_lock:
            while self._tracked_ssd_size() > effective:
                if self._incompatible_index.count > 0:
                    lru = self._incompatible_index.get_lru_entries(1)
                    if lru:
                        victim = lru[0]
                        freed += victim.file_size
                        file_path = self._get_file_path(victim.block_hash)
                        try:
                            if file_path.exists():
                                file_path.unlink()
                        except OSError as e:
                            logger.debug("Enforce incompatible unlink failed: %s", e)
                        self._incompatible_index.remove(victim.block_hash)
                        continue
                if self._index.count == 0:
                    break
                lru = self._index.get_lru_entries(1)
                if not lru:
                    break
                victim = lru[0]
                freed += victim.file_size
                file_path = self._get_file_path(victim.block_hash)
                try:
                    if file_path.exists():
                        file_path.unlink()
                except OSError as e:
                    logger.debug("Enforce unlink failed: %s", e)
                self._index.remove(victim.block_hash)
        return freed

    def preload_matched_blocks(self, block_hashes: list[bytes]) -> int:
        if not block_hashes or self._hot_cache_max_bytes == 0:
            return 0
        cold_hashes = []
        for bh in block_hashes:
            if not self._index.contains(bh):
                continue
            if self._hot_cache_get(bh) is not None:
                continue
            cold_hashes.append(bh)
        if len(cold_hashes) < 4:
            return 0
        remaining = self._hot_cache_max_bytes - self._hot_cache_total_bytes
        if remaining <= 0:
            return 0
        loaded = 0
        t0 = time.time()
        for bh in cold_hashes:
            meta = self._index.get(bh)
            if meta is None:
                continue
            if self._hot_cache_total_bytes + meta.file_size > self._hot_cache_max_bytes:
                break
            if self._hot_cache_budget is not None:
                if self._hot_cache_budget.remaining_bytes < meta.file_size:
                    break
            file_path = self._get_file_path(bh)
            if not file_path.exists():
                continue
            result = self._load_safetensors_raw(str(file_path))
            if result is None:
                continue
            tensors_raw, file_metadata = result
            entry = {
                "tensors_raw": tensors_raw,
                "file_metadata": file_metadata,
                "num_layers": int(file_metadata.get("num_layers", 0)),
                "layer_cache_types": (
                    json.loads(file_metadata["layer_cache_types"])
                    if "layer_cache_types" in file_metadata
                    else None
                ),
                "block_metadata": meta,
                "dirty": False,
            }
            self._hot_cache_put(bh, entry)
            self._stats["hot_cache_promotions"] += 1
            loaded += 1
        elapsed_ms = (time.time() - t0) * 1000
        self._stats["preload_blocks_loaded"] += loaded
        self._stats["preload_calls"] += 1
        self._stats["preload_time_ms"] += elapsed_ms
        return loaded

    def adopt_layer_signature_if_unset(
        self, layer_cache_types: list[str] | None
    ) -> bool:
        if not layer_cache_types:
            return False
        if self._expected_layer_cache_types is not None:
            return False
        self._expected_layer_cache_types = list(layer_cache_types)
        self._signature_sweep_completed = False
        return True

    def invalidate_stale_layer_signature(self) -> int:
        if not self._expected_layer_cache_types:
            return 0
        if not self._expected_model_name:
            return 0
        if self._signature_sweep_completed:
            return 0
        dropped = 0
        with self._state_lock:
            for block_hash in list(self._index.blocks.keys()):
                meta = self._index.get(block_hash)
                if meta is None:
                    continue
                if not meta.model_name or meta.model_name != self._expected_model_name:
                    continue
                if meta.layer_cache_types is None:
                    continue
                if not self._signatures_match(
                    meta.layer_cache_types, self._expected_layer_cache_types
                ):
                    self._index.remove(block_hash)
                    dropped += 1
        self._signature_sweep_completed = True
        return dropped

    def set_expected_layer_signature(self, layer_cache_types: list[str] | None) -> bool:
        if not layer_cache_types:
            return False
        current_canon = (
            _canonicalize_layer_cache_types(self._expected_layer_cache_types)
            if self._expected_layer_cache_types
            else None
        )
        new_canon = _canonicalize_layer_cache_types(layer_cache_types)
        is_canonical_match = current_canon == new_canon
        self._expected_layer_cache_types = list(layer_cache_types)
        if is_canonical_match:
            return False
        self._signature_sweep_completed = False
        return True

    def _signatures_match(self, a: list[str], b: list[str]) -> bool:
        return _canonicalize_layer_cache_types(a) == _canonicalize_layer_cache_types(b)

    def _recover_from_block_error(self, block_hash: bytes):
        file_path = self._get_file_path(block_hash)
        temp_path = file_path.with_suffix(".tmp")
        try:
            if file_path.exists():
                file_path.unlink()
            if temp_path.exists():
                temp_path.unlink()
        except OSError as e:
            logger.debug("Failed to clean up block files: %s", e)
        with self._state_lock:
            self._index.remove(block_hash)

    def store_block(
        self, block_id: int, layers: list[Any], metadata: dict | None = None
    ) -> bool:
        if not layers:
            return False
        block_hash = str(block_id).encode().ljust(20, b"_")[:20]
        return self.save_block(block_hash=block_hash, cache_data=layers, token_count=0)

    def verify_and_repair_index(self) -> dict[str, int]:
        report = {"orphaned_files_removed": 0, "stale_entries_evicted": 0}
        if self._cache_dir is None:
            return report
        for f in self._cache_dir.rglob("*.tmp"):
            if f.is_file():
                try:
                    f.unlink()
                    report["orphaned_files_removed"] += 1
                except OSError:
                    pass
        with self._state_lock:
            for block_hash in list(self._index.blocks.keys()):
                meta = self._index.get(block_hash)
                if meta and meta.file_path and not meta.file_path.exists():
                    file_path = self._get_file_path(block_hash)
                    if not file_path.exists():
                        self._index.remove(block_hash)
                        report["stale_entries_evicted"] += 1
        return report
