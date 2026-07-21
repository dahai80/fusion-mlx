# SPDX-License-Identifier: Apache-2.0
"""
Radix-tree KV cache for diffusion models.

Provides prefix-based KV cache sharing for text encoder outputs and
temporal latent reuse across consecutive video shots. Enables zero-copy
pointer sharing when prompts share common prefixes (e.g. multi-shot
short-drama pipelines where only the action description changes).

Closes #178
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass
class RadixCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    insertions: int = 0
    leaf_count: int = 0
    total_bytes: int = 0


class _RadixNode:
    __slots__ = ("children", "value", "size_bytes", "last_access", "ref_count")

    def __init__(self):
        self.children: dict[str, _RadixNode] = {}
        self.value: object | None = None
        self.size_bytes: int = 0
        self.last_access: float = 0.0
        self.ref_count: int = 0


class DiffusionRadixCache:
    """Radix-tree cache for diffusion model intermediate outputs.

    Supports:
    - Prefix sharing: prompts "a cat sitting on a red sofa, cinematic"
      and "a cat sitting on a blue sofa, cinematic" share the prefix
      "a cat sitting on a " in the tree structure.
    - LRU eviction: when total cached bytes exceed max_bytes, least
      recently used leaves are evicted.
    - Reference counting: nodes can be pinned to prevent eviction.
    - mx.array values: stored values are typically mx.array KV caches
      or latent tensors.

    Usage:
        cache = DiffusionRadixCache(max_mb=512)
        cache.put("prompt:a cat on a sofa", kv_array)
        result = cache.get("prompt:a cat on a sofa")
        stats = cache.stats()
    """

    def __init__(self, max_mb: int = 512):
        self.max_bytes = max_mb * 1024 * 1024
        self._root = _RadixNode()
        self._stats = RadixCacheStats()
        self._clock = 0.0

    def get(self, key: str) -> object | None:
        """Look up a key in the radix tree.

        Returns the cached value on hit, None on miss.
        Updates last_access on hit for LRU tracking.
        """
        self._clock = time.monotonic()
        node = self._walk(key)
        if node is not None and node.value is not None:
            node.last_access = self._clock
            self._stats.hits += 1
            logger.debug("radix cache hit: %s (%d bytes)", key[:32], node.size_bytes)
            return node.value
        self._stats.misses += 1
        return None

    def put(self, key: str, value: object, size_bytes: int | None = None) -> None:
        """Insert or update a key-value pair.

        Args:
            key: Cache key (typically a prompt hash or latent identifier).
            value: Cached data (mx.array KV cache, latent tensor, etc.).
            size_bytes: Optional size hint. If None, attempts to infer
                        from value.shape/value.nbytes for mx.array.
        """
        self._clock = time.monotonic()
        if size_bytes is None:
            size_bytes = self._infer_size(value)

        node = self._walk_or_create(key)
        if node.value is None:
            self._stats.leaf_count += 1
            self._stats.insertions += 1
        else:
            self._stats.total_bytes -= node.size_bytes

        node.value = value
        node.size_bytes = size_bytes
        node.last_access = self._clock
        self._stats.total_bytes += size_bytes

        self._evict_if_needed()

    def pin(self, key: str) -> bool:
        """Increment ref count to prevent eviction."""
        node = self._walk(key)
        if node is not None and node.value is not None:
            node.ref_count += 1
            return True
        return False

    def unpin(self, key: str) -> bool:
        """Decrement ref count. Returns False if key not found."""
        node = self._walk(key)
        if node is not None and node.ref_count > 0:
            node.ref_count -= 1
            return True
        return False

    def stats(self) -> dict:
        s = self._stats
        return {
            "hits": s.hits,
            "misses": s.misses,
            "evictions": s.evictions,
            "insertions": s.insertions,
            "leaf_count": s.leaf_count,
            "total_bytes": s.total_bytes,
            "max_bytes": self.max_bytes,
            "hit_rate": s.hits / max(s.hits + s.misses, 1),
        }

    def clear(self) -> None:
        self._root = _RadixNode()
        self._stats = RadixCacheStats()

    def _walk(self, key: str) -> _RadixNode | None:
        """Walk the radix tree for key lookup. Returns None if not found."""
        node = self._root
        remainder = key
        while remainder:
            matched = False
            for prefix, child in node.children.items():
                common = self._common_prefix(remainder, prefix)
                if not common:
                    continue
                if common == prefix:
                    node = child
                    remainder = remainder[len(prefix) :]
                    matched = True
                    break
                return None
            if not matched:
                return None
        return node

    def _walk_or_create(self, key: str) -> _RadixNode:
        """Walk the radix tree, creating nodes and splitting as needed."""
        node = self._root
        remainder = key
        while remainder:
            matched = False
            for prefix, child in list(node.children.items()):
                common = self._common_prefix(remainder, prefix)
                if not common:
                    continue
                if common == prefix:
                    node = child
                    remainder = remainder[len(prefix) :]
                    matched = True
                    break
                split = _RadixNode()
                old_suffix = prefix[len(common) :]
                new_suffix = remainder[len(common) :]
                split.children[old_suffix] = child
                del node.children[prefix]
                node.children[common] = split
                if new_suffix:
                    leaf = _RadixNode()
                    split.children[new_suffix] = leaf
                    node = leaf
                else:
                    node = split
                remainder = ""
                matched = True
                break
            if not matched:
                leaf = _RadixNode()
                node.children[remainder] = leaf
                node = leaf
                remainder = ""
        return node

    def _evict_if_needed(self) -> None:
        while self._stats.total_bytes > self.max_bytes and self._stats.leaf_count > 1:
            victim = self._find_lru_leaf(self._root, None, "")
            if victim is None:
                break
            parent, edge_key, lru_node = victim
            if lru_node.ref_count > 0:
                break
            self._stats.total_bytes -= lru_node.size_bytes
            self._stats.leaf_count -= 1
            self._stats.evictions += 1
            del parent.children[edge_key]
            self._cleanup_chains(self._root, None, "")
            logger.debug(
                "radix cache evicted %d bytes (leaves=%d, evictions=%d)",
                lru_node.size_bytes,
                self._stats.leaf_count,
                self._stats.evictions,
            )

    def _find_lru_leaf(self, node, parent, edge_key):
        if not node.children and node.value is not None:
            return (parent, edge_key, node)
        best = None
        best_access = float("inf")
        for prefix, child in node.children.items():
            result = self._find_lru_leaf(child, node, prefix)
            if result is not None and result[2].last_access < best_access:
                best = result
                best_access = result[2].last_access
        return best

    def _cleanup_chains(self, node, parent, edge_key):
        for prefix, child in list(node.children.items()):
            self._cleanup_chains(child, node, prefix)
        if parent is not None and len(node.children) == 1 and node.value is None:
            only_prefix = next(iter(node.children))
            only_child = node.children[only_prefix]
            merged = edge_key + only_prefix
            parent.children[merged] = only_child
            del parent.children[edge_key]

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        i = 0
        limit = min(len(a), len(b))
        while i < limit and a[i] == b[i]:
            i += 1
        return a[:i]

    @staticmethod
    def _infer_size(value: object) -> int:
        if HAS_MLX and isinstance(value, mx.array):
            return value.nbytes
        if isinstance(value, dict):
            total = 0
            for v in value.values():
                if HAS_MLX and isinstance(v, mx.array) or hasattr(v, "nbytes"):
                    total += v.nbytes
            return total or 64
        if hasattr(value, "nbytes"):
            return value.nbytes
        return 64
