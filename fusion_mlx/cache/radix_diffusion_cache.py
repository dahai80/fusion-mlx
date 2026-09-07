# SPDX-License-Identifier: Apache-2.0
"""
Radix-tree KV cache for diffusion models.

Provides prefix-based KV cache sharing for text encoder outputs and
temporal latent reuse across consecutive video shots. Enables zero-copy
pointer sharing when prompts share common prefixes (e.g. multi-shot
short-drama pipelines where only the action description changes).

Closes #178
"""

import heapq
import logging
import os
import threading
import time
import weakref
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# #178 Phase-2: module-level registry of live caches so the admin stats
# endpoint can aggregate per-encoder caches without threading references
# through the lazy-loaded server->engine->pipeline chain. WeakSet keeps
# caches alive only while their owning encoder is alive (auto-removed on GC).
_REGISTRY: "weakref.WeakSet[DiffusionRadixCache]" = weakref.WeakSet()


def all_cache_stats() -> list[dict]:
    """Aggregate stats for every live DiffusionRadixCache instance.

    Returns one dict per cache, each = {"name": str|None, **stats()}.
    Caches that have been garbage-collected (encoder unloaded) are absent.
    """
    out = []
    for cache in _REGISTRY:
        try:
            entry = {"name": cache.name}
            entry.update(cache.stats())
            out.append(entry)
        except ReferenceError:
            continue
    return out


@dataclass
class RadixCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    insertions: int = 0
    leaf_count: int = 0
    total_bytes: int = 0


class _RadixNode:
    __slots__ = (
        "children",
        "value",
        "size_bytes",
        "last_access",
        "ref_count",
        "_heap_seq",
    )

    def __init__(self):
        self.children: dict[str, _RadixNode] = {}
        self.value: object | None = None
        self.size_bytes: int = 0
        self.last_access: float = 0.0
        self.ref_count: int = 0
        self._heap_seq: int = 0


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

    def __init__(self, max_mb: int = 512, name: str | None = None):
        self.max_bytes = max_mb * 1024 * 1024
        # PB-12 (#0907 audit): hard cap on live leaves as a backstop for
        # max_bytes. A pathological key set (many tiny/zero-byte values, e.g.
        # empty mx arrays or size-hinted 0) could slip past the byte budget
        # while still leaking _RadixNode objects + heap tuples. The byte
        # budget stays the primary eviction driver; this is a node-count
        # ceiling so a leaky key pattern degrades to evictions, not OOM.
        self.max_nodes = int(os.environ.get("FUSION_RADIX_MAX_NODES", "200000"))
        self._root = _RadixNode()
        self._stats = RadixCacheStats()
        self._clock = 0.0
        # #178 Phase-2: human-readable label surfaced by all_cache_stats()
        self.name = name
        self._lru_heap: list[tuple[float, int, _RadixNode]] = []
        self._heap_seq = 0
        # PB-12 (#0907 audit): compact the LRU heap (drop stale tuples whose
        # node was already evicted or re-touched) every N insertions. Without
        # this the heap accumulates one tuple per historical _touch() and only
        # sheds them lazily during an eviction scan — a long-lived cache with
        # frequent hits grows the heap unboundedly between evictions.
        self._heap_compact_every = 4096
        self._puts_since_compact = 0
        # ARCH-P3-3 (#0907 audit): this cache has no internal lock — safety
        # relies on a single-thread executor contract (image/video/audio are
        # max_workers=1; see engine_core._executor_config). Retrofitting a
        # lock around mx.array-returning ops would serialize the GPU, so we
        # enforce the contract by recording the owning thread and warning
        # loudly on cross-thread access instead of failing silently. A caller
        # that bumps FUSION_MLX_MAX_CONCURRENT_VIDEO>1 (or a future
        # multi-thread caller) hits this before racing the radix tree.
        self._owner_thread = threading.get_ident()
        self._thread_warned = False
        _REGISTRY.add(self)
        logger.debug(
            "radix cache created: name=%s max_mb=%d max_nodes=%d",
            name,
            max_mb,
            self.max_nodes,
        )

    def _check_thread(self) -> None:
        # ARCH-P3-3 (#0907 audit): warn once if a mutating op arrives on a
        # different thread than the one that created the cache. The radix
        # tree is not locked; concurrent access races the LRU heap and node
        # refcounts. This is a detection aid, not a guard — the executor
        # contract (max_workers=1) is the real invariant.
        if self._owner_thread != threading.get_ident() and not self._thread_warned:
            self._thread_warned = True
            logger.error(
                "DiffusionRadixCache(name=%s) accessed from thread %d but "
                "created on thread %d — single-thread contract violated; "
                "radix tree is unlocked and may race. Check "
                "FUSION_MLX_MAX_CONCURRENT_VIDEO and executor max_workers.",
                self.name,
                threading.get_ident(),
                self._owner_thread,
            )

    def get(self, key: str) -> object | None:
        """Look up a key in the radix tree.

        Returns the cached value on hit, None on miss.
        Updates last_access on hit for LRU tracking.
        """
        self._check_thread()
        self._clock = time.monotonic()
        node = self._walk(key)
        if node is not None and node.value is not None:
            node.last_access = self._clock
            self._touch(node)
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
        self._check_thread()
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
        self._touch(node)
        self._stats.total_bytes += size_bytes

        self._evict_if_needed()
        # PB-12 (#0907 audit): node-count backstop (see max_nodes). The byte
        # budget is primary; this catches zero-byte-value leaks that slip past
        # it. Also compact the LRU heap periodically so stale _touch() tuples
        # do not accumulate unboundedly between byte-driven evictions.
        self._puts_since_compact += 1
        if self._puts_since_compact >= self._heap_compact_every:
            self._puts_since_compact = 0
            self._compact_lru_heap()
        if self._stats.leaf_count > self.max_nodes:
            self._evict_to_node_cap()

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
            "max_nodes": self.max_nodes,
            "lru_heap_size": len(self._lru_heap),
            "hit_rate": s.hits / max(s.hits + s.misses, 1),
        }

    def clear(self) -> None:
        self._root = _RadixNode()
        self._stats = RadixCacheStats()
        self._lru_heap.clear()
        self._heap_seq = 0
        self._puts_since_compact = 0

    def drop_prefix(self, prefix: str) -> int:
        self._check_thread()
        # CS-2 (#811 audit 0906): remove every key that starts with *prefix*.
        # Used by per-model session-tail latent invalidation on engine unload:
        # a re-pull/quant swap under the same model_id would otherwise hand a
        # stale tail-frame latent to the next multi-shot request, silently
        # corrupting continuation frames. Returns the number of leaves freed.
        if not prefix:
            return 0
        subtree, parent, edge_key = self._walk_prefix(self._root, prefix, None, "")
        if subtree is None:
            return 0
        freed = self._prune_subtree(subtree)
        if parent is not None and edge_key is not None:
            del parent.children[edge_key]
            self._cleanup_chains(self._root, None, "")
        else:
            # prefix consumed exactly at the root → rebuild an empty root
            self._root = _RadixNode()
            self._lru_heap.clear()
        if freed:
            logger.info(
                "radix cache drop_prefix '%s' freed %d leaf/leaves (%d bytes)",
                prefix[:32],
                freed,
                self._stats.total_bytes,
            )
        return freed

    def _walk_prefix(self, node, remainder, parent, edge_key):
        # Walk consuming *remainder* of the prefix. Returns the subtree root
        # node whose entire descendant set matches the prefix, plus its parent
        # + edge key so the subtree can be detached. (None, None, None) = miss.
        if not remainder:
            return node, parent, edge_key
        for prefix, child in node.children.items():
            common = self._common_prefix(remainder, prefix)
            if not common:
                continue
            if common == prefix:
                # full edge consumed; descend if prefix still has chars,
                # else this child is the subtree root
                return self._walk_prefix(child, remainder[len(common) :], node, prefix)
            # common < prefix → prefix ends mid-edge; child's whole subtree
            # matches (every key under this edge starts with *common* == the
            # remainder we had, which is the trailing part of the prefix).
            if common == remainder:
                return child, node, prefix
            return None, None, None
        return None, None, None

    def _prune_subtree(self, node) -> int:
        # Free every leaf under *node*, fix stats. Does NOT detach node from
        # its parent (caller does). Marks values None so stale LRU-heap
        # tuples are rejected by the pop filter (line ~318).
        freed = 0
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.value is not None:
                self._stats.total_bytes -= cur.size_bytes
                self._stats.leaf_count -= 1
                self._stats.evictions += 1
                freed += 1
                cur.value = None
                cur._heap_seq = -1
            stack.extend(cur.children.values())
        return freed

    # M1: explicit deregister from _REGISTRY on teardown
    @classmethod
    def unregister(cls, cache: "DiffusionRadixCache") -> None:
        _REGISTRY.discard(cache)

    def __del__(self) -> None:
        try:
            _REGISTRY.discard(self)
        except Exception as e:
            # P3 (#811): bare pass hid teardown errors. __del__ runs at GC
            # time so re-raising is uncatchable and just spams stderr; log
            # at debug so a real bug stays traceable while benign shutdown
            # noise stays quiet.
            logger.debug("DiffusionRadixCache.__del__ deregister failed: %s", e)

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
        # E-34: a pinned leaf (ref_count > 0) at the LRU position previously
        # broke the whole eviction loop, so a single pinned leaf let the cache
        # grow past max_bytes unbounded -> OOM. Skip pinned leaves and keep
        # evicting the next LRU candidate instead. Bound consecutive skips by
        # leaf_count so an all-pinned trie terminates instead of spinning.
        skipped = 0
        while self._stats.total_bytes > self.max_bytes and self._stats.leaf_count > 1:
            if skipped >= self._stats.leaf_count:
                logger.warning(
                    "radix cache: all %d leaves pinned (ref_count>0); cannot "
                    "evict below max_bytes (total=%d, max=%d)",
                    self._stats.leaf_count,
                    self._stats.total_bytes,
                    self.max_bytes,
                )
                break
            victim = self._pop_lru_leaf()
            if victim is None:
                break
            parent, edge_key, lru_node = victim
            if lru_node.ref_count > 0:
                skipped += 1
                continue
            skipped = 0
            self._stats.total_bytes -= lru_node.size_bytes
            self._stats.leaf_count -= 1
            self._stats.evictions += 1
            del parent.children[edge_key]
            # P3 (#811): mark the evicted node so the LRU heap pop filter
            # (line ~306: ``node.value is None``) rejects its stale tuples
            # cheaply instead of re-walking the trie via _find_parent. The
            # heap still carries one tuple per historical _touch(); without
            # this mark those dead tuples accumulate unboundedly and each
            # only falls out lazily on the next eviction scan.
            lru_node.value = None
            lru_node._heap_seq = -1
            self._cleanup_chains(self._root, None, "")
            logger.debug(
                "radix cache evicted %d bytes (leaves=%d, evictions=%d)",
                lru_node.size_bytes,
                self._stats.leaf_count,
                self._stats.evictions,
            )

    def _touch(self, node: _RadixNode) -> None:
        self._heap_seq += 1
        node._heap_seq = self._heap_seq
        heapq.heappush(self._lru_heap, (node.last_access, self._heap_seq, node))

    def _compact_lru_heap(self) -> None:
        # PB-12 (#0907 audit): rebuild the LRU heap keeping only live tuples
        # — a node with a non-None value whose _heap_seq matches the tuple's
        # seq is the current canonical entry; everything else is stale (the
        # node was evicted, or re-touched and a newer tuple supersedes it).
        # Without periodic compaction the heap grows one tuple per _touch()
        # and only sheds lazily during eviction, so a hot cache with frequent
        # hits leaks heap memory between evictions.
        live = [
            (ts, seq, node)
            for ts, seq, node in self._lru_heap
            if node.value is not None and node._heap_seq == seq
        ]
        before = len(self._lru_heap)
        self._lru_heap = live
        heapq.heapify(self._lru_heap)
        dropped = before - len(self._lru_heap)
        if dropped:
            logger.debug(
                "radix cache: compacted LRU heap (%d -> %d, dropped %d stale)",
                before,
                len(self._lru_heap),
                dropped,
            )

    def _evict_to_node_cap(self) -> None:
        # PB-12 (#0907 audit): max_nodes backstop. Evict LRU leaves until
        # leaf_count <= max_nodes. Reuses the same pinned-skip loop as
        # _evict_if_needed (ref_count > 0 leaves are skipped, not evicted)
        # so a pinned hot leaf never gets dropped to satisfy a count cap.
        skipped = 0
        while self._stats.leaf_count > self.max_nodes and self._stats.leaf_count > 1:
            if skipped >= self._stats.leaf_count:
                logger.warning(
                    "radix cache: all %d leaves pinned; cannot evict to "
                    "max_nodes=%d (leaf_count=%d)",
                    self._stats.leaf_count,
                    self.max_nodes,
                    self._stats.leaf_count,
                )
                break
            victim = self._pop_lru_leaf()
            if victim is None:
                break
            parent, edge_key, lru_node = victim
            if lru_node.ref_count > 0:
                skipped += 1
                continue
            skipped = 0
            self._stats.total_bytes -= lru_node.size_bytes
            self._stats.leaf_count -= 1
            self._stats.evictions += 1
            del parent.children[edge_key]
            lru_node.value = None
            lru_node._heap_seq = -1
            self._cleanup_chains(self._root, None, "")
        if self._stats.leaf_count > self.max_nodes:
            logger.warning(
                "radix cache: node cap enforced, leaf_count now %d "
                "(max_nodes=%d, evictions=%d)",
                self._stats.leaf_count,
                self.max_nodes,
                self._stats.evictions,
            )

    def _pop_lru_leaf(self):
        while self._lru_heap:
            ts, seq, node = heapq.heappop(self._lru_heap)
            if node.value is None:
                continue
            if node._heap_seq != seq:
                continue
            parent, edge_key = self._find_parent(self._root, node, None, "")
            if parent is not None:
                return (parent, edge_key, node)
        return self._find_lru_leaf(self._root, None, "")

    def _find_parent(self, current, target, parent, edge_key):
        for prefix, child in current.children.items():
            if child is target:
                return (current, prefix)
            result = self._find_parent(child, target, current, prefix)
            if result is not None:
                return result
        return None

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
                # P3 (#811): parenthesize the precedence. The bare
                # `HAS_MLX and isinstance(...) or hasattr(...)` parsed as
                # `(HAS_MLX and isinstance) or hasattr`, which happened to
                # work but is fragile. Match the intent explicitly: an mlx
                # array OR any object exposing nbytes.
                if (HAS_MLX and isinstance(v, mx.array)) or hasattr(v, "nbytes"):
                    total += v.nbytes
            return total or 64
        if hasattr(value, "nbytes"):
            return value.nbytes
        return 64
