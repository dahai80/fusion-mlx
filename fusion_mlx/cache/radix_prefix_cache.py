# SPDX-License-Identifier: Apache-2.0
"""True radix-tree prefix KV cache.

Drop-in alternative to ``BlockAwarePrefixCache``. Instead of hashing whole
block sequences and hoping for an exact prefix collision, this walks a radix
trie over token-id blocks and returns the longest *real* prefix that has
cached KV blocks. Selected via env ``FUSION_MLX_PREFIX_CACHE=radix``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .paged_cache import BlockTable, PagedCacheManager
from .prefix_cache import BlockAwarePrefixCache, BlockCacheEntry

logger = logging.getLogger(__name__)


@dataclass
class _RadixNode:
    # token-id key for the edge into this node (block-aligned tuple)
    key: tuple[int, ...] = ()
    # physical block id backing this node's KV, if materialized
    block_id: int | None = None
    # children keyed by their first token id for O(1) descent
    children: dict[int, _RadixNode] = field(default_factory=dict)
    # last access epoch for LRU eviction
    last_access: float = 0.0
    # number of cached token positions this node represents
    num_tokens: int = 0

    def is_leaf(self) -> bool:
        return not self.children


class RadixPrefixCache:
    """Radix-trie prefix cache over token-id blocks.

    Public surface mirrors ``BlockAwarePrefixCache`` so the scheduler /
    engine can swap implementations with no call-site changes: ``fetch_cache``,
    ``store_cache``, ``release_cache``, ``clear_request_entry``, ``fork_cache``,
    ``block_size``.
    """

    def __init__(
        self,
        model: Any,
        paged_cache_manager: PagedCacheManager,
        paged_ssd_cache_manager: Any | None = None,
    ):
        self.model = model
        self.model_key = id(model)
        self.paged_cache = paged_cache_manager
        self.block_size = paged_cache_manager.block_size
        self.expected_num_layers = self._get_model_num_layers(model)

        # KV persistence (extraction + SSD save + reconstruction) is delegated
        # to a BlockAwarePrefixCache over the SAME paged cache manager. The
        # radix trie below is a pure token-id index over the block_ids
        # BlockAware returns from store_cache — it does not store KV itself.
        # This reuses the 800+ line extract/reconstruct path verbatim (#807).
        self._kv_cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache_manager,
            paged_ssd_cache_manager=paged_ssd_cache_manager,
        )
        self.paged_ssd_cache = self._kv_cache.paged_ssd_cache

        self._root = _RadixNode(last_access=time.monotonic())
        # request_id -> BlockCacheEntry (block table tracking)
        self._request_tables: dict[str, BlockCacheEntry] = {}
        # node -> block_id reverse index for eviction
        self._node_index: dict[int, _RadixNode] = {}

        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._tokens_matched_total = 0
        self._tokens_requested_total = 0
        self._cache_lock = asyncio.Lock()
        self._cold_restore_callback: Callable[[int, bytes], bool] | None = None

    # ------------------------------------------------------------------ utils

    def _now(self) -> float:
        return time.monotonic()

    def _block_slices(self, tokens: list[int]):
        n = len(tokens)
        for start in range(0, n - n % self.block_size, self.block_size):
            yield start, start + self.block_size, tuple(
                tokens[start : start + self.block_size]
            )

    def _get_model_num_layers(self, model: Any) -> int:
        make_cache = getattr(model, "make_cache", None)
        if callable(make_cache):
            try:
                cache_list = make_cache()
                if isinstance(cache_list, list) and len(cache_list) > 0:
                    return len(cache_list)
            except Exception as e:
                logger.debug(f"radix: make_cache() failed: {e}")
        if hasattr(model, "layers"):
            return len(model.layers)
        if hasattr(model, "args") and hasattr(model.args, "num_hidden_layers"):
            return model.args.num_hidden_layers
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
        logger.debug("radix: cannot determine num_layers, validation disabled")
        return 0

    def set_paged_ssd_cache_manager(self, paged_ssd_cache_manager: Any) -> None:
        self.paged_ssd_cache = paged_ssd_cache_manager
        self._kv_cache.set_paged_ssd_cache_manager(paged_ssd_cache_manager)
        if paged_ssd_cache_manager is not None:
            logger.info("PagedSSDCacheManager connected to RadixPrefixCache")

    def set_cold_restore_callback(
        self, callback: Callable[[int, bytes], bool] | None
    ) -> None:
        self._cold_restore_callback = callback
        self._kv_cache.set_cold_restore_callback(callback)

    def preload_blocks(self, block_table: Any) -> int:
        return self._kv_cache.preload_blocks(block_table)

    def reconstruct_cache(
        self, block_table: Any, promote_to_hot_cache: bool = True
    ) -> list[Any] | None:
        return self._kv_cache.reconstruct_cache(
            block_table, promote_to_hot_cache=promote_to_hot_cache
        )

    def get_stats_dict(self) -> dict[str, Any]:
        paged_stats = self.paged_cache.get_memory_usage()
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total > 0 else 0,
            "tokens_saved": self._tokens_saved,
            "block_size": self.block_size,
            "tokens_matched_total": self._tokens_matched_total,
            "tokens_requested_total": self._tokens_requested_total,
            "active_requests": len(self._request_tables),
            **paged_stats,
        }

    def clear(self) -> int:
        cleared = len(self._request_tables) + len(self._node_index)
        self._request_tables.clear()
        self._node_index.clear()
        self._root = _RadixNode(last_access=time.monotonic())
        self._kv_cache.clear()
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._tokens_matched_total = 0
        self._tokens_requested_total = 0
        return cleared

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._tokens_matched_total = 0
        self._tokens_requested_total = 0

    def __len__(self) -> int:
        return len(self._request_tables)

    # ------------------------------------------------------------------ fetch

    def fetch_cache(
        self,
        request_id: str,
        tokens: list[int],
        extra_keys: tuple[Any, ...] | None = None,
        extra_key_token_start: int | None = None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None,
    ) -> tuple[BlockTable | None, list[int]]:
        if not tokens:
            return None, tokens

        matched_block_ids: list[int] = []
        matched_tokens = 0
        node = self._root
        remaining_after = len(tokens)

        for start, end, block_key in self._block_slices(tokens):
            child = node.children.get(block_key[0])
            if child is None or child.key != block_key or child.block_id is None:
                break
            # Only match blocks whose KV was actually persisted by BlockAware
            # (block_hash set means a chain hash + SSD entry exist). A node
            # whose block has no hash carries no restorable KV — stop here so
            # reconstruct_cache gets only fully-backed blocks.
            block = self.paged_cache.allocated_blocks.get(child.block_id)
            if block is None or block.block_hash is None:
                break
            node = child
            node.last_access = self._now()
            matched_block_ids.append(node.block_id)
            matched_tokens += self.block_size
            remaining_after = len(tokens) - (start + self.block_size)

        self._tokens_requested_total += len(tokens)

        if matched_block_ids:
            block_table = self.paged_cache.create_block_table(request_id)
            for bid in matched_block_ids:
                self.paged_cache.increment_ref(bid)
                block = self.paged_cache.allocated_blocks.get(bid)
                if block:
                    block_table.block_ids.append(bid)
                    block_table.num_tokens += block.token_count

            self._request_tables[request_id] = BlockCacheEntry(
                block_table=block_table, last_access=self._now()
            )

            remaining = tokens[matched_tokens:]
            self._hits += 1
            self._tokens_saved += matched_tokens
            self._tokens_matched_total += matched_tokens
            logger.debug(
                "radix cache hit req=%s matched=%d blocks=%d remaining=%d",
                request_id,
                matched_tokens,
                len(matched_block_ids),
                len(remaining),
            )
            return block_table, remaining

        self._misses += 1
        logger.debug("radix cache miss req=%s tokens=%d", request_id, len(tokens))
        return None, tokens

    # ------------------------------------------------------------------ store

    def store_cache(
        self,
        request_id: str,
        tokens: list[int],
        cache_data: list[Any],
        model_cache_config: Any | None = None,
        boundary_snapshots: dict[int, list[Any]] | None = None,
        extra_keys: tuple[Any, ...] | None = None,
        extra_key_token_start: int | None = None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None,
    ) -> BlockTable | None:
        if not tokens:
            return None

        # Delegate KV persistence (block allocation, tensor-slice extraction,
        # SSD save, chain-hash registration) to BlockAwarePrefixCache. It
        # returns a BlockTable over the SAME paged_cache_manager the trie
        # indexes — so the block_ids it returns are the ones we record.
        block_table = self._kv_cache.store_cache(
            request_id,
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            boundary_snapshots=boundary_snapshots,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )
        if block_table is None or not block_table.block_ids:
            return block_table

        # Index the returned full-block ids into the radix trie by token-id
        # block key. Only full blocks (floor of token count) are indexed —
        # matching BlockAware's own full-block-only lookup semantics.
        node = self._root
        inserted = 0
        full_blocks = len(block_table.block_ids)
        for start, end, block_key in self._block_slices(tokens):
            if inserted >= full_blocks:
                break
            child = node.children.get(block_key[0])
            if child is None or child.key != block_key:
                child = _RadixNode(
                    key=block_key, last_access=self._now(), num_tokens=self.block_size
                )
                node.children[block_key[0]] = child

            # Attach the block_id BlockAware allocated for this token range.
            bid = block_table.block_ids[inserted]
            if child.block_id is None:
                child.block_id = bid
                self._node_index[bid] = child
            child.last_access = self._now()
            node = child
            inserted += 1

        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table, last_access=self._now()
        )
        logger.debug(
            "radix store req=%s tokens=%d blocks_indexed=%d total_blocks=%d",
            request_id,
            len(tokens),
            inserted,
            len(block_table.block_ids),
        )
        return block_table

    # ------------------------------------------------------------------ release

    def release_cache(self, request_id: str) -> None:
        # Delegate to BlockAware so its refcount + block-table bookkeeping
        # stays balanced (store_cache registered the request there too).
        self._kv_cache.release_cache(request_id)
        self._request_tables.pop(request_id, None)
        logger.debug("radix released cache for %s", request_id)

    def clear_request_entry(self, request_id: str) -> None:
        self._kv_cache.clear_request_entry(request_id)
        self._request_tables.pop(request_id, None)
        logger.debug("radix cleared request entry for %s (blocks retained)", request_id)

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> BlockTable | None:
        # Delegate: BlockAware forks the block table (refcounts + paged cache).
        # The trie is token-id keyed and shared-prefix-only, so a forked
        # request reuses the same trie nodes — no new trie entry needed.
        new_table = self._kv_cache.fork_cache(source_request_id, new_request_id)
        if new_table is not None:
            self._request_tables[new_request_id] = BlockCacheEntry(
                block_table=new_table, last_access=self._now()
            )
            logger.debug(
                "radix fork %s -> %s blocks=%d",
                source_request_id,
                new_request_id,
                len(new_table.block_ids),
            )
        return new_table

    # ------------------------------------------------------------------ stats

    def get_stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "tokens_saved": self._tokens_saved,
            "tokens_matched": self._tokens_matched_total,
            "tokens_requested": self._tokens_requested_total,
            "nodes": len(self._node_index),
        }
