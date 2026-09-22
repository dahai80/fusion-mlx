# SPDX-License-Identifier: Apache-2.0
"""PR-H: PagedKVCache two-level addressing + CoW + FA-tile alignment.

Extends FusionPagedKVCache with:
  - Two-level addressing: logical block -> page_id -> physical slab
    (decouples sharing from layout, enables prefix page donation)
  - Copy-on-write: refcounted physical blocks, ensure_writable() copies
    on shared write before mutation
  - FA-tile alignment: block_size validated against Metal Flash Attention
    tile granularity (auto round-up when enabled)
  - Prefix page donation: share_pages() adopts physical blocks from a
    prefix-cache hit with refcount increment, skipping KV recompute

Degrade switch (default OFF — prototype):
  FUSION_SHIM_TWO_LEVEL_KV=1  — enable CoW + two-level addressing

When OFF, callers use stock FusionPagedKVCache — zero behavior change.
Golden reference harness (PR-F) verifies KL < 1e-6 vs stock when enabled.
"""

from __future__ import annotations

import logging
import os

from .paged_kv_cache import FusionPagedKVCache
from .paged_kv_pool import FusionPagedKVPool, FusionPagedRequestCache

logger = logging.getLogger(__name__)

_FA_TILE_SIZE = 64


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def is_two_level_kv_enabled() -> bool:
    return _env_on("FUSION_SHIM_TWO_LEVEL_KV")


def align_block_size_to_fa_tile(block_size: int) -> int:
    """Round block_size up to a multiple of the FA tile size."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    remainder = block_size % _FA_TILE_SIZE
    if remainder == 0:
        return block_size
    aligned = block_size + (_FA_TILE_SIZE - remainder)
    logger.info(
        "paged_kv_cow: block_size %d -> %d (FA tile=%d)",
        block_size,
        aligned,
        _FA_TILE_SIZE,
    )
    return aligned


class CoWPagedKVCache(FusionPagedKVCache):
    """FusionPagedKVCache + refcounted CoW + two-level page indirection.

    Stock path (FUSION_SHIM_TWO_LEVEL_KV=0): behaves exactly like
    FusionPagedKVCache — every block is refcount=1, ensure_writable is a
    no-op, share_pages is unused.

    Enabled path: physical blocks carry a refcount. Before any write to a
    block with refcount > 1, ensure_writable() copies it to a fresh slab
    and decrements the original. Prefix-cache hits donate pages via
    share_pages() — multiple requests read the same physical slab until
    one writes (CoW), avoiding KV recompute for shared prefixes.
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 256,
        slab_size: int = 1,
        fa_tile_align: bool = False,
    ):
        if fa_tile_align:
            block_size = align_block_size_to_fa_tile(block_size)
        super().__init__(
            block_size=block_size, num_blocks=num_blocks, slab_size=slab_size
        )
        self._refcount: dict[int, int] = {}
        self._cow_copies = 0
        self._shared_pages = 0

    def _alloc_block(self) -> int:
        idx = super()._alloc_block()
        self._refcount[idx] = 1
        return idx

    def _free_block(self, physical: int) -> None:
        rc = self._refcount.get(physical, 0)
        if rc <= 1:
            self._refcount.pop(physical, None)
            self.free_list.append(physical)
        else:
            self._refcount[physical] = rc - 1

    def ensure_writable(self, physical: int) -> int:
        """CoW: if physical block is shared (refcount > 1), copy to a fresh
        slab and decrement the original. Returns the block to write to
        (same physical if refcount==1, new physical if copied).
        """
        rc = self._refcount.get(physical, 1)
        if rc <= 1:
            return physical
        new_physical = super()._alloc_block()
        self._refcount[new_physical] = 1
        if self.keys_pool is not None:
            self.keys_pool[new_physical] = self.keys_pool[physical]
            self.values_pool[new_physical] = self.values_pool[physical]
        self._refcount[physical] = rc - 1
        self._cow_copies += 1
        logger.debug(
            "paged_kv_cow CoW: phys %d (rc=%d) -> %d",
            physical,
            rc,
            new_physical,
        )
        return new_physical

    def share_pages(self, physical_blocks: list[int]) -> list[int]:
        """Adopt physical blocks from a prefix-cache hit (refcount-shared).

        Caller passes a list of physical block indices from another cache
        (or the prefix page binder). Each gets refcount incremented and is
        appended to this cache's block_table. Returns the list of adopted
        logical block indices.
        """
        if not is_two_level_kv_enabled():
            logger.debug("paged_kv_cow share_pages: disabled, skip")
            return []
        # D5 (audit): cross-pool share silently reads garbage — physical
        # indices are meaningless in a different pool's tensor. A block
        # not in this cache's _refcount is either foreign (cross-pool) or
        # already freed (evicted) — both mean stale/garbage data. Fail
        # visibly instead of silently adopting wrong data.
        for phys in physical_blocks:
            if phys not in self._refcount:
                raise RuntimeError(
                    f"paged_kv_cow share_pages: phys {phys} not owned by this "
                    f"cache (cross-pool or freed block) — refusing to adopt "
                    f"to prevent silent KV corruption (D5)"
                )
        adopted = []
        for phys in physical_blocks:
            self._refcount[phys] = self._refcount[phys] + 1
            lb = len(self.block_table)
            self.block_table.append(phys)
            self._shared_pages += 1
            adopted.append(lb)
        self.offset = len(self.block_table) * self.block_size
        logger.info(
            "paged_kv_cow share_pages: adopted %d blocks, offset=%d",
            len(adopted),
            self.offset,
        )
        return adopted

    def update_and_fetch(self, keys, values):
        """Override: CoW any shared blocks before writing into them."""
        if not is_two_level_kv_enabled():
            return super().update_and_fetch(keys, values)

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
            phys = self.block_table[lb]
            phys = self.ensure_writable(phys)
            self.block_table[lb] = phys
            pos_start = self._pos_in_block(max(prev, block_start_logical))
            self.keys_pool[phys, ..., pos_start : pos_start + n, :] = keys[
                ..., s_start:s_end, :
            ]
            self.values_pool[phys, ..., pos_start : pos_start + n, :] = values[
                ..., s_start:s_end, :
            ]

        self.offset = end
        return self._fetch_logical(end)

    def trim(self, n):
        """Override: refcount-aware trim."""
        if not is_two_level_kv_enabled():
            return super().trim(n)
        n = min(self.offset, n)
        self.offset -= n
        new_num_blocks = (self.offset + self.block_size - 1) // self.block_size
        while len(self.block_table) > new_num_blocks:
            phys = self.block_table.pop()
            self._free_block(phys)
        return n

    def free_all(self) -> int:
        """Override: refcount-aware free."""
        if not is_two_level_kv_enabled():
            return super().free_all()
        freed = len(self.block_table)
        for phys in self.block_table:
            self._free_block(phys)
        self.block_table = []
        self.offset = 0
        return freed

    def stats(self) -> dict:
        s = super().stats()
        s["cow_copies"] = self._cow_copies
        s["shared_pages"] = self._shared_pages
        s["refcount_total"] = sum(self._refcount.values())
        s["two_level_enabled"] = is_two_level_kv_enabled()
        return s


class CoWPagedRequestCache(FusionPagedRequestCache):
    """Pool-backed FusionPagedRequestCache + CoW ensure_writable on write.

    trim / state.setter / free_all are inherited unchanged — they already
    route through pool.free_block(pb, request_id) / pool.free_request which
    are refcount-aware (decrement-and-defer when shared). Only update_and_fetch
    is overridden: before writing into a block, ensure_writable copies it if
    the block is shared (refcount>1) so the donor's KV is not mutated.

    Constructed by install_paged_kv (pool branch) when FUSION_SHIM_TWO_LEVEL_KV=1.
    """

    def __init__(self, pool: FusionPagedKVPool, request_id: str):
        super().__init__(pool, request_id)
        self._cow_copies = 0
        self._shared_pages = 0

    def adopt_donated(self, phys_blocks: list[int]) -> int:
        """Pre-populate this cache's block_table with donated (refcount-shared)
        physical blocks from a concurrent donor. Called by the scheduler on a
        prefix-donation hit. Sets offset to the donated prefix length so the
        engine skips prefill for those tokens. Returns number of blocks adopted.
        """
        if not phys_blocks:
            return 0
        for phys in phys_blocks:
            self.pool.share_block(self.request_id, phys)
            self.block_table.append(phys)
            self._shared_pages += 1
        self.offset = len(self.block_table) * self.pool.block_size
        logger.info(
            "CoWPagedRequestCache adopt_donated request=%s blocks=%d offset=%d",
            self.request_id,
            len(phys_blocks),
            self.offset,
        )
        return len(phys_blocks)

    def update_and_fetch(self, keys, values):
        if self._is_merged:
            raise RuntimeError("cannot update a merged CoWPagedRequestCache")
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        dtype = keys.dtype
        if dtype != self.pool.dtype:
            self.pool._adapt_dtype(dtype)
        self._B = B
        self._n_kv_heads = n_kv_heads
        self._k_head_dim = k_head_dim
        self._v_head_dim = v_head_dim
        self._dtype = dtype

        prev = self.offset
        end = prev + num_steps
        first_block = self._logical_to_block(prev)
        last_block = self._logical_to_block(end - 1)

        allocated_this_call: list[int] = []
        try:
            for lb in range(first_block, last_block + 1):
                while len(self.block_table) <= lb:
                    pb = self.pool.alloc_block(self.request_id)
                    self.block_table.append(pb)
                    allocated_this_call.append(pb)

            for lb in range(first_block, last_block + 1):
                block_start_logical = lb * self.pool.block_size
                block_end_logical = block_start_logical + self.pool.block_size
                s_start = max(prev, block_start_logical) - prev
                s_end = min(end, block_end_logical) - prev
                n = s_end - s_start
                if n <= 0:
                    continue
                phys = self.block_table[lb]
                # CoW: copy shared block before writing (donor's KV protected).
                new_phys = self.pool.ensure_writable(self.request_id, phys)
                if new_phys != phys:
                    self._cow_copies += 1
                    self.block_table[lb] = new_phys
                    phys = new_phys
                pos_start = self._pos_in_block(max(prev, block_start_logical))
                self.pool.keys_pool[phys, ..., pos_start : pos_start + n, :] = keys[
                    ..., s_start:s_end, :
                ]
                self.pool.values_pool[phys, ..., pos_start : pos_start + n, :] = values[
                    ..., s_start:s_end, :
                ]
        except Exception:
            while self.block_table and self.block_table[-1] in allocated_this_call:
                pb = self.block_table.pop()
                self.pool.free_block(pb, self.request_id)
            logger.warning(
                "CoWPagedRequestCache update_and_fetch rolled back %d blocks "
                "request=%s offset unchanged at %d",
                len(allocated_this_call),
                self.request_id,
                self.offset,
            )
            raise
        finally:
            self._block_table_len_before = len(self.block_table)

        self.offset = end
        return self._fetch_logical(end)

    def stats(self) -> dict:
        s = super().stats()
        s["cow_copies"] = self._cow_copies
        s["shared_pages"] = self._shared_pages
        s["refcount_total"] = sum(self.pool._refcount.values())
        s["two_level_enabled"] = is_two_level_kv_enabled()
        return s


class PoolPrefixPageBinder:
    """Pool-level prefix page binder: maps a prefix chain-hash to the GPU
    physical block ids a concurrent donor request allocated, so a later
    same-prefix request can donate (refcount-share) those slabs instead of
    recomputing prefill.

    GPU-resident hot layer parallel to the SSD-backed BlockAwarePrefixCache
    index. Reuses compute_block_hash (same chain hash) so donation aligns
    with the existing prefix index. Concurrent-donation scope only: a donor
    must still be active (slabs resident) when the receiver is admitted; if
    the donor was evicted (refcount=0 / block reused), donate returns [] and
    the receiver falls through to the SSD-reconstruct path.
    """

    def __init__(self, pool: FusionPagedKVPool, max_prefixes: int = 256):
        self.pool = pool
        from collections import OrderedDict

        self._hash_to_pages: OrderedDict[bytes, list[int]] = OrderedDict()
        self._max_prefixes = max_prefixes

    def register_prefix(self, block_hash: bytes, physical_blocks: list[int]) -> None:
        if not physical_blocks:
            return
        self._hash_to_pages[block_hash] = list(physical_blocks)
        self._hash_to_pages.move_to_end(block_hash)
        while len(self._hash_to_pages) > self._max_prefixes:
            evicted_hash, _ = self._hash_to_pages.popitem(last=False)
            logger.debug(
                "pool_prefix_binder: LRU evicted hash %s (cap=%d)",
                (
                    evicted_hash.hex()[:16]
                    if isinstance(evicted_hash, bytes)
                    else str(evicted_hash)
                ),
                self._max_prefixes,
            )
        logger.debug(
            "pool_prefix_binder: registered hash %s -> %d pages",
            block_hash.hex()[:16] if isinstance(block_hash, bytes) else str(block_hash),
            len(physical_blocks),
        )

    def lookup(self, block_hash: bytes) -> list[int] | None:
        pages = self._hash_to_pages.get(block_hash)
        if pages is None:
            return None
        self._hash_to_pages.move_to_end(block_hash)
        return list(pages)

    def donate(self, block_hash: bytes, request_id: str) -> list[int]:
        """Return the donor's physical block ids for block_hash, validating
        each is still resident (refcount>0). If any block was evicted/reused
        (refcount==0 or not in pool._refcount), the entry is stale — drop it
        and return [] so the caller falls back to SSD reconstruct.
        """
        pages = self.lookup(block_hash)
        if pages is None:
            return []
        with self.pool._lock:
            for phys in pages:
                if self.pool._refcount.get(phys, 0) <= 0:
                    logger.info(
                        "pool_prefix_binder: donate miss — phys %d evicted, "
                        "dropping stale hash entry",
                        phys,
                    )
                    self._hash_to_pages.pop(block_hash, None)
                    return []
        logger.info(
            "pool_prefix_binder: donate hash %s -> %d pages request=%s",
            block_hash.hex()[:16] if isinstance(block_hash, bytes) else str(block_hash),
            len(pages),
            request_id,
        )
        return pages


class PrefixPageBinder:
    """Binds prefix-cache block hashes to FusionPagedKVCache physical pages.

    Lightweight adapter: tracks which physical blocks in a CoWPagedKVCache
    correspond to a prefix-cache hit, so a new request can donate those
    pages (refcount-shared) instead of recomputing KV for the shared prefix.

    Cross-pool production wiring (BlockAwarePrefixCache <-> CoWPagedKVCache)
    is Tier-2 deferred — this adapter works within a single CoWPagedKVCache
    pool (intra-request / same-model prefix reuse). The interface is stable
    so a future cross-pool bridge can drop in.
    """

    def __init__(self, cache: CoWPagedKVCache, max_prefixes: int = 256):
        self.cache = cache
        # D7 (audit): cap the prefix dict so it does not grow unbounded.
        # OrderedDict so we can LRU-evict the oldest entry on overflow.
        from collections import OrderedDict

        self._hash_to_pages: OrderedDict[bytes, list[int]] = OrderedDict()
        self._max_prefixes = max_prefixes

    def register_prefix(self, block_hash: bytes, physical_blocks: list[int]) -> None:
        """Register a completed prefix's block hash -> physical pages."""
        # D7: move-to-end on re-register so LRU order reflects recency.
        self._hash_to_pages[block_hash] = list(physical_blocks)
        self._hash_to_pages.move_to_end(block_hash)
        while len(self._hash_to_pages) > self._max_prefixes:
            evicted_hash, _ = self._hash_to_pages.popitem(last=False)
            logger.debug(
                "prefix_page_binder: LRU evicted hash %s (cap=%d)",
                (
                    evicted_hash.hex()[:16]
                    if isinstance(evicted_hash, bytes)
                    else str(evicted_hash)
                ),
                self._max_prefixes,
            )
        logger.debug(
            "prefix_page_binder: registered hash %s -> %d pages",
            block_hash.hex()[:16] if isinstance(block_hash, bytes) else str(block_hash),
            len(physical_blocks),
        )

    def lookup(self, block_hash: bytes) -> list[int] | None:
        """Return physical pages for a prefix hash, or None."""
        pages = self._hash_to_pages.get(block_hash)
        if pages is None:
            return None
        # D7: refresh LRU recency on hit.
        self._hash_to_pages.move_to_end(block_hash)
        logger.debug(
            "prefix_page_binder: hit hash %s -> %d pages",
            block_hash.hex()[:16] if isinstance(block_hash, bytes) else str(block_hash),
            len(pages),
        )
        return list(pages)

    def donate(self, block_hash: bytes) -> list[int]:
        """Donate prefix pages to the cache (refcount-shared via share_pages).

        Returns the list of adopted logical block indices, or [] if the
        hash is unknown or two-level KV is disabled.
        """
        pages = self.lookup(block_hash)
        if pages is None:
            return []
        return self.cache.share_pages(pages)


__all__ = [
    "CoWPagedKVCache",
    "CoWPagedRequestCache",
    "PrefixPageBinder",
    "PoolPrefixPageBinder",
    "align_block_size_to_fa_tile",
    "is_two_level_kv_enabled",
]
