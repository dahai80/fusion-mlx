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
        adopted = []
        for phys in physical_blocks:
            rc = self._refcount.get(phys, 0)
            if rc == 0:
                logger.warning(
                    "paged_kv_cow share_pages: phys %d not in pool, skip", phys
                )
                continue
            self._refcount[phys] = rc + 1
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

    def __init__(self, cache: CoWPagedKVCache):
        self.cache = cache
        self._hash_to_pages: dict[bytes, list[int]] = {}

    def register_prefix(self, block_hash: bytes, physical_blocks: list[int]) -> None:
        """Register a completed prefix's block hash -> physical pages."""
        self._hash_to_pages[block_hash] = list(physical_blocks)
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
    "PrefixPageBinder",
    "align_block_size_to_fa_tile",
    "is_two_level_kv_enabled",
]
