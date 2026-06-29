# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for oMLX continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import logging

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any

import mlx.core as mx

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.


def adjust_store_cache_cap(    self, pressure_level: str) -> None:
    """Resize the store-cache gate based on memory pressure (#1383).

    Called from ProcessMemoryEnforcer on every poll. The cap walks one
    step per poll toward its target so transient spikes don't oscillate
    the cap. Bounded by [1, max_num_seqs]:
    - ok pressure: grow cap back toward max_num_seqs.
    - soft/hard pressure: shrink cap so KV cache backlog fits the system.
    """
    gate = self._store_cache_gate
    if gate is None:
        return
    current = gate.cap
    if pressure_level == "ok":
        new = min(self.config.max_num_seqs, current + 1)
    else:
        new = max(1, current - 1)
    if new != current:
        gate.set_cap(new)
        logger.debug(
            "store-cache queue cap: %d -> %d (pressure=%s)",
            current,
            new,
            pressure_level,
        )

# =========================================================================
# SSD Cache Methods
# =========================================================================

def _set_model_info_for_monitor(self) -> None:
    """Extract model info and set it on memory monitor for estimation."""
    if self.memory_monitor is None:
        return

    try:
        # Try to get model config
        config = None
        if hasattr(self.model, "config"):
            config = self.model.config
        elif hasattr(self.model, "args"):
            config = self.model.args

        if config is None:
            logger.debug("Could not extract model config for memory estimation")
            return

        # VLM / multimodal configs (e.g. Qwen3.6-VL, Gemma-4) nest the
        # language-model dimensions under a sub-config. Prefer
        # ``text_config`` / ``language_config`` / ``llm_config`` when ANY
        # of them exposes the LM layer count, even if the top-level config
        # also has one — on some VLM packs (older Gemma-3, certain Llava /
        # HF auto-wrappers) the top-level field refers to the *vision
        # encoder*, not the LM, and accepting it silently miscalibrates
        # the SDPA-peak estimate by a constant factor (a 40-layer LM
        # wrapped in a 33-layer vision tower under-estimates by ~20 %).
        # Probe both ``num_hidden_layers`` and the legacy ``n_layer`` alias
        # so a GPT-style nested config is also picked up. Falls back to the
        # top-level config only when no sub-config has either field.
        for sub_attr in ("text_config", "language_config", "llm_config"):
            sub = getattr(config, sub_attr, None)
            if sub is not None and (
                getattr(sub, "num_hidden_layers", None)
                or getattr(sub, "n_layer", None)
            ):
                config = sub
                break

        # Extract KV cache dimensions
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(
            config, "n_layer", None
        )
        num_kv_heads = (
            getattr(config, "num_key_value_heads", None)
            or getattr(config, "num_attention_heads", None)
            or getattr(config, "n_head", None)
        )
        head_dim = getattr(config, "head_dim", None)
        hidden_size = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )

        # Calculate head_dim if not directly available
        if head_dim is None and hidden_size and num_kv_heads:
            num_heads = getattr(config, "num_attention_heads", None) or num_kv_heads
            head_dim = hidden_size // num_heads

        # Determine dtype size
        dtype_size = 2  # Default float16
        if hasattr(self.model, "dtype"):
            if self.model.dtype == mx.float32:
                dtype_size = 4
            elif self.model.dtype == mx.bfloat16:
                dtype_size = 2

        # Extract num_attention_heads (query heads) for SDPA peak estimation
        num_attention_heads = (
            getattr(config, "num_attention_heads", None)
            or getattr(config, "n_head", None)
            or num_kv_heads
        )

        # Count KVCache layers for hybrid models
        num_kv_cache_layers = num_layers
        if hasattr(self.model, "make_cache"):
            try:
                cache_list = self.model.make_cache()
                from mlx_lm.models.cache import KVCache

                num_kv_cache_layers = sum(
                    1 for c in cache_list if type(c) is KVCache
                )
                if num_kv_cache_layers == 0:
                    num_kv_cache_layers = num_layers  # fallback
            except Exception:
                logger.debug("swallowed exception at fusion_mlx/scheduler/sched_misc.py:181")

                pass

        if num_layers and num_kv_heads and head_dim:
            self.memory_monitor.set_model_info(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype_size=dtype_size,
                num_attention_heads=num_attention_heads,
                num_kv_cache_layers=num_kv_cache_layers,
            )
            logger.debug(
                f"Model info for memory estimation: "
                f"layers={num_layers} ({num_kv_cache_layers} KVCache), "
                f"kv_heads={num_kv_heads}, q_heads={num_attention_heads}, "
                f"head_dim={head_dim}, dtype_size={dtype_size}"
            )
        else:
            logger.debug(
                f"Incomplete model info: layers={num_layers}, "
                f"kv_heads={num_kv_heads}, head_dim={head_dim}"
            )

    except Exception as e:
        logger.debug(f"Failed to extract model info: {e}")

def _init_tiered_cache(self) -> None:
    """Initialize paged SSD cache components if configured.

    In paged SSD-only mode:
    - All KV cache data is stored on paged SSD via PagedSSDCacheManager
    - PagedCacheManager only stores block metadata (no GPU memory for cache data)
    - BatchGenerator handles GPU memory for active inference
    """
    if not HAS_TIERED_CACHE:
        if self.config.paged_ssd_cache_dir:
            logger.warning(
                "paged SSD cache requested but ssd_cache/memory_monitor modules "
                "not available. Install required dependencies."
            )
        return

    # In paged SSD-only mode, paged_ssd_cache_dir is required
    if not self.config.paged_ssd_cache_dir:
        logger.debug(
            "paged SSD cache not configured (no --ssd-cache-dir specified)"
        )
        return

    try:
        cache_dir = (
            Path(self.config.paged_ssd_cache_dir)
            if self.config.paged_ssd_cache_dir
            else None
        )

        # Pass current model identity so stale blocks from a prior model
        # version (e.g., 30-layer cache after an upgrade to 40 layers via
        # #1404) are unlinked at startup instead of triggering a layer
        # mismatch reject on every prefix lookup. See #1413.
        expected_num_layers = (
            self.block_aware_cache.expected_num_layers
            if self.block_aware_cache is not None
            else 0
        )

        # Initialize paged SSD cache manager for SSD storage
        self.paged_ssd_cache_manager = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=self.config.paged_ssd_cache_max_size,
            hot_cache_max_bytes=self.config.hot_cache_max_size,
            hot_cache_only=self.config.hot_cache_only,
            expected_model_name=self.config.model_name or "",
            expected_num_layers=expected_num_layers,
        )

        # Connect paged SSD cache manager to PagedCacheManager
        if self.paged_cache_manager is not None:
            self.paged_cache_manager.set_paged_ssd_cache_manager(
                self.paged_ssd_cache_manager
            )

        # Connect paged SSD cache manager to BlockAwarePrefixCache for paged SSD-only mode
        if self.block_aware_cache is not None:
            self.block_aware_cache.set_paged_ssd_cache_manager(
                self.paged_ssd_cache_manager
            )

        # Initialize boundary snapshot SSD store for offloading
        # non-sliceable cache snapshots during prefill.
        # Skip in hot_cache_only mode since snapshots would never be written.
        if BoundarySnapshotSSDStore is not None and not self.config.hot_cache_only:
            try:
                self._boundary_snapshot_store = BoundarySnapshotSSDStore(
                    base_dir=Path(self.config.paged_ssd_cache_dir)
                )
            except Exception as e:
                logger.debug(
                    "Failed to initialize boundary snapshot SSD store: %s", e
                )

        logger.info(
            f"paged SSD cache enabled: "
            f"cache_dir={self.config.paged_ssd_cache_dir}, "
            f"max_size={self._format_bytes(self.config.paged_ssd_cache_max_size)}, "
            f"block_size={self.config.paged_cache_block_size} tokens"
        )

    except Exception as e:
        logger.error(f"Failed to initialize paged SSD cache: {e}")
        self.paged_ssd_cache_manager = None

def _check_memory_pressure(self) -> None:
    """Check memory and evict blocks if needed.

    In paged SSD-only mode, memory pressure is not monitored since
    KV cache data is stored on paged SSD, not GPU memory.
    """
    # In paged SSD-only mode, memory_monitor is not used
    # All KV cache data is on paged SSD, so no GPU memory pressure from PagedCache
    pass

def _evict_blocks_permanently(    self, bytes_to_free: int) -> int:
    """
    Evict LRU blocks permanently (metadata cleanup).

    In paged SSD-only mode, blocks don't store data in GPU memory.
    This method just removes block metadata to free up slots.

    Args:
        bytes_to_free: Target bytes to free (used for estimation).

    Returns:
        Number of bytes freed (estimated).
    """
    if self.paged_cache_manager is None or self.memory_monitor is None:
        return 0

    # Estimate how many blocks to evict
    block_size = self.config.paged_cache_block_size
    num_blocks_to_evict = self.memory_monitor.estimate_blocks_to_free(
        bytes_to_free, block_size
    )

    # Get evictable blocks in LRU order
    evictable = self.paged_cache_manager.get_evictable_blocks(num_blocks_to_evict)

    if not evictable:
        logger.debug("No evictable blocks found for permanent eviction")
        return 0

    freed = 0
    evicted_count = 0

    for block in evictable:
        # In paged SSD-only mode, just clear metadata (data is on paged SSD)
        if self.paged_cache_manager.evict_block_permanently(block.block_id):
            freed += self.memory_monitor.estimate_block_memory(block_size)
            evicted_count += 1

        if freed >= bytes_to_free:
            break

    if evicted_count > 0:
        logger.info(
            f"Evicted {evicted_count} blocks permanently "
            f"(~{self._format_bytes(freed)} estimated)"
        )

    return freed

def _evict_blocks_to_cold(    self, bytes_to_free: int) -> int:
    """
    Evict LRU blocks (with paged SSD cache configured).

    In paged SSD-only mode, data is already on paged SSD, so this just evicts
    block metadata from the index. The data remains on paged SSD and can
    be re-discovered if the same token sequence is requested.

    Args:
        bytes_to_free: Target bytes to free (used for estimation).

    Returns:
        Number of bytes freed (estimated).
    """
    if self.paged_cache_manager is None or self.paged_ssd_cache_manager is None:
        return 0

    if self.memory_monitor is None:
        return 0

    # Estimate how many blocks to evict
    block_size = self.config.paged_cache_block_size
    num_blocks_to_evict = self.memory_monitor.estimate_blocks_to_free(
        bytes_to_free, block_size
    )

    # Get evictable blocks in LRU order
    evictable = self.paged_cache_manager.get_evictable_blocks(num_blocks_to_evict)

    if not evictable:
        logger.debug("No evictable blocks found")
        return 0

    evicted_count = 0

    for block in evictable:
        # In paged SSD-only mode, data is already on paged SSD
        # Just evict the block metadata
        if self.paged_cache_manager.evict_block_permanently(block.block_id):
            evicted_count += 1

    # Estimate bytes freed based on block count
    estimated_freed = evicted_count * self.memory_monitor.estimate_block_memory(
        block_size
    )

    if evicted_count > 0:
        logger.info(
            f"Evicted {evicted_count} blocks from index "
            f"(data preserved on paged SSD, ~{self._format_bytes(estimated_freed)} metadata freed)"
        )

    return estimated_freed

def _restore_block_from_cold(    self, block_id: int, block_hash: bytes) -> bool:
    """
    Restore a block from cold storage (deprecated in paged SSD-only mode).

    In paged SSD-only mode, blocks don't store cache_data. Data is loaded
    directly from SSD when needed via reconstruct_cache().

    Kept for API compatibility.

    Args:
        block_id: Block ID to restore.
        block_hash: Block's content hash.

    Returns:
        True if block exists in cold storage.
    """
    if self.paged_ssd_cache_manager is None or self.paged_cache_manager is None:
        return False

    # In paged SSD-only mode, just verify block exists on paged SSD
    if not self.paged_ssd_cache_manager.has_block(block_hash):
        logger.warning(f"Block {block_id} not found in cold storage")
        return False

    # Touch the block to update LRU
    block = (
        self.paged_cache_manager.blocks[block_id]
        if block_id < len(self.paged_cache_manager.blocks)
        else None
    )
    if block:
        block.touch()

    logger.debug(
        f"Block {block_id} verified on paged SSD (hash={block_hash.hex()[:16]}...)"
    )
    return True

def restore_cold_blocks_for_request(    self, request_id: str) -> int:
    """
    Verify all blocks needed for a request exist on paged SSD.

    In paged SSD-only mode, blocks don't store cache_data. This method
    just verifies that blocks exist on paged SSD.

    Args:
        request_id: Request ID.

    Returns:
        Number of blocks verified on paged SSD.
    """
    if self.paged_cache_manager is None or self.paged_ssd_cache_manager is None:
        return 0

    if self.block_aware_cache is None:
        return 0

    # Get block table for request
    block_table = self.paged_cache_manager.request_tables.get(request_id)
    if block_table is None:
        return 0

    verified = 0
    for block_id in block_table.block_ids:
        block = self.paged_cache_manager.blocks[block_id]
        if block.block_hash is not None:
            if self._restore_block_from_cold(block_id, block.block_hash):
                verified += 1

    return verified

def _collect_cache_counters(self) -> dict[str, int] | None:
    if self.block_aware_cache is None:
        return None

    prefix_stats = self.block_aware_cache.get_stats()
    counters = {
        "prefix_hits": prefix_stats.hits,
        "prefix_misses": prefix_stats.misses,
        "prefix_tokens_matched": prefix_stats.tokens_matched_total,
        "prefix_tokens_requested": prefix_stats.tokens_requested_total,
        "prefix_tokens_saved": prefix_stats.tokens_saved,
        "evictions": prefix_stats.evictions,
    }

    if self.paged_ssd_cache_manager is not None:
        ssd = self.paged_ssd_cache_manager.get_stats()
        hot_hits = ssd.hot_cache_hits
        total_loads = ssd.loads
        counters.update({
            "ssd_hot_hits": hot_hits,
            "ssd_disk_loads": max(0, total_loads - hot_hits),
            "ssd_saves": ssd.saves,
            "ssd_errors": ssd.errors,
            "hot_cache_evictions": ssd.hot_cache_evictions,
            "hot_cache_promotions": ssd.hot_cache_promotions,
        })

    return counters

def get_ssd_cache_stats(self) -> dict[str, Any] | None:
    """Get paged SSD + prefix cache observability statistics."""
    stats = {}

    if self.paged_ssd_cache_manager is not None:
        stats["ssd_cache"] = self.paged_ssd_cache_manager.get_stats()

    if self.paged_cache_manager is not None:
        stats["indexed_blocks"] = self.paged_cache_manager.cold_block_count
        stats["block_size"] = self.config.paged_cache_block_size

    if self.block_aware_cache is not None:
        stats["prefix_cache"] = self.block_aware_cache.get_stats_dict()

    counters = self._collect_cache_counters()
    if counters:
        stats["cache_rates"] = self._cache_rate_tracker.snapshot_and_get_rates(
            counters
        )

    return stats if stats else None

# Alias for backwards compatibility
get_tiered_cache_stats = get_ssd_cache_stats

@staticmethod
def _format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_value >= 1024**3:
        return f"{bytes_value / 1024**3:.2f} GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / 1024**2:.2f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f} KB"
    else:
        return f"{bytes_value} B"

def _bypass_hot_cache_under_pressure(self) -> bool:
    if not self._prefill_memory_guard:
        return False
    if self._memory_limit_bytes <= 0:
        return False
    config = getattr(self, "config", None)
    if config is None:
        return False
    if getattr(config, "hot_cache_only", False):
        return False
    if int(getattr(config, "hot_cache_max_size", 0) or 0) <= 0:
        return False
    if getattr(self, "paged_ssd_cache_manager", None) is None:
        return False
    try:
        current = self._current_usage_bytes()
    except Exception:
        logger.debug("Failed to sample memory for hot-cache pressure bypass")
        return False
    return current >= self._memory_limit_bytes
