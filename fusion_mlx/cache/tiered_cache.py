# SPDX-License-Identifier: Apache-2.0
import logging
import threading
import time
from typing import Any

from .interface import CacheManager
from .paged_cache import PagedCacheManager
from .stats import BaseCacheStats

logger = logging.getLogger(__name__)


class TieredCacheStats(BaseCacheStats):
    hot_hits: int = 0
    cold_hits: int = 0
    promotions: int = 0
    demotions: int = 0
    cow_copies_during_demotion: int = 0

    def reset(self) -> None:
        super().reset()
        self.hot_hits = 0
        self.cold_hits = 0
        self.promotions = 0
        self.demotions = 0
        self.cow_copies_during_demotion = 0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["hot_hits"] = self.hot_hits
        d["cold_hits"] = self.cold_hits
        d["promotions"] = self.promotions
        d["demotions"] = self.demotions
        d["cow_copies_during_demotion"] = self.cow_copies_during_demotion
        return d


class TieredCacheManager(CacheManager):
    def __init__(
        self,
        hot: PagedCacheManager,
        cold: Any | None = None,
        demotion_threshold: float = 0.85,
        promotion_on_cold_hit: bool = True,
    ):
        self._hot = hot
        self._cold = cold
        self._demotion_threshold = demotion_threshold
        self._promotion_on_cold_hit = promotion_on_cold_hit
        self._stats = TieredCacheStats()
        self._lock = threading.RLock()
        self._demotion_in_progress = False
        self._last_demotion_time: float = 0.0
        self._demotion_cooldown: float = 2.0
        logger.info(
            "TieredCacheManager init: hot=%s, cold=%s, "
            "demotion_threshold=%.0f%%, promotion=%s",
            type(hot).__name__,
            type(cold).__name__ if cold else "None",
            demotion_threshold * 100,
            promotion_on_cold_hit,
        )

    @property
    def hot(self) -> PagedCacheManager:
        return self._hot

    @property
    def cold(self) -> Any | None:
        return self._cold

    def fetch(self, key: Any) -> tuple[Any | None, bool]:
        block_hash = key
        hot_block = self._hot.get_cached_block(block_hash)
        if hot_block is not None:
            self._stats.record_hit()
            self._stats.hot_hits += 1
            logger.debug(
                "tiered fetch HIT hot: %s",
                block_hash.hex()[:16] if isinstance(block_hash, bytes) else block_hash,
            )
            return hot_block, True

        if self._cold is not None:
            cold_data, found = self._cold.fetch(block_hash)
            if found and cold_data is not None:
                self._stats.record_hit()
                self._stats.cold_hits += 1
                logger.debug(
                    "tiered fetch HIT cold: %s",
                    block_hash.hex()[:16]
                    if isinstance(block_hash, bytes)
                    else block_hash,
                )
                if self._promotion_on_cold_hit:
                    self._promote(block_hash, cold_data)
                return cold_data, True

        self._stats.record_miss()
        logger.debug(
            "tiered fetch MISS: %s",
            block_hash.hex()[:16] if isinstance(block_hash, bytes) else block_hash,
        )
        return None, False

    def store(self, key: Any, value: Any) -> bool:
        block_hash = key
        cache_data = value
        if self._cold is not None:
            saved = self._cold.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
            )
            if saved:
                logger.debug(
                    "tiered store → cold: %s",
                    block_hash.hex()[:16]
                    if isinstance(block_hash, bytes)
                    else block_hash,
                )
                return True

        self._stats.record_miss()
        return False

    def evict(self, key: Any) -> bool:
        block_hash = key
        hot_evicted = False
        cold_evicted = False

        if self._cold is not None:
            cold_evicted = self._cold.evict(block_hash)

        if not cold_evicted:
            hot_block = self._hot.get_cached_block(block_hash)
            if hot_block is not None:
                hot_evicted = True
                self._stats.record_eviction()

        if hot_evicted or cold_evicted:
            logger.debug(
                "tiered evict: %s (hot=%s cold=%s)",
                block_hash.hex()[:16] if isinstance(block_hash, bytes) else block_hash,
                hot_evicted,
                cold_evicted,
            )
        return hot_evicted or cold_evicted

    def clear(self) -> int:
        count = 0
        if self._cold is not None:
            count += self._cold.clear()
        count += self._hot.clear() if hasattr(self._hot, "clear") else 0
        self._stats.reset()
        logger.info("tiered clear: removed %d entries", count)
        return count

    def get_stats(self) -> TieredCacheStats:
        return self._stats

    @property
    def size(self) -> int:
        hot_stats = self._hot.get_stats()
        return hot_stats.total_tokens_cached

    @property
    def max_size(self) -> int:
        return self._hot.max_blocks * self._hot.block_size

    def maybe_demote(self) -> int:
        now = time.monotonic()
        if self._cold is None:
            return 0
        if self._demotion_in_progress:
            return 0
        if now - self._last_demotion_time < self._demotion_cooldown:
            return 0

        hot_stats = self._hot.get_stats()
        utilization = hot_stats.utilization
        if utilization < self._demotion_threshold:
            return 0

        with self._lock:
            if self._demotion_in_progress:
                return 0
            self._demotion_in_progress = True

        try:
            demoted = self._do_demotion()
            self._last_demotion_time = time.monotonic()
            if demoted > 0:
                logger.info("tiered demotion: %d blocks demoted hot→cold", demoted)
            return demoted
        finally:
            self._demotion_in_progress = False

    def _do_demotion(self) -> int:
        evictable = (
            self._hot.get_evictable_blocks()
            if hasattr(self._hot, "get_evictable_blocks")
            else []
        )
        if not evictable:
            return 0

        demoted = 0
        for block in evictable:
            if block.block_hash is None:
                continue

            if block.ref_count > 1:
                self._cow_demote(block)
            else:
                self._simple_demote(block)

            demoted += 1
            self._stats.demotions += 1

        if demoted > 0:
            self._hot.evict_lru_blocks(demoted)

        return demoted

    def _simple_demote(self, block: Any) -> None:
        if self._cold is None or block.block_hash is None:
            return
        logger.debug(
            "simple demote block %d hash=%s",
            block.block_id,
            block.block_hash.hex()[:16]
            if isinstance(block.block_hash, bytes)
            else block.block_hash,
        )

    def _cow_demote(self, block: Any) -> None:
        if self._cold is None or block.block_hash is None:
            return
        self._stats.cow_copies_during_demotion += 1
        logger.debug(
            "CoW demote block %d hash=%s ref_count=%d",
            block.block_id,
            block.block_hash.hex()[:16]
            if isinstance(block.block_hash, bytes)
            else block.block_hash,
            block.ref_count,
        )

    def _promote(self, block_hash: Any, cache_data: Any) -> None:
        if self._cold is None:
            return
        logger.debug(
            "promote cold→hot: %s",
            block_hash.hex()[:16] if isinstance(block_hash, bytes) else block_hash,
        )
        self._stats.promotions += 1

    def get_tier_stats(self) -> dict[str, Any]:
        result = {
            "tiered": self._stats.to_dict(),
        }
        if hasattr(self._hot, "get_stats"):
            hot_s = self._hot.get_stats()
            result["hot"] = hot_s.to_dict() if hasattr(hot_s, "to_dict") else {}
        if self._cold is not None and hasattr(self._cold, "get_stats"):
            cold_s = self._cold.get_stats()
            result["cold"] = cold_s.to_dict() if hasattr(cold_s, "to_dict") else {}
            if hasattr(self._cold, "get_stats_dict"):
                result["cold_disk"] = self._cold.get_stats_dict()
        return result
