# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, asdict, field
from typing import Any, Dict


@dataclass
class BaseCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_queries(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_queries
        if total == 0:
            return 0.0
        return self.hits / total

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def record_eviction(self) -> None:
        self.evictions += 1

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_queries"] = self.total_queries
        d["hit_rate"] = self.hit_rate
        return d


@dataclass
class PrefixCacheStats(BaseCacheStats):
    tokens_saved: int = 0
    partial_block_skips: int = 0
    partial_tokens_skipped: int = 0
    block_size: int = 0
    last_partial_tokens_skipped: int = 0
    last_tokens_to_next_block: int = 0
    tokens_matched_total: int = 0
    tokens_requested_total: int = 0
    _total_queries: int = field(default=0, repr=False)

    @property
    def total_queries(self) -> int:
        if self._total_queries > 0:
            return self._total_queries
        return self.hits + self.misses

    @total_queries.setter
    def total_queries(self, value: int) -> None:
        self._total_queries = value

    def reset(self) -> None:
        super().reset()
        self.tokens_saved = 0
        self.partial_block_skips = 0
        self.partial_tokens_skipped = 0
        self.last_partial_tokens_skipped = 0
        self.last_tokens_to_next_block = 0
        self.tokens_matched_total = 0
        self.tokens_requested_total = 0
        self._total_queries = 0


@dataclass
class PagedCacheStats(BaseCacheStats):
    total_blocks: int = 0
    allocated_blocks: int = 0
    free_blocks: int = 0
    shared_blocks: int = 0
    total_tokens_cached: int = 0
    cow_copies: int = 0

    @property
    def utilization(self) -> float:
        if self.total_blocks == 0:
            return 0.0
        return self.allocated_blocks / self.total_blocks

    def reset(self) -> None:
        super().reset()
        self.cow_copies = 0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["utilization"] = self.utilization
        return d


@dataclass
class VLMCacheStats(BaseCacheStats):
    tokens_saved: int = 0
    image_cache_hits: int = 0

    def record_image_hit(self) -> None:
        self.image_cache_hits += 1

    def reset(self) -> None:
        super().reset()
        self.tokens_saved = 0
        self.image_cache_hits = 0


@dataclass
class PagedSSDCacheStats(BaseCacheStats):
    saves: int = 0
    loads: int = 0
    errors: int = 0
    ssd_write_drops: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0
    configured_max_size_bytes: int = 0
    num_files: int = 0
    hot_cache_entries: int = 0
    hot_cache_size_bytes: int = 0
    hot_cache_max_bytes: int = 0
    hot_cache_hits: int = 0
    hot_cache_evictions: int = 0
    hot_cache_promotions: int = 0

    @property
    def save_rate(self) -> float:
        total = self.saves + self.errors
        if total == 0:
            return 0.0
        return self.saves / total

    def record_save(self) -> None:
        self.saves += 1

    def record_load(self) -> None:
        self.loads += 1
        self.hits += 1

    def record_error(self) -> None:
        self.errors += 1

    def reset(self) -> None:
        super().reset()
        self.saves = 0
        self.loads = 0
        self.errors = 0
        self.ssd_write_drops = 0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["save_rate"] = self.save_rate
        return d
