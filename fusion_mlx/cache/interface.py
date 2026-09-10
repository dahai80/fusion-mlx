# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

from .stats import BaseCacheStats


class CacheManager(ABC):
    """Abstract interface for cache implementations that adopt it.

    R-P2-1: Adoption is OPTIONAL. PagedCacheManager, BlockAwarePrefixCache,
    and TieredCacheManager implement this ABC. RadixPrefixCache,
    DiffusionRadixCache, ResponseCache, MLLMPromptCacheManager, and
    PagedSSDCacheManager do NOT — they predate the ABC or have incompatible
    semantics. Do not assume all caches subclass CacheManager; check with
    isinstance before calling ABC methods on an arbitrary cache.
    """

    @abstractmethod
    def fetch(self, key: Any) -> tuple[Any | None, bool]:
        pass

    @abstractmethod
    def store(self, key: Any, value: Any) -> bool:
        pass

    @abstractmethod
    def evict(self, key: Any) -> bool:
        pass

    @abstractmethod
    def clear(self) -> int:
        pass

    @abstractmethod
    def get_stats(self) -> BaseCacheStats:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def max_size(self) -> int:
        pass

    @property
    def utilization(self) -> float:
        if self.max_size == 0:
            return 0.0
        return self.size / self.max_size
