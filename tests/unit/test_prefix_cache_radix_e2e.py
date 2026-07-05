# SPDX-License-Identifier: Apache-2.0
"""Integration tests for radix-tree prefix-cache index (migrated from Rapid-MLX).

NOTE: fusion-mlx MemoryAwarePrefixCache does not currently support the
radix_index constructor parameter. All radix-coupling tests are skipped.
The fetch-parity and shared-system-prompt tests that don't depend on radix
are preserved with the hash-only (no-radix) path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fusion_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class _FakeCacheLayer:

    def __init__(self, payload_size: int = 1024):
        self.offset = 0
        self._payload = b"\x00" * payload_size

    @property
    def state(self):
        return self._payload

    @property
    def meta_state(self):
        return (str(self.offset),)

    def trim(self, n: int) -> None:
        self.offset = max(0, self.offset - n)

    def is_trimmable(self) -> bool:
        return True


def _make_cache(
    max_memory_mb: int = 64, with_radix: bool = True
) -> MemoryAwarePrefixCache:
    model = MagicMock()
    config = MemoryCacheConfig(max_memory_mb=max_memory_mb, max_entries=1000)
    return MemoryAwarePrefixCache(model=model, config=config)


def _cache_payload(n_layers: int = 2):
    return [_FakeCacheLayer() for _ in range(n_layers)]


class TestRadixCacheCoupling:
    """Radix-coupling tests — skipped (no radix_index support in fusion-mlx)."""

    @pytest.mark.skip(
        reason="fusion-mlx MemoryAwarePrefixCache has no radix_index param"
    )
    def test_store_inserts_into_radix(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx MemoryAwarePrefixCache has no radix_index param"
    )
    def test_remove_drops_from_radix(self):
        pass

    @pytest.mark.skip(
        reason="fusion-mlx MemoryAwarePrefixCache has no radix_index param"
    )
    def test_clear_resets_radix(self):
        pass


class TestRadixFetchParity:
    """Fetch-parity tests — run without radix (hash-only path)."""

    def test_exact_match_hits(self):
        cache = _make_cache()
        cache.store([1, 2, 3, 4], _cache_payload())
        kv, remaining = cache.fetch([1, 2, 3, 4])
        assert kv is not None
        assert remaining == []
        assert cache._last_match_type == "exact"

    def test_prefix_match_returns_same_remaining(self):
        cache = _make_cache()
        cache.store([1, 2, 3], _cache_payload())
        kv, remaining = cache.fetch([1, 2, 3, 4, 5])
        assert kv is not None
        assert remaining == [4, 5]
        assert cache._last_match_type == "prefix"

    def test_miss_returns_full_remaining(self):
        cache = _make_cache()
        kv, remaining = cache.fetch([1, 2, 3])
        assert kv is None
        assert remaining == [1, 2, 3]

    def test_longest_prefix_wins(self):
        cache = _make_cache()
        cache.store([1, 2, 3], _cache_payload())
        cache.store([1, 2, 3, 4, 5], _cache_payload())
        kv, remaining = cache.fetch([1, 2, 3, 4, 5, 6])
        assert kv is not None
        assert remaining == [6]


class TestRadixSharedSystemPromptWorkload:
    """Shared-system-prompt tests — run without radix."""

    def test_n_tenants_shared_preamble(self):
        cache = _make_cache(max_memory_mb=128)
        preamble = list(range(1, 201))
        N = 10
        for tid in range(N):
            suffix = [10_000 + tid, 20_000 + tid, 30_000 + tid]
            cache.store(preamble + suffix, _cache_payload())
        assert len(cache) == N

    def test_new_tenant_with_shared_preamble_is_cache_hit(self):
        cache = _make_cache(max_memory_mb=128)
        preamble = list(range(1, 201))
        cache.store(preamble + [10_001, 20_001], _cache_payload())
        kv, remaining = cache.fetch(preamble + [99_999, 88_888, 77_777])
        assert kv is not None
        assert len(remaining) <= 3


class TestRadixStatsExposure:
    """Radix stats tests — skipped (no radix support)."""

    @pytest.mark.skip(reason="fusion-mlx MemoryAwarePrefixCache has no radix stats")
    def test_get_stats_includes_radix_subdict(self):
        pass

    @pytest.mark.skip(reason="fusion-mlx MemoryAwarePrefixCache has no radix stats")
    def test_get_stats_omits_radix_when_hash_mode(self):
        pass


class TestRadixSurvivesLruEviction:
    """LRU eviction consistency — run without radix."""

    def test_lru_eviction_consistent(self):
        cache = _make_cache(max_memory_mb=1)
        for i in range(50):
            tokens = [i * 100 + j for j in range(50)]
            cache.store(tokens, _cache_payload(n_layers=4))
        assert len(cache) > 0


class TestRadixPersistenceWithCache:
    """Persistence tests — skipped (no radix support)."""

    @pytest.mark.skip(reason="fusion-mlx MemoryAwarePrefixCache has no radix save/load")
    def test_save_load_roundtrip_through_cache(self, tmp_path):
        pass
