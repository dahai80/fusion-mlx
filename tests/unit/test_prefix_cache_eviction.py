# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache eviction counter wiring (migrated from Rapid-MLX).

Adaptations:
- vllm_mlx.memory_cache -> fusion_mlx.memory_cache
- FUSION_MLX_PREFIX_CACHE_MAX_BYTES env var is NOT supported in fusion-mlx;
  use max_memory_mb parameter directly instead.
- Scheduler.evict_prefix_cache_under_pressure does NOT exist in fusion-mlx;
  scheduler-pressure-eviction tests are skipped.
- SchedulerConfig does NOT have enable_prefix_cache / gpu_memory_utilization /
  metal_pressure_evict_fraction in fusion-mlx; those tests are skipped.
"""

from __future__ import annotations

import pytest

from fusion_mlx.memory_cache import (
    _BYTES_PER_MB,
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)

# --- compute_memory_limit via max_memory_mb ---


def test_max_memory_mb_overrides_heuristic():
    cfg = MemoryCacheConfig(max_memory_mb=500)
    assert cfg.compute_memory_limit() == 500 * _BYTES_PER_MB


def test_max_memory_mb_takes_precedence_over_high_percent():
    cfg = MemoryCacheConfig(max_memory_mb=200, max_memory_percent=0.99)
    assert cfg.compute_memory_limit() == 200 * _BYTES_PER_MB


def test_percent_heuristic_when_no_mb_override():
    cfg = MemoryCacheConfig()
    limit = cfg.compute_memory_limit()
    assert limit > 0


# --- LRU evictions counter ticks on cap pressure ---


class _FakeCacheLayer:
    class _FakeDtype:
        size = 4

    class _FakeArr:
        def __init__(self, n: int):
            self.shape = (n,)
            self.dtype = _FakeCacheLayer._FakeDtype()
            self.nbytes = n * 4

    def __init__(self, byte_size: int):
        n = max(1, byte_size // (2 * 4))
        keys = self._FakeArr(n)
        values = self._FakeArr(n)
        self.state = (keys, values)
        self.offset = n

    def is_trimmable(self) -> bool:
        return False


def _make_cache_entry(byte_size: int):
    return [_FakeCacheLayer(byte_size)]


def test_lru_evictions_total_ticks_when_cache_exceeds_cap():
    cfg = MemoryCacheConfig(max_memory_mb=8)
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)
    assert cache.get_stats()["evictions"] == 0

    per_entry_bytes = 3 * 1024 * 1024
    for i in range(5):
        tokens = list(range(i * 100, i * 100 + 64))
        cache.store(tokens, _make_cache_entry(per_entry_bytes))

    stats = cache.get_stats()
    assert (
        stats["evictions"] >= 1
    ), f"LRU-on-cap evictions did not fire; cache stat snapshot: {stats!r}"
    assert stats["current_memory_mb"] <= stats["max_memory_mb"] + 1.0


# --- pressure evictions (not available in fusion-mlx) ---


@pytest.mark.skip(
    reason="fusion-mlx Scheduler has no evict_prefix_cache_under_pressure"
)
def test_pressure_evictions_total_ticks_on_cache_self_pressure():
    pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler has no evict_prefix_cache_under_pressure"
)
def test_pressure_eviction_loop_short_circuits_when_no_trigger_configured():
    pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler has no evict_prefix_cache_under_pressure"
)
def test_pressure_eviction_max_evict_bounds_a_single_tick():
    pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler has no evict_prefix_cache_under_pressure"
)
def test_pressure_eviction_stops_when_cache_drops_below_threshold():
    pass


@pytest.mark.skip(reason="fusion-mlx has no FUSION_MLX_PREFIX_CACHE_MAX_BYTES env var")
def test_cache_self_pressure_respects_env_override(monkeypatch):
    pass


# --- Stats integration: get_stats wires through the counters ---


def test_get_stats_surfaces_evictions():
    cfg = MemoryCacheConfig(max_memory_mb=8)
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)

    for i in range(5):
        cache.store(
            list(range(i * 100, i * 100 + 64)), _make_cache_entry(3 * 1024 * 1024)
        )

    stats = cache.get_stats()
    assert "evictions" in stats
    assert stats["evictions"] >= 1


# --- R7-H7: cap admission MUST evict-LRU-until-fits, not reject-new ---


def test_r7_h7_near_full_cache_admits_fresh_inserts_via_lru_eviction():
    cfg = MemoryCacheConfig(max_memory_mb=8)
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)

    per_entry_bytes = 1 * 1024 * 1024
    for i in range(7):
        cache.store(
            list(range(i * 100, i * 100 + 64)),
            _make_cache_entry(per_entry_bytes),
        )

    preload_stats = cache.get_stats()
    preload_evictions = preload_stats["evictions"]
    assert preload_stats["entry_count"] >= 5, preload_stats

    rejected = 0
    accepted = 0
    for i in range(50):
        tokens = list(range(10_000 + i * 100, 10_000 + i * 100 + 64))
        ok = cache.store(tokens, _make_cache_entry(per_entry_bytes))
        if ok:
            accepted += 1
        else:
            rejected += 1

    final_stats = cache.get_stats()
    assert rejected == 0, (
        f"R7-H7 regression: {rejected}/50 fresh prefix stores rejected"
        f" without LRU eviction. The cap admission policy must"
        f" evict-LRU-until-fits, not reject-new. Final stats:"
        f" {final_stats!r}"
    )
    assert accepted == 50
    new_evictions = final_stats["evictions"] - preload_evictions
    assert new_evictions >= 1, (
        f"R7-H7 regression: evictions counter did not tick across the"
        f" fresh-insert burst. preload_evictions={preload_evictions},"
        f" final_evictions={final_stats['evictions']}"
    )
    assert final_stats["current_memory_mb"] <= final_stats["max_memory_mb"] + 0.5


def test_r7_h7_lru_ordering_least_recently_touched_evicted_first():
    cfg = MemoryCacheConfig(max_memory_mb=3)
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)

    tokens_a = [1, 100, 101, 102, 103]
    tokens_b = [2, 200, 201, 202, 203]
    tokens_c = [3, 300, 301, 302, 303]
    tokens_d = [4, 400, 401, 402, 403]

    per_entry_bytes = 1024 * 1024

    assert cache.store(tokens_a, _make_cache_entry(per_entry_bytes))
    assert cache.store(tokens_b, _make_cache_entry(per_entry_bytes))
    assert cache.store(tokens_c, _make_cache_entry(per_entry_bytes))

    cache.fetch(tokens_a)

    assert cache.store(tokens_d, _make_cache_entry(per_entry_bytes))

    present_keys = set(cache._entries.keys())
    assert (
        tuple(tokens_a) in present_keys
    ), f"LRU regression: just-fetched entry was evicted; present={present_keys}"
    assert tuple(tokens_c) in present_keys, (
        f"LRU regression: middle-aged entry was evicted instead of LRU;"
        f" present={present_keys}"
    )
    assert (
        tuple(tokens_d) in present_keys
    ), f"newly inserted entry missing from cache; present={present_keys}"
    assert tuple(tokens_b) not in present_keys, (
        f"R7-H7 LRU ordering regression: the least-recently-touched"
        f" entry was NOT evicted. Present keys: {present_keys}."
        f" Expected B (tokens_b={tokens_b}) to be the eviction victim."
    )
