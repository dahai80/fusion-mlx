# SPDX-License-Identifier: Apache-2.0
"""#1080: radix prefix cache hit rate / latency percentile + MoE shared
cache stats surfaced in /admin/api/stats.

Covers:
- _percentile nearest-rank helper (empty / single / known dataset).
- BlockAwarePrefixCache.get_stats_dict() carries latency_p50/p99/avg/samples
  and fetch_cache records a sample on every non-empty lookup.
- _build_cache_observability() aggregates per-model prefix stats + MoE
  shared cache flat stats, and returns the two sections.
- get_server_stats() includes prefix_cache + moe_shared_cache keys.
"""

import asyncio
import collections
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import fusion_mlx.admin.stats as admin_stats
from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache, _percentile

BLOCK_SIZE = 4
MODEL_ID = "test-model-1080"


class TestPercentileHelper:
    def test_empty_deque_returns_zero(self):
        d = collections.deque(maxlen=8)
        assert _percentile(d, 50) == 0.0
        assert _percentile(d, 99) == 0.0

    def test_single_value(self):
        d = collections.deque([1.5], maxlen=8)
        assert _percentile(d, 50) == 1.5
        assert _percentile(d, 99) == 1.5

    def test_known_dataset(self):
        # 1..10 in ms
        d = collections.deque(range(1, 11), maxlen=16)
        # nearest-rank p50 of 10 sorted values: ceil(0.5*10)=5 -> idx 4 -> 5
        assert _percentile(d, 50) == 5.0
        # p99: ceil(0.99*10)=10 -> idx 9 -> 10
        assert _percentile(d, 99) == 10.0

    def test_bounded_deque_drops_oldest(self):
        d = collections.deque(maxlen=3)
        for v in range(10):
            d.append(float(v))
        assert len(d) == 3
        # only 7,8,9 remain
        assert _percentile(d, 99) == 9.0


def _make_prefix_cache(block_size=BLOCK_SIZE):
    model = SimpleNamespace()
    paged = MagicMock()
    paged.block_size = block_size
    paged.model_name = "test-model-1080"
    paged.register_block_freed_callback = MagicMock()
    paged.get_memory_usage = MagicMock(return_value={})
    paged.find_shared_prefix = MagicMock(return_value=([], [1, 2, 3, 4]))
    paged.create_block_table = MagicMock(return_value=MagicMock())
    paged.allocated_blocks = {}
    cache = BlockAwarePrefixCache(model, paged)
    return cache, paged


class TestPrefixCacheLatencyStats:
    def test_get_stats_dict_has_latency_fields(self):
        cache, _ = _make_prefix_cache()
        stats = cache.get_stats_dict()
        assert "latency_p50_ms" in stats
        assert "latency_p99_ms" in stats
        assert "latency_avg_ms" in stats
        assert "latency_samples" in stats
        assert stats["latency_samples"] == 0
        assert stats["latency_p50_ms"] == 0.0
        assert stats["latency_p99_ms"] == 0.0

    def test_fetch_cache_records_latency_sample(self):
        cache, _ = _make_prefix_cache()
        # Empty tokens: no sample recorded (early return).
        cache.fetch_cache("req-empty", [])
        assert len(cache._lookup_latencies_ms) == 0
        # Non-empty: miss path records one sample.
        cache.fetch_cache("req-1", [1, 2, 3, 4])
        assert len(cache._lookup_latencies_ms) == 1
        assert cache._misses == 1
        stats = cache.get_stats_dict()
        assert stats["latency_samples"] == 1
        assert stats["latency_p50_ms"] >= 0.0
        assert stats["latency_p99_ms"] >= 0.0

    def test_multiple_lookouts_populate_percentiles(self):
        cache, _ = _make_prefix_cache()
        for i in range(20):
            cache.fetch_cache(f"req-{i}", [1, 2, 3, 4])
        stats = cache.get_stats_dict()
        assert stats["latency_samples"] == 20
        assert stats["hits"] == 0
        assert stats["misses"] == 20
        # p99 >= p50 for any dataset
        assert stats["latency_p99_ms"] >= stats["latency_p50_ms"]

    def test_reset_stats_clears_latencies(self):
        cache, _ = _make_prefix_cache()
        cache.fetch_cache("req-1", [1, 2, 3, 4])
        assert len(cache._lookup_latencies_ms) == 1
        cache.reset_stats()
        assert len(cache._lookup_latencies_ms) == 0
        stats = cache.get_stats_dict()
        assert stats["latency_samples"] == 0


def _make_loaded_entry(prefix_stats_dict):
    """Build an engine entry whose scheduler.block_aware_cache returns
    the given stats dict from get_stats_dict()."""
    block_aware = MagicMock()
    block_aware.get_stats_dict = MagicMock(return_value=prefix_stats_dict)
    scheduler = SimpleNamespace(block_aware_cache=block_aware)
    engine = SimpleNamespace(
        _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
    )
    return SimpleNamespace(engine=engine)


def _pool_with_loaded(entries):
    """entries: dict model_id -> entry. Returns a mock pool whose
    get_status reports each as loaded."""
    pool = MagicMock()
    pool._entries = entries
    pool.get_status = MagicMock(
        return_value={"models": [{"id": mid, "loaded": True} for mid in entries]}
    )
    return pool


class TestBuildCacheObservability:
    def test_no_engine_pool_returns_defaults(self):
        with patch.object(admin_stats, "_get_engine_pool", return_value=None):
            result = admin_stats._build_cache_observability()
        assert "prefix_cache" in result
        assert "moe_shared_cache" in result
        assert result["prefix_cache"]["hits"] == 0
        assert result["prefix_cache"]["latency_samples"] == 0
        assert result["moe_shared_cache"]["enabled"] is False

    def test_aggregates_per_model_prefix_stats(self):
        stats_a = {
            "hits": 10,
            "misses": 5,
            "hit_rate": 0.6667,
            "tokens_saved": 1000,
            "block_size": 4,
            "active_requests": 2,
            "pinned_blocks": 1,
            "latency_p50_ms": 0.5,
            "latency_p99_ms": 2.0,
            "latency_avg_ms": 0.7,
            "latency_samples": 15,
            "hit_buckets": {
                "<1k": {"hits": 8, "misses": 3, "hit_rate": 0.7273},
                "1k-8k": {"hits": 2, "misses": 2, "hit_rate": 0.5},
            },
        }
        stats_b = {
            "hits": 20,
            "misses": 10,
            "hit_rate": 0.6667,
            "tokens_saved": 2000,
            "block_size": 4,
            "active_requests": 3,
            "pinned_blocks": 2,
            "latency_p50_ms": 0.8,
            "latency_p99_ms": 3.5,
            "latency_avg_ms": 1.1,
            "latency_samples": 30,
            "hit_buckets": {
                "<1k": {"hits": 15, "misses": 5, "hit_rate": 0.75},
            },
        }
        entries = {
            "model-a": _make_loaded_entry(stats_a),
            "model-b": _make_loaded_entry(stats_b),
        }
        pool = _pool_with_loaded(entries)
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = admin_stats._build_cache_observability()
        pc = result["prefix_cache"]
        assert pc["hits"] == 30
        assert pc["misses"] == 15
        assert pc["tokens_saved"] == 3000
        assert pc["active_requests"] == 5
        assert pc["pinned_blocks"] == 3
        # aggregate hit_rate = 30/45
        assert pc["hit_rate"] == pytest.approx(30 / 45, rel=1e-3)
        # max p99 across models = 3.5
        assert pc["latency_p99_ms"] == 3.5
        assert pc["latency_p50_ms"] == 0.8
        assert pc["latency_samples"] == 45
        # merged buckets
        assert pc["hit_buckets"]["<1k"]["hits"] == 23
        assert pc["hit_buckets"]["<1k"]["misses"] == 8
        assert pc["hit_buckets"]["1k-8k"]["hits"] == 2
        assert len(pc["models"]) == 2

    def test_model_filter_excludes_other_models(self):
        stats_a = {
            "hits": 10,
            "misses": 0,
            "tokens_saved": 100,
            "block_size": 4,
            "active_requests": 0,
            "pinned_blocks": 0,
            "latency_p50_ms": 0.1,
            "latency_p99_ms": 0.2,
            "latency_avg_ms": 0.15,
            "latency_samples": 10,
            "hit_buckets": {},
        }
        entries = {"model-a": _make_loaded_entry(stats_a)}
        pool = _pool_with_loaded(entries)
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = admin_stats._build_cache_observability(model_filter="model-a")
        assert len(result["prefix_cache"]["models"]) == 1
        assert result["prefix_cache"]["models"][0]["id"] == "model-a"

    def test_moe_shared_cache_section_present(self):
        entries = {}
        pool = _pool_with_loaded(entries)
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = admin_stats._build_cache_observability()
        mc = result["moe_shared_cache"]
        assert "enabled" in mc
        assert "requests" in mc
        assert "hits" in mc
        assert "misses" in mc
        assert "hit_rate" in mc
        assert "layers_tracked" in mc

    def test_entry_without_scheduler_skipped(self):
        # engine._engine is None -> scheduler unreachable -> skipped.
        engine = SimpleNamespace(_engine=None)
        pool = _pool_with_loaded({"bad-model": SimpleNamespace(engine=engine)})
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = admin_stats._build_cache_observability()
        assert result["prefix_cache"]["hits"] == 0
        assert result["prefix_cache"]["models"] == []


class TestGetServerStatsKeys:
    def test_stats_response_includes_cache_sections(self):
        with patch.object(admin_stats, "_get_engine_pool", return_value=None):
            with patch.object(
                admin_stats, "_get_rich_global_settings", return_value=None
            ):
                with patch.object(
                    admin_stats, "_get_global_settings", return_value=None
                ):
                    with patch.object(admin_stats, "_get_engine_info", return_value={}):
                        with patch.object(
                            admin_stats,
                            "_build_active_models_data",
                            return_value={"models": []},
                        ):
                            with patch.object(
                                admin_stats,
                                "_build_runtime_cache_observability",
                                return_value={"models": []},
                            ):
                                result = asyncio.run(
                                    admin_stats.get_server_stats(
                                        model="",
                                        scope="session",
                                        is_admin=True,
                                    )
                                )
        assert "prefix_cache" in result
        assert "moe_shared_cache" in result
        assert result["prefix_cache"]["hits"] == 0
        assert result["moe_shared_cache"]["enabled"] is False
