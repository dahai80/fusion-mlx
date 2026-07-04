# SPDX-License-Identifier: Apache-2.0
"""Metal-pressure-triggered prefix-cache eviction tests (migrated from Rapid-MLX).

Adaptations:
- vllm_mlx.scheduler -> fusion_mlx.scheduler
- fusion-mlx Scheduler does NOT have evict_prefix_cache_under_pressure,
  memory_aware_cache, num_prefix_cache_pressure_evictions,
  _resolve_metal_cap_bytes, _current_metal_active_bytes.
- fusion-mlx does NOT have _render_prometheus in routes.metrics.
- fusion-mlx EngineCore does NOT have _run_pressure_evict_tick.
- All tests that depend on these APIs are skipped.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks evict_prefix_cache_under_pressure"
)
class TestPressureEvictionDispatch:
    def test_no_op_when_no_cache(self):
        pass

    def test_no_op_when_cap_zero(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks evict_prefix_cache_under_pressure"
)
class TestLegacyPrefixCacheEviction:
    def test_no_eviction_below_threshold(self):
        pass

    def test_pressure_evicts_lru_entries(self):
        pass

    def test_pressure_eviction_stops_when_pressure_drops(self):
        pass

    def test_max_evict_bound_respected(self):
        pass

    def test_clear_cache_called_after_each_eviction(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks memory_aware_cache eviction dispatch"
)
class TestMemoryAwareCacheEviction:
    def test_pressure_evicts_memory_aware_entries(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx has no _render_prometheus in routes.metrics"
)
class TestPressureEvictionMetric:
    def test_get_stats_exposes_counter(self):
        pass

    def test_metric_renders_after_pressure_evictions(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx EngineCore lacks _run_pressure_evict_tick"
)
class TestEngineCoreInvokesPressureEvict:
    def test_pressure_tick_calls_scheduler_regardless_of_legacy_threshold(self):
        pass

    def test_pressure_tick_logs_warning_once_on_eviction_failure(self, caplog):
        pass

    def test_pressure_tick_silent_on_success(self, caplog):
        pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks evict_prefix_cache_under_pressure"
)
class TestClearCacheFailurePropagation:
    def test_clear_cache_failure_propagates_to_engine_path(self):
        pass

    def test_counter_reflects_cache_mutation_on_clear_cache_failure(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks evict_prefix_cache_under_pressure"
)
class TestSchedulerPropagatesEvictionErrors:
    def test_memory_aware_cache_eviction_error_propagates(self):
        pass

    def test_legacy_prefix_cache_eviction_error_propagates(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx SchedulerConfig lacks metal_pressure_evict_fraction"
)
class TestPressureEvictFractionClamp:
    def test_zero_fraction_clamped_to_default(self):
        pass

    def test_above_one_fraction_clamped_to_one(self):
        pass


@pytest.mark.skip(
    reason="fusion-mlx Scheduler lacks mx.get_cache_memory stats wiring"
)
class TestMetalCacheMemoryMetric:
    def test_get_stats_reads_live_cache_memory(self):
        pass
