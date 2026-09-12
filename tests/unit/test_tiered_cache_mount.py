# SPDX-License-Identifier: Apache-2.0
"""D2.1: TieredCache main-path mount tests.

Verifies the tiered cache coordinator is wired into the factory stack
and the scheduler activation helper, with env-gate default ON.
"""

import os
from unittest.mock import MagicMock


class TestFactoryTieredMount:
    def test_stack_returns_tiered_cache_when_hot_present(self):
        from fusion_mlx.cache.factory import CacheConfig, CacheFactory

        cfg = CacheConfig(block_size=64, max_num_blocks=8, initial_blocks=4)
        stack = CacheFactory.create_full_cache_stack(cfg)
        assert stack["paged_cache"] is not None
        assert stack["tiered_cache"] is not None

    def test_stack_tiered_cache_none_when_disabled(self, monkeypatch):
        from fusion_mlx.cache.factory import CacheConfig, CacheFactory

        monkeypatch.setenv("FUSION_MLX_TIERED_CACHE", "0")
        cfg = CacheConfig(block_size=64, max_num_blocks=8, initial_blocks=4)
        stack = CacheFactory.create_full_cache_stack(cfg)
        assert stack["tiered_cache"] is None

    def test_tiered_cache_wraps_hot_and_cold(self):
        from fusion_mlx.cache.factory import CacheConfig, CacheFactory

        cfg = CacheConfig(
            block_size=64,
            max_num_blocks=8,
            initial_blocks=4,
            paged_ssd_cache_dir=None,
        )
        stack = CacheFactory.create_full_cache_stack(cfg)
        tc = stack["tiered_cache"]
        assert tc is not None
        assert tc.hot is stack["paged_cache"]
        # cold is None when no SSD dir (pure-memory mode)
        assert tc.cold is None or tc.cold is stack["paged_ssd_cache"]


class TestSchedulerActivation:
    def _make_sched(self):
        sched = MagicMock()
        sched.paged_cache_manager = MagicMock()
        sched.paged_ssd_cache_manager = MagicMock()
        return sched

    def test_activation_mounts_tiered_manager(self, monkeypatch):
        from fusion_mlx.scheduler.sched_misc import _activate_tiered_cache_manager

        monkeypatch.setenv("FUSION_MLX_TIERED_CACHE", "1")
        sched = self._make_sched()
        _activate_tiered_cache_manager(sched)
        assert sched._tiered_cache_manager is not None

    def test_activation_disabled_when_env_off(self, monkeypatch):
        from fusion_mlx.scheduler.sched_misc import _activate_tiered_cache_manager

        monkeypatch.setenv("FUSION_MLX_TIERED_CACHE", "0")
        sched = self._make_sched()
        _activate_tiered_cache_manager(sched)
        assert sched._tiered_cache_manager is None

    def test_activation_skips_when_no_paged_cache(self, monkeypatch):
        from fusion_mlx.scheduler.sched_misc import _activate_tiered_cache_manager

        monkeypatch.setenv("FUSION_MLX_TIERED_CACHE", "1")
        sched = self._make_sched()
        sched.paged_cache_manager = None
        _activate_tiered_cache_manager(sched)
        assert sched._tiered_cache_manager is None


class TestRuntimeConfigExposure:
    def test_runtime_config_reports_tiered_enabled(self):
        from fusion_mlx.routes_internal.runtime_config import runtime_config

        os.environ.pop("FUSION_MLX_TIERED_CACHE", None)
        snap = __import__("asyncio").run(runtime_config())
        cache = snap.get("cache", {})
        assert "tiered_cache_enabled" in cache
        assert cache["tiered_cache_enabled"] is True

    def test_runtime_config_reports_tiered_disabled(self, monkeypatch):
        from fusion_mlx.routes_internal.runtime_config import runtime_config

        monkeypatch.setenv("FUSION_MLX_TIERED_CACHE", "0")
        snap = __import__("asyncio").run(runtime_config())
        cache = snap.get("cache", {})
        assert cache["tiered_cache_enabled"] is False
