# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import fusion_mlx.admin.stats as admin_stats


MODEL_ID = "test-model"


def _run_clear():
    return asyncio.run(admin_stats.clear_hot_cache(is_admin=True))


def _pool(models, entries=None):
    pool = MagicMock(spec=[])
    pool.get_status = MagicMock(return_value={"models": models})
    pool._entries = entries or {}
    return pool


def _loaded_entry(clear_hot_cache_mock, stream="engine-stream"):
    scheduler = SimpleNamespace(
        paged_ssd_cache_manager=SimpleNamespace(
            clear_hot_cache=clear_hot_cache_mock,
        ),
        _cache_rate_tracker=None,
        _stream=stream,
    )
    core = SimpleNamespace(scheduler=scheduler)
    return SimpleNamespace(
        engine=SimpleNamespace(
            _engine=SimpleNamespace(engine=core),
        )
    )


class TestHotCacheClear:
    def test_clears_no_model_loaded(self):
        pool = _pool(models=[])
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = _run_clear()

        assert result["total_cleared"] == 0
        assert result["status"] == "ok"

    def test_clears_loaded_model(self):
        clear_mock = MagicMock(return_value=7)
        entry = _loaded_entry(clear_mock, stream="loaded-stream")
        pool = _pool(
            models=[{"id": MODEL_ID, "loaded": True}],
            entries={MODEL_ID: entry},
        )
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = _run_clear()

        assert clear_mock.called
        assert result["total_cleared"] == 7
        assert result["status"] == "ok"

    def test_response_shape(self):
        pool = _pool(models=[])
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = _run_clear()

        assert set(result.keys()) == {"status", "total_cleared"}
        assert result["status"] == "ok"


class TestClearReachesOrphans:
    def test_clears_orphan_via_budget_with_no_model_loaded(self):
        orphan_clear = MagicMock(return_value=5)
        pool = _pool(models=[])
        pool._scheduler_config = SimpleNamespace(
            hot_cache_budget=SimpleNamespace(clear_all_owners=orphan_clear)
        )
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = _run_clear()

        assert result["total_cleared"] == 0

    def test_no_budget_is_tolerated(self):
        pool = _pool(models=[])
        pool._scheduler_config = SimpleNamespace(hot_cache_budget=None)
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = _run_clear()

        assert result["total_cleared"] == 0


class TestClearReclaimBranches:
    def test_mixed_loaded_and_orphan_no_double_count(self):
        entry = _loaded_entry(MagicMock(return_value=7), stream="mixed-stream")
        pool = _pool(
            models=[{"id": MODEL_ID, "loaded": True}],
            entries={MODEL_ID: entry},
        )
        pool._scheduler_config = SimpleNamespace(
            hot_cache_budget=SimpleNamespace(clear_all_owners=MagicMock(return_value=5))
        )
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = _run_clear()

        assert result["total_cleared"] == 7
        assert result["status"] == "ok"

    def test_two_loaded_engines(self):
        e1 = _loaded_entry(MagicMock(return_value=1), stream="stream-a")
        e2 = _loaded_entry(MagicMock(return_value=1), stream="stream-b")
        pool = _pool(
            models=[{"id": "m1", "loaded": True}, {"id": "m2", "loaded": True}],
            entries={"m1": e1, "m2": e2},
        )
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = _run_clear()

        assert result["total_cleared"] == 2
        assert result["status"] == "ok"
