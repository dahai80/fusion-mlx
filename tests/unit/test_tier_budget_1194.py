# SPDX-License-Identifier: Apache-2.0
"""#1194: VramAllocator tier-budget enforcement on model-load paths."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_mlx.config import MemoryConfig
from fusion_mlx.exceptions import ModelTooLargeError
from fusion_mlx.pool.engine_pool import EnginePool
from fusion_mlx.server_metrics import get_server_metrics


def _make_pool(ceiling: int | None = None) -> EnginePool:
    pool = EnginePool()
    if ceiling is None or ceiling <= 0:
        pool._get_final_ceiling = lambda: 0
    else:
        pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


def _small_model_dir(tmp_path):
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_a / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path


@pytest.fixture(autouse=True)
def restore_tier_budgets():
    # _tier_budgets_bytes is a class-level dict; restore it after each test
    # so budgets never leak into other test files in the same pytest process.
    saved = dict(EnginePool._tier_budgets_bytes)
    yield
    EnginePool._tier_budgets_bytes.clear()
    EnginePool._tier_budgets_bytes.update(saved)


def _mock_engine():
    eng = MagicMock(spec=["start", "stop", "safe_evict", "has_active_requests"])
    eng.start = AsyncMock()
    eng.stop = AsyncMock()
    eng.safe_evict = AsyncMock()
    eng.has_active_requests = MagicMock(return_value=False)
    return eng


def _tier_budget(pool: EnginePool, p0: int, p1: int) -> None:
    pool._tier_budgets_bytes = {"P0": p0, "P1": p1}


GB = 1024**3


class TestLoadEngineTierBudget:
    @pytest.mark.asyncio
    async def test_load_engine_rejects_over_tier_budget(self, tmp_path):
        _small_model_dir(tmp_path)
        pool = _make_pool(ceiling=1024 * GB)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("model-a")
        entry.estimated_size = 100 * GB
        pool.set_tier_budget("P0", 50 * GB)
        mock_engine = _mock_engine()
        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine", return_value=mock_engine
        ):
            with pytest.raises(ModelTooLargeError):
                await pool._load_engine("model-a")
        assert entry.engine is None
        assert entry.is_loading is False
        sm = get_server_metrics()
        assert sm.tier_budget_rejections_total >= 1

    @pytest.mark.asyncio
    async def test_load_engine_within_budget_loads(self, tmp_path):
        _small_model_dir(tmp_path)
        pool = _make_pool(ceiling=1024 * GB)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("model-a")
        entry.estimated_size = 100 * GB
        pool.set_tier_budget("P0", 200 * GB)
        mock_engine = _mock_engine()
        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine", return_value=mock_engine
        ):
            await pool._load_engine("model-a")
        assert entry.engine is mock_engine

    @pytest.mark.asyncio
    async def test_load_engine_unbounded_default(self, tmp_path):
        _small_model_dir(tmp_path)
        pool = _make_pool(ceiling=1024 * GB)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("model-a")
        entry.estimated_size = 100 * GB
        mock_engine = _mock_engine()
        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine", return_value=mock_engine
        ):
            await pool._load_engine("model-a")
        assert entry.engine is mock_engine

    @pytest.mark.asyncio
    async def test_pinned_bypass_on_load_path(self, tmp_path):
        _small_model_dir(tmp_path)
        pool = _make_pool(ceiling=1024 * GB)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("model-a")
        entry.estimated_size = 100 * GB
        entry.priority_tier = "P0_NEVER_EVICT"
        pool.set_tier_budget("P0", 1 * GB)
        mock_engine = _mock_engine()
        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine", return_value=mock_engine
        ):
            await pool._load_engine("model-a")
        assert entry.engine is mock_engine

    @pytest.mark.asyncio
    async def test_rejection_metric_increments(self, tmp_path):
        _small_model_dir(tmp_path)
        pool = _make_pool(ceiling=1024 * GB)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("model-a")
        entry.estimated_size = 100 * GB
        pool.set_tier_budget("P0", 50 * GB)
        sm = get_server_metrics()
        before_total = sm.tier_budget_rejections_total
        before_p0 = sm.tier_budget_rejections_by_tier.get("P0", 0)
        mock_engine = _mock_engine()
        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine", return_value=mock_engine
        ):
            with pytest.raises(ModelTooLargeError):
                await pool._load_engine("model-a")
        assert sm.tier_budget_rejections_total == before_total + 1
        assert sm.tier_budget_rejections_by_tier.get("P0") == before_p0 + 1

    @pytest.mark.asyncio
    async def test_register_engine_estimated_size_accounted(self):
        pool = EnginePool()
        pool._tier_budgets_bytes = {"P0": 0, "P1": 0}
        mock_engine = _mock_engine()
        size = 10 * GB
        before = pool._current_model_memory
        pool.register_engine("a", mock_engine, model_path="/tmp/x", estimated_size=size)
        assert pool.tier_resident_bytes()["P0"] >= size
        assert pool._current_model_memory == before + size
        before2 = pool._current_model_memory
        # Alias registration (second name for the same engine) must not
        # double-count: _load_single_model omits the size on the alias call.
        pool.register_engine("alias-b", mock_engine, model_path="/tmp/x")
        assert pool._current_model_memory == before2
        assert pool.tier_resident_bytes()["P0"] >= size

    @pytest.mark.asyncio
    async def test_set_tier_budget_wiring_from_env(self):
        from fusion_mlx.server import _resolve_tier_budget_gb

        assert _resolve_tier_budget_gb(None, 5.0) == 5.0
        assert _resolve_tier_budget_gb("20", 5.0) == 20.0
        assert _resolve_tier_budget_gb("abc", 5.0) == 5.0
        assert _resolve_tier_budget_gb("", 5.0) == 5.0
        assert _resolve_tier_budget_gb("-5", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_single_model_over_budget_rejected(self, tmp_path):
        fake_dir = tmp_path / "fake"
        fake_dir.mkdir()
        (fake_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        pool = _make_pool(ceiling=1024 * GB)
        _tier_budget(pool, 1 * GB, 0)
        import fusion_mlx.server as server_mod

        srv = server_mod.Server.__new__(server_mod.Server)
        # #1301: _load_single_model reads self.config.scheduler (single
        # source of truth) — a bare __new__ stub needs the config.
        srv.config = server_mod.ServerConfig()
        srv.pool = pool
        srv.engine_cores = {}
        sm = get_server_metrics()
        before_total = sm.tier_budget_rejections_total
        before_p0 = sm.tier_budget_rejections_by_tier.get("P0", 0)
        with (
            patch("fusion_mlx.engines.batched.BatchedEngine"),
            patch(
                "fusion_mlx.pool.model_discovery.estimate_model_size",
                return_value=100 * GB,
            ),
        ):
            with pytest.raises(ModelTooLargeError):
                await srv._load_single_model(
                    {"model_path": str(fake_dir), "served_model_name": "m"}
                )
        assert "m" not in pool._entries
        assert sm.tier_budget_rejections_total == before_total + 1
        assert sm.tier_budget_rejections_by_tier.get("P0") == before_p0 + 1


def test_memory_config_clamp():
    cfg = MemoryConfig(tier_budget_p0_gb=-5, tier_budget_p1_gb=-1)
    assert cfg.tier_budget_p0_gb == 0.0
    assert cfg.tier_budget_p1_gb == 0.0
    cfg2 = MemoryConfig(tier_budget_p0_gb=12.5, tier_budget_p1_gb=3.0)
    assert cfg2.tier_budget_p0_gb == 12.5
    assert cfg2.tier_budget_p1_gb == 3.0
