# SPDX-License-Identifier: Apache-2.0
# #779: stale _current_model_memory accumulator must not block admission when
# no engine is loaded and live gauges show memory free.
#
# Symptom from the issue: a 5 GB model is rejected as "insufficient memory"
# while the error payload reports available_memory_mb=122995 (122 GB). That
# field is actually the admission `current` gauge, not system free memory.
# When all engines are unloaded, a drifted accumulator (decrement skipped on
# a cancelled/aborted unload) inflates `current` past the ceiling, so a model
# that genuinely fits is rejected. The fix reconciles the accumulator to the
# live gauges when no engine is loaded (gated on has_loaded==False so the
# #1623 under-reporting case — a loaded engine whose live gauge reads low —
# still trusts the accumulator).
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_mlx.pool.engine_pool import EnginePool


def _make_pool(ceiling: int | None = None) -> EnginePool:
    pool = EnginePool()
    if ceiling is None or ceiling <= 0:
        pool._get_final_ceiling = lambda: 0
    else:
        pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


@pytest.fixture
def small_model_dir(tmp_path):
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_a / "model.safetensors").write_bytes(b"0" * 1024)

    model_b = tmp_path / "model-b"
    model_b.mkdir()
    (model_b / "config.json").write_text(json.dumps({"model_type": "qwen"}))
    (model_b / "model.safetensors").write_bytes(b"0" * 2048)
    return tmp_path


def _mock_engine():
    eng = MagicMock(spec=["start", "stop", "safe_evict", "has_active_requests"])
    eng.start = AsyncMock()
    eng.stop = AsyncMock()
    eng.safe_evict = AsyncMock()
    eng.has_active_requests = MagicMock(return_value=False)
    return eng


class TestStaleAccumulatorNoEngineLoaded:
    # #779: when no engine is loaded, a drifted _current_model_memory must not
    # dominate the admission `current` gauge. Live gauges reflect physical
    # reality; the accumulator is a bookkeeping estimate that can drift when a
    # decrement is skipped (cancelled settle / aborted unload before the
    # counter update).

    @pytest.mark.asyncio
    async def test_loads_when_accumulator_drifted_no_engine_loaded(
        self, small_model_dir, monkeypatch
    ):
        # ceiling 112 GB; model ~1KB fits trivially.
        ceiling = 112 * 1024**3
        pool = _make_pool(ceiling=ceiling)
        pool.discover_models(str(small_model_dir))

        # live gauges: machine is actually idle (no Metal cache resident).
        monkeypatch.setenv("FUSION_MLX_ADMISSION_KV_HEADROOM_GB", "0")
        monkeypatch.setattr(
            "fusion_mlx.pool.engine_pool.get_phys_footprint",
            lambda: int(0.5 * 1024**3),
        )
        monkeypatch.setattr(
            "fusion_mlx.pool.engine_pool.mx.get_active_memory", lambda: 0
        )

        # Simulate a drifted accumulator: a prior large model (122 GB) was
        # unloaded but the decrement was skipped (cancelled settle / aborted
        # unload). No engine is loaded now.
        pool._current_model_memory = 122 * 1024**3
        assert all(e.engine is None for e in pool._entries.values())

        mock_engine = _mock_engine()
        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine",
            return_value=mock_engine,
        ):
            engine = await pool.get_engine("model-a")

        assert engine is mock_engine
        assert pool._entries["model-a"].engine is mock_engine
        # accumulator reconciled toward the live gauge (no engine loaded).
        assert pool._current_model_memory < 122 * 1024**3

    @pytest.mark.asyncio
    async def test_rejects_when_accumulator_drifted_engine_still_loaded(
        self, small_model_dir, monkeypatch
    ):
        # #1623 guard: when an engine IS loaded, the accumulator is trusted
        # (live gauges under-report a loaded model). A drifted-high
        # accumulator must still block a second load that would over-commit.
        ceiling = 2500
        pool = _make_pool(ceiling=ceiling)
        pool.discover_models(str(small_model_dir))
        monkeypatch.setenv("FUSION_MLX_ADMISSION_KV_HEADROOM_GB", "0")
        monkeypatch.setattr("fusion_mlx.pool.engine_pool.get_phys_footprint", lambda: 0)
        monkeypatch.setattr(
            "fusion_mlx.pool.engine_pool.mx.get_active_memory", lambda: 0
        )

        # Load model-a first (real engine loaded).
        mock_engine_a = _mock_engine()
        mock_engine_b = _mock_engine()
        mock_engine_b.has_active_requests.return_value = False

        def create_engine(*args, **kwargs):
            name = str(kwargs.get("model_name", args[0] if args else ""))
            if "model-a" in name:
                return mock_engine_a
            return mock_engine_b

        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine",
            side_effect=create_engine,
        ):
            await pool.get_engine("model-a")
            assert pool.loaded_model_count == 1

            # model-a loaded, accumulator reflects it. model-b fits alone but
            # the pair exceeds the ceiling; with an engine loaded the
            # accumulator is trusted, so model-a is evicted (NOT silently
            # co-located because live gauges read 0).
            await pool.get_engine("model-b")

        assert pool._entries["model-a"].engine is None
        assert pool._entries["model-b"].engine is mock_engine_b
        assert pool.loaded_model_count == 1
