# SPDX-License-Identifier: Apache-2.0
# LoRA adapter wiring (Phase B).
# Slice 1 (boot-time): --lora-path threads from server.load_model into
# BatchedEngine and reaches mlx_lm.load(adapter_path=...).
# Slice 2 (multi-model): per-model lora_path via ModelSettings threads
# through engine_pool's BatchedEngine construction.
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mlx.utils")  # real mlx runtime; Linux CI ships a flat mlx

from fusion_mlx.engines.batched import BatchedEngine
from fusion_mlx.model_profiles import MODEL_SPECIFIC_PROFILE_FIELDS
from fusion_mlx.model_settings import ModelSettings, ModelSettingsManager
from fusion_mlx.pool.engine_pool import EnginePool


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestBatchedEngineLoraPath:
    def test_stores_lora_path(self):
        eng = BatchedEngine(model_name="dummy/repo", lora_path="/adapters/x")
        assert eng._lora_path == "/adapters/x"

    def test_default_lora_path_none(self):
        eng = BatchedEngine(model_name="dummy/repo")
        assert eng._lora_path is None

    def test_load_receives_adapter_path(self):
        eng = BatchedEngine(model_name="dummy/repo", lora_path="/adapters/x")
        captured = {}

        def fake_load(name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            model = MagicMock(name="model")
            model.parameters.return_value = {}
            return model, MagicMock(name="tokenizer")

        fake_core = MagicMock()
        fake_core.engine.start = AsyncMock()

        with (
            patch("mlx_lm.load", side_effect=fake_load),
            patch("fusion_mlx.engines.batched.AsyncEngineCore", return_value=fake_core),
        ):
            _run(eng.start())

        assert captured["name"] == "dummy/repo"
        assert captured["kwargs"]["adapter_path"] == "/adapters/x"

    def test_load_omits_adapter_path_when_no_lora(self):
        eng = BatchedEngine(model_name="dummy/repo")
        captured = {}

        def fake_load(name, **kwargs):
            captured["kwargs"] = kwargs
            model = MagicMock()
            model.parameters.return_value = {}
            return model, MagicMock()

        fake_core = MagicMock()
        fake_core.engine.start = AsyncMock()

        with (
            patch("mlx_lm.load", side_effect=fake_load),
            patch("fusion_mlx.engines.batched.AsyncEngineCore", return_value=fake_core),
        ):
            _run(eng.start())

        assert "adapter_path" not in captured["kwargs"]


class TestLoadModelStagesLoraPath:
    def test_load_model_stages_lora_path(self):
        from fusion_mlx import server
        from fusion_mlx.config import get_config

        saved = server._pending_single_model
        cfg = get_config()
        saved_name = cfg.model_name
        saved_path = cfg.model_path
        saved_alias = cfg.model_alias
        try:
            with (
                patch.object(
                    server, "_resolve_single_model_path", return_value="/resolved"
                ),
                patch.object(server, "get_app"),
                patch.object(server, "_sync_config"),
            ):
                server.load_model("dummy/repo", lora_path="/adapters/x")
            assert server._pending_single_model["lora_path"] == "/adapters/x"
        finally:
            server._pending_single_model = saved
            cfg.model_name = saved_name
            cfg.model_path = saved_path
            cfg.model_alias = saved_alias


def _make_mock_model_dir(tmp_path):
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_a / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path


class TestModelSettingsLoraPath:
    def test_roundtrip(self):
        ms = ModelSettings(lora_path="/adapters/x")
        d = ms.to_dict()
        assert d["lora_path"] == "/adapters/x"
        ms2 = ModelSettings.from_dict(d)
        assert ms2.lora_path == "/adapters/x"

    def test_default_none(self):
        assert ModelSettings().lora_path is None

    def test_from_dict_ignores_unknown_keys(self):
        ms = ModelSettings.from_dict({"lora_path": "/a", "bogus": 1})
        assert ms.lora_path == "/a"

    def test_in_profile_allowlist(self):
        assert "lora_path" in MODEL_SPECIFIC_PROFILE_FIELDS


class TestEnginePoolLoraWiring:
    @pytest.mark.asyncio
    async def test_threads_lora_path_from_settings(self, tmp_path):
        _make_mock_model_dir(tmp_path)
        pool = EnginePool()
        pool._get_final_ceiling = lambda: 0
        pool.discover_models(str(tmp_path))

        mgr = ModelSettingsManager(tmp_path / "settings")
        mgr.set_settings("model-a", ModelSettings(lora_path="/adapters/x"))
        pool._settings_manager = mgr

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine",
            return_value=mock_engine,
        ) as mock_ctor:
            await pool._load_engine("model-a")

        mock_ctor.assert_called_once()
        assert mock_ctor.call_args.kwargs.get("lora_path") == "/adapters/x"

    @pytest.mark.asyncio
    async def test_no_lora_path_when_settings_unset(self, tmp_path):
        _make_mock_model_dir(tmp_path)
        pool = EnginePool()
        pool._get_final_ceiling = lambda: 0
        pool.discover_models(str(tmp_path))

        pool._settings_manager = ModelSettingsManager(tmp_path / "settings")

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch(
            "fusion_mlx.pool.engine_pool.BatchedEngine",
            return_value=mock_engine,
        ) as mock_ctor:
            await pool._load_engine("model-a")

        mock_ctor.assert_called_once()
        assert mock_ctor.call_args.kwargs.get("lora_path") is None
