# SPDX-License-Identifier: Apache-2.0
# Boot-time LoRA adapter wiring (Phase B LoRA slice 1).
# Verifies --lora-path threads from server.load_model into BatchedEngine and
# reaches mlx_lm.load(adapter_path=...).
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fusion_mlx.engines.batched import BatchedEngine


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

        saved = server._pending_single_model
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
