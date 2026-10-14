# SPDX-License-Identifier: Apache-2.0
"""Tests for #1182: SFT resume from adapter checkpoint.

FineTuneConfig gains init_adapter_path (same naming as DPOService's
#1149 field). _execute_training loads the checkpoint weights into the
freshly-built LoRA layers via _resume_adapter_weights() before train();
the step counter restarts at 0 (optimizer-state persistence is a
follow-up). Missing checkpoint fails fast with a clear error; resuming
in-place warns that the first save overwrites the source checkpoint.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from fusion_mlx.training.service import FineTuneConfig, _resume_adapter_weights


class TestConfigSurface:
    def test_field_default_empty(self):
        assert FineTuneConfig().init_adapter_path == ""

    def test_to_mlx_args_none_when_unset(self):
        args = FineTuneConfig().to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert args.resume_adapter_file is None

    def test_to_mlx_args_points_at_checkpoint(self):
        cfg = FineTuneConfig(init_adapter_path="/data/adapters/m1/run1")
        args = cfg.to_mlx_args("/tmp/adt", "/tmp/data", "/tmp/model")
        assert args.resume_adapter_file == (
            "/data/adapters/m1/run1/adapters.safetensors"
        )

    def test_field_accepted_in_payload_shape(self):
        # Admin route does FineTuneConfig(**config_body) — the field must
        # survive a payload-style kwargs construction alongside rank knobs.
        cfg = FineTuneConfig(**{"init_adapter_path": "/data/x", "lora_rank": 16})
        assert cfg.init_adapter_path == "/data/x"
        assert cfg.lora_rank == 16


class TestResumeHelper:
    def test_noop_when_unset(self):
        model = SimpleNamespace(touched=False)
        out = _resume_adapter_weights(model, FineTuneConfig(), Path("/tmp/adt"))
        assert out is model

    def test_loads_weights_from_checkpoint(self, tmp_path, monkeypatch):
        pytest.importorskip("mlx_lm.tuner.utils")
        import mlx_lm.tuner.utils as tuner_utils

        ckpt = tmp_path / "run1"
        ckpt.mkdir()
        (ckpt / "adapters.safetensors").write_bytes(b"stub")

        calls = {}

        def fake_load_adapters(model, adapter_path):
            calls["adapter_path"] = adapter_path
            return model

        monkeypatch.setattr(tuner_utils, "load_adapters", fake_load_adapters)

        model = SimpleNamespace(touched=False)
        out = _resume_adapter_weights(
            model, FineTuneConfig(init_adapter_path=str(ckpt)), tmp_path / "out"
        )
        assert out is model
        assert calls["adapter_path"] == str(ckpt)

    def test_missing_checkpoint_fails_fast(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no adapters.safetensors"):
            _resume_adapter_weights(
                SimpleNamespace(),
                FineTuneConfig(init_adapter_path=str(empty)),
                tmp_path / "out",
            )

    def test_in_place_resume_warns(self, tmp_path, caplog, monkeypatch):
        pytest.importorskip("mlx_lm.tuner.utils")
        import mlx_lm.tuner.utils as tuner_utils

        ckpt = tmp_path / "run1"
        ckpt.mkdir()
        (ckpt / "adapters.safetensors").write_bytes(b"stub")
        monkeypatch.setattr(
            tuner_utils, "load_adapters", lambda model, adapter_path: model
        )
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.training.service"):
            _resume_adapter_weights(
                SimpleNamespace(),
                FineTuneConfig(init_adapter_path=str(ckpt), steps_per_save=100),
                ckpt,
            )
        assert any(
            "overwrite the resumed checkpoint" in r.message for r in caplog.records
        )


class TestExecuteWiring:
    def test_execute_training_calls_resume_helper(self):
        # Pins the call site: resume must run inside _execute_training
        # AFTER linear_to_lora_layers and BEFORE train()/save_config.
        from fusion_mlx.training.service import FineTuneService

        src = inspect.getsource(FineTuneService._execute_training)
        assert "_resume_adapter_weights" in src
        assert src.index("linear_to_lora_layers") < src.index("_resume_adapter_weights")
        assert src.index("_resume_adapter_weights") < src.index("save_config(")
