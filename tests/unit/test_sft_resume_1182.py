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


class _StubModel:
    # Stand-in exposing load_weights like mlx.nn.Module; records the call.
    def __init__(self):
        self.calls = []

    def load_weights(self, file_or_weights, strict=True):
        self.calls.append((str(file_or_weights), strict))
        return self


class TestResumeHelper:
    def test_noop_when_unset(self):
        model = _StubModel()
        out = _resume_adapter_weights(model, FineTuneConfig(), Path("/tmp/adt"))
        assert out is model
        assert model.calls == []

    def test_loads_weights_from_checkpoint(self, tmp_path):
        ckpt = tmp_path / "run1"
        ckpt.mkdir()
        (ckpt / "adapters.safetensors").write_bytes(b"stub")

        model = _StubModel()
        out = _resume_adapter_weights(
            model, FineTuneConfig(init_adapter_path=str(ckpt)), tmp_path / "out"
        )
        assert out is model
        assert model.calls == [(str(ckpt / "adapters.safetensors"), False)]

    def test_not_load_adapters_double_apply(self):
        # mlx_lm load_adapters() re-applies linear_to_lora_layers from the
        # saved config — double-apply on our already-built LoRA tree
        # ("Can't convert layer of type LoRALinear to LoRA", caught in
        # real-model verification). The helper must not call it.
        import fusion_mlx.training.service as svc

        src = inspect.getsource(svc._resume_adapter_weights)
        assert "load_adapters" not in src
        assert "load_weights" in src
        assert "strict=False" in src

    def test_missing_checkpoint_fails_fast(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no adapters.safetensors"):
            _resume_adapter_weights(
                _StubModel(),
                FineTuneConfig(init_adapter_path=str(empty)),
                tmp_path / "out",
            )

    def test_in_place_resume_warns(self, tmp_path, caplog):
        ckpt = tmp_path / "run1"
        ckpt.mkdir()
        (ckpt / "adapters.safetensors").write_bytes(b"stub")
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.training.service"):
            _resume_adapter_weights(
                _StubModel(),
                FineTuneConfig(init_adapter_path=str(ckpt), steps_per_save=100),
                ckpt,
            )
        assert any(
            "overwrite the resumed checkpoint" in r.message for r in caplog.records
        )


class TestExecuteWiring:
    def test_execute_training_calls_resume_helper(self):
        # Pins the call site: resume must run inside execute_training_core
        # (#1186: body shared by thread path and spawned child) AFTER
        # linear_to_lora_layers and BEFORE train()/save_config.
        import fusion_mlx.training.service as svc

        src = inspect.getsource(svc.execute_training_core)
        assert "_resume_adapter_weights" in src
        assert src.index("linear_to_lora_layers") < src.index("_resume_adapter_weights")
        assert src.index("_resume_adapter_weights") < src.index("save_config(")

    def test_execute_paths_share_core(self):
        # #1186: both execution modes must go through the same core so the
        # spawned child cannot drift from the legacy thread path.
        import fusion_mlx.training.service as svc

        thread_src = inspect.getsource(svc.FineTuneService._execute_training)
        assert "execute_training_core" in thread_src
        child_src = inspect.getsource(svc.execute_training_spec)
        assert "execute_training_core" in child_src
