# SPDX-License-Identifier: Apache-2.0
"""Tests for the --enable-lsa / --lsa-map CLI plumbing into the LSA patch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from fusion_mlx.config import SchedulerConfig
from fusion_mlx.model_settings import ModelSettings

# The post-load transform path imports the sibling fusion-lsa package
# (pip install -e ~/fusion/fusion-lsa), which CI runners don't have.
_fusion_lsa_missing = importlib.util.find_spec("fusion_lsa") is None


def test_scheduler_config_carries_lsa_fields():
    config = SchedulerConfig()
    assert config.lsa_enabled is False
    assert config.lsa_map_path is None
    config = SchedulerConfig(lsa_enabled=True, lsa_map_path="m.safetensors")
    assert config.lsa_enabled is True
    assert config.lsa_map_path == "m.safetensors"


def test_serve_parser_accepts_lsa_flags(monkeypatch):
    import sys

    from fusion_mlx import cli as cli_mod

    captured = {}

    def _fake_serve(args):
        captured["enable_lsa"] = args.enable_lsa
        captured["lsa_map_path"] = args.lsa_map_path
        raise SystemExit(0)

    monkeypatch.setattr(cli_mod, "serve_command", _fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fusion-mlx",
            "serve",
            "--enable-lsa",
            "--lsa-map",
            "/tmp/m.safetensors",
            "mlx-community/Qwen3.5-9B-4bit",
        ],
    )
    with pytest.raises(SystemExit):
        cli_mod.main()
    assert captured["enable_lsa"] is True
    assert captured["lsa_map_path"] == "/tmp/m.safetensors"


def test_serve_command_rejects_lsa_with_turboquant(monkeypatch, capsys):
    import sys

    from fusion_mlx import cli as cli_mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fusion-mlx",
            "serve",
            "--enable-lsa",
            "--lsa-map",
            "/tmp/m.safetensors",
            "--kv-cache-turboquant",
            "mlx-community/Qwen3.5-9B-4bit",
        ],
    )
    with pytest.raises(SystemExit):
        cli_mod.main()
    assert "turboquant" in capsys.readouterr().err.lower()


def test_batched_engine_overlays_scheduler_lsa_onto_model_settings(
    tmp_path, monkeypatch
):
    from fusion_mlx.engines.batched import BatchedEngine

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "hidden_size": 256})
    )
    captured = {}

    def _fake_pre_load(model_path, model_settings=None, for_vlm=False):
        captured["settings"] = model_settings
        raise RuntimeError("stop before real load")

    monkeypatch.setattr(
        "fusion_mlx.utils.model_loading.maybe_apply_pre_load_patches", _fake_pre_load
    )
    engine = BatchedEngine(
        model_name=str(tmp_path),
        scheduler_config=SchedulerConfig(
            lsa_enabled=True, lsa_map_path="m.safetensors"
        ),
        model_settings=ModelSettings(),
    )
    import asyncio

    with pytest.raises(RuntimeError, match="stop before real load"):
        asyncio.run(engine.start())
    settings = captured["settings"]
    assert settings is not None
    assert settings.lsa_enabled is True
    assert settings.lsa_map_path == "m.safetensors"


def test_batched_engine_keeps_settings_when_lsa_off(tmp_path, monkeypatch):
    from fusion_mlx.engines.batched import BatchedEngine

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "hidden_size": 256})
    )
    captured = {}

    def _fake_pre_load(model_path, model_settings=None, for_vlm=False):
        captured["settings"] = model_settings
        raise RuntimeError("stop before real load")

    monkeypatch.setattr(
        "fusion_mlx.utils.model_loading.maybe_apply_pre_load_patches", _fake_pre_load
    )
    engine = BatchedEngine(
        model_name=str(tmp_path),
        scheduler_config=SchedulerConfig(),
        model_settings=None,
    )
    import asyncio

    with pytest.raises(RuntimeError, match="stop before real load"):
        asyncio.run(engine.start())
    assert captured["settings"] is None


def test_start_sh_whitelists_lsa_flags():
    script = Path(__file__).resolve().parents[2].joinpath("start.sh").read_text()
    assert "--enable-lsa" in script
    assert "--lsa-map" in script


def _fake_qwen35_model():
    from lsa_attention_support import make_attn

    class _Layer:
        def __init__(self, full):
            if full:
                self.self_attn = make_attn()

    class _TextModel:
        def __init__(self):
            self.layers = [_Layer(i % 4 == 3) for i in range(16)]

    class _LanguageModel:
        def __init__(self):
            self.model = _TextModel()

    class _Args:
        text_config = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "partial_rotary_factor": 0.25,
            "full_attention_interval": 4,
        }

    class _Model:
        def __init__(self):
            self.args = _Args()
            self.model_type = "qwen3_5"
            self.language_model = _LanguageModel()

    return _Model()


@pytest.mark.skipif(
    _fusion_lsa_missing, reason="fusion-lsa sibling package not installed"
)
def test_post_load_transforms_wraps_qwen35_layers(tmp_path):
    from lsa_attention_support import write_map

    map_path = write_map(tmp_path, layers=(3, 7))
    from fusion_mlx.model_settings import ModelSettings
    from fusion_mlx.utils.model_loading import apply_post_load_transforms

    model = _fake_qwen35_model()
    settings = ModelSettings(lsa_enabled=True, lsa_map_path=map_path)
    out = apply_post_load_transforms(model, settings)
    assert out is model
    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention

    assert isinstance(model.language_model.model.layers[3].self_attn, LsaAttention)
    assert isinstance(model.language_model.model.layers[7].self_attn, LsaAttention)
    assert not isinstance(model.language_model.model.layers[11].self_attn, LsaAttention)


@pytest.mark.skipif(
    _fusion_lsa_missing, reason="fusion-lsa sibling package not installed"
)
def test_post_load_transforms_noop_without_lsa(tmp_path):
    from fusion_mlx.model_settings import ModelSettings
    from fusion_mlx.utils.model_loading import apply_post_load_transforms

    model = _fake_qwen35_model()
    out = apply_post_load_transforms(model, ModelSettings())
    assert out is model
    from fusion_mlx.patches.qwen35_lsa.attention import LsaAttention

    assert not isinstance(model.language_model.model.layers[3].self_attn, LsaAttention)


def test_lsa_disables_auto_turboquant(monkeypatch, caplog):
    import logging

    from fusion_mlx.config import SchedulerConfig
    from fusion_mlx.engines.batched import BatchedEngine
    from fusion_mlx.model_settings import ModelSettings

    engine = BatchedEngine(
        model_name="dummy",
        scheduler_config=SchedulerConfig(
            lsa_enabled=True, lsa_map_path="m.safetensors"
        ),
        model_settings=ModelSettings(),
    )
    with caplog.at_level(logging.INFO):
        assert engine._turboquant_should_enable() is False
        assert engine._turboquant_kv_will_enable() is False
    assert any(
        "LSA needs explicit k8v4 mode" in r.message for r in caplog.records
    )

    plain = BatchedEngine(
        model_name="dummy",
        scheduler_config=SchedulerConfig(),
        model_settings=ModelSettings(),
    )
    assert plain._turboquant_should_enable() is True


def test_serve_command_rejects_lsa_with_quantized_kv_dtype(monkeypatch, capsys):
    import sys

    from fusion_mlx import cli as cli_mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fusion-mlx",
            "serve",
            "--enable-lsa",
            "--lsa-map",
            "/tmp/m.safetensors",
            "--kv-cache-dtype",
            "int4",
            "mlx-community/Qwen3.5-9B-4bit",
        ],
    )
    with pytest.raises(SystemExit):
        cli_mod.main()
    assert "kv-cache-dtype" in capsys.readouterr().err.lower()


def test_lsa_with_explicit_k8v4_enables_turboquant():
    # coexistence: explicit k8v4 keeps 8-bit keys, so the TQ selection
    # index rebuild reads dequantized rows at probe-validated quality
    from fusion_mlx.config import SchedulerConfig
    from fusion_mlx.engines.batched import BatchedEngine
    from fusion_mlx.model_settings import ModelSettings

    engine = BatchedEngine(
        model_name="dummy",
        scheduler_config=SchedulerConfig(
            lsa_enabled=True,
            lsa_map_path="m.safetensors",
            kv_cache_turboquant_mode="k8v4",
        ),
        model_settings=ModelSettings(turboquant_kv_enabled=True),
    )
    assert engine._turboquant_should_enable() is True
    assert engine._turboquant_kv_will_enable() is True

    v4 = BatchedEngine(
        model_name="dummy",
        scheduler_config=SchedulerConfig(
            lsa_enabled=True,
            lsa_map_path="m.safetensors",
            kv_cache_turboquant_mode="v4",
        ),
        model_settings=ModelSettings(turboquant_kv_enabled=True),
    )
    assert v4._turboquant_should_enable() is False
    assert v4._turboquant_kv_will_enable() is False

    # the CLI flag is operator-explicit and wins over a settings-file False
    flagged = BatchedEngine(
        model_name="dummy",
        scheduler_config=SchedulerConfig(
            lsa_enabled=True,
            lsa_map_path="m.safetensors",
            kv_cache_turboquant=True,
            kv_cache_turboquant_mode="k8v4",
        ),
        model_settings=ModelSettings(turboquant_kv_enabled=False),
    )
    assert flagged._turboquant_should_enable() is True
    assert flagged._turboquant_kv_will_enable() is True
