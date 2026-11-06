# SPDX-License-Identifier: Apache-2.0
"""Tests for the --enable-lsa / --lsa-map CLI plumbing into the LSA patch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_mlx.config import SchedulerConfig
from fusion_mlx.model_settings import ModelSettings


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
