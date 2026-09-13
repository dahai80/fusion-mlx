# SPDX-License-Identifier: Apache-2.0
"""Regression for F-4 (#0912 audit): model_settings.json slash/hyphen dual-key.

HF repo id ("mlx-community/Qwen3.8-27B-4bit", slash form, returned by
resolve_model from aliases.json) and the EnginePool registered dir name
("mlx-community--Qwen3.8-27B-4bit", hyphen form, the on-disk model dir)
resolved to DIFFERENT dict keys in model_settings.json. Admin writes
used the pool entry id (hyphen); serve-time lookups used the resolved
HF repo id (slash) -> the same model stored under two keys that could
diverge silently. Operator sets TTL via admin (hyphen key), serve reads
slash key -> default, TTL ignored.

Fix: ModelSettingsManager canonicalizes every model_id key to hyphen
form ("/" -> "--") on load (merging duplicate keys with a fail-visible
ERROR log) and on every public read/write, so the two forms can never
coexist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fusion_mlx.model_settings import (
    ModelSettings,
    ModelSettingsManager,
    _canonical_model_id,
)


def test_canonical_slash_to_hyphen():
    assert _canonical_model_id("mlx-community/Qwen3.8-27B-4bit") == (
        "mlx-community--Qwen3.8-27B-4bit"
    )


def test_canonical_hyphen_unchanged():
    assert _canonical_model_id("mlx-community--Qwen3.8-27B-4bit") == (
        "mlx-community--Qwen3.8-27B-4bit"
    )


def test_canonical_bare_name_unchanged():
    assert _canonical_model_id("Qwen3.8-27B-4bit") == "Qwen3.8-27B-4bit"


def test_canonical_empty_passthrough():
    assert _canonical_model_id("") == ""


def test_dual_key_collapsed_on_load(tmp_path: Path):
    settings_file = tmp_path / "model_settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "mlx-community/Qwen3.8-27B-4bit": {"ttl_seconds": 300},
                    "mlx-community--Qwen3.8-27B-4bit": {"is_pinned": True},
                },
            }
        ),
        encoding="utf-8",
    )
    mgr = ModelSettingsManager(tmp_path)
    keys = list(mgr._settings.keys())
    assert keys == ["mlx-community--Qwen3.8-27B-4bit"]
    assert len(mgr._settings) == 1


def test_dual_key_conflict_logs_error(tmp_path: Path, caplog):
    settings_file = tmp_path / "model_settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "mlx-community/Qwen3.8-27B-4bit": {"ttl_seconds": 300},
                    "mlx-community--Qwen3.8-27B-4bit": {"is_pinned": True},
                },
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR, logger="fusion_mlx.model_settings"):
        ModelSettingsManager(tmp_path)
    assert any("F-4" in r.message and "dual-key" in r.message for r in caplog.records)


def test_write_slash_read_hyphen(tmp_path: Path):
    mgr = ModelSettingsManager(tmp_path)
    ms = ModelSettings(ttl_seconds=600)
    mgr.set_settings("mlx-community/Qwen3.8-27B-4bit", ms)
    assert "mlx-community--Qwen3.8-27B-4bit" in mgr._settings
    assert "mlx-community/Qwen3.8-27B-4bit" not in mgr._settings
    got = mgr.get_settings("mlx-community--Qwen3.8-27B-4bit")
    assert got.ttl_seconds == 600


def test_write_hyphen_read_slash(tmp_path: Path):
    mgr = ModelSettingsManager(tmp_path)
    mgr.set_settings("mlx-community--Qwen3.8-27B-4bit", ModelSettings(is_pinned=True))
    got = mgr.get_settings("mlx-community/Qwen3.8-27B-4bit")
    assert got.is_pinned is True


def test_delete_slash_removes_hyphen(tmp_path: Path):
    mgr = ModelSettingsManager(tmp_path)
    mgr.set_settings("mlx-community--Qwen3.8-27B-4bit", ModelSettings(is_pinned=True))
    assert mgr.delete_settings("mlx-community/Qwen3.8-27B-4bit") is True
    assert "mlx-community--Qwen3.8-27B-4bit" not in mgr._settings


def test_no_dual_key_after_roundtrip(tmp_path: Path):
    mgr = ModelSettingsManager(tmp_path)
    mgr.set_settings("mlx-community/Qwen3.8-27B-4bit", ModelSettings(ttl_seconds=120))
    mgr.set_settings("mlx-community--Qwen3.8-27B-4bit", ModelSettings(is_default=True))
    assert len(mgr._settings) == 1
    assert "mlx-community--Qwen3.8-27B-4bit" in mgr._settings
    data = json.loads((tmp_path / "model_settings.json").read_text())
    assert "mlx-community/Qwen3.8-27B-4bit" not in data["models"]


def test_profile_dual_key_collapsed_on_load(tmp_path: Path):
    profiles_file = tmp_path / "model_profiles.json"
    profiles_file.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "mlx-community/Qwen3.8-27B-4bit": {
                        "fast": {
                            "name": "fast",
                            "display_name": "fast",
                            "api_name": "fast",
                            "settings": {"temperature": 0.5},
                        }
                    },
                    "mlx-community--Qwen3.8-27B-4bit": {
                        "fast": {
                            "name": "fast",
                            "display_name": "fast",
                            "api_name": "fast",
                            "settings": {"temperature": 0.7},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    mgr = ModelSettingsManager(tmp_path)
    assert list(mgr._profiles.keys()) == ["mlx-community--Qwen3.8-27B-4bit"]


def test_profile_write_slash_read_hyphen(tmp_path: Path):
    mgr = ModelSettingsManager(tmp_path)
    mgr.save_profile(
        "mlx-community/Qwen3.8-27B-4bit",
        "fast",
        "fast",
        None,
        {"temperature": 0.3},
    )
    assert "mlx-community--Qwen3.8-27B-4bit" in mgr._profiles
    prof = mgr.get_profile("mlx-community--Qwen3.8-27B-4bit", "fast")
    assert prof is not None
    assert prof["settings"]["temperature"] == 0.3


def test_get_settings_for_request_slash_resolves(tmp_path: Path):
    mgr = ModelSettingsManager(tmp_path)
    mgr.set_settings("mlx-community--Qwen3.8-27B-4bit", ModelSettings(ttl_seconds=90))
    got = mgr.get_settings_for_request("mlx-community/Qwen3.8-27B-4bit")
    assert got.ttl_seconds == 90
