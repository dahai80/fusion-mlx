# #1439: config_seam is the only place fusion_mlx reads FUSION_MLX_API_KEY
# or settings.json auth.api_key. Resolution policy lives in
# fusion_core.config when installed (monorepo); an identical stdlib
# fallback covers standalone PyPI installs. These tests pin the precedence
# contract on BOTH paths (fusion_core path skipped when not installed, e.g.
# CI, since fusion-core is in-tree only and not on PyPI).

from __future__ import annotations

import json
import logging

import pytest

from fusion_mlx import config_seam


def _with_core(monkeypatch, present: bool):
    # Force the seam onto the requested path regardless of how the test
    # env is installed. present=True only works when fusion_core actually
    # exists (CI does not install it).
    if present:
        pytest.importorskip("fusion_core.config")
        import fusion_core.config as core

        monkeypatch.setattr(config_seam, "_fusion_core_config", core)
    else:
        monkeypatch.setattr(config_seam, "_fusion_core_config", None)


@pytest.fixture
def home_settings(monkeypatch, tmp_path):
    # Isolate the fallback path's settings read from the operator's real
    # ~/.fusion-mlx/settings.json (which holds a real key).
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / ".fusion-mlx"


@pytest.mark.parametrize("present", [True, False], ids=["fusion_core", "fallback"])
class TestPrecedence:
    def test_explicit_beats_env_and_settings(self, monkeypatch, home_settings, present):
        _with_core(monkeypatch, present)
        home_settings.mkdir()
        (home_settings / "settings.json").write_text(
            json.dumps({"auth": {"api_key": "settings-secret"}})
        )
        monkeypatch.setenv("FUSION_MLX_API_KEY", "env-secret")
        assert config_seam.api_key("cli-secret") == "cli-secret"

    def test_env_beats_settings(self, monkeypatch, home_settings, present):
        _with_core(monkeypatch, present)
        home_settings.mkdir()
        (home_settings / "settings.json").write_text(
            json.dumps({"auth": {"api_key": "settings-secret"}})
        )
        monkeypatch.setenv("FUSION_MLX_API_KEY", "env-secret")
        assert config_seam.api_key() == "env-secret"

    def test_settings_used_when_no_cli_no_env(
        self, monkeypatch, home_settings, present
    ):
        _with_core(monkeypatch, present)
        home_settings.mkdir()
        (home_settings / "settings.json").write_text(
            json.dumps({"auth": {"api_key": "settings-secret"}})
        )
        monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
        assert config_seam.api_key() == "settings-secret"

    def test_empty_when_nothing_configured(self, monkeypatch, home_settings, present):
        _with_core(monkeypatch, present)
        monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
        assert config_seam.api_key() == ""


def test_fallback_missing_settings_file_returns_empty(monkeypatch, tmp_path):
    _with_core(monkeypatch, False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    assert config_seam.api_key() == ""


def test_fallback_malformed_settings_warns_not_raises(monkeypatch, tmp_path, caplog):
    _with_core(monkeypatch, False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    settings = tmp_path / ".fusion-mlx"
    settings.mkdir()
    (settings / "settings.json").write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.config_seam"):
        assert config_seam.api_key() == ""
    assert any("settings.json" in r.message for r in caplog.records)


def test_fallback_non_dict_auth_returns_empty(monkeypatch, tmp_path):
    _with_core(monkeypatch, False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    settings = tmp_path / ".fusion-mlx"
    settings.mkdir()
    (settings / "settings.json").write_text(json.dumps({"auth": "not-a-dict"}))
    assert config_seam.api_key() == ""


def test_env_api_key_reads_env_only(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_API_KEY", "env-only-secret")
    assert config_seam.env_api_key() == "env-only-secret"
    monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
    assert config_seam.env_api_key() == ""


def test_fusion_core_hook_present_in_monorepo():
    # On the monorepo dev machine fusion_core is installed; the seam must
    # have picked it up at import. On CI (no fusion_core) this pins the
    # documented degradation instead.
    if config_seam._fusion_core_config is None:
        pytest.skip("fusion_core not installed (standalone/CI env)")
    assert hasattr(config_seam._fusion_core_config, "resolve_api_key")
