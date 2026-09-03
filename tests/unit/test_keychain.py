# SPDX-License-Identifier: Apache-2.0
"""Tests for the macOS Keychain API-key backend (#770).

Mocks the `security` CLI via subprocess so tests never touch the real
Keychain. Covers: enabled/available gating, set/get/delete round-trip,
plaintext-clearing migration in Settings._load_sync, and save-path
leaving no plaintext on disk.
"""

import json
from unittest import mock

import pytest


def _make_proc(stdout=b"", stderr=b"", rc=0):
    proc = mock.Mock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = rc
    return proc


@pytest.fixture
def kc(monkeypatch):
    monkeypatch.setenv("FUSION_KEYCHAIN", "on")
    from fusion_mlx.admin import keychain

    monkeypatch.setattr(keychain.platform, "system", lambda: "Darwin")
    return keychain


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FUSION_KEYCHAIN", raising=False)
    from fusion_mlx.admin import keychain

    assert keychain.is_enabled() is False


def test_get_key_missing_returns_none(kc, monkeypatch):
    monkeypatch.setattr(
        kc.subprocess,
        "run",
        lambda *a, **k: _make_proc(stderr=b"could not be found", rc=1),
    )
    assert kc.get_key() is None


def test_set_get_delete_roundtrip(kc, monkeypatch):
    calls = []

    def fake_run(cmd, input=None, capture_output=False, check=False, timeout=None):
        calls.append(cmd)
        if cmd[1] == "find-generic-password":
            return _make_proc(stdout=b"secret123\n", rc=0)
        return _make_proc(rc=0)

    monkeypatch.setattr(kc.subprocess, "run", fake_run)
    assert kc.set_key("secret123") is True
    assert kc.get_key() == "secret123"
    assert kc.delete_key() is True
    # add must be preceded by a delete to avoid collisions
    assert any(c[1] == "delete-generic-password" for c in calls)
    assert any(c[1] == "add-generic-password" for c in calls)


def test_set_empty_deletes(kc, monkeypatch):
    deleted = []

    def fake_run(cmd, input=None, capture_output=False, check=False, timeout=None):
        if cmd[1] == "delete-generic-password":
            deleted.append(True)
        return _make_proc(rc=0)

    monkeypatch.setattr(kc.subprocess, "run", fake_run)
    assert kc.set_key("") is True
    assert deleted


def test_unavailable_platform(monkeypatch):
    monkeypatch.setenv("FUSION_KEYCHAIN", "on")
    from fusion_mlx.admin import keychain

    monkeypatch.setattr(keychain.platform, "system", lambda: "Linux")
    assert keychain.is_available() is False
    assert keychain.get_key() is None


def test_load_migrates_plaintext_to_keychain(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_KEYCHAIN", "on")
    from fusion_mlx.admin import keychain

    monkeypatch.setattr(keychain.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(keychain, "is_available", lambda: True)
    monkeypatch.setattr(keychain, "is_enabled", lambda: True)
    stored = {}

    def fake_set_key(key):
        stored["key"] = key
        return True

    monkeypatch.setattr(keychain, "set_key", fake_set_key)
    monkeypatch.setattr(keychain, "get_key", lambda: None)

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"api_key": "plain-secret"}))

    from fusion_mlx.settings import Settings

    s = Settings._load_sync(settings_path)
    assert s.api_key == "plain-secret"
    assert stored["key"] == "plain-secret"
    on_disk = json.loads(settings_path.read_text())
    assert on_disk.get("api_key") is None


def test_load_prefers_keychain_over_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_KEYCHAIN", "on")
    from fusion_mlx.admin import keychain

    monkeypatch.setattr(keychain.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(keychain, "is_available", lambda: True)
    monkeypatch.setattr(keychain, "is_enabled", lambda: True)
    monkeypatch.setattr(keychain, "get_key", lambda: "kc-secret")
    monkeypatch.setattr(keychain, "set_key", lambda key: True)

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"api_key": "stale-plain"}))

    from fusion_mlx.settings import Settings

    s = Settings._load_sync(settings_path)
    assert s.api_key == "kc-secret"
    on_disk = json.loads(settings_path.read_text())
    assert on_disk.get("api_key") is None


def test_save_writes_keychain_no_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_KEYCHAIN", "on")
    from fusion_mlx.admin import keychain

    monkeypatch.setattr(keychain.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(keychain, "is_available", lambda: True)
    monkeypatch.setattr(keychain, "is_enabled", lambda: True)
    saved = {}

    def fake_set_key(key):
        saved["key"] = key
        return True

    monkeypatch.setattr(keychain, "set_key", fake_set_key)

    from fusion_mlx.settings import Settings

    s = Settings(api_key="will-be-keychain")
    s._save_sync(tmp_path / "settings.json")
    on_disk = json.loads((tmp_path / "settings.json").read_text())
    assert on_disk["api_key"] is None
    assert saved["key"] == "will-be-keychain"


def test_save_falls_back_when_keychain_off(tmp_path, monkeypatch):
    monkeypatch.delenv("FUSION_KEYCHAIN", raising=False)
    from fusion_mlx.settings import Settings

    s = Settings(api_key="plain-kept")
    s._save_sync(tmp_path / "settings.json")
    on_disk = json.loads((tmp_path / "settings.json").read_text())
    assert on_disk["api_key"] == "plain-kept"
