# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.settings — the rewritten single-``Settings`` surface.

The legacy per-section dataclass settings (ServerSettings/GlobalSettings/
AuthSettings/... and the burst-decode env mapping) were removed when settings
collapsed into one persistent ``Settings`` dataclass (#770 keychain work).
These tests pin the current contract:

- save/load round-trip through settings.json (0o600 perms)
- flat api_key vs nested auth.api_key read compatibility
- ``Settings.auth`` live view propagation
- IntegrationSettings from_dict/to_dict round-trip
- plaintext api_key scrubbing (``_clear_plaintext_api_key``)
"""

import json
from pathlib import Path

from fusion_mlx.settings import (
    IntegrationSettings,
    Settings,
    SubKeyEntry,
    _clear_plaintext_api_key,
)

# ---------------------------------------------------------------------------
# IntegrationSettings
# ---------------------------------------------------------------------------


class TestIntegrationSettings:
    def test_defaults(self):
        s = IntegrationSettings()
        assert s.markitdown_enabled is True
        assert s.markitdown_expose_model is False
        assert s.markitdown_max_file_size_mb == 25
        assert s.markitdown_max_files_per_request == 5
        assert s.markitdown_pdf_processing_engine == "markitdown"

    def test_round_trip(self):
        s = IntegrationSettings(
            markitdown_enabled=False,
            markitdown_expose_model=True,
            markitdown_max_file_size_mb=50,
        )
        d = s.to_dict()
        assert d["markitdown_enabled"] is False
        assert d["markitdown_expose_model"] is True
        assert d["markitdown_max_file_size_mb"] == 50
        s2 = IntegrationSettings.from_dict(d)
        assert s2 == s

    def test_from_dict_partial(self):
        s = IntegrationSettings.from_dict({"markitdown_enabled": False})
        assert s.markitdown_enabled is False
        assert s.markitdown_max_file_size_mb == 25  # default preserved


# ---------------------------------------------------------------------------
# SubKeyEntry
# ---------------------------------------------------------------------------


class TestSubKeyEntry:
    def test_defaults(self):
        e = SubKeyEntry(name="k1", key_hash="abc", created_at="2026-01-01")
        assert e.expires_at is None
        assert e.usage_count == 0
        assert e.is_active is True


# ---------------------------------------------------------------------------
# Settings save/load round-trip
# ---------------------------------------------------------------------------


class TestSettingsPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        s = Settings(
            api_key="sk-test",
            model_settings={"m": {"max_tokens": 100}},
            global_settings={"skip_api_key_verification": True},
        )
        s.save(path)
        loaded = Settings.load(path)
        assert loaded.api_key == "sk-test"
        assert loaded.model_settings == {"m": {"max_tokens": 100}}
        assert loaded.global_settings == {"skip_api_key_verification": True}

    def test_save_sets_0600(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        Settings(api_key="sk-test").save(path)
        assert (path.stat().st_mode & 0o777) == 0o600

    def test_load_missing_file_returns_defaults(self, tmp_path: Path):
        loaded = Settings.load(tmp_path / "nonexistent.json")
        assert loaded.api_key is None
        assert loaded.sub_keys == []
        assert loaded.model_settings == {}

    def test_load_corrupt_json_returns_defaults(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text("{not json")
        loaded = Settings.load(path)
        assert loaded.api_key is None

    def test_load_accepts_nested_auth_api_key(self, tmp_path: Path):
        """Legacy nested ``auth.api_key`` shape still loads (compat)."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"auth": {"api_key": "sk-nested"}}))
        loaded = Settings.load(path)
        assert loaded.api_key == "sk-nested"

    def test_sub_keys_round_trip(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        s = Settings(
            sub_keys=[
                SubKeyEntry(
                    name="k1",
                    key_hash="h1",
                    created_at="2026-01-01",
                    expires_at="2027-01-01",
                    usage_count=3,
                )
            ]
        )
        s.save(path)
        loaded = Settings.load(path)
        assert len(loaded.sub_keys) == 1
        sk = loaded.sub_keys[0]
        assert sk.name == "k1"
        assert sk.usage_count == 3
        assert sk.expires_at == "2027-01-01"


# ---------------------------------------------------------------------------
# Auth live view
# ---------------------------------------------------------------------------


class TestAuthView:
    def test_auth_view_propagates_mutation(self, tmp_path: Path):
        s = Settings(api_key="sk-old")
        s.auth.api_key = "sk-new"
        assert s.api_key == "sk-new"

    def test_auth_view_reads_sub_keys(self):
        s = Settings(sub_keys=[SubKeyEntry(name="a", key_hash="h", created_at="t")])
        assert s.auth.sub_keys[0].name == "a"

    def test_auth_view_skip_verification_from_global(self):
        s = Settings(global_settings={"skip_api_key_verification": True})
        assert s.auth.skip_api_key_verification is True
        assert Settings().auth.skip_api_key_verification is False


# ---------------------------------------------------------------------------
# Plaintext scrubbing
# ---------------------------------------------------------------------------


class TestPlaintextScrub:
    def test_clears_top_level_and_nested_keys(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"api_key": "sk-a", "auth": {"api_key": "sk-b"}}))
        _clear_plaintext_api_key(path)
        data = json.loads(path.read_text())
        assert data["api_key"] is None
        assert data["auth"]["api_key"] is None

    def test_noop_when_no_key(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"model_settings": {}}))
        _clear_plaintext_api_key(path)
        data = json.loads(path.read_text())
        assert "api_key" not in data

    def test_saves_still_0600_after_scrub(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"api_key": "sk-a"}))
        _clear_plaintext_api_key(path)
        assert (path.stat().st_mode & 0o777) == 0o600
