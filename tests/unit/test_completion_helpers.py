# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _completion.py shell completion helpers.

Covers:
- _is_safe_alias_name validation (printable, non-whitespace)
- _load_alias_names file I/O, cache, size cap, corrupt JSON
- alias_completer prefix matching
- alias_csv_completer CSV prefix matching
"""

from __future__ import annotations

# Load _completion directly to avoid fusion_mlx/__init__.py MLX chain
import importlib.util
import json
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "_completion", "fusion_mlx/_completion.py"
)
comp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comp)


class TestIsSafeAliasName:
    """_is_safe_alias_name validation."""

    def test_normal_name_passes(self):
        assert comp._is_safe_alias_name("Qwen3-4B-Q4_K_M") is True

    def test_empty_string_rejected(self):
        assert comp._is_safe_alias_name("") is False

    def test_non_string_rejected(self):
        assert comp._is_safe_alias_name(123) is False
        assert comp._is_safe_alias_name(None) is False
        assert comp._is_safe_alias_name([]) is False

    def test_whitespace_rejected(self):
        assert comp._is_safe_alias_name("my model") is False

    def test_control_char_rejected(self):
        assert comp._is_safe_alias_name("model\nname") is False
        assert comp._is_safe_alias_name("model\tname") is False

    def test_unicode_printable_allowed(self):
        assert comp._is_safe_alias_name("claude-4.5-sonnet") is True


class TestLoadAliasNames:
    """_load_alias_names file I/O and caching."""

    def test_missing_file_returns_empty(self, monkeypatch):
        monkeypatch.setattr(comp, "_ALIASES_PATH", Path("/nonexistent/aliases.json"))
        assert comp._load_alias_names() == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "aliases.json"
        f.write_text("{}")
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        assert comp._load_alias_names() == []
        monkeypatch.undo()

    def test_loads_valid_aliases(self, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"model-a": "path/a", "model-b": "path/b"}
        f.write_text(json.dumps(data))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        names = comp._load_alias_names()
        assert names == ["model-a", "model-b"]
        monkeypatch.undo()

    def test_returns_sorted_order(self, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"z-model": "z", "a-model": "a", "m-model": "m"}
        f.write_text(json.dumps(data))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        names = comp._load_alias_names()
        assert names == ["a-model", "m-model", "z-model"]
        monkeypatch.undo()

    def test_cache_hit_on_same_file(self, tmp_path):
        """Without file changes, cache returns same result."""
        f = tmp_path / "aliases.json"
        data = {"only-model": "path"}
        f.write_text(json.dumps(data))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        comp._CACHE = None  # reset

        # First call loads and caches
        first = comp._load_alias_names()
        assert first == ["only-model"]

        # Second call on same file (no changes) should use cache
        second = comp._load_alias_names()
        assert second == ["only-model"]  # still cached!

        monkeypatch.undo()
        comp._CACHE = None

    def test_cache_invalidated_on_mtime_change(self, tmp_path):
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps({"old": "path"}))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        comp._CACHE = None

        comp._load_alias_names()  # prime cache

        # Change content and touch
        f.write_text(json.dumps({"new": "path"}))
        os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 1))

        names = comp._load_alias_names()
        assert names == ["new"]  # re-read because mtime changed

        monkeypatch.undo()
        comp._CACHE = None

    def test_size_capped_file_rejected(self, tmp_path):
        f = tmp_path / "aliases.json"
        # Write data just over MAX_ALIASES_BYTES
        big_data = {f"{'x' * 1000}_{i}": "y" for i in range(2000)}
        f.write_text(json.dumps(big_data))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        monkeypatch.setattr(comp, "_MAX_ALIASES_BYTES", 100)
        names = comp._load_alias_names()
        assert names == []  # file too large
        monkeypatch.undo()

    def test_corrupt_json_returns_empty(self, tmp_path):
        f = tmp_path / "aliases.json"
        f.write_bytes(b"{broken json")
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        names = comp._load_alias_names()
        assert names == []
        monkeypatch.undo()

    def test_non_dict_json_returns_empty(self, tmp_path):
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps(["list", "not", "dict"]))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        names = comp._load_alias_names()
        assert names == []
        monkeypatch.undo()

    def test_unsafe_names_filtered_out(self, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"good-name": "a", "bad name": "b", "bad\nname": "c"}
        f.write_text(json.dumps(data))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        names = comp._load_alias_names()
        assert names == ["good-name"]
        monkeypatch.undo()


class TestAliasCompleter:
    """alias_completer prefix matching."""

    def test_empty_prefix_returns_all(self, monkeypatch, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"abc": "1", "abd": "2", "xyz": "3"}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        comp._CACHE = None

        result = comp.alias_completer(prefix="")
        assert result == ["abc", "abd", "xyz"]

        monkeypatch.undo()
        comp._CACHE = None

    def test_prefix_filters(self, monkeypatch, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"model-a": "1", "model-b": "2", "other": "3"}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        comp._CACHE = None

        result = comp.alias_completer(prefix="model")
        assert result == ["model-a", "model-b"]

        monkeypatch.undo()
        comp._CACHE = None

    def test_no_match_returns_empty(self, monkeypatch, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"abc": "1"}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        comp._CACHE = None

        result = comp.alias_completer(prefix="xyz")
        assert result == []

        monkeypatch.undo()
        comp._CACHE = None


class TestAliasCsvCompleter:
    """alias_csv_completer handles comma-separated prefixes."""

    def test_csv_last_segment_matches(self, monkeypatch, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"qwen-4b": "1", "qwen-8b": "2", "gemma-2b": "3"}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        comp._CACHE = None

        # prefix "qwen" after "model-a," should match qwen-4b and qwen-8b
        result = comp.alias_csv_completer(prefix="model-a,qwen")
        assert result == ["model-a,qwen-4b", "model-a,qwen-8b"]

        monkeypatch.undo()
        comp._CACHE = None

    def test_csv_single_segment_falls_through(self, monkeypatch, tmp_path):
        f = tmp_path / "aliases.json"
        data = {"model-x": "1"}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(comp, "_ALIASES_PATH", f)
        comp._CACHE = None

        # No comma: falls through to alias_completer
        result = comp.alias_csv_completer(prefix="model")
        assert result == ["model-x"]

        monkeypatch.undo()
        comp._CACHE = None
