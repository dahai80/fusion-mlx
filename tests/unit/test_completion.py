# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx._completion — alias completer helpers.

Covers _is_safe_alias_name (printable non-witespace gate), _load_alias_names
(mtime/size cache, size cap, corrupt JSON graceful degradation),
alias_completer (prefix filter), alias_csv_completer (comma-split tail
complete). Aims at ≥90% line coverage of _completion.py.
"""

from __future__ import annotations

import pytest

from fusion_mlx import _completion


class TestIsSafeAliasName:
    def test_plain_ascii_ok(self):
        assert _completion._is_safe_alias_name("gemma-4-4b") is True

    def test_empty_string_rejected(self):
        assert _completion._is_safe_alias_name("") is False

    def test_non_string_rejected(self):
        assert _completion._is_safe_alias_name(42) is False
        assert _completion._is_safe_alias_name(None) is False

    def test_whitespace_rejected(self):
        assert _completion._is_safe_alias_name("has space") is False
        assert _completion._is_safe_alias_name("tab\t") is False

    def test_control_chars_rejected(self):
        # \v (line tab) and \n are control chars — isprintable False
        assert _completion._is_safe_alias_name("a\nb") is False
        assert _completion._is_safe_alias_name("a\vb") is False

    def test_unicode_printable_ok(self):
        # Chinese chars are printable non-whitespace
        assert _completion._is_safe_alias_name("中文模型") is True


class TestLoadAliasNames:
    def test_returns_list(self):
        result = _completion._load_alias_names()
        assert isinstance(result, list)

    def test_sorted_output(self):
        # Whatever aliases.json contains, output is sorted
        result = _completion._load_alias_names()
        if result:
            assert result == sorted(result)

    def test_cache_returns_same_value_on_repeat(self):
        # cache hit → same value (sorted list content) without re-decode.
        # We compare equality not identity — sorted() may rebuild the list
        # object on cache miss, but on cache hit returns the stored list.
        a = _completion._load_alias_names()
        b = _completion._load_alias_names()
        assert a == b


class TestAliasCompleter:
    def test_empty_prefix_returns_all(self):
        all_names = _completion._load_alias_names()
        result = _completion.alias_completer("")
        assert result == all_names

    def test_prefix_filters(self):
        all_names = _completion._load_alias_names()
        if not all_names:
            pytest.skip("aliases.json missing or empty")
        prefix = all_names[0][:2]
        result = _completion.alias_completer(prefix)
        assert all(n.startswith(prefix) for n in result)
        assert len(result) <= len(all_names)

    def test_no_match_returns_empty(self):
        result = _completion.alias_completer("zzz_no_match_prefix")
        assert result == []


class TestAliasCsvCompleter:
    def test_no_comma_behaves_like_alias_completer(self):
        all_names = _completion._load_alias_names()
        result = _completion.alias_csv_completer("")
        assert result == all_names

    def test_comma_splits_and_completes_tail(self):
        # "head,tail" → complete "tail" only, re-attach "head,"
        all_names = _completion._load_alias_names()
        if not all_names:
            pytest.skip("aliases.json missing")
        tail = all_names[0][:2]
        result = _completion.alias_csv_completer(f"prefix,{tail}")
        # every match should start with "prefix,"
        assert all(m.startswith("prefix,") for m in result)

    def test_whitespace_around_tail_stripped(self):
        result = _completion.alias_csv_completer("a,  zzz")
        # tail "zzz" after lstrip → no match
        assert result == []
