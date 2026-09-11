# SPDX-License-Identifier: Apache-2.0
"""S1.3: tool-call JSON auto-repairer tests (5 categories + ≥90% repair rate)."""

import json

import pytest

from fusion_mlx.api.tool_json_repair import (
    repair_arguments,
    repair_tool_call_json,
)


class TestFastPath:
    def test_well_formed_passes_through(self):
        raw = '{"name": "get_weather", "location": "Paris"}'
        out = repair_tool_call_json(raw)
        assert json.loads(out) == {"name": "get_weather", "location": "Paris"}

    def test_empty_returns_empty_object(self):
        assert repair_tool_call_json("") == "{}"
        assert repair_tool_call_json("   ") == "{}"

    def test_non_object_wraps_in_value(self):
        out = repair_tool_call_json("42")
        assert json.loads(out) == {"value": 42}


class TestTrailingComma:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"a": 1,}', {"a": 1}),
            ('{"a": 1, "b": 2,}', {"a": 1, "b": 2}),
            ("[1, 2, 3,]", {"value": [1, 2, 3]}),
            ('{"a": [1, 2,], "b": 3,}', {"a": [1, 2], "b": 3}),
        ],
    )
    def test_trailing_comma_removed(self, raw, expected):
        assert json.loads(repair_tool_call_json(raw)) == expected


class TestUnclosedBrackets:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"name": "foo"', {"name": "foo"}),
            ('{"a": {"b": 1', {"a": {"b": 1}}),
            ('{"a": [1, 2', {"a": [1, 2]}),
            ('{"a": {"b": [1, 2]', {"a": {"b": [1, 2]}}),
            ("[1, 2, 3", {"value": [1, 2, 3]}),
        ],
    )
    def test_unclosed_brackets_closed(self, raw, expected):
        assert json.loads(repair_tool_call_json(raw)) == expected


class TestUnclosedString:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"msg": "hello world', {"msg": "hello world"}),
            ('{"msg": "hello \\"world', {"msg": 'hello "world'}),
            ('{"a": "foo", "b": "bar', {"a": "foo", "b": "bar"}),
        ],
    )
    def test_unclosed_string_closed(self, raw, expected):
        assert json.loads(repair_tool_call_json(raw)) == expected


class TestTruncatedMidValue:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"name": "foo", "args": {"q": "search ter', {"name": "foo"}),
            ('{"a": 1, "b": "incomp', {"a": 1}),
            ('{"a": 1, "b": 2, "c": [1, 2, {"d": "xx', {"a": 1, "b": 2}),
        ],
    )
    def test_truncated_rewinds_to_last_complete(self, raw, expected):
        out = json.loads(repair_tool_call_json(raw))
        for k, v in expected.items():
            assert out.get(k) == v


class TestScalarWrap:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", {"value": True}),
            ("false", {"value": False}),
            ("null", {"value": None}),
            ("3.14", {"value": 3.14}),
            ('"bare string', {"value": "bare string"}),
        ],
    )
    def test_scalar_wrapped(self, raw, expected):
        assert json.loads(repair_tool_call_json(raw)) == expected


class TestUnrecoverable:
    def test_garbage_returns_empty_object(self):
        assert repair_tool_call_json("}{][") == "{}"

    def test_nested_garbage_returns_empty(self):
        assert repair_tool_call_json("{{{}}}{") == "{}"


class TestRepairArguments:
    def test_dict_passes_through(self):
        d = {"x": 1}
        assert json.loads(repair_arguments(d)) == d

    def test_string_repaired(self):
        assert json.loads(repair_arguments('{"a": 1,')) == {"a": 1}

    def test_none_returns_empty(self):
        assert repair_arguments(None) == "{}"

    def test_int_wrapped(self):
        assert json.loads(repair_arguments(42)) == {"value": 42}


class TestRepairRate:
    CORPUS = [
        (
            '{"name": "get_weather", "location": "Paris"}',
            {"name": "get_weather", "location": "Paris"},
        ),
        ('{"a": 1,}', {"a": 1}),
        ('{"a": 1, "b": 2,}', {"a": 1, "b": 2}),
        ('{"a": [1, 2,], "b": 3,}', {"a": [1, 2], "b": 3}),
        ('{"name": "foo"', {"name": "foo"}),
        ('{"a": {"b": 1', {"a": {"b": 1}}),
        ('{"a": [1, 2', {"a": [1, 2]}),
        ('{"a": {"b": [1, 2]', {"a": {"b": [1, 2]}}),
        ("[1, 2, 3", {"value": [1, 2, 3]}),
        ('{"msg": "hello world', {"msg": "hello world"}),
        ('{"msg": "hello \\"world', {"msg": 'hello "world'}),
        ('{"a": "foo", "b": "bar', {"a": "foo", "b": "bar"}),
        ('{"name": "foo", "args": {"q": "search ter', {"name": "foo"}),
        ('{"a": 1, "b": "incomp', {"a": 1}),
        ('{"a": 1, "b": 2, "c": [1, 2, {"d": "xx', {"a": 1, "b": 2}),
        ("true", {"value": True}),
        ("false", {"value": False}),
        ("null", {"value": None}),
        ("3.14", {"value": 3.14}),
        ('"bare string', {"value": "bare string"}),
        (
            '{"tool": "search", "query": "test", "limit": 5}',
            {"tool": "search", "query": "test", "limit": 5},
        ),
        ('{"tool": "search", "query": "test",', {"tool": "search", "query": "test"}),
        ('{"items": [1, 2, 3, 4, 5', {"items": [1, 2, 3, 4, 5]}),
        ('{"nested": {"deep": {"value": 42', {"nested": {"deep": {"value": 42}}}),
        (
            '{"name": "calc", "operands": [10, 20, 30], "op": "add"',
            {"name": "calc", "operands": [10, 20, 30], "op": "add"},
        ),
        (
            '{"path": "/usr/local/bin", "flags": ["a", "b", "c",',
            {"path": "/usr/local/bin", "flags": ["a", "b", "c"]},
        ),
        (
            '{"text": "hello \\"world\\"", "count": 3',
            {"text": 'hello "world"', "count": 3},
        ),
        (
            '{"a": "b", "c": "d", "e": "f", "g": "h',
            {"a": "b", "c": "d", "e": "f", "g": "h"},
        ),
        ('{"matrix": [[1, 2], [3, 4', {"matrix": [[1, 2], [3, 4]]}),
        (
            '{"flag": true, "data": null, "num": 0',
            {"flag": True, "data": None, "num": 0},
        ),
        ('{"esc": "line1\\nline2', {"esc": "line1\nline2"}),
    ]

    def test_repair_rate_at_least_90_percent(self):
        repaired = 0
        total = len(self.CORPUS)
        for raw, expected in self.CORPUS:
            try:
                out = json.loads(repair_tool_call_json(raw))
                if out == expected:
                    repaired += 1
            except (json.JSONDecodeError, ValueError):
                pass
        rate = repaired / total
        assert rate >= 0.90, f"repair rate {rate:.0%} < 90% ({repaired}/{total})"
