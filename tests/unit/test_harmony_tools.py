# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.api.harmony_tools — JSON Schema → TypeScript converter."""

from __future__ import annotations

from fusion_mlx.api.harmony_tools import convert_tools_to_typescript


class TestConvertEmpty:
    def test_none_returns_none(self):
        assert convert_tools_to_typescript(None) is None

    def test_empty_list_returns_none(self):
        assert convert_tools_to_typescript([]) is None

    def test_no_function_tools_returns_none(self):
        assert convert_tools_to_typescript([{"type": "other"}]) is None

    def test_function_without_name_skipped(self):
        tools = [{"type": "function", "function": {"description": "x"}}]
        assert convert_tools_to_typescript(tools) is None


class TestSimpleFunction:
    def test_function_with_no_params(self):
        ts = convert_tools_to_typescript(
            [{"type": "function", "function": {"name": "ping"}}]
        )
        assert "namespace functions {" in ts
        assert "type ping = () => any;" in ts

    def test_function_with_description(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {"name": "ping", "description": "Ping the server"},
                }
            ]
        )
        assert "// Ping the server" in ts

    def test_required_param(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ]
        )
        assert "location: string" in ts
        assert "location?" not in ts

    def test_optional_param(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get",
                        "parameters": {
                            "properties": {"unit": {"type": "string"}},
                        },
                    },
                }
            ]
        )
        assert "unit?: string" in ts


class TestTypeConversion:
    def test_enum_union(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "properties": {"u": {"type": "string", "enum": ["c", "f"]}}
                        },
                    },
                }
            ]
        )
        assert '"c" | "f"' in ts

    def test_array_with_items(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "properties": {
                                "arr": {"type": "array", "items": {"type": "number"}}
                            }
                        },
                    },
                }
            ]
        )
        assert "Array<number>" in ts

    def test_array_without_items(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"properties": {"arr": {"type": "array"}}},
                    },
                }
            ]
        )
        assert "Array<any>" in ts

    def all_type_mappings(self):
        for jtype, ts_type in [
            ("string", "string"),
            ("number", "number"),
            ("integer", "number"),
            ("boolean", "boolean"),
            ("null", "null"),
            ("object", "object"),
        ]:
            ts = convert_tools_to_typescript(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"properties": {"p": {"type": jtype}}},
                        },
                    }
                ]
            )
            assert f"p: {ts_type}" in ts, f"failed {jtype}→{ts_type}"

    def test_unknown_type_falls_to_any(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"properties": {"p": {"type": "weird"}}},
                    },
                }
            ]
        )
        # param not in required → optional "?" suffix: "p?: any"
        assert "p?: any" in ts

    def test_missing_type_falls_to_any(self):
        ts = convert_tools_to_typescript(
            [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"properties": {"p": {}}}},
                }
            ]
        )
        assert "p?: any" in ts


class TestMultipleFunctions:
    def test_two_functions_in_namespace(self):
        ts = convert_tools_to_typescript(
            [
                {"type": "function", "function": {"name": "a"}},
                {"type": "function", "function": {"name": "b"}},
            ]
        )
        assert "type a = () => any;" in ts
        assert "type b = () => any;" in ts
        assert ts.count("namespace functions {") == 1
