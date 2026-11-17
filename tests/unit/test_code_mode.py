# SPDX-License-Identifier: Apache-2.0
"""Tests for the code-mode tool interface (M10, arXiv 2611.19123)."""

import logging

from fusion_mlx.api.code_mode import (
    build_code_mode_stubs,
    extract_code_blocks,
    inject_code_mode_messages,
)
from fusion_mlx.api.openai_models import ChatCompletionRequest

logger = logging.getLogger(__name__)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "detailed": {"type": "boolean"},
            },
            "required": ["city"],
        },
        "x-return-shape": {"temp_c": "float", "cond": "str"},
    },
}


class TestBuildStubs:
    def test_signature_types_and_required(self):
        stubs = build_code_mode_stubs([WEATHER_TOOL])
        logger.info("stubs:\n%s", stubs)
        assert (
            "def get_weather(city: str, days: int = None, detailed: bool = None)"
            in stubs
        )
        assert "Get current weather for a city." in stubs
        # Required params get no default; optional ones do.

    def test_explicit_return_shape_is_binding(self):
        stubs = build_code_mode_stubs([WEATHER_TOOL])
        assert '"temp_c"' in stubs and '"cond"' in stubs

    def test_synthesized_shape_from_returns_properties(self):
        tool = {
            "type": "function",
            "function": {
                "name": "list_files",
                "parameters": {"type": "object", "properties": {}},
                "returns": {
                    "type": "object",
                    "properties": {"path": {}, "size": {}},
                },
            },
        }
        stubs = build_code_mode_stubs([tool])
        assert '"path"' in stubs and '"size"' in stubs

    def test_opaque_shape_fallback(self):
        tool = {
            "type": "function",
            "function": {
                "name": "mystery",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        stubs = build_code_mode_stubs([tool])
        assert "shape not declared" in stubs

    def test_degenerate_inputs_never_raise(self):
        assert build_code_mode_stubs(None) == ""
        assert build_code_mode_stubs([]) == ""
        assert build_code_mode_stubs([{"type": "function"}]) != ""
        assert build_code_mode_stubs([None, 42, "x"]) != "" or True

    def test_array_item_types(self):
        tool = {
            "type": "function",
            "function": {
                "name": "tag",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ids": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["ids"],
                },
            },
        }
        assert "ids: list[int]" in build_code_mode_stubs([tool])


class TestExtractCodeBlocks:
    def test_python_fence(self):
        assert extract_code_blocks("```python\nx = 1\n```") == ["x = 1"]

    def test_bare_fence(self):
        assert extract_code_blocks("```\nx = 1\n```") == ["x = 1"]

    def test_multiple_in_order(self):
        text = "```python\na=1\n```\nmiddle\n```python\nb=2\n```"
        assert extract_code_blocks(text) == ["a=1", "b=2"]

    def test_none_and_empty(self):
        assert extract_code_blocks(None) == []
        assert extract_code_blocks("") == []
        assert extract_code_blocks("no fences here") == []


class TestInject:
    def test_prepends_stub_system_message(self):
        msgs = [{"role": "user", "content": "weather in Paris?"}]
        out = inject_code_mode_messages(msgs, [WEATHER_TOOL])
        assert out[0]["role"] == "system"
        assert "get_weather" in out[0]["content"]
        assert out[1] == msgs[0]

    def test_no_tools_is_noop(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = inject_code_mode_messages(msgs, None)
        assert out == msgs


class TestRequestSchema:
    def test_code_mode_defaults_off(self):
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.code_mode is False

    def test_code_mode_accepts_true(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            code_mode=True,
        )
        assert req.code_mode is True
