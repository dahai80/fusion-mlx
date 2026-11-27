# SPDX-License-Identifier: Apache-2.0
"""Unit tests for default-on tool-call grammar forcing (#1730).

Standalone pytest file testing fusion_mlx.api.openai.tool_grammar and the
chat.py wiring. Spec-level validation only — matcher creation is covered by
test_grammar.py against the same llguidance machinery.
"""

from types import SimpleNamespace
from unittest.mock import patch

import llguidance as lg

from fusion_mlx.api.openai import tool_grammar as tg


class FakeTok:
    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"


class FakeTokUnsupported:
    has_tool_calling = True
    tool_call_start = "<minimax:tool_call>"
    tool_call_end = "</minimax:tool_call>"


class FakeTokNoTools:
    has_tool_calling = False


def _tool(name="get_weather", city_required=True):
    params = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    if city_required:
        params["required"] = ["city"]
    return {"type": "function", "function": {"name": name, "parameters": params}}


def _request(**kw):
    base = dict(
        tools=[_tool()],
        code_mode=False,
        tool_choice="auto",
        structured_outputs=None,
        response_format=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestBuildToolGrammarSpec:
    def test_spec_parses_as_lark(self):
        spec = tg.build_tool_grammar_spec([_tool()], FakeTok())
        assert spec is not None
        assert "start:" in spec and "%json" in spec
        assert lg.grammar_from("lark", spec)

    def test_structure_begin_embeds_name_and_schema(self):
        spec = tg.build_tool_grammar_spec([_tool("my_func")], FakeTok())
        assert "my_func" in spec and "arguments" in spec
        assert "%json" in spec
        assert "}\\n" in spec

    def test_multiple_tools_one_structure_each(self):
        spec = tg.build_tool_grammar_spec([_tool("a"), _tool("b")], FakeTok())
        assert spec.count("%json") == 2
        assert '"a"' in spec.replace("\\", "")
        assert '"b"' in spec.replace("\\", "")

    def test_tool_without_parameters_gets_empty_object_schema(self):
        tool = {"type": "function", "function": {"name": "ping"}}
        spec = tg.build_tool_grammar_spec([tool], FakeTok())
        assert "'properties': {}" in spec or '"properties": {}' in spec

    def test_unsupported_markup_returns_none(self):
        assert tg.build_tool_grammar_spec([_tool()], FakeTokUnsupported()) is None

    def test_no_tool_markers_returns_none(self):
        assert tg.build_tool_grammar_spec([_tool()], FakeTokNoTools()) is None

    def test_empty_tools_returns_none(self):
        assert tg.build_tool_grammar_spec([], FakeTok()) is None


class TestCompileToolGrammar:
    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_TOOL_GRAMMAR", "0")
        assert tg.compile_tool_grammar(object(), _request()) is None

    def test_disabled_via_env_false(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_TOOL_GRAMMAR", "false")
        assert tg.compile_tool_grammar(object(), _request()) is None

    def test_enabled_default(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_TOOL_GRAMMAR", raising=False)
        assert tg.tool_grammar_enabled() is True

    def test_no_tools_skips(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_TOOL_GRAMMAR", raising=False)
        assert tg.compile_tool_grammar(object(), _request(tools=None)) is None

    def test_code_mode_skips(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_TOOL_GRAMMAR", raising=False)
        assert tg.compile_tool_grammar(object(), _request(code_mode=True)) is None

    def test_tool_choice_none_skips(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_TOOL_GRAMMAR", raising=False)
        assert tg.compile_tool_grammar(object(), _request(tool_choice="none")) is None

    def test_explicit_response_format_skips(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_TOOL_GRAMMAR", raising=False)
        assert (
            tg.compile_tool_grammar(
                object(), _request(response_format={"type": "json_object"})
            )
            is None
        )

    def test_unsupported_tokenizer_skips(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_TOOL_GRAMMAR", raising=False)
        assert tg.compile_tool_grammar(object(), _request()) is None

    def test_calls_matcher_factory_with_spec(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_TOOL_GRAMMAR", raising=False)
        engine = SimpleNamespace(
            _tokenizer=FakeTok(),
        )
        with patch(
            "fusion_mlx.api.grammar.create_llguidance_matcher",
            return_value=object(),
        ) as factory:
            matcher = tg.compile_tool_grammar(engine, _request())
        assert matcher is not None
        assert factory.call_count == 1
        args, kwargs = factory.call_args
        assert args[0] is engine._tokenizer
        assert args[1]["format"] == "lark"
        assert isinstance(args[1]["grammar"], str)
        assert "%json" in args[1]["grammar"]
