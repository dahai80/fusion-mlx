from __future__ import annotations

import json
import logging

import pytest

from fusion_mlx.tool_parsers import LlamaToolParser

logger = logging.getLogger(__name__)


@pytest.fixture
def parser() -> LlamaToolParser:
    return LlamaToolParser()


class TestXmlWrapperShape:
    def test_single_call(self, parser: LlamaToolParser):
        text = '<function=multiply>{"x": 3, "y": 4}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "multiply"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"x": 3, "y": 4}

    def test_multiple_calls(self, parser: LlamaToolParser):
        text = '<function=add>{"a": 1}</function><function=multiply>{"x": 3}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert [tc["name"] for tc in result.tool_calls] == ["add", "multiply"]

    def test_content_before(self, parser: LlamaToolParser):
        text = 'Computing result<function=calc>{"n": 5}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Computing result"


class TestPythonTagShape:
    def test_basic(self, parser: LlamaToolParser):
        text = '<|python_tag|>{"name": "web_search", "parameters": {"query": "你好"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "web_search"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "你好"}
        assert result.content is None

    def test_tag_with_preceding_prose(self, parser: LlamaToolParser):
        text = (
            "Sure, let me search.<|python_tag|>"
            '{"name": "web_search", "parameters": {"query": "weather"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Sure, let me search."

    def test_tag_with_arguments_alias(self, parser: LlamaToolParser):
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}


class TestBareJsonShape:
    def test_f008_repro_chinese_greeting(self, parser: LlamaToolParser):
        text = '{"name": "web_search", "parameters": {"query": "你好"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called, (
            "Bare JSON tool-call leaked as content — F-008 regression"
        )
        assert result.tool_calls[0]["name"] == "web_search"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "你好"}
        assert result.content is None

    def test_bare_json_with_arguments_alias(self, parser: LlamaToolParser):
        text = '{"name": "calc", "arguments": {"n": 5}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"n": 5}

    def test_bare_json_no_arg_uses_empty_parameters(self, parser: LlamaToolParser):
        text = '{"name": "get_current_time", "parameters": {}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {}

    def test_bare_json_with_nested_args(self, parser: LlamaToolParser):
        args = {"filter": {"city": "Tokyo", "limit": 10, "tags": ["a", "b"]}}
        text = json.dumps({"name": "search", "parameters": args["filter"]})
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == args["filter"]

    def test_bare_json_preserves_prefix_content(self, parser: LlamaToolParser):
        text = (
            "Let me look that up. "
            '{"name": "web_search", "parameters": {"query": "weather"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Let me look that up."

    def test_string_with_brace_inside(self, parser: LlamaToolParser):
        text = '{"name": "echo", "parameters": {"msg": "hello } world"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"msg": "hello } world"}


class TestNoFalsePositives:
    def test_plain_prose_is_not_a_tool_call(self, parser: LlamaToolParser):
        text = "Hello! How can I help you today?"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_prose_json_without_name_key(self, parser: LlamaToolParser):
        text = 'Here is an example: {"city": "Tokyo", "temp": 22}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_prose_json_with_name_but_no_args(self, parser: LlamaToolParser):
        text = 'Here is a user: {"name": "Alice", "age": 30}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_json_with_empty_name(self, parser: LlamaToolParser):
        text = '{"name": "", "parameters": {}}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_malformed_json_left_as_content(self, parser: LlamaToolParser):
        text = '{"name": "broken", "parameters": {oops'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_empty_input(self, parser: LlamaToolParser):
        result = parser.extract_tool_calls("")
        assert not result.tools_called

    def test_plain_long_prefix_pending_fast_path_skips_json_scan(self, monkeypatch):
        import fusion_mlx.tool_parsers.llama_tool_parser as llama_mod

        parser = LlamaToolParser()

        def fail_json_scan(*_args, **_kwargs):
            raise AssertionError("plain text should not enter JSON scanner")

        monkeypatch.setattr(llama_mod, "_find_top_level_json_object", fail_json_scan)

        text = ""
        for _ in range(128):
            text += "ordinary assistant prose without anchors "
            assert parser.has_pending_tool_call(text) is False

    def test_plain_prefix_cache_reset_does_not_cross_request_boundary(self):
        parser = LlamaToolParser()
        first = "ordinary assistant prose " * 32
        second = 'different prefix {"name": "search"'

        assert parser.has_pending_tool_call(first) is False
        parser.reset()
        assert parser.has_pending_tool_call(second) is True


class TestStreaming:
    def test_bare_json_streams_content_until_close(self, parser: LlamaToolParser):
        partial = '{"name": "web_search", "parameters": {"query": "wea'
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=partial,
            delta_text=partial,
        )
        assert result is None or "tool_calls" not in result

    def test_first_brace_token_is_pending_not_content(self, parser: LlamaToolParser):
        for delta in ("{", '{"', '{"na'):
            result = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=delta,
                delta_text=delta,
            )
            assert result is None or "content" not in result, (
                f"streaming leaked partial JSON prefix as content: "
                f"delta={delta!r} -> {result!r}"
            )

    def test_streaming_buffered_prefix_flushed_if_not_tool(
        self, parser: LlamaToolParser
    ):
        previous = '{"city": "Tokyo", "temp"'
        mid = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=previous,
            delta_text=previous,
        )
        assert mid is None or "content" not in mid

        delta = ": 22}"
        current = previous + delta
        final = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert final is not None
        assert "content" in final
        assert final["content"] == current

    def test_bare_json_emits_tool_calls_on_close(self, parser: LlamaToolParser):
        previous = '{"name": "web_search", "parameters": {"query": "weather"'
        delta = "}}"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_python_tag_emits_on_close(self, parser: LlamaToolParser):
        previous = (
            '<|python_tag|>{"name": "web_search", "parameters": {"query": "weather"'
        )
        delta = "}}"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_xml_wrapper_still_works_streaming(self, parser: LlamaToolParser):
        previous = '<function=add>{"a": 1}'
        delta = "</function>"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "add"

    def test_plain_content_streams_as_content(self, parser: LlamaToolParser):
        delta = "Hello there"
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
        )
        assert result == {"content": delta}

    def test_prose_prefix_then_bare_json_tool_call(self, parser: LlamaToolParser):
        d1 = "Let me check. "
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 == {"content": d1}

        d2 = '{"name": "search", "parameters": {}}'
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        assert r2 is not None
        assert "tool_calls" in r2
        assert r2["tool_calls"][0]["function"]["name"] == "search"

    def test_post_json_content_streams_through(self, parser: LlamaToolParser):
        d1 = '{"city": "Tokyo"}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is not None
        assert "content" in r1
        assert r1["content"] == d1

        d2 = " is nice."
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        assert r2 == {"content": d2}


class TestCodexR3Regressions:
    def test_no_reemit_after_tool_close(self, parser: LlamaToolParser):
        prev = '{"name": "a", "parameters": {}}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=prev, delta_text=prev
        )
        assert r1 is not None and "tool_calls" in r1

        delta = " some prose"
        r2 = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev + delta,
            delta_text=delta,
        )
        assert r2 == {"content": " some prose"}

    def test_back_to_back_bare_json_tool_calls(self, parser: LlamaToolParser):
        d1 = '{"name": "a", "parameters": {}}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is not None and "tool_calls" in r1
        assert r1["tool_calls"][0]["function"]["name"] == "a"
        assert r1["tool_calls"][0]["index"] == 0

        d2 = '{"name": "b", "parameters": {}}'
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        assert r2 is not None and "tool_calls" in r2
        assert len(r2["tool_calls"]) == 1
        assert r2["tool_calls"][0]["function"]["name"] == "b"
        assert r2["tool_calls"][0]["index"] == 1

    def test_idempotent_double_call_on_same_prefix(self, parser: LlamaToolParser):
        prev = '{"name": "a", "parameters": {}}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=prev, delta_text=prev
        )
        assert r1 is not None and "tool_calls" in r1

        r1_again = parser.extract_tool_calls_streaming(
            previous_text=prev, current_text=prev, delta_text=""
        )
        assert r1_again is None

    @pytest.mark.parametrize(
        "split_at",
        [1, 2, 4, 6, 8, 10, 12],
    )
    def test_python_tag_split_across_deltas(
        self, parser: LlamaToolParser, split_at: int
    ):
        full = '<|python_tag|>{"name": "x", "parameters": {}}'
        d1 = full[:split_at]
        d2 = full[split_at:]

        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is None or "content" not in r1

        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=full, delta_text=d2
        )
        assert r2 is not None and "tool_calls" in r2
        assert r2["tool_calls"][0]["function"]["name"] == "x"

    @pytest.mark.parametrize("split_at", [1, 3, 5, 8])
    def test_function_open_split_across_deltas(
        self, parser: LlamaToolParser, split_at: int
    ):
        full = '<function=greet>{"name": "Alice"}</function>'
        d1 = full[:split_at]
        d2 = full[split_at:]

        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is None or "content" not in r1

        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=full, delta_text=d2
        )
        assert r2 is not None and "tool_calls" in r2
        assert r2["tool_calls"][0]["function"]["name"] == "greet"

    def test_xml_arg_value_contains_function_closer(self, parser: LlamaToolParser):
        text = '<function=echo>{"msg": "close }</function> trick"}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "echo"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "msg": "close }</function> trick"
        }
        assert result.content is None

    def test_flush_held_content_at_stream_end(self, parser: LlamaToolParser):
        assert parser.flush_held_content("abc<|python") == "<|python"
        assert parser.flush_held_content("hello world") == ""

    def test_mixed_content_and_tool_in_one_delta_returns_both(
        self, parser: LlamaToolParser
    ):
        cur = 'Let me check. {"name": "search", "parameters": {}}'
        r = parser.extract_tool_calls_streaming(
            previous_text="", current_text=cur, delta_text=cur
        )
        assert r is not None
        assert r.get("content") == "Let me check. "
        assert "tool_calls" in r
        assert r["tool_calls"][0]["function"]["name"] == "search"
        assert r["tool_calls"][0]["index"] == 0

    def test_mixed_tool_then_trailing_content_in_one_delta(
        self, parser: LlamaToolParser
    ):
        cur = '{"name": "a", "parameters": {}} tail'
        r = parser.extract_tool_calls_streaming(
            previous_text="", current_text=cur, delta_text=cur
        )
        assert r is not None
        assert "tool_calls" in r
        assert r["tool_calls"][0]["function"]["name"] == "a"
        assert r.get("content") == " tail"

    def test_has_pending_on_prose_with_unclosed_brace(self, parser: LlamaToolParser):
        assert parser.has_pending_tool_call("Let me check. {")
        assert parser.has_pending_tool_call('Let me check. {"na')
        assert not parser.has_pending_tool_call('Result: {"x": 1}')
        assert not parser.has_pending_tool_call("Hello world")

    def test_has_pending_on_prose_with_whitespace_before_name(
        self, parser: LlamaToolParser
    ):
        assert parser.has_pending_tool_call(
            'Let me check. { "name": "search", "parameters": {}}'
        )
        assert parser.has_pending_tool_call(
            'Calling tool:\n{\n  "name": "search",\n  "parameters": {}\n}'
        )
        assert parser.has_pending_tool_call(
            '{"type": "function", "name": "search", "parameters": {}}'
        )

    def test_streaming_postprocessor_invariant_violation(self, parser: LlamaToolParser):
        previous = "old prefix "
        delta = "new tail"
        current = "DIFFERENT old prefix " + delta
        r = parser.extract_tool_calls_streaming(
            previous_text=previous, current_text=current, delta_text=delta
        )
        if r is not None:
            assert "content" in r
            assert isinstance(r["content"], str)


def test_expected_wire_formats_declared():
    fmts = LlamaToolParser.EXPECTED_WIRE_FORMATS
    assert "function_bare" in fmts
    assert "llama_python_tag" in fmts
    assert "raw_json" in fmts
