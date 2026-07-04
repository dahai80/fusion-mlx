import json
import logging

import pytest

from fusion_mlx.api.harmony_tools import convert_tools_to_typescript
from fusion_mlx.reasoning import get_parser
from fusion_mlx.reasoning.harmony_parser import HarmonyReasoningParser
from fusion_mlx.tool_parsers import ToolParserManager
from fusion_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

logger = logging.getLogger(__name__)


class TestHarmonyToolParser:

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_registration(self):
        assert ToolParserManager.get_tool_parser("harmony") is HarmonyToolParser
        assert ToolParserManager.get_tool_parser("gpt-oss") is HarmonyToolParser

    def test_single_tool_call(self, parser):
        text = (
            "<|start|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "San Francisco", "unit": "celsius"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

    def test_tool_call_with_analysis_and_final(self, parser):
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>The user wants weather. I should call get_weather.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_final_response_only(self, parser):
        text = (
            "<|start|>\n"
            "<|channel|>final\n"
            "<|message|>The weather in San Francisco is 72F and sunny!\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "The weather in San Francisco is 72F and sunny!"

    def test_multiple_tool_calls(self, parser):
        text = (
            "<|start|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.get_time\n"
            "<|constrain|>json\n"
            '<|message|>{"timezone": "PST"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_tool_call_without_constrain(self, parser):
        text = (
            "<|channel|>commentary to=functions.simple_func\n"
            '<|message|>{"arg": "value"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "simple_func"

    def test_malformed_json_arguments(self, parser):
        text = (
            "<|channel|>commentary to=functions.broken_func\n"
            "<|constrain|>json\n"
            "<|message|>{invalid json here}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "broken_func"
        assert result.tool_calls[0]["arguments"] == "{invalid json here}"

    def test_tool_call_with_final_content(self, parser):
        text = (
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"query": "python"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>Here are the results.\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content == "Here are the results."

    def test_empty_input(self, parser):
        result = parser.extract_tool_calls("")
        assert not result.tools_called
        assert result.tool_calls == []

    def test_plain_text_input(self, parser):
        result = parser.extract_tool_calls("Just a regular response.")
        assert not result.tools_called
        assert result.content == "Just a regular response."

    def test_unique_tool_ids(self, parser):
        text = (
            "<|channel|>commentary to=functions.func_a\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>\n"
            "<|channel|>commentary to=functions.func_b\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        ids = [tc["id"] for tc in result.tool_calls]
        assert len(set(ids)) == 2
        assert all(id_.startswith("call_") for id_ in ids)

    def test_nested_json_arguments(self, parser):
        args = {"filter": {"type": "range", "min": 0, "max": 100}, "sort": "asc"}
        text = (
            "<|channel|>commentary to=functions.query\n"
            "<|constrain|>json\n"
            f"<|message|>{json.dumps(args)}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        parsed_args = json.loads(result.tool_calls[0]["arguments"])
        assert parsed_args["filter"]["type"] == "range"

    def test_streaming_no_tool_markers(self, parser):
        result = parser.extract_tool_calls_streaming("", "Hello", "Hello")
        assert result == {"content": "Hello"}

    def test_streaming_tool_call_complete(self, parser):
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls_streaming("", current, "<|call|>")

        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "func"

    def test_streaming_building_tool_call(self, parser):
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a":'
        )
        result = parser.extract_tool_calls_streaming("", current, '{"a":')
        assert result is None


class TestHarmonyReasoningParser:

    @pytest.fixture()
    def parser(self):
        return HarmonyReasoningParser()

    def test_registration(self):
        parser_cls = get_parser("harmony")
        assert parser_cls is HarmonyReasoningParser

    def test_extract_analysis_and_final(self, parser):
        output = (
            "<|channel|>analysis\n"
            "<|message|>Let me think step by step.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>The answer is 42.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me think step by step."
        assert content == "The answer is 42."

    def test_extract_final_terminated_by_end_token(self, parser):
        output = (
            "<|channel|>analysis<|message|>The user asks 2+2.<|end|>"
            "<|channel|>final<|message|>4<|end|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "The user asks 2+2."
        assert content == "4"

    def test_literal_end_token_in_content_does_not_truncate_return_terminated(
        self, parser
    ):
        output = "<|channel|>final<|message|>prefix <|end|> suffix<|return|>"
        _, content = parser.extract_reasoning(output)

        assert content == "prefix <|end|> suffix"

    def test_literal_end_token_in_content_end_only_terminator(self, parser):
        output = "<|channel|>final<|message|>prefix <|end|> suffix<|end|>"
        _, content = parser.extract_reasoning(output)

        assert content == "prefix <|end|> suffix"

    def test_multiple_analysis_blocks(self, parser):
        output = (
            "<|channel|>analysis\n"
            "<|message|>First thought.\n"
            "<|end|>\n"
            "<|channel|>analysis\n"
            "<|message|>Second thought.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>Result.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert "First thought." in reasoning
        assert "Second thought." in reasoning
        assert content == "Result."

    def test_no_analysis_channel(self, parser):
        output = "<|channel|>final\n<|message|>Direct answer.\n<|return|>"
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is None
        assert content == "Direct answer."

    def test_analysis_only_no_final(self, parser):
        output = "<|channel|>analysis\n<|message|>Just thinking...\n<|end|>"
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Just thinking..."
        assert content is None

    def test_empty_input(self, parser):
        reasoning, content = parser.extract_reasoning("")
        assert reasoning is None
        assert content is None

    def test_analysis_with_commentary_and_final(self, parser):
        output = (
            "<|channel|>analysis\n"
            "<|message|>Need to call a tool.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"q": "test"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>Found results.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Need to call a tool."
        assert content == "Found results."

    def test_streaming_analysis_to_final(self, parser):
        parser.reset_state()

        r1 = parser.extract_reasoning_streaming(
            "", "<|channel|>analysis\n", "<|channel|>analysis\n"
        )
        assert r1 is None

        r2 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n",
            "<|channel|>analysis\n<|message|>",
            "<|message|>",
        )
        assert r2 is None

        r3 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>",
            "<|channel|>analysis\n<|message|>Thinking",
            "Thinking",
        )
        assert r3 is not None
        assert r3.reasoning == "Thinking"
        assert r3.content is None

        r4 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking",
            "<|channel|>analysis\n<|message|>Thinking<|end|>",
            "<|end|>",
        )
        assert r4 is None

        r5 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking<|end|>",
            "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n",
            "\n<|channel|>final\n",
        )
        assert r5 is None

        prev = "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n<|message|>"
        parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n",
            prev,
            "<|message|>",
        )
        r6 = parser.extract_reasoning_streaming(
            prev,
            prev + "Answer",
            "Answer",
        )
        assert r6 is not None
        assert r6.content == "Answer"
        assert r6.reasoning is None

    def test_streaming_reset(self, parser):
        parser._current_channel = "analysis"
        parser._in_message = True
        parser.reset_state()
        assert parser._current_channel is None
        assert parser._in_message is False

    def test_streaming_commentary_passed_through(self, parser):
        parser.reset_state()

        r = parser.extract_reasoning_streaming(
            "",
            "<|channel|>commentary to=functions.f\n",
            "<|channel|>commentary to=functions.f\n",
        )
        assert r is not None
        assert r.content == "<|channel|>commentary to=functions.f\n"

        r = parser.extract_reasoning_streaming(
            "<|channel|>commentary to=functions.f\n",
            "<|channel|>commentary to=functions.f\n<|message|>",
            "<|message|>",
        )
        assert r is not None
        assert r.content == "<|message|>"

        r = parser.extract_reasoning_streaming(
            "<|channel|>commentary to=functions.f\n<|message|>",
            '<|channel|>commentary to=functions.f\n<|message|>{"a":1}',
            '{"a":1}',
        )
        assert r is not None
        assert r.content == '{"a":1}'


class TestHarmonyToolDefinitionConverter:

    def test_simple_tool(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "namespace functions" in result
        assert "get_weather" in result
        assert "location: string," in result
        assert "// Get weather for a location" in result

    def test_optional_parameters(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "required_param": {"type": "string"},
                            "optional_param": {"type": "number"},
                        },
                        "required": ["required_param"],
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "required_param: string," in result
        assert "optional_param?: number," in result

    def test_enum_type(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_unit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert '"celsius" | "fahrenheit"' in result

    def test_multiple_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func_a",
                    "description": "First function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "func_b",
                    "description": "Second function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_typescript(tools)

        assert "func_a" in result
        assert "func_b" in result
        assert "// First function" in result
        assert "// Second function" in result

    def test_no_tools(self):
        assert convert_tools_to_typescript(None) is None
        assert convert_tools_to_typescript([]) is None

    def test_no_parameters(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "type ping = () => any;" in result

    def test_array_type(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "Array<string>" in result

    def test_boolean_and_integer_types(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "config",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "count": {"type": "integer"},
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "enabled?: boolean," in result
        assert "count?: number," in result

    def test_no_description(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "no_desc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "//" not in result
        assert "no_desc" in result

    def test_skips_non_function_types(self):
        tools = [
            {"type": "retrieval"},
            {
                "type": "function",
                "function": {
                    "name": "real_func",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_typescript(tools)

        assert "real_func" in result
        assert "retrieval" not in result


class TestHarmonyEnginePipeline:

    def test_commentary_only_output_extracts_tool_call(self):
        from fusion_mlx.api.utils import clean_output_text

        raw = (
            "<|channel|>analysis<|message|>We need to call get_weather "
            'with city "Tokyo".<|end|>'
            "<|start|>assistant<|channel|>commentary "
            "to=functions.get_weather <|constrain|>json<|message|>"
            '{"city":"Tokyo"}'
        )

        engine_text = clean_output_text(raw)
        assert "<|channel|>commentary" in engine_text, (
            "Engine layer stripped commentary structure — tool parser "
            "will see plain text and extract zero calls. "
            f"Got: {engine_text!r}"
        )

        parser = HarmonyToolParser()
        result = parser.extract_tool_calls(engine_text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Tokyo"

    def test_final_only_output_still_strips_normally(self):
        from fusion_mlx.api.utils import clean_output_text

        raw = (
            "<|channel|>analysis<|message|>thinking<|end|>"
            "<|channel|>final<|message|>The answer is 42.<|return|>"
        )
        cleaned = clean_output_text(raw)
        assert cleaned == "The answer is 42."

    def test_hyphenated_tool_name_extracted(self):
        from fusion_mlx.api.utils import clean_output_text

        raw = (
            "<|channel|>analysis<|message|>thinking<|end|>"
            "<|channel|>commentary to=functions.get-weather "
            '<|constrain|>json<|message|>{"city":"Tokyo"}'
        )
        cleaned = clean_output_text(raw)
        assert "to=functions.get-weather" in cleaned, (
            "Engine-layer guard failed to detect hyphenated tool name "
            f"(cleaned={cleaned!r})"
        )
        parser = HarmonyToolParser()
        result = parser.extract_tool_calls(cleaned)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get-weather"
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Tokyo"

    def test_commentary_isolation_from_trailing_text(self):
        from fusion_mlx.api.utils import clean_output_text

        raw = (
            "<|channel|>analysis<|message|>SHOULD_NOT_LEAK<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city":"Tokyo"}'
        )
        cleaned = clean_output_text(raw)
        parser = HarmonyToolParser()
        result = parser.extract_tool_calls(cleaned)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"city": "Tokyo"}, f"args payload leaked trailing text: {args!r}"
        assert "SHOULD_NOT_LEAK" not in result.tool_calls[0]["arguments"]
        assert "<|channel|>analysis" in cleaned
        assert "SHOULD_NOT_LEAK" in cleaned


class TestHarmonyEdgeCases:

    def test_tool_parser_unterminated_call_is_now_parsed(self):
        parser = HarmonyToolParser()
        text = '<|channel|>commentary to=functions.func\n<|message|>{"arg": "value"}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "func"
        assert json.loads(result.tool_calls[0]["arguments"])["arg"] == "value"

    def test_tool_parser_unicode_content(self):
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.translate\n"
            "<|constrain|>json\n"
            '<|message|>{"text": "日本語テスト"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["text"] == "日本語テスト"

    def test_reasoning_parser_unicode_content(self):
        parser = HarmonyReasoningParser()
        output = (
            "<|channel|>analysis\n"
            "<|message|>让我想想...\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>答案是42。\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "让我想想..."
        assert content == "答案是42。"

    def test_mixed_channels_full_flow(self):
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>Think 1.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"q": "test"}\n'
            "<|call|>\n"
            "<|channel|>analysis\n"
            "<|message|>Think 2.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>Done.\n"
            "<|return|>"
        )

        tool_parser = HarmonyToolParser()
        tool_result = tool_parser.extract_tool_calls(text)
        assert tool_result.tools_called
        assert len(tool_result.tool_calls) == 1
        assert tool_result.tool_calls[0]["name"] == "search"
        assert tool_result.content == "Done."

        reasoning_parser = HarmonyReasoningParser()
        reasoning, content = reasoning_parser.extract_reasoning(text)
        assert "Think 1." in reasoning
        assert "Think 2." in reasoning
        assert content == "Done."

    def test_tool_parser_empty_arguments(self):
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.ping\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {}

    def test_tool_parser_whitespace_handling(self):
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>  {"key": "value"}  \n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["key"] == "value"


class TestHarmonyExtractToolCalls:

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_single_tool_with_analysis_and_commentary(self, parser):
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>User wants weather info for London.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"city": "London", "units": "metric"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "London"
        assert args["units"] == "metric"
        assert result.content is None

    def test_tool_call_with_constrain_token(self, parser):
        text = (
            "<|channel|>commentary to=functions.calculate\n"
            "<|constrain|>json\n"
            '<|message|>{"expression": "2+2"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_multiple_tool_calls_with_final(self, parser):
        text = (
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"query": "python tutorials"}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"query": "rust tutorials"}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.bookmark\n"
            "<|constrain|>json\n"
            '<|message|>{"url": "https://example.com"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>I've searched for both and bookmarked the result.\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 3
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[1]["name"] == "search"
        assert result.tool_calls[2]["name"] == "bookmark"
        assert result.content == "I've searched for both and bookmarked the result."

    def test_no_tool_call_final_channel_only(self, parser):
        text = "<|channel|>final\n<|message|>Sure, I can help with that.\n<|return|>"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "Sure, I can help with that."

    def test_no_tool_call_no_final_channel(self, parser):
        text = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help you today?"

    def test_no_tool_call_with_control_tokens_stripped(self, parser):
        text = "<|start|>\nHere is some text with tokens.\n<|end|>"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert "Here is some text with tokens." in result.content

    def test_empty_string(self, parser):
        result = parser.extract_tool_calls("")
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == ""

    def test_only_whitespace(self, parser):
        result = parser.extract_tool_calls("   \n\n  ")
        assert not result.tools_called
        assert result.tool_calls == []

    def test_missing_call_tag_is_now_parsed(self, parser):
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"key": "value"}\n'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "func"
        assert json.loads(result.tool_calls[0]["arguments"])["key"] == "value"

    def test_malformed_missing_message_tag(self, parser):
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '{"key": "value"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_tool_id_format(self, parser):
        text = (
            "<|channel|>commentary to=functions.test_func\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tool_calls[0]["id"].startswith("call_")
        assert len(result.tool_calls[0]["id"]) > len("call_")


class TestHarmonyStreaming:

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_token_by_token_analysis_commentary_call(self, parser):
        chunks = [
            "<|channel|>analysis\n",
            "<|message|>Let me think.\n",
            "<|end|>\n",
            "<|channel|>commentary to=functions.get_weather\n",
            "<|constrain|>json\n",
            '<|message|>{"location": "NYC"}\n',
            "<|call|>",
        ]

        previous = ""
        results = []
        for chunk in chunks:
            current = previous + chunk
            result = parser.extract_tool_calls_streaming(previous, current, chunk)
            results.append(result)
            previous = current

        for r in results[:-1]:
            assert r is None, f"Expected None during build-up, got {r}"

        final = results[-1]
        assert final is not None
        assert "tool_calls" in final
        assert final["tool_calls"][0]["function"]["name"] == "get_weather"
        args = json.loads(final["tool_calls"][0]["function"]["arguments"])
        assert args["location"] == "NYC"

    def test_final_channel_streaming_emits_content(self, parser):
        chunks = [
            "<|channel|>final\n",
            "<|message|>",
            "The ",
            "weather ",
            "is ",
            "sunny.",
        ]

        previous = ""
        content_parts = []
        for chunk in chunks:
            current = previous + chunk
            result = parser.extract_tool_calls_streaming(previous, current, chunk)
            if result and result.get("content"):
                content_parts.append(result["content"])
            previous = current

        joined = "".join(content_parts)
        assert joined == "The weather is sunny."

    def test_final_channel_empty_content_before_message(self, parser):
        current = "<|channel|>final\n"
        result = parser.extract_tool_calls_streaming("", current, current)
        assert result == {"content": ""}

    def test_final_channel_control_tokens_suppressed(self, parser):
        prev = "<|channel|>final\n<|message|>Done."
        current = prev + "<|return|>"
        result = parser.extract_tool_calls_streaming(prev, current, "<|return|>")
        if result is not None:
            assert result.get("content", "") == ""

    def test_call_in_delta_triggers_extraction(self, parser):
        current = (
            "<|channel|>commentary to=functions.add\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1, "b": 2}\n'
            "<|call|>"
        )
        prev = current[: -len("<|call|>")]
        result = parser.extract_tool_calls_streaming(prev, current, "<|call|>")

        assert result is not None
        assert "tool_calls" in result
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "add"
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert "id" in tc

    def test_no_channel_markers_pass_through(self, parser):
        result = parser.extract_tool_calls_streaming("", "Hello world", "Hello world")
        assert result == {"content": "Hello world"}

        result2 = parser.extract_tool_calls_streaming(
            "Hello world", "Hello world!", "!"
        )
        assert result2 == {"content": "!"}

    def test_analysis_channel_suppressed(self, parser):
        current = "<|channel|>analysis\n<|message|>Thinking..."
        result = parser.extract_tool_calls_streaming("", current, current)
        assert result is None

    def test_commentary_channel_suppressed(self, parser):
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"partial":'
        )
        result = parser.extract_tool_calls_streaming("", current, current)
        assert result is None

    def test_streaming_multiple_tool_calls(self, parser):
        text_before_second_call = (
            "<|channel|>commentary to=functions.func_a\n"
            "<|constrain|>json\n"
            '<|message|>{"x": 1}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.func_b\n"
            "<|constrain|>json\n"
            '<|message|>{"y": 2}\n'
        )
        current = text_before_second_call + "<|call|>"
        result = parser.extract_tool_calls_streaming(
            text_before_second_call, current, "<|call|>"
        )
        assert result is not None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "func_a"
        assert result["tool_calls"][1]["function"]["name"] == "func_b"

    def test_streaming_final_channel_incremental_content(self, parser):
        base = "<|channel|>final\n<|message|>"

        prev1 = base
        curr1 = base + "Hello"
        r1 = parser.extract_tool_calls_streaming(prev1, curr1, "Hello")
        assert r1 == {"content": "Hello"}

        prev2 = curr1
        curr2 = curr1 + " world"
        r2 = parser.extract_tool_calls_streaming(prev2, curr2, " world")
        assert r2 == {"content": " world"}

    def test_streaming_tool_call_format(self, parser):
        current = (
            "<|channel|>commentary to=functions.my_tool\n"
            "<|constrain|>json\n"
            '<|message|>{"key": "val"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls_streaming(
            current[: -len("<|call|>")], current, "<|call|>"
        )
        tc = result["tool_calls"][0]
        assert "id" in tc
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "my_tool"
        assert json.loads(tc["function"]["arguments"]) == {"key": "val"}


class TestHarmonyHasPendingToolCall:

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_returns_true_for_commentary_to_functions(self, parser):
        text = (
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}'
        )
        assert parser.has_pending_tool_call(text) is True

    def test_returns_true_partial_commentary(self, parser):
        text = "<|channel|>commentary to=functions.something"
        assert parser.has_pending_tool_call(text) is True

    def test_returns_false_for_normal_text(self, parser):
        assert parser.has_pending_tool_call("Hello, how can I help?") is False

    def test_returns_false_for_final_channel_only(self, parser):
        text = "<|channel|>final\n<|message|>Here is the answer.\n<|return|>"
        assert parser.has_pending_tool_call(text) is False

    def test_returns_false_for_analysis_channel(self, parser):
        text = "<|channel|>analysis\n<|message|>Let me think about this.\n<|end|>"
        assert parser.has_pending_tool_call(text) is False

    def test_returns_false_after_call_completed(self, parser):
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1}\n'
            "<|call|>"
        )
        assert parser.has_pending_tool_call(text) is True

    def test_returns_false_empty_string(self, parser):
        assert parser.has_pending_tool_call("") is False

    def test_does_not_use_base_class_check(self, parser):
        text = "<tool_call>some content"
        assert parser.has_pending_tool_call(text) is False


class TestHarmonyHelperFunctions:

    def test_strip_control_tokens_removes_all(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "<|start|>Hello<|end|>"
        result = _strip_control_tokens(text)
        assert "<|start|>" not in result
        assert "<|end|>" not in result
        assert "Hello" in result

    def test_strip_control_tokens_all_types(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        tokens = [
            "<|start|>",
            "<|end|>",
            "<|message|>",
            "<|channel|>",
            "<|constrain|>",
            "<|return|>",
            "<|call|>",
        ]
        for token in tokens:
            result = _strip_control_tokens(f"before{token}after")
            assert token not in result

    def test_strip_control_tokens_cleans_channel_names(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "<|channel|>analysis <|message|>Some reasoning<|end|>"
        result = _strip_control_tokens(text)
        assert "analysis" not in result.split()
        assert "Some reasoning" in result

    def test_strip_control_tokens_cleans_function_references(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "commentary to=functions.get_weather some text"
        result = _strip_control_tokens(text)
        assert "to=functions.get_weather" not in result
        assert "some text" in result

    def test_strip_control_tokens_plain_text(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "Just some regular text."
        result = _strip_control_tokens(text)
        assert result == "Just some regular text."

    def test_strip_control_tokens_empty(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        assert _strip_control_tokens("") == ""

    def test_is_control_token_valid_tokens(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        valid_tokens = [
            "<|start|>",
            "<|end|>",
            "<|message|>",
            "<|channel|>",
            "<|constrain|>",
            "<|return|>",
            "<|call|>",
        ]
        for token in valid_tokens:
            assert _is_control_token(token) is True, f"{token} should be recognized"

    def test_is_control_token_with_whitespace(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        assert _is_control_token("  <|start|>  ") is True
        assert _is_control_token("\n<|call|>\n") is True

    def test_is_control_token_non_tokens(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        assert _is_control_token("hello") is False
        assert _is_control_token("<|unknown|>") is False
        assert _is_control_token("") is False
        assert _is_control_token("<|start|>extra") is False

    def test_is_control_token_partial(self):
        from fusion_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        assert _is_control_token("<|start") is False
        assert _is_control_token("start|>") is False


class TestHarmonyCLIIntegration:

    def test_harmony_in_cli_choices(self):
        assert "harmony" in ToolParserManager.tool_parsers

    def test_gpt_oss_in_cli_choices(self):
        assert "gpt-oss" in ToolParserManager.tool_parsers

    def test_registry_has_both_names(self):
        cls_harmony = ToolParserManager.get_tool_parser("harmony")
        cls_gpt_oss = ToolParserManager.get_tool_parser("gpt-oss")
        assert cls_harmony is HarmonyToolParser
        assert cls_gpt_oss is HarmonyToolParser
        assert cls_harmony is cls_gpt_oss

    def test_harmony_in_registered_list(self):
        registered = ToolParserManager.list_registered()
        assert "harmony" in registered
        assert "gpt-oss" in registered

    def test_invalid_parser_not_registered(self):
        with pytest.raises(KeyError):
            ToolParserManager.get_tool_parser("nonexistent_parser")


@pytest.mark.skip(reason="fusion_mlx.cli unavailable")
class TestServeLogLevelFlags:
    def test_cli_serve_has_log_level_flag(self):
        pass

    def test_module_server_has_log_level_flag(self):
        pass


class TestHarmonyNativeFormat:

    def test_supports_native_format_true(self):
        assert HarmonyToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True
        assert HarmonyToolParser.supports_native_format() is True

    def test_instance_supports_native_format(self):
        parser = HarmonyToolParser()
        assert parser.supports_native_format() is True
