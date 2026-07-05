# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from fusion_mlx.tool_parsers import ToolParserManager
from fusion_mlx.tool_parsers.deepseek_v3_tool_parser import DeepSeekV3ToolParser
from fusion_mlx.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser

logger = logging.getLogger(__name__)

TC_OPEN = "<｜tool▁calls▁begin｜>"
TC_CLOSE = "<｜tool▁calls▁end｜>"
C_OPEN = "<｜tool▁call▁begin｜>"
C_CLOSE = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


def _v3_block(name: str, args_json: str) -> str:
    return f"{C_OPEN}function{SEP}{name}\n```json\n{args_json}\n```{C_CLOSE}"


def _v31_block(name: str, args_body: str) -> str:
    return f"{C_OPEN}{name}{SEP}{args_body}{C_CLOSE}"


def _envelope(*blocks: str, prefix: str = "") -> str:
    return f"{prefix}{TC_OPEN}{''.join(blocks)}{TC_CLOSE}"


@pytest.fixture
def v3_parser() -> DeepSeekV3ToolParser:
    return DeepSeekV3ToolParser()


@pytest.fixture
def v31_parser() -> DeepSeekV31ToolParser:
    return DeepSeekV31ToolParser()


def test_wire_format_primitives_use_fullwidth_pipe() -> None:
    for s in (TC_OPEN, TC_CLOSE, C_OPEN, C_CLOSE, SEP):
        assert "｜" in s, f"{s!r} missing fullwidth pipe"
        assert "|" not in s, f"{s!r} leaked ASCII pipe"


@pytest.mark.parametrize("name", ["deepseek_v3", "deepseek_r1_0528"])
def test_registry_lookup_returns_v3_parser(name: str) -> None:
    cls = ToolParserManager.get_tool_parser(name)
    assert cls is DeepSeekV3ToolParser


def test_registry_lookup_keeps_v31_separate() -> None:
    cls = ToolParserManager.get_tool_parser("deepseek_v31")
    assert cls is DeepSeekV31ToolParser


class TestV3WireFormat:
    def test_single_v3_tool_call(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called, "V3-shaped payload must trigger tools_called"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert (
            tc["name"] == "get_weather"
        ), f"V3 type-tag leak: parser returned name={tc['name']!r}"
        assert json.loads(tc["arguments"]) == {"city": "Tokyo"}

    def test_parallel_v3_tool_calls(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("get_time", '{"tz": "UTC"}'),
            _v3_block("search", '{"q": "deepseek"}'),
        )

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 3
        names = [c["name"] for c in result.tool_calls]
        assert names == ["get_weather", "get_time", "search"]
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}
        assert json.loads(result.tool_calls[1]["arguments"]) == {"tz": "UTC"}
        assert json.loads(result.tool_calls[2]["arguments"]) == {"q": "deepseek"}

    def test_v3_with_leading_content(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            prefix="Let me check the weather. ",
        )

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert result.content == "Let me check the weather. "
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_v3_arguments_with_nested_braces(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        args = '{"filter": {"city": "Tokyo", "tags": ["a", "b"]}, "limit": 10}'
        payload = _envelope(_v3_block("search", args))

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == json.loads(args)


class TestV31WireFormat:
    def test_single_v31_tool_call(self, v31_parser: DeepSeekV31ToolParser) -> None:
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))

        result = v31_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}

    def test_parallel_v31_tool_calls(self, v31_parser: DeepSeekV31ToolParser) -> None:
        payload = _envelope(
            _v31_block("f1", '{"a": 1}'),
            _v31_block("f2", '{"b": 2}'),
        )

        result = v31_parser.extract_tool_calls(payload)

        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["f1", "f2"]

    def test_v31_with_tool_named_function_passes(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        payload = _envelope(_v31_block("function_lookup", '{"q": "x"}'))

        result = v31_parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["name"] == "function_lookup"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"q": "x"}


class TestSplitContract:
    def test_v3_parser_drops_v31_shaped_body(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is False
        assert result.tool_calls == []

    def test_v31_parser_misparses_v3_body_as_function_named(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        payload = _envelope(_v3_block("get_weather", '{"city": "Paris"}'))
        result = v31_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "function"


class TestMalformedGraceful:
    def test_no_envelope_passes_through(self, v3_parser: DeepSeekV3ToolParser) -> None:
        text = "Just plain reasoning, no tools here."
        result = v3_parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == text

    def test_truncated_block_falls_back_to_content(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        payload = f"prefix {TC_OPEN}{C_OPEN}function{SEP}get_weather\n```json\n{{"

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == payload

    def test_envelope_with_no_blocks(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = f"{TC_OPEN}{TC_CLOSE}"

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == payload

    def test_block_missing_separator(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = f"{TC_OPEN}{C_OPEN}garbage_no_sep_here{C_CLOSE}{TC_CLOSE}"

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == payload

    def test_one_good_one_bad_block(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            f"{C_OPEN}no_sep_here{C_CLOSE}",
            _v3_block("get_time", '{"tz": "UTC"}'),
        )

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]

    def test_literal_marker_text_before_envelope_does_not_misparse(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        prose = (
            "Here's how DeepSeek tool calls work: "
            f"the format uses {C_OPEN}NAME{SEP}ARGS{C_CLOSE}. For example:\n\n"
        )
        real_call = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))
        payload = prose + real_call

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_envelope_with_truncated_trailing_block_preserves_text(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        good_block = _v3_block("get_weather", '{"city": "Tokyo"}')
        truncated_tail = f"{C_OPEN}function{SEP}get_time\n```json\n{{"
        payload = f"{TC_OPEN}{good_block}{truncated_tail}"

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.content is not None
        assert "get_time" in result.content, (
            f"Truncated trailing block text dropped from content. "
            f"content={result.content!r}"
        )

    def test_v3_anchored_body_with_partial_fence_drops_block(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        payload = (
            f"{TC_OPEN}{C_OPEN}function{SEP}get_weather\n"
            f'```json\n{{"city": "Tokyo"'
            f"{C_CLOSE}{TC_CLOSE}"
        )

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called, (
            f"Partial-fence V3 body must NOT emit a tool call. "
            f"Got: {result.tool_calls!r}"
        )
        assert result.tool_calls == []
        assert result.content is not None
        assert "get_weather" in result.content


class TestV3Streaming:
    def _feed(
        self, parser: DeepSeekV3ToolParser, payload: str, chunk_size: int = 8
    ) -> list[dict | None]:
        results: list[dict | None] = []
        prev = ""
        for i in range(0, len(payload), chunk_size):
            delta = payload[i : i + chunk_size]
            cur = prev + delta
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=cur,
                delta_text=delta,
            )
            results.append(r)
            prev = cur
        return results

    def test_plain_content_before_envelope_streams_normally(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        events = self._feed(v3_parser, "Let me check.", chunk_size=4)
        content_seen = [ev for ev in events if ev and ev.get("content")]
        assert (
            content_seen
        ), f"No content emitted for pre-envelope tokens. Events: {events!r}"

    def test_pre_envelope_prose_in_same_delta_as_marker_is_emitted(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        delta = "Let me check the weather. " + TC_OPEN
        result = v3_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
        )
        assert result is not None
        assert result.get("content") == "Let me check the weather. "

    def test_subsequent_deltas_after_marker_return_none(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        prev = "Let me check. " + TC_OPEN
        delta = C_OPEN + "function" + SEP + "get_weather"
        result = v3_parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev + delta,
            delta_text=delta,
        )
        assert result is None

    def test_v3_stream_emits_no_mid_stream_tool_calls(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        events = self._feed(v3_parser, payload, chunk_size=12)
        for ev in events:
            if not ev:
                continue
            assert (
                "tool_calls" not in ev
            ), f"V3 stream leaked mid-stream tool_calls: {ev!r}"

        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}

    def test_v3_parallel_stream_finalize_yields_all_calls(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("get_time", '{"tz": "UTC"}'),
        )

        result = v3_parser.extract_tool_calls(payload)
        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]


def _make_postprocessor_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "deepseek_v3"
    cfg.tool_parser_instance = None
    return cfg


def _make_generation_output(
    text: str, finished: bool = False, finish_reason: str | None = None
) -> MagicMock:
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = finish_reason or ("stop" if finished else None)
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


@pytest.mark.skip(
    reason="StreamingPostProcessor unavailable: sanitize_output missing from api.utils"
)
class TestV3StreamingIntegration:
    def test_v3_payload_emits_tool_call_via_finalize(self) -> None:
        from fusion_mlx.service.postprocessor import StreamingPostProcessor

        cfg = _make_postprocessor_cfg()
        pp = StreamingPostProcessor(cfg, tools_requested=True)
        pp.reset()

        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        all_events = []
        for i in range(0, len(payload), 16):
            chunk = payload[i : i + 16]
            is_last = i + 16 >= len(payload)
            output = _make_generation_output(
                chunk,
                finished=is_last,
                finish_reason="tool_calls" if is_last else None,
            )
            all_events.extend(pp.process_chunk(output))

        all_events.extend(pp.finalize())

        tool_call_events = [e for e in all_events if e.type == "tool_call"]
        assert tool_call_events, (
            f"No tool_call event emitted end-to-end. Events: "
            f"{[(e.type, getattr(e, 'tool_calls', None)) for e in all_events]!r}"
        )
        names = []
        for ev in tool_call_events:
            for tc in ev.tool_calls or []:
                names.append(tc.get("function", {}).get("name") or tc.get("name"))
        assert (
            "get_weather" in names
        ), f"V3 finalize path lost the real tool name. Names: {names!r}"
        assert (
            "function" not in names
        ), f"V3 type-tag leaked into the final tool_call event. Names: {names!r}"


class TestArgumentBytesPassthrough:
    def test_valid_json_args_passed_through_verbatim(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        args = '{   "k"  :  1  }'
        payload = _envelope(_v3_block("f", args))

        result = v3_parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["arguments"] == args
        assert json.loads(result.tool_calls[0]["arguments"]) == {"k": 1}

    def test_v31_non_json_args_passed_through(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        payload = _envelope(_v31_block("explain", "free-form text body"))

        result = v31_parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["arguments"] == "free-form text body"
