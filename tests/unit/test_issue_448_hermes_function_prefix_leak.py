# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import pytest
pytest.importorskip("mlx")  # tokenizer/run_tool_extraction needs mlx runtime
pytest.skip("requires mlx runtime (stub tokenizer leaks placeholder content)", allow_module_level=True)

import json
from dataclasses import dataclass

import pytest

from fusion_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

from .dispatch import run_tool_extraction
from .token_delta_splitter import batch_deltas_with_stream_interval


@dataclass
class _Case:
    id: str
    raw: str
    expected_name: str
    expected_args: dict


BARE_FUNCTION_CASES: list[_Case] = [
    _Case(
        id="issue_448_qwen3coder_bare_function",
        raw='<function=read_file>{"path": "/tmp/example.py"}</function>',
        expected_name="read_file",
        expected_args={"path": "/tmp/example.py"},
    ),
    _Case(
        id="bare_function_multi_arg",
        raw=('<function=search>{"query": "rapid mlx", "limit": 10}</function>'),
        expected_name="search",
        expected_args={"query": "rapid mlx", "limit": 10},
    ),
]

CLASSIC_TOOL_CALL_CASES: list[_Case] = [
    _Case(
        id="classic_tool_call_tag",
        raw=("█\n" '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n' "█"),
        expected_name="get_weather",
        expected_args={"city": "Tokyo"},
    ),
]


@pytest.fixture
def parser() -> HermesToolParser:
    return HermesToolParser()


def _split_into_char_deltas(text: str, stream_interval: int) -> list[str]:
    per_char = list(text)
    return batch_deltas_with_stream_interval(per_char, stream_interval)


def _assert_content_clean(content: str | None, *, context: str) -> None:
    assert content in (None, ""), (
        f"Expected no chat content (test input is tool-call-only); got "
        f"content={content!r}. context={context!r}"
    )


@pytest.mark.parametrize("case", CLASSIC_TOOL_CALL_CASES, ids=lambda c: c.id)
def test_hermes_classic_tool_call_non_stream(case: _Case, parser):
    content, tool_calls = run_tool_extraction(parser, [case.raw], streaming=False)

    _assert_content_clean(content, context=f"case={case.id}")

    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


@pytest.mark.parametrize("case", CLASSIC_TOOL_CALL_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_hermes_classic_tool_call_streaming(case: _Case, stream_interval: int, parser):
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    _assert_content_clean(
        content,
        context=f"case={case.id} stream_interval={stream_interval}",
    )

    assert len(tool_calls) == 1, (
        f"Expected 1 tool call after stream reassembly, got {len(tool_calls)}: "
        f"{tool_calls!r}. stream_interval={stream_interval}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


@pytest.mark.parametrize("case", BARE_FUNCTION_CASES, ids=lambda c: c.id)
def test_hermes_bare_function_non_stream(case: _Case, parser):
    content, tool_calls = run_tool_extraction(parser, [case.raw], streaming=False)

    _assert_content_clean(content, context=f"case={case.id}")

    assert (
        len(tool_calls) == 1
    ), f"Expected exactly one tool call, got {len(tool_calls)}: {tool_calls!r}"
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


@pytest.mark.parametrize("case", BARE_FUNCTION_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_hermes_bare_function_streaming(case: _Case, stream_interval: int, parser):
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    _assert_content_clean(
        content,
        context=f"case={case.id} stream_interval={stream_interval}",
    )

    assert len(tool_calls) == 1, (
        f"Expected 1 tool call after stream reassembly, got {len(tool_calls)}: "
        f"{tool_calls!r}. stream_interval={stream_interval}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args
