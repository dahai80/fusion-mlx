# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from fusion_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

from ._harmony_markers import assert_no_harmony_marker_leak
from .dispatch import run_tool_extraction
from .token_delta_splitter import batch_deltas_with_stream_interval


@dataclass
class _Case:
    id: str
    raw: str
    expected_name: str
    expected_args: dict


TEST_CASES: list[_Case] = [
    _Case(
        id="simple_single_arg",
        raw=(
            "<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city": "Tokyo"}<|call|>'
        ),
        expected_name="get_weather",
        expected_args={"city": "Tokyo"},
    ),
    _Case(
        id="multi_arg_object",
        raw=(
            "<|channel|>commentary to=functions.search "
            '<|constrain|>json<|message|>{"query": "rapid mlx", '
            '"limit": 10}<|call|>'
        ),
        expected_name="search",
        expected_args={"query": "rapid mlx", "limit": 10},
    ),
    _Case(
        id="empty_args_object",
        raw=(
            "<|channel|>commentary to=functions.list_models "
            "<|constrain|>json<|message|>{}<|call|>"
        ),
        expected_name="list_models",
        expected_args={},
    ),
    _Case(
        id="issue_480_paris",
        raw=(
            "<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city": "Paris"}<|call|>'
        ),
        expected_name="get_weather",
        expected_args={"city": "Paris"},
    ),
]


@pytest.fixture
def parser() -> HarmonyToolParser:
    return HarmonyToolParser()


def _split_into_char_deltas(text: str, stream_interval: int) -> list[str]:
    per_char = list(text)
    return batch_deltas_with_stream_interval(per_char, stream_interval)


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.id)
def test_harmony_tool_extraction_non_stream(case: _Case, parser):
    content, tool_calls = run_tool_extraction(parser, [case.raw], streaming=False)

    assert content in (None, ""), (
        f"Expected non-stream extraction to consume all input as a tool "
        f"call; got leftover content={content!r}"
    )
    assert (
        len(tool_calls) == 1
    ), f"Expected exactly one tool call, got {len(tool_calls)}: {tool_calls!r}"
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_harmony_tool_extraction_streaming(case: _Case, stream_interval: int, parser):
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    assert_no_harmony_marker_leak(
        content,
        context=f"case={case.id} stream_interval={stream_interval}",
    )

    assert content in (None, ""), (
        f"Expected no chat content (tool-call-only input); got "
        f"content={content!r}. stream_interval={stream_interval} "
        f"case={case.id}"
    )

    assert len(tool_calls) == 1, (
        f"Expected 1 tool call after stream reassembly, got {len(tool_calls)}: "
        f"{tool_calls!r}. stream_interval={stream_interval}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args
