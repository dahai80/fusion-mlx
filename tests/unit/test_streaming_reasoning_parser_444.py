# SPDX-License-Identifier: Apache-2.0
"""Regression tests for streaming reasoning_parser wiring (#444)."""

import pytest

pytest.importorskip("mlx")  # suite needs mlx runtime; skip if absent

from fusion_mlx.api.openai_routes import (
    _CHANNEL_REASONING_PARSERS,
    _resolve_streaming_reasoning_parser,
)
from fusion_mlx.reasoning.harmony_parser import HarmonyReasoningParser


def test_channel_reasoning_parsers_set_contents():
    assert "harmony" in _CHANNEL_REASONING_PARSERS
    assert "gpt_oss" in _CHANNEL_REASONING_PARSERS
    assert "gemma4" in _CHANNEL_REASONING_PARSERS
    # tag-based parsers must NOT be in the channel set (they stay on
    # ThinkingParser, which already handles them)
    assert "qwen3" not in _CHANNEL_REASONING_PARSERS
    assert "deepseek_r1" not in _CHANNEL_REASONING_PARSERS


def test_resolve_returns_none_for_unknown_model():
    # no profile / no auto-config -> None (ThinkingParser fallback)
    assert (
        _resolve_streaming_reasoning_parser("definitely-not-a-real-model-xyz") is None
    )


def test_resolve_returns_none_for_tag_based_model_name():
    # qwen3-style names resolve a parser but it's tag-based, so the
    # streaming resolver must return None (ThinkingParser handles it)
    result = _resolve_streaming_reasoning_parser("Qwen3-0.6B")
    assert result is None


def test_harmony_parser_streams_reasoning_not_content():
    # The core #444 invariant: a Harmony channel marker must route to
    # reasoning, NOT leak as content. ThinkingParser fails this; the
    # named harmony parser's extract_reasoning_streaming passes it.
    # Deltas are split at marker boundaries to match real token-by-token
    # streaming (the parser sets channel/message state per-delta).
    parser = HarmonyReasoningParser(tokenizer=None)
    parser.reset_state()
    reasoning_out = ""
    content_out = ""
    deltas = [
        "<|channel|>analysis",
        "<|message|>",
        "Let me think step by step. ",
        "17 * 23 = 391. ",
        "<|channel|>final",
        "<|message|>",
        "The answer is 391.",
    ]
    prev = ""
    for d in deltas:
        cur = prev + d
        msg = parser.extract_reasoning_streaming(prev, cur, d)
        prev = cur
        if msg is not None:
            if msg.reasoning:
                reasoning_out += msg.reasoning
            if msg.content:
                content_out += msg.content
    assert "Let me think" in reasoning_out
    assert "391" in reasoning_out
    assert "The answer is 391" in content_out
    # the channel marker prose must NOT leak into content
    assert "<|channel|>analysis" not in content_out


def test_thinking_parser_leaks_channel_marker_as_content():
    # Documents the pre-#444 bug: ThinkingParser passes channel markers
    # through as content (it only knows .ndim tags). This is WHY the
    # streaming route must use the named parser for channel models.
    from fusion_mlx.api.thinking import ThinkingParser

    tp = ThinkingParser()
    content = ""
    for d in ["<|channel|>analysis<|message|>Let me think. ", "17*23=391"]:
        _, c = tp.feed(d)
        content += c
    # ThinkingParser does not strip the channel marker -> leaks as content
    assert "<|channel|>analysis" in content
