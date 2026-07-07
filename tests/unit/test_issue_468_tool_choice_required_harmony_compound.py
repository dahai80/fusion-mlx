# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pytest

from fusion_mlx.output_router import OutputRouter

from .fake_tokenizer import HARMONY_VOCAB, harmony_fake_tokenizer


@dataclass
class _Case:
    id: str
    token_ids: list[int]
    expected_reasoning_marker: str
    expected_function_name: str
    expected_args_marker: str


_V = HARMONY_VOCAB


BUG_CASES: list[_Case] = [
    _Case(
        id="issue_468_analysis_then_commentary_get_weather",
        token_ids=[
            _V["<|channel|>"],
            _V["analysis"],
            _V["<|message|>"],
            _V["Reason"],
            _V["ing"],
            _V["<|end|>"],
            _V["<|start|>"],
            _V["assistant"],
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.get_weather"],
            _V[" json"],
            _V["<|message|>"],
            _V['{"city":"Tokyo"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_reasoning_marker="Reasoning",
        expected_function_name="get_weather",
        expected_args_marker='"Tokyo"',
    ),
    _Case(
        id="issue_468_analysis_then_commentary_multi_token_body",
        token_ids=[
            _V["<|channel|>"],
            _V["analysis"],
            _V["<|message|>"],
            _V["Reason"],
            _V["ing"],
            _V["<|end|>"],
            _V["<|start|>"],
            _V["assistant"],
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.calculate"],
            _V[" json"],
            _V["<|message|>"],
            _V["{"],
            _V['"expr'],
            _V['ession":"'],
            _V["17"],
            _V["*"],
            _V["23"],
            _V['"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_reasoning_marker="Reasoning",
        expected_function_name="calculate",
        expected_args_marker='"17*23"',
    ),
]


@pytest.fixture
def router() -> OutputRouter:
    fake_tok = harmony_fake_tokenizer()
    r = OutputRouter.from_tokenizer(fake_tok)
    assert r is not None
    assert r.map.format_tag == "harmony"
    return r


def _normalize_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _stringify_structured(entry: object) -> str:
    if isinstance(entry, dict):
        parts: list[str] = []
        for value in entry.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                parts.append(_stringify_structured(value))
        return " ".join(parts)
    return repr(entry)


@pytest.mark.parametrize("case", BUG_CASES, ids=lambda c: c.id)
@pytest.mark.xfail(
    reason=(
        "Issue #468 (router-level portion) — compound analysis + "
        "commentary sequence leaks the commentary block as CONTENT "
        "text. Same family-wide gap as #455. Live verification on "
        "gpt-oss-20b-mxfp4-q8 (2026-06-04) confirmed the symptom AND surfaced "
        "the deeper constraint that breaks naive single-token fixes: "
        "production ``commentary`` is two tokens (``comment``+``ary``). "
        "Eventual fix must lookahead-decode the channel-type word or "
        "preserve structural markers in CONTENT during tool-call paths "
        "(see #455 reason for the marker-preserving TOOL_CALL_TEXT "
        "design). The shape contract asserted below (empty content, "
        "reasoning + structured tool_calls) is ONE valid post-fix "
        "outcome; an alternative valid outcome is content containing "
        "the full markered tool-call text (parser handles the call "
        "downstream). Tracked as a followup PR. "
        'tool_choice="required" enforcement (the other #468 symptom) '
        "is out of scope here — covered by FSM PR #132."
    ),
    strict=True,
)
def test_harmony_router_compound_analysis_then_commentary(case: _Case, router):
    result = router.feed_sequence(case.token_ids)

    assert _normalize_str(result["content"]) is None, (
        f"content leaked for case={case.id}: got={result['content']!r}. "
        "Compound sequence emitted analysis or commentary text as content."
    )

    reasoning = _normalize_str(result["reasoning"]) or ""
    assert case.expected_reasoning_marker in reasoning, (
        f"reasoning missing {case.expected_reasoning_marker!r} for "
        f"case={case.id}; got reasoning={result['reasoning']!r}"
    )
    for tool_marker in (
        case.expected_function_name,
        case.expected_args_marker,
        "commentary",
        " to=functions.",
    ):
        assert tool_marker not in reasoning, (
            f"Tool metadata marker {tool_marker!r} leaked into reasoning "
            f"for case={case.id}; got reasoning={result['reasoning']!r}. "
            "Channel partition contract violated — tool-call tokens "
            "should not appear in the reasoning channel."
        )

    tool_calls = result["tool_calls"]
    assert (
        tool_calls
    ), f"tool_calls is empty/None for case={case.id}; got={tool_calls!r}"
    assert len(tool_calls) == 1, (
        f"Expected ONE aggregated tool_calls entry for case={case.id}; got "
        f"{len(tool_calls)}: {tool_calls!r}"
    )
    entry = tool_calls[0]
    payload = entry if isinstance(entry, str) else _stringify_structured(entry)
    assert case.expected_function_name in payload, (
        f"Function name {case.expected_function_name!r} missing from "
        f"tool_calls entry for case={case.id}; got={entry!r}"
    )
    assert case.expected_args_marker in payload, (
        f"Args marker {case.expected_args_marker!r} missing from "
        f"tool_calls entry for case={case.id}; got={entry!r}"
    )
