# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pytest

from fusion_mlx.output_router import OutputRouter

from .fake_tokenizer import HARMONY_VOCAB, harmony_fake_tokenizer


@dataclass
class _SanityCase:
    id: str
    token_ids: list[int]
    expected_content: str | None
    expected_reasoning: str | None
    expected_tool_calls: list[str] | None


@dataclass
class _BugCase:
    id: str
    token_ids: list[int]
    expected_function_name: str
    expected_args_marker: str


_V = HARMONY_VOCAB


SANITY_CASES: list[_SanityCase] = [
    _SanityCase(
        id="canonical_analysis_only",
        token_ids=[
            _V["<|channel|>"],
            _V["analysis"],
            _V["<|message|>"],
            _V["Reason"],
            _V["ing"],
            _V["<|end|>"],
        ],
        expected_content=None,
        expected_reasoning="Reasoning",
        expected_tool_calls=None,
    ),
    _SanityCase(
        id="canonical_final_only",
        token_ids=[
            _V["<|channel|>"],
            _V["final"],
            _V["<|message|>"],
            _V["Answer"],
            _V["<|return|>"],
        ],
        expected_content="Answer",
        expected_reasoning=None,
        expected_tool_calls=None,
    ),
    _SanityCase(
        id="canonical_analysis_then_final",
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
            _V["final"],
            _V["<|message|>"],
            _V["Answer"],
            _V["<|return|>"],
        ],
        expected_content="Answer",
        expected_reasoning="Reasoning",
        expected_tool_calls=None,
    ),
]


BUG_CASES: list[_BugCase] = [
    _BugCase(
        id="issue_455_calculate_single_token_body",
        token_ids=[
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.calculate"],
            _V[" json"],
            _V["<|message|>"],
            _V['{"expression":"17*23"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_function_name="calculate",
        expected_args_marker='"17*23"',
    ),
    _BugCase(
        id="issue_455_calculate_multi_token_body",
        token_ids=[
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
        expected_function_name="calculate",
        expected_args_marker='"17*23"',
    ),
    _BugCase(
        id="get_weather_single_token_body",
        token_ids=[
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.get_weather"],
            _V[" json"],
            _V["<|message|>"],
            _V['{"city":"Tokyo"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_function_name="get_weather",
        expected_args_marker='"Tokyo"',
    ),
]


@pytest.fixture
def router() -> OutputRouter:
    fake_tok = harmony_fake_tokenizer()
    r = OutputRouter.from_tokenizer(fake_tok)
    assert r is not None, (
        "OutputRouter.from_tokenizer returned None on the synthetic "
        "harmony vocab — discovery is broken or the vocab is missing "
        "required tokens (<|channel|>, <|message|>)."
    )
    assert (
        r.map.format_tag == "harmony"
    ), f"Expected harmony discovery; got format_tag={r.map.format_tag!r}"
    return r


def _normalize_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@pytest.mark.parametrize("case", SANITY_CASES, ids=lambda c: c.id)
def test_harmony_router_sanity(case: _SanityCase, router):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize_str(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize_str(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize_str(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize_str(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )
    assert result["tool_calls"] == case.expected_tool_calls, (
        f"tool_calls mismatch for case={case.id}: expected="
        f"{case.expected_tool_calls!r}, got={result['tool_calls']!r}"
    )


@pytest.mark.parametrize("case", BUG_CASES, ids=lambda c: c.id)
@pytest.mark.xfail(
    reason=(
        "Issue #455 — OutputRouter's harmony AWAITING_CHANNEL_TYPE "
        "handling (output_router.py:222-238) only recognizes "
        "``analysis``/``final`` as channel-type words. The tool-call "
        "channel emits ``commentary`` followed by `` to=functions.X`` + "
        "an optional `` json`` constrain directive, then the body. The "
        "router falls into the default branch, transitions to CONTENT, "
        "and leaks ``commentary`` + recipient + body as content text. "
        "Live verification on gpt-oss-20b-mxfp4-q8 (2026-06-04) revealed an "
        "additional constraint not modeled by these BUG_CASES synthetic "
        "vocab: production ``commentary`` is TWO tokens — ``comment`` "
        "(12606) + ``ary`` (815) — so naive single-token channel-type "
        "matching can never fire on real gpt-oss-20b. The eventual fix "
        "is the marker-preserving TOOL_CALL_TEXT router redesign "
        "(tracked in followup PR / issue #513): structural "
        "``<|channel|>`` / ``<|message|>`` / ``<|call|>`` markers stay "
        "in CONTENT during tool-call paths and the existing text-based "
        "parser (HarmonyToolParser) extracts the call downstream — at "
        "router level this manifests as router-emitted tool_calls "
        "non-empty with name + args (matching the assertions below). "
        "The alternative shape (content containing the markered text, "
        "tool_calls empty) requires asserting in the parser-level "
        "regression file, not here — keeping this contract single-"
        "outcome avoids a passing-test-but-wrong-router-fix loophole "
        "(codex re-review NIT)."
    ),
    strict=True,
)
def test_harmony_router_commentary_tool_call(case: _BugCase, router):
    result = router.feed_sequence(case.token_ids)

    assert _normalize_str(result["content"]) is None, (
        f"content leaked for case={case.id}: got={result['content']!r}. "
        "Recipient/body must not be emitted to CONTENT channel."
    )
    assert (
        _normalize_str(result["reasoning"]) is None
    ), f"reasoning leaked for case={case.id}: got={result['reasoning']!r}"

    tool_calls = result["tool_calls"]
    assert tool_calls, (
        f"tool_calls is empty/None for case={case.id}; got={tool_calls!r}. "
        "Router must emit at least one TOOL_CALL event."
    )
    assert len(tool_calls) == 1, (
        f"Expected ONE aggregated tool_calls entry for case={case.id}; got "
        f"{len(tool_calls)}: {tool_calls!r}. The fix must accumulate "
        "multi-token bodies into a single entry rather than emitting per-"
        "token fragments."
    )
    entry = tool_calls[0]
    payload = entry if isinstance(entry, str) else _stringify_structured(entry)

    assert case.expected_function_name in payload, (
        f"Function name {case.expected_function_name!r} missing from "
        f"tool_calls entry for case={case.id}; got={entry!r}. Args-only "
        "emission drops tool_use.name and is insufficient downstream."
    )
    assert case.expected_args_marker in payload, (
        f"Args marker {case.expected_args_marker!r} missing from "
        f"tool_calls entry for case={case.id}; got={entry!r}."
    )


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
