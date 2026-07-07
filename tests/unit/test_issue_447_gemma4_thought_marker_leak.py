# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pytest

from fusion_mlx.output_router import OutputRouter

from .fake_tokenizer import GEMMA4_VOCAB, gemma4_fake_tokenizer


@dataclass
class _Case:
    id: str
    token_ids: list[int]
    expected_content: str | None
    expected_reasoning: str | None


_V = GEMMA4_VOCAB


SANITY_CASES: list[_Case] = [
    _Case(
        id="canonical_thought_then_final",
        token_ids=[
            _V["<|channel>"],
            _V["thought"],
            _V["analysis_body"],
            _V["<channel|>"],
            _V["<|channel>"],
            _V["final"],
            _V["message_body"],
            _V["<eos>"],
        ],
        expected_content="message_body",
        expected_reasoning="analysis_body",
    ),
    _Case(
        id="canonical_final_only",
        token_ids=[
            _V["<|channel>"],
            _V["final"],
            _V["Hello"],
            _V[" world"],
            _V["<eos>"],
        ],
        expected_content="Hello world",
        expected_reasoning=None,
    ),
]


BUG_CASES: list[_Case] = [
    _Case(
        id="issue_447_thought_word_only_no_final",
        token_ids=[
            _V["thought"],
            _V["\n"],
            _V["message_body"],
            _V["<eos>"],
        ],
        expected_content=None,
        expected_reasoning="message_body",
    ),
]


LOOKAHEAD_ROLLBACK_CASES: list[_Case] = [
    _Case(
        id="legit_final_then_content_not_swallowed",
        token_ids=[
            _V["final"],
            _V[" world"],
            _V["<eos>"],
        ],
        expected_content="final world",
        expected_reasoning=None,
    ),
    _Case(
        id="legit_content_then_body_not_swallowed",
        token_ids=[
            _V["content"],
            _V["Hello"],
            _V["<eos>"],
        ],
        expected_content="contentHello",
        expected_reasoning=None,
    ),
    _Case(
        id="legit_thought_then_body_not_swallowed",
        token_ids=[
            _V["thought"],
            _V["Hello"],
            _V["<eos>"],
        ],
        expected_content="thoughtHello",
        expected_reasoning=None,
    ),
    _Case(
        id="bare_word_only_no_followup",
        token_ids=[
            _V["final"],
            _V["<eos>"],
        ],
        expected_content="final",
        expected_reasoning=None,
    ),
]


KNOWN_LIMITATION_CASES: list[_Case] = [
    _Case(
        id="issue_447_compound_bare_words_known_limitation",
        token_ids=[
            _V["thought"],
            _V["\n"],
            _V["analysis_body"],
            _V["\n"],
            _V["final"],
            _V["\n"],
            _V["message_body"],
            _V["<eos>"],
        ],
        expected_content="message_body",
        expected_reasoning="analysis_body",
    ),
]


@pytest.fixture
def router() -> OutputRouter:
    fake_tok = gemma4_fake_tokenizer()
    r = OutputRouter.from_tokenizer(fake_tok)
    assert r is not None, (
        "OutputRouter.from_tokenizer returned None on the synthetic "
        "Gemma 4 vocab — discovery is broken or the vocab is missing "
        "required tokens (<|channel>, <|tool_call>)."
    )
    assert (
        r.map.format_tag == "gemma4"
    ), f"Expected Gemma 4 discovery; got format_tag={r.map.format_tag!r}"
    return r


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@pytest.mark.parametrize("case", SANITY_CASES, ids=lambda c: c.id)
def test_gemma4_router_sanity(case: _Case, router):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )


@pytest.mark.parametrize("case", BUG_CASES, ids=lambda c: c.id)
def test_gemma4_router_no_channel_marker_leaks(case: _Case, router):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )


@pytest.mark.parametrize("case", LOOKAHEAD_ROLLBACK_CASES, ids=lambda c: c.id)
def test_gemma4_router_lookahead_rollback_preserves_legit_first_token(
    case: _Case, router
):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )


@pytest.mark.parametrize("case", KNOWN_LIMITATION_CASES, ids=lambda c: c.id)
@pytest.mark.xfail(
    reason=(
        "Compound bare-word sequence (#447 Case A): the model emits "
        "bare ``thought`` followed later by bare ``final`` mid-stream "
        "after exiting the first channel. The INIT-only bare-word gate "
        "in output_router.py deliberately does NOT fire transitions "
        "outside INIT state — broadening the gate regresses canonical "
        "Gemma 4 bodies whose first content token happens to be "
        "``final`` / ``content`` / ``thought`` (codex round-2 BLOCKING). "
        "Marker-preserving router followup will resolve this without "
        "the trade-off."
    ),
    strict=True,
)
def test_gemma4_router_compound_bare_words_known_limitation(case: _Case, router):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content)
    assert result["reasoning"] == _normalize(case.expected_reasoning)
