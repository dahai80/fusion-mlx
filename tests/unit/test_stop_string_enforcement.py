# SPDX-License-Identifier: Apache-2.0
"""Tests for server-side stop-string truncation in the text Scheduler.

The OpenAI-API contract says ``stop`` (list of strings) is honoured by the
server: if any string in the list appears in the generated output, the
server must truncate at that point and report ``finish_reason="stop"``.
mlx-lm's BatchGenerator only honours integer ``stop_tokens`` — so the
scheduler has to scan the decoded output itself.

This was missing on the text path (MLLMScheduler had it, Scheduler did
not), which surfaced as 4 failing regression-suite tests on qwen3.5-4b-4bit:
tests 1 (newline), 2 (literal word), 4 (Unicode), 5 (streaming).

These unit tests exercise ``Scheduler._process_batch_responses`` directly with
a mocked tokenizer so the truncation contract is pinned without needing
a real model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fusion_mlx.request import Request, RequestStatus, SamplingParams
from fusion_mlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler() -> Scheduler:
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s.split())))
    config = SchedulerConfig(max_num_seqs=4)
    return Scheduler(model, tokenizer, config)


def _make_request(
    rid: str,
    stop_strings: list[str] | None,
    prefilled_tokens: list[int] | None = None,
) -> Request:
    """Build a Request already in the ``running`` state with the given
    output token ids — _process_batch_responses appends one more token from the
    incoming Response, so prefilled_tokens is the prefix and the test
    drives the final token via the Response."""
    sp = SamplingParams(max_tokens=100, stop=stop_strings or [])
    req = Request(request_id=rid, prompt="ignored", sampling_params=sp)
    req.num_prompt_tokens = 4
    req.status = RequestStatus.RUNNING
    if prefilled_tokens:
        for t in prefilled_tokens:
            req.append_output_token(t)
    return req


def _run_step(
    scheduler: Scheduler,
    request: Request,
    *,
    last_token: int,
    decoded_full: str,
    finish_reason: str | None = None,
):
    """Drive ``_process_batch_responses`` once for a single request and return
    the resulting RequestOutput.

    ``decoded_full`` is the full decoded text AFTER the new token is
    appended. The production path decodes through a streaming detokenizer
    (``_get_detokenizer`` → ``detokenizer.add_token`` / ``detokenizer.text``
    / ``detokenizer.last_segment``), then re-decodes the full token list on
    finish via ``tokenizer.decode``. We stub both seams so the test fully
    controls the surface the stop-string check scans without a real
    tokenizer backend.
    """
    rid = request.request_id
    uid = 0
    scheduler.running[rid] = request
    scheduler.uid_to_request_id[uid] = rid

    # Stub the streaming detokenizer. Each test drives a single new token,
    # so after add_token the detokenizer's full text == decoded_full and
    # last_segment (the delta from the prior state) == decoded_full too —
    # no prior text was fed. Production computes prev_len =
    # len(text) - len(last_segment) = 0, so the stop scan starts at 0.
    class _FakeDetok:
        def __init__(self):
            self._text = ""
            self._finalized = False

        def reset(self):
            self._text = ""
            self._finalized = False

        def add_token(self, _tok):
            self._text = decoded_full

        @property
        def text(self):
            return self._text

        @property
        def last_segment(self):
            # add_token already published the full segment; finalize() has
            # nothing left to flush, so post-finalize reads are empty (the
            # production NaiveStreamingDetokenizer clears its buffer).
            return "" if self._finalized else self._text

        def finalize(self):
            self._finalized = True
            return None

    detok = _FakeDetok()
    scheduler._get_detokenizer = lambda _rid: detok  # type: ignore[method-assign]

    # Finished path re-decodes the full token list via tokenizer.decode.
    scheduler.tokenizer.decode = lambda _tokens: decoded_full  # type: ignore[method-assign]

    # Build a minimal Response stub matching BatchGenerator's contract.
    response = MagicMock()
    response.uid = uid
    response.token = last_token
    response.finish_reason = finish_reason
    response.logprobs = None
    # No prompt_cache attr — _process_batch_responses checks hasattr.
    del response.prompt_cache

    outputs, finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    return outputs[0], finished


def test_stop_string_truncates_output_at_match():
    """Output contains the stop string → finish_reason becomes "stop"
    and output_text is the prefix BEFORE the stop marker."""
    scheduler = _make_scheduler()
    req = _make_request("r1", stop_strings=["World"], prefilled_tokens=[10, 11])

    output, finished = _run_step(
        scheduler,
        req,
        last_token=12,
        decoded_full="Hello World!",
        finish_reason=None,
    )

    assert output.finished is True
    assert output.finish_reason == "stop"
    assert "World" not in output.output_text
    assert output.output_text == "Hello "
    assert "r1" in finished


def test_stop_string_first_match_wins():
    """With multiple stop strings, the one that matches earliest in the
    output wins. The user's ["World", "!"] case should stop at "World"
    not at "!"."""
    scheduler = _make_scheduler()
    req = _make_request("r2", stop_strings=["World", "!"], prefilled_tokens=[10])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full="Hello World! Goodbye World!",
    )

    assert output.finished is True
    assert output.finish_reason == "stop"
    # Both stop strings should be absent; truncated at first ("World")
    assert "World" not in output.output_text
    assert "!" not in output.output_text
    assert output.output_text == "Hello "


def test_stop_string_unicode_match():
    """Stop string with non-ASCII chars must match the same way ASCII does."""
    scheduler = _make_scheduler()
    req = _make_request("r3", stop_strings=["世界"], prefilled_tokens=[10])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full="你好世界 goodbye",
    )

    assert output.finished is True
    assert output.finish_reason == "stop"
    assert "世界" not in output.output_text
    assert output.output_text == "你好"


def test_stop_string_no_match_does_not_truncate():
    """If no stop string matches, generation continues — finish_reason
    stays None and output_text contains the full decoded surface."""
    scheduler = _make_scheduler()
    req = _make_request("r4", stop_strings=["XYZ"], prefilled_tokens=[10])

    output, finished = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full="Hello there",
    )

    assert output.finished is False
    assert output.finish_reason is None
    assert finished == set()


def test_empty_stop_list_skips_check():
    """Empty stop list → no truncation, even when the output happens
    to contain content matching empty string semantics."""
    scheduler = _make_scheduler()
    req = _make_request("r5", stop_strings=[], prefilled_tokens=[10])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full="anything goes",
    )

    assert output.finished is False
    assert output.finish_reason is None


def test_empty_string_in_stop_list_is_ignored():
    """A stop string of "" would otherwise match anywhere; the guard in
    the scheduler must skip empty strings to avoid truncating at offset
    0 on every step."""
    scheduler = _make_scheduler()
    req = _make_request("r6", stop_strings=[""], prefilled_tokens=[10])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full="Hello",
    )

    assert output.finished is False
    assert output.finish_reason is None


def test_stop_string_finishes_with_already_set_finish_reason():
    """When BatchGenerator already flagged finish_reason="length", the
    stop-string check must not override it (length wins)."""
    scheduler = _make_scheduler()
    req = _make_request("r7", stop_strings=["World"], prefilled_tokens=[10])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full="Hello World",
        finish_reason="length",
    )

    assert output.finished is True
    # length sticks — the new check only runs when finish_reason is None
    assert output.finish_reason == "length"


def test_stop_string_at_position_zero():
    """Stop marker appears at the very start of the decoded output —
    trimmed_total is "" and new_text must be "" (not the un-trimmed
    surface). Pinned because DeepSeek flagged this edge case as
    untested in round 1."""
    scheduler = _make_scheduler()
    req = _make_request("r0", stop_strings=["Hello"], prefilled_tokens=[10])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full="Hello World",
    )

    assert output.finished is True
    assert output.finish_reason == "stop"
    assert output.output_text == ""
    # new_text must not echo the stop marker
    assert "Hello" not in output.new_text


def test_stop_string_truncates_new_text_for_streaming():
    """The streaming new_text on the truncating step must NEVER contain
    the stop marker, regardless of how prev_text was estimated. This is
    the contract streaming HTTP handlers depend on (test_5 in the
    regression suite)."""
    scheduler = _make_scheduler()
    req = _make_request("rs", stop_strings=[", 5"], prefilled_tokens=[10, 11])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=12,
        decoded_full="1, 2, 3, 4, 5, 6",
    )

    assert output.finish_reason == "stop"
    assert "5" not in output.output_text
    # The new_text emitted on the truncating step must also not leak the
    # stop marker — that's the bug streaming clients hit if the slice
    # logic is wrong.
    assert ", 5" not in output.new_text


@pytest.mark.parametrize(
    "decoded_full,stop_strings,expected_prefix",
    [
        ("foo\nbar", ["\n"], "foo"),
        ("Hello, World!", ["World"], "Hello, "),
        ("a, b, c, 5, 6", [", 5"], "a, b, c"),
    ],
)
def test_stop_string_truncation_parametrized(
    decoded_full, stop_strings, expected_prefix
):
    """Parametrised across the patterns exercised by regression_suite
    tests 1, 2, 5 — each pins that the trimmed output is the prefix
    immediately before the first stop-marker occurrence."""
    scheduler = _make_scheduler()
    req = _make_request("rp", stop_strings=stop_strings, prefilled_tokens=[10])

    output, _ = _run_step(
        scheduler,
        req,
        last_token=11,
        decoded_full=decoded_full,
    )

    assert output.finish_reason == "stop"
    assert output.output_text == expected_prefix
