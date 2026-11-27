# SPDX-License-Identifier: Apache-2.0
# Issue #1694: AutoToolParser streaming loses tool calls when the end marker
# is split across deltas, and nothing flushed a call completed in the final
# tokens. Regression tests for the accumulated-text completion check +
# finish() end-of-stream flush.

from fusion_mlx.tool_parsers import AutoToolParser

OPEN = "<tool_call>"
CLOSE = "\u2009>"
BODY = '{"name": "Read", "arguments": {"f": "a.py"}}'


def feed(parser, chunks):
    results = []
    prev = ""
    acc = ""
    for delta in chunks:
        prev = acc
        acc += delta
        results.append(parser.extract_tool_calls_streaming(prev, acc, delta))
    return results


def collect(results):
    return [tc for r in results if r for tc in (r.get("tool_calls") or [])]


def test_end_marker_in_single_delta_emits():
    parser = AutoToolParser()
    calls = collect(feed(parser, [OPEN, BODY, CLOSE]))
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "Read"


def test_split_end_marker_still_emits_tool_call():
    parser = AutoToolParser()
    calls = collect(feed(parser, [OPEN, BODY, CLOSE[:1], CLOSE[1:]]))
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "Read"


def test_finish_without_text_is_noop():
    parser = AutoToolParser()
    assert parser.finish("") is None


def test_finish_emits_call_completed_without_trailing_delta():
    parser = AutoToolParser()
    # Only the open marker streamed; the complete call (body + close) arrives
    # via finish() -- no delta ever carried a complete end marker.
    feed(parser, [OPEN])
    pending = parser.finish(OPEN + BODY + CLOSE)
    assert pending is not None
    assert pending["tool_calls"][0]["function"]["name"] == "Read"


def test_no_duplicate_emission_across_deltas_and_finish():
    parser = AutoToolParser()
    full = OPEN + BODY + CLOSE
    collect(feed(parser, [full[:6], full[6:22], full[22:]]))
    assert parser.finish(full) is None


def test_incomplete_call_suppressed_not_leaked():
    parser = AutoToolParser()
    results = feed(parser, [OPEN, "still no valid call", "tail"])
    for r in results[1:]:
        assert r is None or not r.get("content")


def test_plain_content_passthrough_unaffected():
    parser = AutoToolParser()
    results = feed(parser, ["hello ", "world"])
    assert results[-1] == {"content": "world"}
