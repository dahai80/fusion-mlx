# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processor — unified reasoning + tool call + sanitization pipeline.

Replaces 500+ lines of duplicated logic across stream_chat_completion,
_stream_anthropic_messages, and stream_completion. NOT a filter chain —
one cohesive orchestrator, because reasoning/tool/sanitize are tightly coupled.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

try:
    from ...domain.events import StreamEvent
except ImportError:
    StreamEvent = None

if TYPE_CHECKING:
    from ...engine.base import GenerationOutput

logger = logging.getLogger(__name__)


def _find_json_start(text: str) -> int:
    """Find the first `{` or `[` that is NOT inside `<think>...</think>` tags.

    Returns the index in ``text``, or -1 if no JSON delimiter found outside
    think blocks.  Handles unclosed `<think>` (still accumulating) by
    treating everything after it as inside the block.
    """
    in_think = False
    i = 0
    while i < len(text):
        # Check for <think> open tag
        if text[i : i + 7] == "<think>":
            in_think = True
            i += 7
            continue
        # Check for </think> close tag
        if text[i : i + 8] == "</think>":
            in_think = False
            i += 8
            continue
        # Outside think block — check for JSON delimiter
        if not in_think and text[i] in ("{", "["):
            return i
        i += 1
    return -1


def _find_json_fence_opener(text: str) -> int:
    """Return the index of the OPENING JSON fence in ``text``, or -1.

    Used by the H-07 scan phase to anchor the JSON-start search past
    any preamble fences. The OPENING JSON fence is the last
    triple-backtick whose payload starts (after an optional ``json``
    language tag and whitespace) with ``{`` or ``[`` — i.e., the
    fence whose body is actual JSON.

    Codex r7 BLOCKING: a preamble may include NON-JSON fenced
    examples (``\\n```python\\nx=1\\n``` ``) before the actual JSON
    fence; the earlier ``buf.find("```")`` anchored on the python
    fence and skipped the real ``` ```json `` opener. Scanning for
    a fence whose payload begins with a JSON delimiter eliminates
    that ambiguity — language-tagged code blocks (python, bash,
    etc.) and string-content fences don't match.

    Codex r10 BLOCKING: the scan must NOT look past the
    matching CLOSING fence of a NON-JSON block. Otherwise a preamble
    like ``\\n```python\\nx\\n```\\n{"k":1}`` would treat the python
    block's closing ``` ``` `` (followed by ``\\n{`` in the next text)
    as an opening JSON fence. We pair each ``` ``` `` with its
    matching closer and skip past the closer before scanning the
    next fence — only the OPENING fences can win, and only those
    whose immediately-following payload begins with a JSON delimiter.

    Returns the index of the first backtick of the chosen fence,
    or -1 if no JSON-bearing fence is found. Multiple matches: the
    LAST one wins (preferring the most recent fence — the model is
    most likely to wrap the FINAL answer).
    """
    best = -1
    i = 0
    n = len(text)
    while i < n:
        pos = text.find("```", i)
        if pos < 0:
            break
        # Skip past the fence + optional ``json`` tag + whitespace.
        cur = pos + 3
        is_json_tagged = text[cur : cur + 4].lower() == "json"
        if is_json_tagged:
            cur += 4
        while cur < n and text[cur] in " \t\r\n":
            cur += 1
        # If the next non-whitespace char is a JSON delimiter, this
        # fence opens a JSON block — eligible as the opener.
        if cur < n and text[cur] in "{[":
            best = pos
        # Codex r10 BLOCKING: advance past the matching CLOSING
        # fence so we don't treat its trailing whitespace + a later
        # JSON delimiter as a fresh opener. If no closer exists yet
        # (streaming: closer hasn't arrived), advance one char past
        # the opener so we don't loop forever on the same position.
        closer = text.find("```", pos + 3)
        i = closer + 3 if closer >= 0 else pos + 3
    return best


def _json_fence_suffix_hold_len(text: str) -> int:
    """Return how many trailing bytes of ``text`` MIGHT start a ``` fence.

    Used by the H-07 streaming fence-strip state machine in
    ``StreamingPostProcessor._guard_closing_fence``. A closing fence on
    the wire is one of ``\\n```\\n``, ```\\n``, or ``` ``` `` alone; the
    longest legitimate fence-prefix this function recognizes at a
    chunk boundary is ``\\n`` + up to two backticks (the next chunk
    would carry the third backtick to complete the fence).

    Returns ``0`` (release everything) on the bare-JSON fast path —
    chunks ending in ``}``, a digit, a quote, etc. flush immediately.
    Only chunks ending in ``\\n``, ``\\r``, or ``` ` `` pay a one-chunk
    delay so the state machine can decide whether the suffix becomes
    a fence.

    Codex r2 BLOCKING: when the trailing suffix is ``\\n```` ``,
    ``\\n```` ``, or ``\\n``` ``, the hold MUST include the leading
    ``\\n`` together with the backticks. Otherwise the next chunk's
    closing-fence completion swallows the backticks but the ``\\n``
    is already on the wire, leaving the stream output ``...}\\n``
    instead of the bare ``...}`` the non-stream path produces — a
    deviation that breaks byte-identical equality with the non-stream
    response shape.
    """
    if not text:
        return 0

    # Walk from the right counting trailing backticks (up to 3).
    trailing_backticks = 0
    while trailing_backticks < 3 and trailing_backticks < len(text):
        if text[-(trailing_backticks + 1)] == "`":
            trailing_backticks += 1
        else:
            break

    if trailing_backticks > 0:
        # Hold ``trailing_backticks`` backticks AND any immediately
        # preceding newline. The newline is part of the canonical
        # closing fence ``\\n``` `` and must not slip onto the wire
        # before the rest of the fence arrives.
        pre = len(text) - trailing_backticks
        if pre > 0 and text[pre - 1] in "\r\n":
            return trailing_backticks + 1
        return trailing_backticks

    # No trailing backticks. A lone ``\\n`` at the end could be the
    # start of ``\\n```\\n``; hold ONE byte. The next chunk's ``` ` ``
    # will trigger the combined re-scan, and we'll re-evaluate the
    # hold above with the backtick(s) appended.
    if text[-1] in "\r\n":
        return 1
    return 0


class StreamingPostProcessorFormatterMixin:
    """Formatter methods for StreamingPostProcessor — SSE formatting / output serialization."""

    # ------------------------------------------------------------------
    # H-07: ```json markdown-fence strip for streaming json_mode
    # ------------------------------------------------------------------
    #
    # Mirrors the non-streaming ``extract_json_from_response`` behaviour
    # (fusion_mlx/api/utils.py) on the SSE delta path. The non-stream
    # response calls that helper after assembling the full text; the
    # stream path concatenated raw tokens without any fence scrub, so
    # joined ``delta.content`` parsed as ```json\n{...}\n``` and clients
    # had to de-fence manually (H-07 / Marisol repro).
    #
    # Design: a per-instance state machine, NOT a post-join regex. Two
    # constraints forced the state-machine shape:
    #
    # 1. Fence tokens are split across delta chunks. Tokenizers fragment
    #    ``\n``` `` arbitrarily ("``", "`json", "\n"); a post-emission
    #    regex would not help because we need to SUPPRESS bytes BEFORE
    #    they reach the wire.
    # 2. The bare-JSON path (model returns ``{...}`` with no fence at
    #    all) must pass through unchanged — we can't unconditionally
    #    buffer.
    #
    # No-op when ``json_mode`` is False (``response_format`` absent or
    # ``"text"``); the gate sits inside ``_apply_json_fence_strip`` so
    # all call sites can call it unconditionally.
    #
    # ``_json_fence_state`` transitions:
    #   "scan"   → "inside"  when the first JSON delimiter (``{``/``[``)
    #                        is seen, with any preceding ``` ```json ``` /
    #                        ``` ``` `` / whitespace / think-content
    #                        bytes suppressed.
    #   "inside" → "done"    when a closing ``` ``` `` is detected (with
    #                        the preceding ``\n`` also dropped).
    #
    # Bounded buffers: ``_json_fence_buffer`` is capped at 4096 bytes.
    # Codex r9 NIT: when the cap is exceeded the implementation TRIMS
    # the buffer to the trailing ``_JSON_FENCE_SCAN_KEEP_SUFFIX`` bytes
    # (just enough for a split opening fence to still be detected on
    # the next chunk) and KEEPS scanning. Older preamble bytes are
    # dropped from memory but NEVER released onto the wire — the
    # json-mode contract is "suppress everything before the first
    # ``{``/``[``" and runaway preambles do not relax that contract
    # (codex r3 BLOCKING).

    # Max bytes to accumulate while scanning for the JSON start. Past
    # this point the buffer is trimmed to the last
    # ``_JSON_FENCE_SCAN_KEEP_SUFFIX`` bytes — JUST enough to detect
    # an opening fence split across the trim boundary — while older
    # preamble bytes are dropped from the buffer. We never RELEASE the
    # preamble onto the wire (codex r3 BLOCKING: doing so would leak
    # the wrapper that the non-stream path strips); we just stop
    # holding the entire history in memory.
    _JSON_FENCE_SCAN_CAP = 4096
    # When the scan cap is hit, retain this many trailing bytes so
    # a split ``...\\n``` `` opening fence can still be detected on
    # the next chunk. ``"```json\n"`` is 8 bytes; 32 gives slack for
    # rare opener variants like ``` ```json   \n ``` and is still
    # negligible vs. the dropped 4KB.
    _JSON_FENCE_SCAN_KEEP_SUFFIX = 32

    def _apply_json_fence_strip(self, content: str) -> str:
        """Strip ```json...``` markdown fence from streaming content.

        See block comment above for design rationale. Returns the
        bytes that are safe to emit on the wire RIGHT NOW; any
        deferred tail bytes are held in ``self._json_fence_tail`` and
        flushed by ``_flush_json_fence_tail`` at stream end.

        No-op when ``json_mode`` is False — the call sites pass content
        through unchanged in that case.
        """
        if not self.json_mode or not content:
            return content

        state = self._json_fence_state

        if state == "done":
            # Closing fence already consumed; any trailing model bytes
            # (often a stray newline / whitespace before EOS) are
            # suppressed so the joined stream stays parseable JSON.
            return ""

        if state == "scan":
            self._json_fence_buffer += content
            buf = self._json_fence_buffer
            # Codex r6 BLOCKING #2: when an opening fence is present
            # in the preamble, the REAL JSON answer starts AFTER the
            # fence, not at the first ``{``/``[`` we see. A preamble
            # like ``Example shape: {"k":...}\n```json\n{"answer":42}\n``` ``
            # has TWO JSON delimiters; the first is illustrative
            # content. Prefer the JSON delimiter that appears after
            # the LAST opening fence in the preamble.
            # Find the first JSON delimiter AND the first ``` ``` ``.
            # The order matters: a fence BEFORE the JSON delimiter
            # is the OPENING fence (we anchor search after it to
            # skip an illustrative-example JSON in the preamble —
            # codex r6 BLOCKING #2); a fence AFTER the first JSON
            # delimiter (or no fence at all) is irrelevant to the
            # scan-phase anchor — that's the closing fence (the
            # ``_guard_closing_fence`` walker handles it later).
            json_start = _find_json_start(buf)
            fence_pos = _find_json_fence_opener(buf)
            # Codex r8 BLOCKING #1: re-anchor whenever a JSON-bearing
            # fence opener exists ANYWHERE in the buffer — not only
            # when ``fence_pos < json_start``. A preamble like
            # ``Example: {"k":1}\n```json\n{"answer":42}\n``` `` has
            # the example JSON BEFORE the fence; without unconditional
            # re-anchoring we'd land on the example. ``_find_json_fence_opener``
            # already requires the fence's payload to start with
            # ``{``/``[``, so the candidate is reliable.
            if fence_pos >= 0:
                # Opening fence in preamble. Re-anchor the JSON
                # search to after the fence + optional ``json`` tag
                # + whitespace, so an illustrative example JSON
                # before the fence does NOT win.
                #
                # Codex r7 BLOCKING: ``_find_json_fence_opener`` looks
                # for the LAST ``` ```json `` (case-insensitive) before
                # the first JSON delimiter, then falls back to a bare
                # ``` ``` ``. This handles preambles that include
                # NON-JSON fenced examples (``\\n```python\\n...\\n``` ``)
                # before the real JSON fence — those earlier fences
                # don't anchor the search.
                search_from = fence_pos + 3
                if buf[search_from : search_from + 4].lower() == "json":
                    search_from += 4
                while search_from < len(buf) and buf[search_from] in " \t\r\n":
                    search_from += 1
                rel_start = _find_json_start(buf[search_from:])
                if rel_start < 0:
                    # Opener seen but no JSON delimiter yet — keep
                    # scanning. Apply the scan-cap trim if needed.
                    if len(buf) > self._JSON_FENCE_SCAN_CAP:
                        self._json_fence_buffer = buf[
                            -self._JSON_FENCE_SCAN_KEEP_SUFFIX :
                        ]
                    return ""
                json_start = search_from + rel_start
                # Codex r8 BLOCKING #2: record that an opening fence
                # was actually consumed in the scan phase. The
                # closing-fence walker uses this flag to decide
                # whether to suppress a later ``` `` ``` — a bare-JSON
                # stream (no opening fence) must pass markdown after
                # the JSON root through unchanged.
                self._json_fence_opener_consumed = True
            elif json_start < 0:
                # No JSON delimiter AND no opening fence yet. Keep
                # scanning. If the buffer grew past the cap, drop
                # the OLD bytes — but keep enough of the suffix to
                # catch a fence-opener split across the boundary.
                # Codex r3 BLOCKING: the earlier draft RELEASED
                # the entire >4KB buffer raw, which leaked the
                # preamble + opening fence onto the wire (the
                # opposite of what response_format=json_* requires).
                # The contract for json_mode is "suppress everything
                # before the first ``{``/``[``", and that contract
                # must hold regardless of preamble length.
                if len(buf) > self._JSON_FENCE_SCAN_CAP:
                    self._json_fence_buffer = buf[-self._JSON_FENCE_SCAN_KEEP_SUFFIX :]
                return ""
            # else: json_start is set; fence (if any) was AFTER the
            # JSON delimiter — the ``_guard_closing_fence`` walker
            # will suppress it. Found the JSON start. Strip everything
            # before it (preamble + opening fence). Symmetric with the
            # non-stream ``extract_json_from_response``'s
            # ``rfind('{') ... endswith('}')`` peel: bytes BEFORE the
            # first ``{``/``[`` are the wrapper, and we are done
            # with them.
            payload = buf[json_start:]
            self._json_fence_state = "inside"
            self._json_fence_buffer = ""
            return self._guard_closing_fence(payload)

        # state == "inside" — pass content through, guarding against the
        # closing fence.
        return self._guard_closing_fence(content)

    def _guard_closing_fence(self, content: str) -> str:
        """Hold back the last few bytes that might start a closing ``` fence.

        The streaming wire MUST suppress the trailing ``\\n```\\n``
        before it lands in the SSE delta. Bytes that COULD be the
        beginning of such a fence are held in ``_json_fence_tail`` and
        only released once we know they are not a fence.

        Tail-hold size is ``_JSON_FENCE_TAIL_HOLD`` so the typical
        ``\\n```\\n`` / ``\\r\\n```\\r\\n`` patterns fit entirely in
        the deferred buffer.

        Bare-JSON streams pay no latency: ``_json_fence_suffix_hold_len``
        returns 0 when the chunk does not end in a fence-prefix
        character (``\\n`` / ``\\r`` / ``` ` ``), so a stream of
        ``{"k": 1}`` chunks flushes immediately. Only chunks whose
        last byte LOOKS like the start of a closing fence are
        deferred — that one chunk's bytes are held until the next
        chunk arrives or ``_flush_json_fence_tail`` runs in
        ``finalize()``.
        """
        # Prepend any previously-held tail so we re-examine the full
        # suffix as a single string.
        combined = self._json_fence_tail + content
        self._json_fence_tail = ""

        # Codex r8 BLOCKING #2: bare-JSON streams (no opening fence
        # consumed in the scan phase) must not have closing-fence
        # detection or fence-tail hold applied. The non-stream
        # ``extract_json_from_response`` leaves unfenced text alone;
        # streaming has to match. Without this fast-path, a model
        # that returns ``{...}\n\nHere's how I did it:\n```python...```
        # would get truncated at the first ``` ``` ``. Flush any held
        # tail and pass the rest through unchanged.
        if not self._json_fence_opener_consumed:
            return combined

        # Walk the buffer character-by-character, tracking JSON-string
        # state, so the FIRST ``` `` ` ``` we treat as a closing fence
        # is actually OUTSIDE a string literal. Codex r1 BLOCKING #1:
        # the previous leftmost-``find("```")`` truncated valid JSON
        # whose VALUES happened to contain triple-backticks (e.g.
        # ``{"markdown": "```python\\nx\\n```"}``).
        #
        # The walker starts from the per-instance snapshot of the
        # (in_string, escape) flags taken at the held-tail boundary on
        # the previous call. ``combined`` is structured as
        # ``[previously-held tail] + [fresh content]`` and the snapshot
        # is exactly the flag state AT the start of that previously-
        # held tail — so the walker over ``combined`` from index 0
        # produces the correct flag state at every position.
        in_string = self._json_fence_in_string
        escape = self._json_fence_string_escape
        depth = self._json_fence_bracket_depth
        # ``json_root_closed_at`` records the index IMMEDIATELY AFTER
        # the brace/bracket that closed the JSON root (depth returned
        # to 0 from > 0). For json_mode the contract is "emit ONLY the
        # JSON object", matching the non-stream
        # ``extract_json_from_response`` shape. Everything after the
        # closing brace is wrapper/explanation/fence/whitespace and
        # gets suppressed — whether or not a triple-backtick follows.
        # Codex r5 BLOCKING: the earlier draft only suppressed AT the
        # first ``` ``` ``, leaving trailing prose like
        # ``\nHere is code:`` on the wire.
        # Walker tracks JSON-root close AND the closing ``` ``` ``
        # fence. ``json_root_closed_at`` is the index of the brace
        # that returned depth to 0 (top-level close). ``fence_idx``
        # is the index of the FIRST ``` ``` `` that appears OUTSIDE
        # a JSON string literal AND AT depth==0 (i.e. after the JSON
        # root has fully closed). For the saw-open-fence path we
        # truncate at fence_idx; if the buffer contains a root-close
        # but no fence yet, we MUST continue holding (the model may
        # still be emitting whitespace between root-close and the
        # closing fence — that whitespace is suppressed regardless,
        # but truncating at root-close would lose the chance to
        # recognise a JSON value that contains a literal terminating
        # ``}`` followed by more content the model still wants to
        # emit, e.g. the codex r6 #1 multi-value concern). The
        # contract: opening fence seen => terminator is the closing
        # fence, not the root close.
        # Codex r9 BLOCKING #1: track the FIRST index at which the JSON
        # root closes (depth returns from 1 to 0). For fenced-mode
        # streams the contract is "emit the JSON object only" — bytes
        # between the root close and the closing fence are
        # wrapper / explanation that the non-stream
        # ``extract_json_from_response`` strips along with the fence.
        # We must NOT emit those bytes onto the wire as they arrive in
        # an earlier chunk than the closing ``` ``` ``. Track the
        # ROOT_CLOSE position so we can suppress everything from it
        # onward when no fence is found in this chunk (the next chunk
        # might carry both extra prose AND the fence — we hold both).
        fence_idx = -1
        # Codex r9 BLOCKING #1: ``root_close_at`` tracks the FIRST
        # index in ``combined`` at which the JSON root closed. When
        # the persistent ``_json_root_closed`` latch is already set
        # (from a PRIOR call's walker), every byte of ``combined`` is
        # post-root-close — root_close_at = 0 so we suppress from the
        # start. Otherwise we scan for the first depth-1→0
        # transition and record its position+1 (the byte AFTER the
        # closing brace/bracket).
        root_close_at = 0 if self._json_root_closed else -1
        i = 0
        n = len(combined)
        while i < n:
            c = combined[i]
            if escape:
                escape = False
                i += 1
                continue
            if c == "\\" and in_string:
                escape = True
                i += 1
                continue
            if c == '"':
                in_string = not in_string
                i += 1
                continue
            if not in_string:
                if c in "{[":
                    depth += 1
                    i += 1
                    continue
                if c in "}]":
                    # Defensive clamp on negative depth (malformed
                    # unbalanced output).
                    prev_depth = depth
                    depth = max(depth - 1, 0)
                    if prev_depth == 1 and depth == 0 and root_close_at < 0:
                        # First top-level close — record the position
                        # right AFTER this closing brace/bracket.
                        root_close_at = i + 1
                    i += 1
                    continue
                # Triple-backtick OUTSIDE a JSON string AND OUTSIDE
                # the JSON body (depth==0). Codex r1 + r5 + r6
                # combined: a backtick inside a string literal is
                # value content, a backtick inside the structural
                # body (between matched braces) is also content
                # (e.g. JSON containing a stringified code block);
                # the ONLY position that means "closing fence" is
                # at depth 0 after the root has closed.
                if c == "`" and depth == 0 and combined[i : i + 3] == "```":
                    fence_idx = i
                    break
            i += 1

        if fence_idx >= 0:
            # Closing fence found at depth 0. Trim payload at the
            # FIRST root close (codex r9 BLOCKING #1: drop any
            # explanation prose between the JSON root and the fence,
            # symmetric with the non-stream
            # ``_strip_markdown_code_block`` peel), then drop the
            # newline whitespace.
            cut = root_close_at if 0 <= root_close_at <= fence_idx else fence_idx
            payload = combined[:cut].rstrip("\r\n")
            self._json_fence_state = "done"
            return payload

        # Codex r9 BLOCKING #1: in fenced mode, once the JSON root has
        # closed we MUST suppress every byte after the close until the
        # closing fence arrives. Otherwise a chunk-boundary like
        # ``{"k":1}\nextra`` (chunk N) + ``` ``` `` (chunk N+1) leaks
        # ``\nextra`` onto the wire before the fence terminator is
        # seen — the joined stream would be ``{"k":1}\nextra``,
        # invalid JSON for any client that runs ``json.loads`` on
        # the assembled deltas. Emit only the bytes UP TO root close
        # (the JSON object itself) and HOLD all post-close bytes as
        # tail. The next call's walker re-examines the full
        # tail+content buffer for the fence; the tail is bounded by
        # one chunk's worth of post-close bytes per call.
        if root_close_at >= 0:
            head = combined[:root_close_at]
            self._json_fence_tail = combined[root_close_at:]
            # Snapshot flags AT the root-close boundary. ``head`` ends
            # at depth 0 outside any string, so reset the snapshot to
            # that baseline. The persistent ``_json_root_closed`` latch
            # ensures the next call's walker treats ``combined``'s very
            # first byte as already past the close — so any new
            # post-close bytes are also held until the fence arrives.
            self._json_fence_in_string = False
            self._json_fence_string_escape = False
            self._json_fence_bracket_depth = 0
            self._json_root_closed = True
            return head

        # No complete fence yet. Compute the minimum suffix-hold so the
        # NEXT chunk can still detect a fence that straddles the chunk
        # boundary. A closing fence on the wire is one of:
        #
        #   ``\\n```\\n``  (canonical)
        #   ```\\n``       (no trailing newline; ``` could land at EOS)
        #   ```            (fence-only line, no newlines)
        #
        # The longest prefix that could legitimately appear at the END
        # of a non-fence emission is 4 bytes: ``\\n`` followed by up to
        # two backticks (the next chunk would carry the third backtick
        # to complete the fence). Anything longer than that we KNOW is
        # real JSON body and can release immediately. Anything shorter
        # that ends in ``\\n`` / ``` ` `` / ``` `` `` we hold; anything
        # else we release wholesale.
        #
        # Codex r1 BLOCKING #1 redux: when the trailing fence-prefix
        # chars are INSIDE a JSON string literal (the running
        # ``in_string`` flag from the walker says so), they can't be
        # the start of a closing fence — release them too.
        # At this point the walker advanced ``in_string`` / ``escape``
        # to the END of the entire ``combined`` buffer (no fence found).
        # We need to snapshot the flags as they were at the START of
        # the soon-to-be-held tail (so the next chunk's walker can
        # resume there). The held-tail length depends on whether we're
        # inside a string literal: a trailing ``` ` `` / ``\\n`` inside
        # a string can't begin a fence and should flush immediately.
        if in_string:
            hold_len = 0
        else:
            hold_len = _json_fence_suffix_hold_len(combined)

        if hold_len == 0:
            # Snapshot the END-of-buffer flags (== start of next chunk).
            self._json_fence_in_string = in_string
            self._json_fence_string_escape = escape
            self._json_fence_bracket_depth = depth
            return combined
        if hold_len >= len(combined):
            # The whole buffer is suspicious tail. The flags at the
            # start of this tail are the snapshot we entered with —
            # leave instance fields untouched (they already reflect
            # that boundary).
            self._json_fence_tail = combined
            return ""
        emit = combined[:-hold_len]
        self._json_fence_tail = combined[-hold_len:]
        # Snapshot the flags AT the start of the held tail by
        # re-walking from the prior snapshot through ``emit``. Held
        # tail will never include a quote / brace (the hold chars
        # are ``\\n`` / ``\\r`` / ``` ` ``), so this is a defensive
        # replay rather than load-bearing — but it keeps the
        # next-chunk walker mechanically correct.
        prior_in_string, prior_escape, prior_depth = (
            self._json_fence_in_string,
            self._json_fence_string_escape,
            self._json_fence_bracket_depth,
        )
        for c in emit:
            if prior_escape:
                prior_escape = False
                continue
            if c == "\\" and prior_in_string:
                prior_escape = True
                continue
            if c == '"':
                prior_in_string = not prior_in_string
                continue
            if not prior_in_string:
                if c in "{[":
                    prior_depth += 1
                elif c in "}]":
                    prior_depth = max(prior_depth - 1, 0)
        self._json_fence_in_string = prior_in_string
        self._json_fence_string_escape = prior_escape
        self._json_fence_bracket_depth = prior_depth
        return emit

    def _filter_events_for_json_fence(
        self, events: list[StreamEvent], *, drain_tail: bool = False
    ) -> list[StreamEvent]:
        """Run ``_apply_json_fence_strip`` over a list of StreamEvents.

        Walks the event list and rewrites every ``content`` field
        (whether on a ``type="content"`` event or on a ``type="finish"``
        event with merged content). When the strip pass empties the
        content of a plain ``type="content"`` event, the event is
        dropped — pristine ``content`` deltas with empty payload would
        otherwise emit an empty SSE chunk.

        Codex r4 BLOCKING #1: rewrites use ``dataclasses.replace`` so
        all other ``StreamEvent`` fields the inner processors may have
        attached (``metadata``, ``finish_reason``, ``tool_calls_detected``,
        future fields) are preserved. The earlier draft constructed a
        minimal ``StreamEvent(type=..., content=...)`` and dropped the
        rest.

        Codex r4 BLOCKING #2: when ``drain_tail=True`` (set by
        ``finalize()``), any held tail bytes are merged into the
        LAST emitted content/finish event in a single pass — avoids
        emitting tail content AFTER a finish marker. When no such
        event exists, the tail is appended as its own content event
        at the END of the list (still before any terminal-finish
        chunk the caller will assemble).

        No-op fast path when ``json_mode`` is False — caller treats the
        list as already-filtered.
        """
        if not self.json_mode:
            return events

        from dataclasses import replace as _dc_replace

        filtered: list[StreamEvent] = []
        for ev in events:
            if ev.type == "content":
                stripped = self._apply_json_fence_strip(ev.content or "")
                if stripped:
                    filtered.append(_dc_replace(ev, content=stripped))
                # else: fully suppressed (fence/preamble/closer) — drop.
            elif ev.type == "finish":
                # Finish events can carry merged content (the route's
                # buffered-finish merge path). Strip it the same way.
                # Tail draining happens BELOW (after the walk) so we
                # don't double-drain if the caller also passed
                # ``drain_tail=True``.
                terminal = ev.content or ""
                if terminal:
                    terminal = self._apply_json_fence_strip(terminal)
                filtered.append(_dc_replace(ev, content=terminal or None))
            else:
                # reasoning / tool_call / other event types: pass through.
                filtered.append(ev)

        if drain_tail:
            tail = self._flush_json_fence_tail()
            if tail:
                # Merge into the LAST content-bearing event in one pass
                # (codex r4 BLOCKING #2 — avoid ordering finalize tail
                # AFTER a finish event the inner branch emitted). Walk
                # from the right; prefer ``finish`` (merges into the
                # terminal SSE chunk), fall back to ``content`` (extends
                # the last content delta), else append a new content
                # event at the END.
                merged = False
                for i in range(len(filtered) - 1, -1, -1):
                    if filtered[i].type in ("finish", "content"):
                        prev = filtered[i].content or ""
                        filtered[i] = _dc_replace(filtered[i], content=prev + tail)
                        merged = True
                        break
                if not merged:
                    filtered.append(StreamEvent(type="content", content=tail))

        return filtered

    def _flush_json_fence_tail(self) -> str:
        """Release any deferred tail bytes at stream end.

        Called from ``finalize()`` so the bare-JSON path (model returned
        ``{...}`` with NO closing fence) still flushes the final few
        bytes that were held back in case they were the start of a
        fence. Idempotent: clears the tail.

        Codex r1 BLOCKING #2: flush the tail UNCHANGED unless the state
        machine has already transitioned to ``"done"`` (closing fence
        detected). The earlier draft rstripped backticks at EOS, which
        would corrupt a valid bare JSON whose final string value
        legitimately ends with backticks (``{"text":"```"}`` streamed
        with the trailing ``\\"}`` arriving in the same chunk as a
        leading ``` ` `` in the value). When ``state == "done"`` the
        closing fence was already structurally detected and the tail
        is dead bytes — we still drop them.
        """
        if not self.json_mode:
            return ""
        if self._json_fence_state == "done":
            # Closing fence detected; whatever sat in the tail belongs
            # AFTER the fence and is suppressed.
            self._json_fence_tail = ""
            return ""
        # Codex r9 BLOCKING #1: in fenced mode, if the JSON root has
        # closed but the closing ``` ``` `` never arrived (truncated
        # stream / model stopped mid-fence), the held tail is
        # post-root-close prose that the non-stream
        # ``extract_json_from_response`` would have peeled. Drop it
        # so the streaming bytes match the non-stream shape.
        if self._json_fence_opener_consumed and self._json_root_closed:
            self._json_fence_tail = ""
            return ""
        tail = self._json_fence_tail
        self._json_fence_tail = ""
        return tail

    # R10-C8 (Mira r10-R1) — UI-TARS-style tool-prose-prefix scrubber.
    #
    # Mira's r10 dogfood evidence (E SSE) shows the chat lane streaming
    # 12 plain-content chunks like ``"Tool"``, ``":"``, ``" get"``,
    # ``"_weather"``, ``"\n"``, ``"Parameters"``, ``":"``, ``" location"``,
    # ``"="``, ``"Paris"``, ``"\n"`` BEFORE the single structured
    # ``delta.tool_calls`` chunk. A client rendering ``delta.content``
    # live shows garbage prefixing the actual tool call. The wire-scrub
    # from PR #806 only filters ``<tool_call>...</tool_call>`` literal
    # spans; the UI-TARS action parser's natural-language preamble
    # falls through to the standard content path.
    #
    # Strategy: lightweight buffer-and-emit state machine. When tools
    # are requested AND a content event matches a tool-prose-prefix
    # pattern, hold it back. Discard the buffer when a tool_call event
    # arrives in the same stream (the model just confirmed the prose
    # was a tool-dispatch preamble). Release the buffer back to the
    # wire when the buffer exceeds the cap (a model legitimately
    # discussing ``"Tool: foo"`` in prose isn't censored). Active
    # ONLY when ``tools_requested`` is True so non-tool requests pay
    # zero overhead.
    #
    # The pattern set is intentionally narrow — only the UI-TARS /
    # function-calling stylesheet prefixes that have NO other plausible
    # plain-prose use at the start of a streamed reply. Each entry is
    # a compiled regex that must match the FULL leading content
    # buffer (re.match anchored at position 0).
    _TOOL_PROSE_PREFIX_RES: tuple = (
        # ``Tool: <name>\nParameters: ...`` (UI-TARS evidence — Mira E)
        re.compile(
            r"^(?:Tool|Action|Function)\s*:\s*[A-Za-z_][A-Za-z0-9_-]*", re.DOTALL
        ),
        # Bare ``Tool:`` / ``Action:`` / ``Function:`` with no name yet —
        # the prose hasn't progressed to the name token yet but the
        # opener is unambiguous.
        re.compile(r"^(?:Tool|Action|Function)\s*:\s*$", re.DOTALL),
        # Partial prefix while the name token is still streaming
        # (chunk boundary split ``"Tool"`` ``":"`` ``" get_weather"``).
        re.compile(r"^(?:Tool|Action|Function)$", re.DOTALL),
        re.compile(r"^(?:Tool|Action|Function)\s*$", re.DOTALL),
    )
    # Soft cap on buffer growth. The longest plausible UI-TARS preamble
    # is ``Tool: <name>\nParameters: <kv-pairs>\n`` — for a 64-char name
    # plus 256 chars of param prose that's ~330 bytes. 512 leaves slack
    # for parser variants without holding back legitimate prose
    # indefinitely.
    _TOOL_PROSE_MAX_HOLD: int = 512

    def _matches_tool_prose_prefix(self, text: str) -> bool:
        """True iff ``text`` matches a tool-dispatch prose preamble.

        Used to gate hold-back: a buffer that no longer matches any of
        the patterns is released back to the wire because the model
        moved on to legitimate content. A buffer that grows past
        ``_TOOL_PROSE_MAX_HOLD`` is also released (the tool parser
        clearly isn't going to consume it).
        """
        if not text:
            return False
        for pattern in self._TOOL_PROSE_PREFIX_RES:
            if pattern.match(text):
                return True
        return False

    def _filter_events_for_tool_prose(
        self, events: list[StreamEvent]
    ) -> list[StreamEvent]:
        """Hold back tool-prose-prefix content events; discard on tool_call.

        Walks the event list once. ``content`` events extend the
        prose buffer when (a) the buffer is empty and the content
        matches a prose-prefix pattern, OR (b) the buffer is non-
        empty (we're already inside a hold). ``tool_call`` events
        clear the buffer (the model confirmed the prose was a tool-
        dispatch preamble — discard so it never reaches the wire,
        matching the R10-M4 ``\\n\\n`` trailing whitespace requirement).
        Other event types pass through unchanged.

        When the held buffer grows past ``_TOOL_PROSE_MAX_HOLD`` OR no
        longer matches any prefix pattern (the model moved on to
        legitimate prose), flush it as a single content event in the
        position the LAST content event would have occupied. This
        preserves event ordering for downstream consumers.

        No-op when ``tools_requested`` is False — non-tool requests
        pay zero cost. R10-M4 trailing-whitespace handling: when the
        buffer trails with ``\\n\\n`` AND a tool_call is later detected,
        the buffer (including the trailing whitespace) is discarded
        as part of the prose preamble — no separate scrubber needed.
        """
        if not self.tools_requested:
            return events

        from dataclasses import replace as _dc_replace

        out: list[StreamEvent] = []
        for ev in events:
            if ev.type == "tool_call":
                # Tool call confirmed — every byte we held was a
                # dispatch preamble. Discard the buffer and pass the
                # tool_call through. ``_tool_prose_active`` stays
                # latched True so later content events in the same
                # turn (e.g. an explanatory tail the parser surfaces)
                # also get the prefix check, but most parsers go
                # straight to ``finish`` after the call so this is a
                # short-lived state.
                self._tool_prose_buffer = ""
                out.append(ev)
                continue
            if ev.type != "content":
                out.append(ev)
                continue
            chunk = ev.content or ""
            # R10-M4: when the prose buffer is non-empty AND this
            # chunk is just trailing whitespace, fold it into the
            # buffer so a later tool_call discards it too. The
            # alternative (emit ``"\\n\\n"`` to the wire) is the
            # exact leak r10-R1 surfaced on the canonical-tag scrub.
            combined = self._tool_prose_buffer + chunk
            if self._tool_prose_active or self._matches_tool_prose_prefix(combined):
                self._tool_prose_active = True
                self._tool_prose_buffer = combined
                # Soft cap — release as legitimate prose if we held
                # too much. A model legitimately discussing the word
                # ``Tool:`` in prose passes through here.
                if len(
                    self._tool_prose_buffer
                ) > self._TOOL_PROSE_MAX_HOLD or not self._matches_tool_prose_prefix(
                    self._tool_prose_buffer
                ):
                    released = self._tool_prose_buffer
                    self._tool_prose_buffer = ""
                    self._tool_prose_active = False
                    out.append(_dc_replace(ev, content=released))
                # Else: keep holding; emit nothing for this chunk.
                continue
            out.append(ev)
        return out

    def _flush_tool_prose_buffer(self) -> str:
        """Drain the tool-prose buffer at stream end.

        Called from ``finalize()`` so a held buffer that never saw a
        matching tool_call IS released to the client (the model
        legitimately ended a turn with ``"Tool: ...\\n"`` prose —
        rare, but censoring it would be a silent drop).
        """
        if not self.tools_requested:
            return ""
        # If a tool_call was detected this turn, every byte we held
        # was a dispatch preamble — drop it silently. This is the
        # canonical R10-C8 + R10-M4 path: model emitted the prose
        # preamble, parser surfaced the structured call, prose is
        # discarded as wire residue.
        if self.tool_calls_detected:
            self._tool_prose_buffer = ""
            self._tool_prose_active = False
            return ""
        tail = self._tool_prose_buffer
        self._tool_prose_buffer = ""
        self._tool_prose_active = False
        return tail

    def _build_tool_call_event(self, items) -> StreamEvent:
        """Build a tool_call StreamEvent from an iterable of {id, name, arguments} dicts.

        Used by both finalize() branches (configured parser succeeded, and the
        cross-format ``parse_tool_calls`` fallback) so the two paths can't drift
        in wire shape.

        R11-A: also bumps ``_tool_calls_emitted_to_wire`` so the finalize-
        recovered path satisfies the invariant
        ``finish_reason="tool_calls" ⇒ ≥1 tool_call delta on the wire``.
        The route layer reads the finalize-produced event and treats it as
        ``fallback_tool_calls``, splicing it into the buffered_finish
        terminal chunk (see ``routes/chat.py`` SSE-FALLBACK-TC-MERGED) — so
        a wire ``delta.tool_calls`` IS emitted, and the counter reflects
        that.
        """
        tcs = [
            {
                "index": i,
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for i, tc in enumerate(items)
        ]
        self._tool_calls_emitted_to_wire += len(tcs)
        return StreamEvent(
            type="tool_call",
            tool_calls=tcs,
            finish_reason="tool_calls",
            tool_calls_detected=True,
        )

    def _compute_finish_reason(self, output: GenerationOutput) -> str | None:
        if not output.finished:
            return None
        # R11-A invariant: ``finish_reason="tool_calls"`` is emitted ONLY
        # when at least one ``tool_call`` StreamEvent has actually reached
        # the wire on this turn. Pre-fix this gated on
        # ``tool_calls_detected``, which the forced-``tool_choice`` filter
        # flipped to True even when it dropped every anchor as spec-
        # violating scratch (qwen3 emitting ``arguments="20230805"`` —
        # bare-int root). The wire then carried zero ``delta.tool_calls``
        # chunks but a ``finish_reason="tool_calls"`` terminal — bug
        # R11-V1, ~50% reproducible on Qwen3-0.6B + ``tool_choice="required"``.
        # The route layer separately re-stamps ``finish_reason`` to
        # ``"tool_calls"`` when ``finalize()`` recovers a call via the
        # cross-format fallback (see ``routes/chat.py`` buffered-finish
        # merge), so this gate is the floor — the route-layer override
        # only fires UP from ``output.finish_reason``, never DOWN.
        if self._tool_calls_emitted_to_wire > 0:
            return "tool_calls"
        return output.finish_reason

    def _make_finish_event(self, output: GenerationOutput) -> StreamEvent:
        return StreamEvent(
            type="finish",
            finish_reason=self._compute_finish_reason(output),
            tool_calls_detected=self.tool_calls_detected,
        )
