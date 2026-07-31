# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processor — unified reasoning + tool call + sanitization pipeline.

Replaces 500+ lines of duplicated logic across stream_chat_completion,
_stream_anthropic_messages, and stream_completion. NOT a filter chain —
one cohesive orchestrator, because reasoning/tool/sanitize are tightly coupled.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from ...api.tool_calling import parse_tool_calls
from ...api.utils import sanitize_output, strip_special_tokens

try:
    from ...domain.events import StreamEvent
except ImportError:
    StreamEvent = None

if TYPE_CHECKING:
    try:
        from ...config.server_config import ServerConfig
    except ImportError:
        from ...config import ServerConfig  # type: ignore[no-redef]
    from ...engine.base import GenerationOutput

logger = logging.getLogger(__name__)


from .formatters import (
    StreamingPostProcessorFormatterMixin,
    _find_json_fence_opener,
    _find_json_start,
)
from .parsers import StreamingPostProcessorParserMixin


class StreamingPostProcessor(
    StreamingPostProcessorParserMixin, StreamingPostProcessorFormatterMixin
):
    """Processes streaming engine output into StreamEvents.

    Handles:
    1. Channel routing (OutputRouter models like Gemma 4)
    2. Reasoning extraction (text-based parsers for Qwen3, DeepSeek, MiniMax)
    3. Tool call streaming detection (incremental parser)
    4. Output sanitization (strip special tokens, markup)

    Usage::

        processor = StreamingPostProcessor(cfg, request)
        processor.reset()
        async for output in engine.stream_chat(...):
            for event in processor.process_chunk(output):
                yield format_for_my_api_spec(event)
        for event in processor.finalize():
            yield format_for_my_api_spec(event)
    """

    def __init__(
        self,
        cfg: ServerConfig,
        tools_requested: bool = False,
        enable_thinking: bool | None = None,
        json_mode: bool = False,
        request: dict | None = None,
        reasoning_max_tokens: int | None = None,
    ):
        self.cfg = cfg
        self.tools_requested = tools_requested
        self.json_mode = json_mode
        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # When set and the model is still emitting on the reasoning
        # channel after this many tokens, the processor force-closes
        # the channel: text-parser engines see an injected ``</think>``
        # marker on the next chunk so subsequent text routes to content;
        # channel-routed engines (gemma4 / harmony) reclassify further
        # reasoning deltas as content. ``None`` means "no cap" and is
        # the documented default.
        self._reasoning_max_tokens = reasoning_max_tokens
        # Approximate count of reasoning tokens we've emitted so far.
        # Engine deltas don't always carry per-channel token counts, so
        # we approximate from the text length divided by 4 (the OpenAI
        # spec's documented chars→tokens heuristic — same constant
        # ``_build_usage`` uses in helpers.py for the reasoning_tokens
        # split). This intentionally tracks EMITTED reasoning, not the
        # raw model output, so the cap counts what the client sees.
        self._reasoning_tokens_emitted = 0
        # Flag: cap was hit. Set once the running count crosses the
        # threshold; once True, the channel-routed branch reclassifies
        # subsequent reasoning chunks as content and the text-parser
        # branch injects ``</think>`` into the parser stream so it
        # flips to content. Single-bit latch — never reset within a
        # request (the cap is monotonic).
        self._reasoning_cap_hit = False
        # Whether the text-parser injection has already fired. Idempotent
        # guard so we don't keep stuffing ``</think>`` into every
        # subsequent chunk after the cap fired.
        self._reasoning_close_injected = False
        # Forwarded to streaming tool parsers — qwen3_coder needs request.tools
        # for schema-driven type conversion (#171). Without it, raw XML leaks
        # into delta.content instead of structured tool_calls deltas.
        self.request = request
        # When the client explicitly sets enable_thinking=False, the chat
        # template suppresses the <think> generation prompt and the model
        # answers directly. The streaming reasoning parser's implicit-think
        # heuristic (treat ambiguous tokens as reasoning until </think> is
        # seen) misclassifies that direct answer as reasoning_content,
        # leaving content empty. Track the explicit signal so process_chunk
        # can skip the reasoning path in that case.
        self.enable_thinking = enable_thinking
        # R8-M2 (2026-06-22): one-way latch tracking whether the model
        # has emitted an explicit ``<think>`` token despite
        # ``enable_thinking=False``. Set by
        # ``_should_route_through_reasoning`` when the opener shows up
        # in the accumulated buffer; promotes the bypass path back to
        # the reasoning lane so the wrapper bytes split correctly
        # instead of leaking into ``delta.content``. Latched so the
        # decision doesn't oscillate as the accumulator grows past the
        # opener.
        self._explicit_think_seen = False
        # R10-C7 (2026-06-23): one-way latch tracking whether the
        # ``_process_standard`` path has already emitted plain content
        # for this request. Once True, ``_should_route_through_reasoning``
        # refuses to promote a mid-content ``<think>`` token back into
        # the reasoning lane — the R8-M2 head-of-buffer anchor relies
        # on ``accumulated_text`` reflecting everything streamed so far,
        # but ``_process_standard`` does NOT mutate ``accumulated_text``
        # (only ``_process_with_reasoning`` does, at lines 2158/2163).
        # Without this latch, a plain answer that mentions a literal
        # ``<think>`` token (e.g. "Reply with ```<think>``` as code")
        # would, after the FIRST plain-content chunk was emitted via
        # ``_process_standard`` (accumulator still empty), see the
        # subsequent ``<think>`` chunk's ``probe = "" + "<think>..."``
        # match ``head.startswith("<think>")`` and latch ``_explicit_think_seen``,
        # rerouting the rest of the response to ``delta.reasoning``.
        # Mira r10-R1 root cause for the post-r8-C regression. The
        # latch is reset by ``reset_for_new_request`` so a re-used
        # processor doesn't carry the prior request's state.
        self._standard_content_observed = False

        # Per-request parser instances — each streaming request gets its
        # own parser to avoid state corruption under concurrent
        # BatchedEngine requests.
        #
        # Production path: reasoning_parser_name / tool_call_parser are set
        # at startup → _create_*() builds a fresh instance per request.
        #
        # Legacy/test path: cfg.reasoning_parser / cfg.tool_parser_instance
        # may be pre-built (mocks in tests, or singleton from server.py).
        # When reasoning_parser_name is set, always create fresh.
        if cfg.reasoning_parser_name:
            self.reasoning_parser = self._create_reasoning_parser(cfg)
        else:
            self.reasoning_parser = cfg.reasoning_parser  # None or injected mock

        # R10-M1 (2026-06-23): propagate the request-level
        # ``enable_thinking`` to parsers that expose ``set_enable_thinking``
        # (UI-TARS). Defense in depth so the streaming bypass survives a
        # future dispatcher refactor that calls
        # ``extract_reasoning_streaming`` outside the
        # ``_should_route_through_reasoning`` gate. Parsers that don't
        # expose the setter (qwen3 / deepseek_r1 / gemma4 / harmony /
        # gpt_oss / think_parser) are unaffected — they consume
        # ``enable_thinking`` via the non-streaming ``extract_reasoning``
        # kwarg path or via their own template-driven branching.
        if self.reasoning_parser is not None:
            _set = getattr(self.reasoning_parser, "set_enable_thinking", None)
            if callable(_set):
                try:
                    _set(enable_thinking)
                except Exception:
                    # Setter is best-effort — a buggy override must not
                    # block request construction. The non-stream extract
                    # path still honours the flag explicitly.
                    pass

        self._tool_parser_request_local = False
        if cfg.tool_call_parser:
            self.tool_parser = self._create_tool_parser(cfg, tools_requested)
            self._tool_parser_request_local = self.tool_parser is not None
        elif cfg.tool_parser_instance:
            self.tool_parser = self._clone_injected_tool_parser(
                cfg.tool_parser_instance
            )
            self._tool_parser_request_local = (
                self.tool_parser is not None
                and self.tool_parser is not cfg.tool_parser_instance
            )
        else:
            self.tool_parser = self._create_tool_parser(cfg, tools_requested)
            self._tool_parser_request_local = self.tool_parser is not None
        if self.tool_parser and self._tool_parser_request_local:
            reset_parser = getattr(self.tool_parser, "reset", None)
            if callable(reset_parser):
                reset_parser()

        # State
        self.accumulated_text = ""
        self.tool_accumulated_text = ""
        # Accumulated reasoning content (split out by the reasoning parser
        # from the raw model output). Surfaced on the streaming Usage
        # chunk so clients see ``completion_tokens_details.reasoning_tokens``
        # in parity with the non-streaming response shape. v0.6.63
        # onboarding sweep finding #5.
        self.accumulated_reasoning = ""
        self.tool_calls_detected = False
        # R11-A invariant tracker (PR 0.8.13 hotfix). Counts ``tool_call``
        # StreamEvents the postprocessor has actually emitted to the wire
        # this turn (i.e. the route layer will serialize a ``delta.tool_calls``
        # SSE chunk for each). DISTINCT from ``tool_calls_detected``, which
        # is a "parser saw a tool-call shape" signal that stays True even
        # when the forced-``tool_choice`` filter dropped every anchor as
        # spec-violating scratch. Bug R11-V1: pre-fix the cap-empty / filter-
        # empty branches set ``tool_calls_detected=True`` AND emitted
        # ``finish_reason="tool_calls"`` even though zero ``tool_call``
        # deltas had reached the wire — clients saw the promised tool call
        # never materialise and the agent loop deadlocked. The invariant
        # this counter enforces: ``finish_reason="tool_calls"`` is emitted
        # IFF at least one ``tool_call`` StreamEvent (or finalize-recovered
        # ``fallback_tool_calls`` in the route layer) was/will be on the
        # wire before ``[DONE]``. Incremented at every streaming-path emit
        # site (channel-routed, reasoning, standard) AND in
        # ``_build_tool_call_event``. Read by ``_compute_finish_reason``.
        self._tool_calls_emitted_to_wire: int = 0
        self.tool_markup_possible = False
        # R10-C8 (Mira r10-R1): tool-prose-prefix hold-back. When the
        # request declares ``tools`` and the model emits a UI-TARS-style
        # natural-language preamble like ``"Tool: get_weather\n
        # Parameters: location=Paris\n"`` BEFORE the structured
        # ``delta.tool_calls`` chunk, those prose tokens used to leak
        # into ``delta.content`` and clients rendering content live saw
        # garbage prefixing the tool dispatch. Buffer content events
        # that match the ``^(Tool|Action|Function):`` prefix pattern;
        # release the buffer either when a tool_call arrives (discard,
        # because the model just confirmed it was tool-prose) OR when
        # the buffer grows past ``_TOOL_PROSE_MAX_HOLD`` bytes (release
        # as legitimate prose so a model legitimately discussing the
        # word "Tool:" isn't censored). ``_tool_prose_buffer`` is the
        # staging area; ``_tool_prose_active`` is the state latch.
        # Both fields are no-ops when ``tools_requested`` is False.
        self._tool_prose_buffer: str = ""
        self._tool_prose_active: bool = False
        # #447 round-3 (PR #948): 1-chunk hold-forward buffer for
        # ambiguous routing heads (``<`` or ``<t`` — strict prefix of
        # BOTH ``<think>`` AND ``<tool_call>``). Populated by
        # ``process_chunk`` when ``enable_thinking is False`` and the
        # head can't yet be disambiguated; prepended to the next chunk
        # so the merged head routes unambiguously. Cleared by every
        # successful flush + by ``reset()``. No-op when reasoning is
        # disabled or thinking is on by default.
        self._ambiguous_prefix_held: str = ""
        # Monotonic counter for structured tool-call indices across the
        # whole response. Each TOOL_CALL channel ``GenerationOutput`` may
        # carry a single structured call; if multiple chunks fire
        # separately (router emits one per ``<|call|>``) the index field
        # must keep counting up so clients can disambiguate them
        # (OpenAI spec: tool_calls deltas merge on ``index``). Codex
        # round-15 BLOCKING #1.
        self._structured_tool_call_count = 0
        # Set of tool_call indices we've already admitted under the
        # ``parallel_tool_calls`` cap. Text-parser streaming paths
        # (hermes, qwen3_coder, etc.) emit MANY deltas per logical call:
        # name first, then argument fragments, all with the same
        # ``index``. The cap consumes a slot only on the FIRST sighting
        # of a new index; subsequent deltas for an already-admitted
        # index are continuations and must pass through so the client
        # can reassemble the JSON. PR #518 codex round-1 BLOCKING.
        self._admitted_tool_call_indices: set[int] = set()
        # Parallel to the indexed-set above, but for parsers that emit
        # continuation deltas without an ``index`` field. Treated as a
        # single in-flight call: first no-index delta admits, every
        # subsequent no-index delta is forwarded as a continuation.
        # PR #518 round-2 codex BLOCKING: without this, no-index
        # continuations were re-classified as new calls and dropped
        # once the cap was full, silently truncating arguments.
        self._no_index_call_admitted: bool = False
        # Identity of the admitted no-index call. Some parsers re-emit
        # the same ``id`` / function ``name`` on every cumulative
        # argument-update delta (rather than emitting an anchor once
        # and bare-argument continuations after). Round-10 codex
        # BLOCKING: without remembering the admitted identity, the
        # repeated anchor was misclassified as a NEW call and dropped
        # under ``parallel_tool_calls=false``, truncating the JSON.
        # Set together with ``_no_index_call_admitted`` on admit;
        # cleared on ``reset()``.
        self._no_index_admitted_id: str | None = None
        self._no_index_admitted_name: str | None = None
        # R10-H2 (Sven r10-R1): single-call enforcement under forced
        # ``tool_choice={"type":"function","function":{"name":X}}``.
        # The OpenAI spec mandates that a forced named tool_choice
        # produces exactly ONE tool_call for the chosen function;
        # pre-fix the streaming postprocessor admitted every anchor
        # that matched the forced name, so models that re-emitted the
        # same call shape across two chunks (qwen3-bf16 reasoning
        # scratch + final emit) shipped TWO indices to the wire with
        # different ``call_id`` values for the same function. Agent
        # loops then executed the tool twice.
        #
        # State: ``_forced_anchor_admitted_id`` records the id of the
        # FIRST admitted anchor for the forced choice. Subsequent
        # anchors whose ``id`` matches are re-emissions of the same
        # call (cumulative-arguments parsers — same call_id, growing
        # arguments JSON) and pass through. Subsequent anchors with
        # DIFFERENT ``id`` values are duplicate calls — dropped per
        # the OpenAI single-call spec. ``None`` value means "no
        # forced anchor admitted yet"; once set, the latch is
        # monotonic for the rest of the request. Reset only at
        # ``reset()`` between requests.
        #
        # When an admitted anchor has no ``id`` field at all (some
        # parsers emit only ``index``+``function.name``), we use the
        # sentinel ``""`` to record "admitted but identity unknown"
        # — the next anchor without an id matches by sentinel; an
        # anchor WITH an id but a different value is treated as a
        # new call (cap-violating).
        self._forced_anchor_admitted_id: str | None = None
        # Tracks whether the MOST RECENT anchor delta (one carrying a
        # fresh ``id`` / function ``name`` / new ``index``) was DROPPED
        # because the cap was full. Subsequent argument-only no-index
        # fragments belong to whichever anchor came last — so if the
        # last anchor was dropped, the fragments must be dropped too,
        # not silently appended to the admitted call's arguments.
        # Reset on every admit (indexed or no-index). Set on every
        # cap-full drop (indexed or no-index). PR #518 round-3 first
        # surfaced the leak; round-6 codex widened the set to also
        # cover indexed dropped anchors (name kept ``no_index`` for
        # backwards refs, but semantically tracks "last anchor was
        # dropped"). Assumes sequential parser emission — interleaved
        # no-index continuations of distinct admitted indexed calls
        # are indistinguishable from delta shape alone; well-behaved
        # parsers either disambiguate via ``index``/``id`` or emit
        # sequentially.
        self._no_index_last_dropped: bool = False

        # Nemotron thinking prefix
        self._is_thinking_model = False
        self._think_prefix_sent = False

        # JSON mode: suppress thinking preamble before JSON content (#46).
        # When json_mode=True and no reasoning parser, buffer content until
        # the first JSON delimiter ({ or [) is seen, then emit from there.
        self._json_preamble_stripped = False
        self._json_preamble_buffer = ""

        # JSON mode: ```json markdown-fence strip (H-07).
        # The non-streaming chat response builder calls
        # ``extract_json_from_response`` to peel a ```json\n{...}\n```
        # wrapper off the model output when ``response_format`` is set so
        # downstream clients see bare JSON. The streaming path concatenated
        # raw model tokens without the same scrub — joined SSE deltas
        # decoded as ```json\n{...}\n``` and ``json.loads`` failed for any
        # SDK consumer assembling ``delta.content`` into a string.
        #
        # State machine (driven by ``_apply_json_fence_strip``) swallows an
        # opening fence (with any leading whitespace / pre-JSON think
        # content), passes the JSON body through, and suppresses a trailing
        # closing fence. Active only when ``json_mode=True``; absent or
        # ``"text"`` ``response_format`` leaves these fields cold and the
        # state machine is a pass-through.
        #
        # ``_json_fence_state`` values:
        #   "scan"   — pre-JSON: buffering until we see ``{``/``[`` or a
        #              ``` fence. Holds bytes in ``_json_fence_buffer``.
        #   "inside" — JSON body streaming. Holds a small tail in
        #              ``_json_fence_tail`` to defer emission of bytes that
        #              might be the start of a closing ``\n``` ``.
        #   "done"   — closing fence consumed; suppress all further bytes.
        self._json_fence_state: str = "scan"
        self._json_fence_buffer: str = ""
        self._json_fence_tail: str = ""
        # Lightweight JSON-string awareness for fence detection.
        # Tracks whether the cursor (running over emitted JSON body
        # bytes only) is currently INSIDE a JSON ``"..."`` string
        # literal — backticks inside a string literal are content, not
        # fence markers, so we MUST skip them when looking for the
        # closing ``` ``` ``. The flag flips on every unescaped ``"``
        # we see in the streamed payload. The escape flag handles
        # ``\\"`` so the next ``"`` does NOT flip the state.
        #
        # Codex r1 BLOCKING #1: without this, a valid JSON value like
        # ``{"text": "```"}`` would be truncated by the leftmost-find
        # behavior of the original ``_guard_closing_fence``.
        self._json_fence_in_string: bool = False
        self._json_fence_string_escape: bool = False
        # Bracket-depth tracker (running over emitted JSON body bytes
        # only). Increments on ``{``/``[`` outside string literals,
        # decrements on ``}``/``]``. The closing fence ``` ``` `` is
        # only recognized when ``depth == 0`` — i.e. AFTER the
        # top-level JSON root has fully closed. Without this, a
        # response like
        # ``{"k": 1}\nHere is code:\n```python\nx = 1\n``` ``
        # would truncate at the FIRST triple-backtick after the JSON
        # root and emit ``...\nHere is code:`` as content; with this,
        # the fence still fires only at depth 0 (after ``}``) and the
        # trailing markdown still gets suppressed AS the wrapper that
        # json_mode promises to strip. The state lives on the
        # instance because the walker re-scans ``combined`` from
        # index 0 on every call and must resume with the depth value
        # snapshotted at the start of the held tail. Codex r5
        # BLOCKING.
        self._json_fence_bracket_depth: int = 0
        # Whether the scan phase actually consumed an opening
        # ``` ```json `` (or bare ``` ``` `` wrapping JSON) fence.
        # Codex r8 BLOCKING #2: ``_guard_closing_fence`` only
        # suppresses a closing ``` ``` `` when an opening fence was
        # consumed; bare-JSON streams (model returned ``{...}`` straight
        # with no markdown wrapper) pass markdown content after the
        # root close THROUGH UNCHANGED, mirroring the non-stream
        # ``extract_json_from_response`` which leaves unfenced text
        # alone. Without this gate, a model that legitimately
        # continues with prose containing ``` ``` `` after the JSON
        # would have the prose truncated.
        self._json_fence_opener_consumed: bool = False
        # Codex r9 BLOCKING #1: persistent flag — has the JSON root
        # closed (depth returned to 0 from >0 at some point)? Once
        # this latches in fenced mode, every byte after that point
        # is wrapper/prose/whitespace/fence; we suppress all of it
        # until the closing ``` ``` `` fence is confirmed. Without
        # this latch a chunk boundary between root-close and the
        # fence leaks the intervening bytes onto the wire.
        self._json_root_closed: bool = False

        # Forced ``tool_choice`` assistant-prefix replay swallow (PR #716
        # codex r9 BLOCKING #1). When the route layer forces a function via
        # the OpenAI ``tool_choice`` contract (#673), the chat-template
        # renderer suffixes the prompt with a parser-shaped wire opener
        # (e.g. ``<tool_call>\n{"name": "X", "arguments":``) and
        # ``BatchedEngine.stream_chat`` yields that prefix back as a
        # SYNTHETIC first chunk so plain-text consumers (and parser state)
        # see the full envelope from the very first delta. Without the
        # swallow below, the postprocessor would route that synthetic chunk
        # through the reasoning parser (``BaseThinkingReasoningParser``
        # Case-3 ``no <think> seen yet → classify as reasoning``),
        # polluting ``accumulated_reasoning`` with the prefix bytes and —
        # on every chunk-boundary edge case the MiniMax-style tool-markup
        # redirect (``_process_with_reasoning`` lines ~1045) doesn't cover
        # (split tag across chunks, future parser variants) — risking a
        # raw-``<tool_call>``-byte leak into ``delta.reasoning_content``.
        #
        # ``seed_forced_assistant_prefix(prefix)`` is called by the
        # streaming route BEFORE ``process_chunk`` ever fires. It primes
        # the tool-parser state with the prefix (so the parser sees the
        # complete opener as already-accumulated context) and arms a
        # one-shot match buffer that swallows the synthetic chunk(s) from
        # ``process_chunk`` BEFORE they hit reasoning routing. The buffer
        # is BYTE-COUNT stateful so partial-chunk splits (synthetic chunk
        # shorter than prefix) drain incrementally across calls; overshoot
        # (chunk carries prefix + tail bytes) emits the post-prefix tail
        # through the normal pipeline. ``None`` / empty string ≡ not
        # armed; once drained to zero the swallow is inert.
        self._forced_prefix_pending: str = ""

    def seed_forced_assistant_prefix(self, prefix: str | None) -> None:
        """Prime tool-parser state with the forced ``tool_choice`` prefix.

        The streaming chat route calls this when the engine's
        ``chat_kwargs`` carry ``forced_assistant_prefix``. The prefix
        bytes are appended to ``tool_accumulated_text`` so the hermes /
        qwen3coder streaming parsers see the full wire envelope before
        the first model continuation chunk arrives, AND the same prefix
        is stored in ``_forced_prefix_pending`` so ``process_chunk`` can
        swallow the synthetic-replay chunk(s) without re-routing them
        through the reasoning parser (see ``__init__`` for the BLOCKING
        leak rationale).

        Safe to call with ``None`` / empty string — a no-op. Idempotent
        within a request: the second call REPLACES the buffer (the
        route never sets two distinct prefixes in one request, but
        replacement matches the ``__init__`` semantics).
        """
        if not prefix:
            self._forced_prefix_pending = ""
            return
        # Seed parser context. ``tool_accumulated_text`` is the buffer
        # the parser's ``previous_text`` argument is read from on the
        # first ``_detect_tool_calls`` call — without this seeding,
        # the parser would see ``previous_text=""`` and ``current_text
        # = prefix + first_model_chunk`` on chunk 1 (which works for
        # ``<tool_call>``-counting parsers like hermes; but for parsers
        # that look at ``delta_text`` boundaries it leaks).
        self.tool_accumulated_text = prefix
        self._forced_prefix_pending = prefix

    def set_thinking_model(self, model_name: str):
        """Enable Nemotron-style thinking prefix injection."""
        self._is_thinking_model = (
            "nemotron" in model_name.lower() and not self.reasoning_parser
        )

    def reset(self):
        """Reset all parser states for a new stream.

        Safe for concurrent BatchedEngine requests when parser instances
        are request-local. Injected singleton parsers are not reset here
        because that would clear another active stream's parser state.
        """
        self.accumulated_text = ""
        self.tool_accumulated_text = ""
        self.accumulated_reasoning = ""
        self.tool_calls_detected = False
        # R11-A invariant tracker — see ``__init__`` for the contract. The
        # reset MUST clear this so a re-used processor doesn't carry the
        # prior turn's emitted-count into a new stream and lie to
        # ``_compute_finish_reason`` about the wire state.
        self._tool_calls_emitted_to_wire = 0
        self.tool_markup_possible = False
        # R10-C8: clear the tool-prose hold-back so a re-used processor
        # doesn't ship the prior turn's buffered preamble bytes.
        self._tool_prose_buffer = ""
        self._tool_prose_active = False
        # #447 round-3 (PR #948): clear the ambiguous-prefix hold-buffer
        # so a re-used processor doesn't prepend the prior turn's held
        # ``<`` / ``<t`` into the new stream's first delta.
        self._ambiguous_prefix_held = ""
        self._think_prefix_sent = False
        self._json_preamble_stripped = False
        self._json_preamble_buffer = ""
        # H-07: ```json fence-strip state machine — reset to baseline.
        # ``_apply_json_fence_strip`` is a no-op when ``json_mode`` is
        # False, but clearing the buffers keeps a reused processor
        # instance (legacy singleton path) from carrying tail bytes into
        # the next request.
        self._json_fence_state = "scan"
        self._json_fence_buffer = ""
        self._json_fence_tail = ""
        self._json_fence_in_string = False
        self._json_fence_string_escape = False
        self._json_fence_bracket_depth = 0
        self._json_fence_opener_consumed = False
        self._json_root_closed = False
        # Forced-prefix swallow buffer reset to baseline. The route layer
        # re-seeds via ``seed_forced_assistant_prefix`` after ``reset()``
        # when the request carries ``forced_assistant_prefix``; without
        # the explicit clear here, a reused processor instance (legacy
        # singleton path) would carry stale swallow bytes into the next
        # request and corrupt the first non-forced chunk.
        self._forced_prefix_pending = ""
        self._structured_tool_call_count = 0
        self._admitted_tool_call_indices = set()
        self._no_index_call_admitted = False
        self._no_index_admitted_id = None
        self._no_index_admitted_name = None
        self._no_index_last_dropped = False
        # R10-H2: clear the forced-anchor latch on reset so a re-used
        # processor doesn't carry the prior request's admit state into
        # the next forced-choice stream.
        self._forced_anchor_admitted_id = None
        # Per-request reasoning-cap counters reset to baseline. The
        # configured cap itself (``self._reasoning_max_tokens``) is
        # immutable — it was set at __init__ from the request.
        self._reasoning_tokens_emitted = 0
        self._reasoning_cap_hit = False
        self._reasoning_close_injected = False
        # R8-M2: clear the explicit-think latch so a re-used processor
        # doesn't carry the prior request's promotion into this one.
        self._explicit_think_seen = False
        # R10-C7: clear the plain-content-emitted latch so the next
        # request's first chunk is evaluated against a genuinely empty
        # accumulator (mirrors the ``accumulated_text = ""`` reset).
        self._standard_content_observed = False

        if self.reasoning_parser:
            self.reasoning_parser.reset_state()
            # R10-M1: ``reset_state`` clears the parser's per-request
            # ``enable_thinking`` override; re-propagate the dispatcher's
            # captured value so a re-used processor continues to honour
            # the off-flag after a reset (the ``StreamingPostProcessor``
            # itself is bound to ONE request, but the contract here is
            # symmetric with the ``__init__`` propagation).
            _set = getattr(self.reasoning_parser, "set_enable_thinking", None)
            if callable(_set):
                try:
                    _set(self.enable_thinking)
                except Exception:
                    pass
        if self.tool_parser and self._tool_parser_request_local:
            self.tool_parser.reset()

    def process_chunk(self, output: GenerationOutput) -> list[StreamEvent]:
        """Process a single engine output chunk.

        Returns a list of StreamEvents (may be empty if content is suppressed).
        """
        delta_text = output.new_text
        if not delta_text:
            # Handle finish-only chunks
            if output.finished:
                # #447 round-4 NIT (PR #948): if a stream emitted an
                # ambiguous head (``<`` / ``<t``) on a non-finished
                # chunk and then an EMPTY finish-only chunk, the held
                # byte was previously dropped because this early
                # ``not delta_text`` branch ran before the hold-replay
                # logic in the main path. Without a disambiguating
                # second chunk we can't resolve to reasoning or tool
                # routing — flush the held bytes as plain content on
                # the FINISH event itself so the wire still receives
                # the model's literal tail. ``_make_finish_event``
                # supports an optional ``content`` payload; merging
                # here keeps the terminal chunk count unchanged.
                if self._ambiguous_prefix_held:
                    flush = self._ambiguous_prefix_held
                    self._ambiguous_prefix_held = ""
                    finish_event = self._make_finish_event(output)
                    # Prepend the held bytes to whatever content the
                    # finish event already carries (usually empty) so
                    # downstream finalize() merges keep working.
                    existing = finish_event.content or ""
                    finish_event = StreamEvent(
                        type=finish_event.type,
                        content=flush + existing,
                        reasoning=finish_event.reasoning,
                        tool_calls=finish_event.tool_calls,
                        finish_reason=finish_event.finish_reason,
                        tool_calls_detected=finish_event.tool_calls_detected,
                        metadata=finish_event.metadata,
                    )
                    return [finish_event]
                return [self._make_finish_event(output)]
            return []

        # Forced ``tool_choice`` synthetic-prefix replay swallow (codex
        # r9 BLOCKING #1). See ``seed_forced_assistant_prefix`` for the
        # full rationale. The engine yields the prefix as a synthetic
        # first chunk so plain-text consumers see the wire envelope; the
        # tool-parser state was already seeded with the same bytes by
        # the route, so feeding this chunk through the reasoning parser
        # would (a) double-count the prefix in ``accumulated_reasoning``
        # and (b) risk a raw-byte leak into ``delta.reasoning_content``
        # on parser variants that don't currently hit the MiniMax tool-
        # markup redirect.
        #
        # Drain the swallow buffer byte-by-byte across chunks: if the
        # synthetic chunk is shorter than the pending prefix (engine
        # split the prefix across multiple yields), consume what's here
        # and wait for more; if the chunk is longer (engine merged the
        # prefix with a trailing token), strip the prefix and forward
        # the tail through the normal pipeline. ``finished`` chunks
        # still need their finish event emitted even when the body is
        # fully swallowed.
        if self._forced_prefix_pending and delta_text.startswith(
            self._forced_prefix_pending[: len(delta_text)]
        ):
            consumed = min(len(delta_text), len(self._forced_prefix_pending))
            self._forced_prefix_pending = self._forced_prefix_pending[consumed:]
            tail = delta_text[consumed:]
            if not tail:
                if output.finished:
                    return [self._make_finish_event(output)]
                return []
            # Overshoot: rewrite ``new_text`` (and ``text`` if present) to
            # the post-prefix tail in place and fall through so reasoning
            # / tool parsing run on the GENUINE model bytes only. We
            # avoid ``dataclasses.replace`` so the swallow stays
            # compatible with MagicMock outputs used in unit tests AND
            # with the real ``GenerationOutput`` dataclass — both expose
            # writable ``new_text`` / ``text`` attributes.
            output.new_text = tail
            if hasattr(output, "text"):
                output.text = tail
            delta_text = tail

        # #447 round-3 MAJOR (PR #948): ambiguous-prefix hold-forward.
        # When ``enable_thinking is False`` AND a delta head reduces to
        # ``<`` or ``<t`` (strict prefix of BOTH ``<think>`` AND
        # ``<tool_call>``/``<minimax:tool_call>``), neither routing
        # choice is safe — eager routing to reasoning leaks the outer
        # opener when the next chunk completes ``<tool_call>`` (codex
        # r2 reproducer), and eager routing to the standard path leaks
        # ``<think>`` into ``delta.content`` when the next chunk
        # completes ``<think>`` (codex r3 reproducer).
        #
        # Hold the ambiguous prefix for exactly one chunk, prepend it
        # to the next delta, and re-evaluate. When the next chunk
        # arrives, the merged head is unambiguous in one direction:
        # ``<th...`` only matches ``<think>`` (reasoning lane);
        # ``<to...`` / ``<m...`` only matches a tool envelope (standard
        # lane). On the rare terminal-only chunk that carries the
        # ambiguous head (``output.finished`` True), flush the buffer
        # through the standard path so the wire envelope still closes.
        # Skipped entirely when ``enable_thinking is not False`` —
        # the default-route policy at the top of
        # ``_should_route_through_reasoning`` already locks reasoning.
        if self._ambiguous_prefix_held:
            delta_text = self._ambiguous_prefix_held + delta_text
            self._ambiguous_prefix_held = ""
            output.new_text = delta_text
            if hasattr(output, "text"):
                output.text = delta_text
        if (
            self.enable_thinking is False
            and self.reasoning_parser is not None
            and not output.finished
        ):
            _probe_head = (self.accumulated_text + delta_text).lstrip()
            if _probe_head and self._THINK_OPEN_TOKEN.startswith(_probe_head):
                # Head is a strict prefix of ``<think>``. If it is
                # ALSO a strict prefix of any tool envelope opener,
                # the routing choice is ambiguous — hold and wait.
                for _opener, _ in self._TOOL_ENVELOPE_OPENERS:
                    if _opener.startswith(_probe_head):
                        self._ambiguous_prefix_held = delta_text
                        return []

        # Step 1: Separate content from reasoning
        if output.channel is not None:
            events = self._process_channel_routed(delta_text, output)
        elif self.reasoning_parser and self._should_route_through_reasoning(delta_text):
            # When enable_thinking is explicitly False, the model is told to
            # skip thinking and answer directly. Bypass the reasoning parser
            # so its implicit-think heuristic doesn't reroute the answer to
            # reasoning_content.
            #
            # R8-M2 (2026-06-22): the bypass was overzealous. When
            # ``tool_choice="auto"`` is set on a thinking-capable model
            # AND the client explicitly set ``enable_thinking=False``,
            # the model can STILL emit explicit ``<think>...</think>``
            # wrapper tokens (Qwen3-thinking sometimes ignores the
            # chat-template hint when tools are in the prompt). The
            # pre-fix bypass routed the literal ``<think>`` bytes to
            # ``delta.content`` BEFORE the tool-call chunk. Detect the
            # explicit wrapper here and re-enter the reasoning lane so
            # the gate splits BEFORE content emit. See
            # ``_should_route_through_reasoning`` for the policy.
            events = self._process_with_reasoning(delta_text, output)
        else:
            events = self._process_standard(delta_text, output)

        # H-07: ```json markdown-fence strip for streaming json_mode.
        # The non-stream chat response builder peels the fence via
        # ``extract_json_from_response`` AFTER assembling the full
        # text; the stream path concatenated raw tokens without the
        # same scrub. Filter content here, AFTER all reasoning / tool /
        # sanitize passes have run so we only see the bytes that would
        # land on the wire as ``delta.content``. No-op when
        # ``json_mode`` is False — call sites in ``_filter_events_for_json_fence``
        # short-circuit there. Tool-call deltas and reasoning_content
        # are untouched (the fence only ever shows up in plain content
        # for json_mode requests).
        events = self._filter_events_for_json_fence(events)
        # R10-C8 (Mira r10-R1) — UI-TARS-style tool-prose-prefix
        # hold-back. Runs AFTER the json-fence filter so the two
        # buffers can't interfere; runs BEFORE the caller's wire
        # emission so prose preambles like ``"Tool: get_weather\n
        # Parameters: location=Paris\n"`` are suppressed when the
        # parser surfaces the structured call in the same turn.
        # No-op when ``tools_requested`` is False.
        events = self._filter_events_for_tool_prose(events)
        return events

    def _process_channel_routed(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle OutputRouter models (Gemma 4 etc.) with token-level routing."""
        # Engine-surfaced structured tool calls (HarmonyStreamingRouter
        # via openai-harmony's StreamableParser). Emit a structured
        # StreamEvent directly — the router has already done the
        # parse and re-running text-based extraction over the wire
        # representation would re-introduce the round-trip lossy path
        # this refactor exists to eliminate (PR #515 codex round-12 /
        # round-14 BLOCKING — tool calls whose JSON args contain
        # literal harmony sentinels were corrupted by sentinel-
        # anchored regex parsing).
        engine_tool_calls = getattr(output, "tool_calls", None) or []
        # F-200: when ``tool_choice`` forces a named function, route
        # the channel-routed structured calls through the SHARED
        # filter so the wire-shape variants (flat
        # ``{"name":"X","arguments":...}`` for HarmonyStreamableParser,
        # wrapped ``{"function":{"name":...}}`` for any future
        # router) are handled identically to the text-parser path.
        # Codex r3 BLOCKING #2: the earlier inline filter accepted
        # only the flat shape and would have silently dropped a
        # wrapped-shape channel emission. Reusing the helper also
        # picks up the JSON-object-root validation for free, which
        # closes the same scratch-with-primitive-args leak on the
        # channel-routed path.
        if engine_tool_calls:
            engine_tool_calls = self._apply_forced_tool_choice_filter(engine_tool_calls)
        if output.channel == "tool_call" and engine_tool_calls:
            # ``parallel_tool_calls=false`` is a hard external contract:
            # the non-streaming path caps the parsed list at one
            # (routes/chat.py); the streaming path must do the same or
            # clients with the flag set get extra calls they explicitly
            # opted out of. Drop everything past the cap on this chunk
            # AND mark ``tool_calls_detected`` so subsequent chunks
            # short-circuit before emission. Codex round-15 BLOCKING #2.
            #
            # Engine surfaces ONE complete structured call per
            # ``<|call|>`` boundary (openai-harmony StreamableParser),
            # so each entry here is a distinct logical call — no
            # continuation-delta concern (that's the text-parser path,
            # see ``_apply_parallel_cap``). PR #518 round-1: keep this
            # branch's per-entry counting but share the admitted-set
            # with the text-parser path so the response-wide counter
            # stays consistent.
            parallel_allowed = self._parallel_tool_calls_allowed()
            allowed_calls: list[dict] = []
            for tc in engine_tool_calls:
                # Defense in depth: include the no-index slot in the
                # cap total even though a single stream rarely hits
                # both the channel-routed AND text-parser paths
                # (channel-routed is gated on ``output.channel`` being
                # set, which only happens for OutputRouter models).
                # Round-5 codex BLOCKING #1: if any future flow lets
                # cross-pollination happen, the cap would leak.
                already_admitted = len(self._admitted_tool_call_indices) + (
                    1 if self._no_index_call_admitted else 0
                )
                if not parallel_allowed and already_admitted >= 1:
                    break
                new_idx = self._structured_tool_call_count
                self._admitted_tool_call_indices.add(new_idx)
                self._structured_tool_call_count = new_idx + 1
                allowed_calls.append(tc)
            if not allowed_calls:
                # Cap exhausted — preserve the parser-saw-tool-call signal
                # (``tool_calls_detected``) for downstream gates like the
                # tool-prose buffer drop AND the harmony cut-short rescue,
                # but DO NOT lie about the wire state with
                # ``finish_reason="tool_calls"`` — no ``tool_call`` delta
                # ever made it to the wire on this turn. R11-A invariant:
                # ``_compute_finish_reason`` reads ``_tool_calls_emitted_to_wire``
                # and falls back to ``output.finish_reason`` when zero.
                # The route layer's ``buffered_finish`` + ``fallback_tool_calls``
                # merge re-stamps the finish_reason if ``finalize()``
                # recovers a call via the cross-format fallback.
                self.tool_calls_detected = True
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason=self._compute_finish_reason(output),
                            tool_calls_detected=True,
                        )
                    ]
                return []
            # Monotonic indices across the whole response so clients
            # can disambiguate calls that arrive in separate router
            # chunks. ``OpenAI`` clients merge ``tool_calls`` deltas
            # on ``index`` — colliding indices cause one call to
            # overwrite another. Codex round-15 BLOCKING #1.
            structured = []
            for offset, tc in enumerate(allowed_calls):
                idx = self._structured_tool_call_count - len(allowed_calls) + offset
                structured.append(
                    {
                        "index": idx,
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                )
            self.tool_calls_detected = True
            # R11-A: increment the wire-truth counter BEFORE returning the
            # tool_call StreamEvent. The route layer turns this into a
            # ``delta.tool_calls`` SSE chunk on the wire, so the invariant
            # "finish_reason=tool_calls ⇒ ≥1 tool_call delta sent" holds.
            self._tool_calls_emitted_to_wire += len(structured)
            return [
                StreamEvent(
                    type="tool_call",
                    tool_calls=structured,
                    finish_reason="tool_calls" if output.finished else None,
                    tool_calls_detected=True,
                )
            ]

        if output.channel == "reasoning":
            content, reasoning = None, delta_text
        elif output.channel == "tool_call":
            content, reasoning = delta_text, None
        else:
            content, reasoning = delta_text, None

        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # When the reasoning budget is exhausted, route the overflow
        # portion of the current chunk — and every subsequent reasoning
        # chunk — to the content channel instead of dropping it.
        # Channel-routed engines (gemma4 / harmony) DON'T need a
        # ``</think>`` injection since channels are tracked at the
        # token level upstream; reclassifying the chunk is enough.
        if reasoning is not None:
            kept_reasoning, overflow_content = self._consume_reasoning_budget(reasoning)
            reasoning = kept_reasoning or None
            if overflow_content:
                content = (content or "") + overflow_content

        # Tool call detection on content
        if self.tool_parser and content:
            result = self._detect_tool_calls(content)
            if result is None:
                # Suppressed (inside tool markup OR prefix-held partial
                # sentinel). If this was ALSO the finished chunk, we
                # still must emit a finish event so the chat route's
                # buffered_finish gate fires — otherwise the
                # defensive-elif synthetic chunk path would re-emit
                # ``accumulated_text + finalize_content``, double-counting
                # already-streamed deltas (codex round-6 BLOCKING).
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason=self._compute_finish_reason(output),
                            tool_calls_detected=self.tool_calls_detected,
                        )
                    ]
                return []
            if result.get("tool_calls"):
                # When the streaming parser carries BOTH a content
                # delta AND a tool-call delta in one return (one
                # delta carried ``preface + tool_close`` — codex r4
                # BLOCKING on llama parser), the content half must
                # be emitted regardless of how the parallel-cap
                # rules out the tool half — otherwise enabling
                # ``parallel_tool_calls=false`` silently drops
                # assistant prose (codex r6 MAJOR). Apply the same
                # strip_special_tokens + sanitize_output pipeline the
                # plain-content branch (lines 850-852) uses so mixed
                # preface/trailing content can't leak special markup
                # to the client — codex r7 MAJOR.
                mixed_content = result.get("content")
                events: list[StreamEvent] = []
                if isinstance(mixed_content, str) and mixed_content:
                    mixed_content = strip_special_tokens(mixed_content)
                    if mixed_content:
                        mixed_content = sanitize_output(mixed_content)
                    if mixed_content:
                        events.append(
                            StreamEvent(type="content", content=mixed_content)
                        )

                # Issue #517 — apply ``parallel_tool_calls=false`` cap
                # uniformly across all streaming paths. Round-1 codex
                # BLOCKING: admit by ``index`` so continuation deltas
                # (incremental argument fragments for the same call)
                # don't each consume a slot.
                # F-200: forced ``tool_choice`` name filter MUST run
                # before the parallel cap — otherwise a scratch-call
                # delta inside ``<think>`` (qwen3-thinking / phi-4-
                # mini-reasoning hit the MiniMax tool-markup redirect
                # which promotes those scratch ``<tool_call>`` bodies
                # to content + tool_call detection) takes the only
                # cap slot and the real forced call is dropped as
                # ``parallel_tool_calls=false`` overflow. The forced-
                # name filter drops the scratch anchor first so the
                # cap admits the legitimate forced call.
                _tc_list = self._apply_forced_tool_choice_filter(result["tool_calls"])
                allowed_tcs = self._apply_parallel_cap(_tc_list)
                if not allowed_tcs:
                    # R11-A invariant: parser saw a tool-call shape but
                    # the forced-``tool_choice`` filter / parallel cap
                    # dropped every entry. Keep ``tool_calls_detected``
                    # set so the tool-prose buffer drop + harmony rescue
                    # gates fire on the parser-saw-call signal, but DO
                    # NOT emit ``finish_reason="tool_calls"`` — no
                    # ``delta.tool_calls`` chunk reached the wire on
                    # this turn. Bug R11-V1 pre-fix shipped a terminal
                    # ``finish_reason="tool_calls"`` with zero tool_call
                    # deltas; clients broke their agent loop waiting for
                    # the promised call. ``_compute_finish_reason``
                    # downgrades to ``output.finish_reason`` (typically
                    # ``"stop"``) when ``_tool_calls_emitted_to_wire``
                    # is zero. The route layer's buffered-finish merge
                    # still re-stamps the reason to ``"tool_calls"`` if
                    # ``finalize()`` recovers the call via the
                    # cross-format fallback path.
                    self.tool_calls_detected = True
                    if output.finished:
                        events.append(
                            StreamEvent(
                                type="finish",
                                finish_reason=self._compute_finish_reason(output),
                                tool_calls_detected=True,
                            )
                        )
                    return events
                self.tool_calls_detected = True
                # R11-A: increment the wire-truth counter BEFORE appending
                # the tool_call StreamEvent. The route layer turns this
                # into a ``delta.tool_calls`` SSE chunk so the invariant
                # "finish_reason=tool_calls ⇒ ≥1 tool_call delta sent"
                # holds across all three streaming-emit branches
                # (channel-routed, reasoning, standard).
                self._tool_calls_emitted_to_wire += len(allowed_tcs)
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                )
                return events
            content = result.get("content", "")

        if self.tool_calls_detected:
            if output.finished:
                # R11-A: route the finish through ``_compute_finish_reason``
                # so the wire-truth gate (``_tool_calls_emitted_to_wire``)
                # decides whether ``"tool_calls"`` is honest. When the
                # parser saw a call shape but every entry was dropped by
                # the forced-``tool_choice`` filter / parallel cap on a
                # prior chunk, ``tool_calls_detected`` is True but zero
                # ``delta.tool_calls`` chunks were sent — the terminal
                # finish must downgrade to ``output.finish_reason`` so
                # the client doesn't wait for a tool call that never
                # materialised (bug R11-V1, ~50% reproducible on Qwen3
                # + ``tool_choice="required"`` pre-fix).
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason=self._compute_finish_reason(output),
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Sanitize
        if content:
            content = strip_special_tokens(content)
        if reasoning:
            reasoning = strip_special_tokens(reasoning)

        finish_reason = self._compute_finish_reason(output)
        if not content and not reasoning and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        # Accumulate post-sanitize so the final usage chunk can compute
        # ``completion_tokens_details.reasoning_tokens`` via _build_usage's
        # proportional split (PR #453 logic). Without this, OutputRouter
        # models (Gemma 4, harmony/gpt-oss) emit reasoning_content deltas
        # to the client but leave both accumulators empty — _build_usage
        # then sees ``reasoning_text=None`` and omits the field entirely,
        # creating stream/non-stream usage shape drift. Verified on
        # gemma-4-26b-4bit + gpt-oss-20b-mxfp4-q8 during the v0.6.66 onboarding sweep.
        if content:
            self.accumulated_text += content
        if reasoning:
            self.accumulated_reasoning += reasoning

        # When finish_reason is set, emit ONE finish event with content/reasoning
        # merged in to avoid double-emission.
        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    reasoning=reasoning,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        events = []
        if content:
            events.append(StreamEvent(type="content", content=content))
        if reasoning:
            events.append(StreamEvent(type="reasoning", reasoning=reasoning))
        return events

    def _process_with_reasoning(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle models with text-based reasoning parsers."""
        # If the reasoning cap fired on a prior chunk, splice ``</think>``
        # into the parser's view of the stream so it flips to content on
        # this call. Idempotent — only fires once per request.
        #
        # Codex round-8 BLOCKING #1: keep the synthetic ``</think>``
        # marker OUT of the shared ``self.accumulated_text``. The
        # earlier draft mutated ``delta_text`` to ``"</think>" +
        # delta_text`` and then appended that mutated value to
        # ``self.accumulated_text`` — poisoning the buffer with forged
        # model bytes that downstream (usage chars-÷4 in chat.py, the
        # ``finalize()`` tool-call fallback) would see and account
        # against. Build the parser's ``current`` argument LOCALLY
        # from the (true) ``previous_text`` + the injected marker +
        # the ORIGINAL ``delta_text``. The shared buffer only ever
        # holds real model output. Symmetric with the routes-side
        # local-buffer pattern (round-6 fix).
        original_delta_text = delta_text
        previous_text = self.accumulated_text
        parser_delta_text = self._maybe_inject_reasoning_close(original_delta_text)
        injected_this_chunk = parser_delta_text is not original_delta_text
        if not injected_this_chunk:
            # No injection — common path. Keep the shared buffer
            # update minimal.
            self.accumulated_text += original_delta_text
            parser_current = self.accumulated_text
        else:
            # Injection fired this chunk: parser sees ``</think>`` +
            # ``original_delta``; shared buffer only gets the original.
            self.accumulated_text += original_delta_text
            parser_current = previous_text + parser_delta_text
        try:
            delta_msg = self.reasoning_parser.extract_reasoning_streaming(
                previous_text, parser_current, parser_delta_text
            )
        except Exception:
            # Codex round-10 BLOCKING #1: if the parser raises on a
            # chunk that carried the injected ``</think>``, do NOT
            # flip the ``_reasoning_close_injected`` latch — let the
            # NEXT chunk retry the forced transition. Re-raise so the
            # caller can decide (a transient parser bug is still a
            # bug; just don't lose retry on the cap-flush path).
            raise
        if injected_this_chunk:
            # Parser flip succeeded this chunk — latch so subsequent
            # chunks don't re-inject. Latch flip lives HERE (not in
            # ``_maybe_inject_reasoning_close``) so a parser exception
            # on the injection-carrying chunk leaves the latch clear
            # and the next chunk retries.
            self._reasoning_close_injected = True

        if delta_msg is None:
            # Skip (e.g., <think> token itself)
            if output.finished:
                return [self._make_finish_event(output)]
            return []

        content = delta_msg.content
        reasoning = delta_msg.reasoning

        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # Account for any reasoning bytes this chunk produced. Overflow
        # is rerouted to content so no model output is silently dropped
        # and the SSE stream gets a clean transition from reasoning to
        # content even when the parser hasn't actually seen ``</think>``.
        #
        # Codex round-9 BLOCKING #1: when overflow is produced on the
        # cap-crossing chunk and the parser hasn't yet seen
        # ``</think>``, the parser is still LOGICALLY mid-think.
        # Emitting overflow as content from that state leaks
        # still-in-thinking bytes onto the wire as
        # ``delta.content`` on the chat stream. Symmetric with the
        # routes-side fix (round-7 + round-8): force the parser flip
        # in THIS same chunk with a synthetic ``</think>`` against a
        # LOCAL ``current`` (don't pollute ``self.accumulated_text`` —
        # round-8 invariant). Only promote overflow to content when
        # the flip succeeds; suppress on flip failure rather than
        # mixing channels under a broken state machine.
        if reasoning:
            # Capture the FULL original reasoning text the parser
            # returned BEFORE the cap truncates it. We need this to
            # position the synthetic ``</think>`` marker at the
            # CAP BOUNDARY (between kept and overflow) on the flip
            # call below — not after the full over-budget chunk.
            full_reasoning = reasoning
            kept_reasoning, overflow_content = self._consume_reasoning_budget(reasoning)
            reasoning = kept_reasoning or None
            if overflow_content:
                flip_succeeded = self._reasoning_close_injected
                if not self._reasoning_close_injected:
                    # Codex round-10 BLOCKING #1: only mark the close-
                    # injected latch AFTER a SUCCESSFUL parser flip.
                    # If the parser raises, we want the NEXT chunk to
                    # retry the forced transition rather than skipping
                    # it forever — otherwise a transient parser bug
                    # leaves the parser permanently mid-think for the
                    # rest of the request.
                    #
                    # Codex round-13 BLOCKING #1: position the
                    # synthetic ``</think>`` AT THE CAP BOUNDARY (not
                    # after the full over-budget chunk). The earlier
                    # draft built ``flip_previous = self.accumulated_text``
                    # which included the OVERFLOW bytes — the parser
                    # was asked to close AFTER the over-budget bytes
                    # rather than at the kept-reasoning boundary,
                    # which would let stateful parsers mis-classify
                    # the overflow as still-in-thinking. Build the
                    # flip from ``previous_text + kept_reasoning`` —
                    # this represents the model output "up to the cap
                    # firing point" from the parser's POV.
                    flip_previous = previous_text + kept_reasoning
                    flip_delta = "</think>"
                    flip_current = flip_previous + flip_delta
                    try:
                        flip_msg = self.reasoning_parser.extract_reasoning_streaming(
                            flip_previous, flip_current, flip_delta
                        )
                        self._reasoning_close_injected = True
                        flip_succeeded = True
                    except Exception as e:
                        logger.warning(
                            "postprocessor in-chunk close-marker flip raised "
                            "on %r: %s — parser state may stay mid-think; "
                            "suppressing %d-byte overflow on this chunk; "
                            "next chunk will retry the forced transition",
                            type(self.reasoning_parser).__name__,
                            e,
                            len(overflow_content),
                        )
                        flip_msg = None
                    flip_content = (
                        getattr(flip_msg, "content", None)
                        if flip_msg is not None
                        else None
                    )
                    if isinstance(flip_content, str) and flip_content:
                        content = (content or "") + flip_content
                if flip_succeeded:
                    content = (content or "") + overflow_content
            # ``full_reasoning`` only needed within this block; release
            # the reference to drop the temporary view.
            del full_reasoning

        if reasoning:
            self.accumulated_reasoning += reasoning

        # MiniMax redirect: tool calls wrapped in <think> blocks.
        # Also load-bearing for hermes / qwen3-thinking when the chat
        # template pre-injects ``<think>`` AND a forced ``tool_choice``
        # prefix lands the model inside an in-think tool envelope —
        # the reasoning parser would otherwise leave the model's
        # continuation of the prefix in the reasoning channel and the
        # tool_call would never surface.
        if self.tool_parser and reasoning:
            _check = self.tool_accumulated_text + reasoning
            if (
                "<minimax:tool_call>" in _check
                or "<tool_call>" in _check
                or '<invoke name="' in _check
            ):
                content = (content or "") + reasoning
                reasoning = None

        # Tool call detection
        if self.tool_parser and content:
            result = self._detect_tool_calls(content)
            if result is None:
                # Suppressed (inside tool markup OR prefix-held). When
                # also the finished chunk, emit finish so the chat
                # route's buffered_finish gate fires (codex round-6
                # BLOCKING — defensive-elif duplication path).
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason=self._compute_finish_reason(output),
                            tool_calls_detected=self.tool_calls_detected,
                        )
                    ]
                return []
            if result.get("tool_calls"):
                # Combined content+tool delta — emit content half
                # regardless of how the parallel-cap rules out the
                # tool half (codex r6 MAJOR: enabling
                # ``parallel_tool_calls=false`` used to silently drop
                # the preface when cap rejected the call). Apply the
                # same strip_special_tokens + sanitize_output pipeline
                # the plain-content branch (lines 835-839) uses so
                # mixed preface/trailing content can't leak special
                # markup to the client — codex r7 MAJOR.
                mixed_content = result.get("content")
                events: list[StreamEvent] = []
                if isinstance(mixed_content, str) and mixed_content:
                    mixed_content = strip_special_tokens(mixed_content)
                    if mixed_content:
                        mixed_content = sanitize_output(mixed_content)
                    if mixed_content:
                        events.append(
                            StreamEvent(type="content", content=mixed_content)
                        )

                # Issue #517 — apply ``parallel_tool_calls=false`` cap
                # uniformly across all streaming paths. Round-1 codex
                # BLOCKING: admit by ``index`` so continuation deltas
                # (incremental argument fragments for the same call)
                # don't each consume a slot.
                # F-200: forced ``tool_choice`` name filter MUST run
                # before the parallel cap — otherwise a scratch-call
                # delta inside ``<think>`` (qwen3-thinking / phi-4-
                # mini-reasoning hit the MiniMax tool-markup redirect
                # which promotes those scratch ``<tool_call>`` bodies
                # to content + tool_call detection) takes the only
                # cap slot and the real forced call is dropped as
                # ``parallel_tool_calls=false`` overflow. The forced-
                # name filter drops the scratch anchor first so the
                # cap admits the legitimate forced call.
                _tc_list = self._apply_forced_tool_choice_filter(result["tool_calls"])
                allowed_tcs = self._apply_parallel_cap(_tc_list)
                if not allowed_tcs:
                    # R11-A invariant: parser saw a tool-call shape but
                    # the forced-``tool_choice`` filter / parallel cap
                    # dropped every entry. Keep ``tool_calls_detected``
                    # set so the tool-prose buffer drop + harmony rescue
                    # gates fire on the parser-saw-call signal, but DO
                    # NOT emit ``finish_reason="tool_calls"`` — no
                    # ``delta.tool_calls`` chunk reached the wire on
                    # this turn. Bug R11-V1 pre-fix shipped a terminal
                    # ``finish_reason="tool_calls"`` with zero tool_call
                    # deltas; clients broke their agent loop waiting for
                    # the promised call. ``_compute_finish_reason``
                    # downgrades to ``output.finish_reason`` (typically
                    # ``"stop"``) when ``_tool_calls_emitted_to_wire``
                    # is zero. The route layer's buffered-finish merge
                    # still re-stamps the reason to ``"tool_calls"`` if
                    # ``finalize()`` recovers the call via the
                    # cross-format fallback path.
                    self.tool_calls_detected = True
                    if output.finished:
                        events.append(
                            StreamEvent(
                                type="finish",
                                finish_reason=self._compute_finish_reason(output),
                                tool_calls_detected=True,
                            )
                        )
                    return events
                self.tool_calls_detected = True
                # R11-A: increment the wire-truth counter BEFORE appending
                # the tool_call StreamEvent. The route layer turns this
                # into a ``delta.tool_calls`` SSE chunk so the invariant
                # "finish_reason=tool_calls ⇒ ≥1 tool_call delta sent"
                # holds across all three streaming-emit branches
                # (channel-routed, reasoning, standard).
                self._tool_calls_emitted_to_wire += len(allowed_tcs)
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                )
                return events
            content = result.get("content", "")

        if self.tool_calls_detected:
            if output.finished:
                # R11-A: route the finish through ``_compute_finish_reason``
                # so the wire-truth gate (``_tool_calls_emitted_to_wire``)
                # decides whether ``"tool_calls"`` is honest. When the
                # parser saw a call shape but every entry was dropped by
                # the forced-``tool_choice`` filter / parallel cap on a
                # prior chunk, ``tool_calls_detected`` is True but zero
                # ``delta.tool_calls`` chunks were sent — the terminal
                # finish must downgrade to ``output.finish_reason`` so
                # the client doesn't wait for a tool call that never
                # materialised (bug R11-V1, ~50% reproducible on Qwen3
                # + ``tool_choice="required"`` pre-fix).
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason=self._compute_finish_reason(output),
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Sanitize
        if content:
            content = strip_special_tokens(content)
        if reasoning:
            reasoning = strip_special_tokens(reasoning)

        finish_reason = self._compute_finish_reason(output)
        if not content and not reasoning and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    reasoning=reasoning,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        events = []
        if content:
            events.append(StreamEvent(type="content", content=content))
        if reasoning:
            events.append(StreamEvent(type="reasoning", reasoning=reasoning))
        return events

    def _process_standard(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle standard models (no reasoning parser, no channel router)."""
        content = strip_special_tokens(delta_text)

        # JSON mode preamble stripping (#46): when response_format is set and
        # no reasoning parser is active, the model may emit a thinking preamble
        # (e.g. "Let me think...\n{json}") before the actual JSON. Suppress
        # everything before the first JSON delimiter.
        if (
            self.json_mode
            and not self.reasoning_parser
            and not self._json_preamble_stripped
        ):
            if content:
                self._json_preamble_buffer += content
                json_start = _find_json_start(self._json_preamble_buffer)
                if json_start >= 0:
                    self._json_preamble_stripped = True
                    # Codex r8 BLOCKING #2: if the preamble we're about
                    # to strip ends in an opening ``` ```json `` /
                    # ``` ``` `` fence (whose payload IS the JSON we
                    # just landed on), the downstream fence-walker
                    # must know an opening fence WAS consumed so it
                    # will suppress the matching closing fence.
                    # Without this signal the bare-JSON pass-through
                    # fast-path fires and the closing ``` ``` `` leaks
                    # onto the wire. ``_find_json_fence_opener`` needs
                    # the JSON delimiter visible to recognise the
                    # fence's payload, so we run it over the FULL
                    # buffer (preamble + JSON) and check whether the
                    # found fence sits inside the about-to-be-stripped
                    # preamble.
                    fence_in_full = _find_json_fence_opener(self._json_preamble_buffer)
                    if 0 <= fence_in_full < json_start:
                        self._json_fence_opener_consumed = True
                    content = self._json_preamble_buffer[json_start:]
                else:
                    return []

        # Nemotron thinking prefix injection moved to AFTER sanitize_output
        # below - sanitize_output strips ``<think>`` tags as special tokens,
        # so injecting here made the prefix dead code.

        # Tool call detection
        if self.tool_parser and delta_text:
            result = self._detect_tool_calls(delta_text)
            if result is None:
                # Suppressed. When also finished, emit finish so the
                # chat route's buffered_finish gate fires (codex
                # round-6 BLOCKING).
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason=self._compute_finish_reason(output),
                            tool_calls_detected=self.tool_calls_detected,
                        )
                    ]
                return []
            if result.get("tool_calls"):
                # Combined content+tool delta — emit content half
                # regardless of how the parallel-cap rules out the
                # tool half (codex r6 MAJOR). Match the plain-content
                # branch (line 1265) with ``sanitize_output`` so mixed
                # preface/trailing content can't leak special markup
                # — codex r7 MAJOR.
                mixed_content = result.get("content")
                events: list[StreamEvent] = []
                if isinstance(mixed_content, str) and mixed_content:
                    mixed_content = strip_special_tokens(mixed_content)
                    if mixed_content:
                        mixed_content = sanitize_output(mixed_content)
                    if mixed_content:
                        # R10-C7: mixed content+tool deltas also count
                        # as standard-path plain-content emission for
                        # the router-latch (see
                        # ``_should_route_through_reasoning``). Codex
                        # r10-F HIGH: gate on a non-whitespace byte —
                        # the router's head check uses
                        # ``probe.lstrip()`` so leading whitespace
                        # alone must NOT block a subsequent legitimate
                        # ``<think>`` promotion (Sven r8-M2 evidence:
                        # Qwen3-thinking sometimes prefixes the
                        # wrapper with whitespace bytes).
                        if mixed_content.strip():
                            self._standard_content_observed = True
                        events.append(
                            StreamEvent(type="content", content=mixed_content)
                        )

                # Apply ``parallel_tool_calls=false`` cap (issue #517).
                # Round-1 codex BLOCKING: admit by ``index`` so
                # incremental argument fragments don't each consume a
                # cap slot (qwen3_coder pattern — header delta + N
                # argument-fragment deltas all share the same index).
                # F-200: forced ``tool_choice`` name filter MUST run
                # before the parallel cap — otherwise a scratch-call
                # delta inside ``<think>`` (qwen3-thinking / phi-4-
                # mini-reasoning hit the MiniMax tool-markup redirect
                # which promotes those scratch ``<tool_call>`` bodies
                # to content + tool_call detection) takes the only
                # cap slot and the real forced call is dropped as
                # ``parallel_tool_calls=false`` overflow. The forced-
                # name filter drops the scratch anchor first so the
                # cap admits the legitimate forced call.
                _tc_list = self._apply_forced_tool_choice_filter(result["tool_calls"])
                allowed_tcs = self._apply_parallel_cap(_tc_list)
                if not allowed_tcs:
                    # R11-A invariant: parser saw a tool-call shape but
                    # the forced-``tool_choice`` filter / parallel cap
                    # dropped every entry. Keep ``tool_calls_detected``
                    # set so the tool-prose buffer drop + harmony rescue
                    # gates fire on the parser-saw-call signal, but DO
                    # NOT emit ``finish_reason="tool_calls"`` — no
                    # ``delta.tool_calls`` chunk reached the wire on
                    # this turn. Bug R11-V1 pre-fix shipped a terminal
                    # ``finish_reason="tool_calls"`` with zero tool_call
                    # deltas; clients broke their agent loop waiting for
                    # the promised call. ``_compute_finish_reason``
                    # downgrades to ``output.finish_reason`` (typically
                    # ``"stop"``) when ``_tool_calls_emitted_to_wire``
                    # is zero. The route layer's buffered-finish merge
                    # still re-stamps the reason to ``"tool_calls"`` if
                    # ``finalize()`` recovers the call via the
                    # cross-format fallback path.
                    self.tool_calls_detected = True
                    if output.finished:
                        events.append(
                            StreamEvent(
                                type="finish",
                                finish_reason=self._compute_finish_reason(output),
                                tool_calls_detected=True,
                            )
                        )
                    return events
                self.tool_calls_detected = True
                # R11-A: increment the wire-truth counter BEFORE appending
                # the tool_call StreamEvent. The route layer turns this
                # into a ``delta.tool_calls`` SSE chunk so the invariant
                # "finish_reason=tool_calls ⇒ ≥1 tool_call delta sent"
                # holds across all three streaming-emit branches
                # (channel-routed, reasoning, standard).
                self._tool_calls_emitted_to_wire += len(allowed_tcs)
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                )
                return events
            content = strip_special_tokens(result.get("content", ""))

        if self.tool_calls_detected:
            if output.finished:
                # R11-A: route the finish through ``_compute_finish_reason``
                # so the wire-truth gate (``_tool_calls_emitted_to_wire``)
                # decides whether ``"tool_calls"`` is honest. When the
                # parser saw a call shape but every entry was dropped by
                # the forced-``tool_choice`` filter / parallel cap on a
                # prior chunk, ``tool_calls_detected`` is True but zero
                # ``delta.tool_calls`` chunks were sent — the terminal
                # finish must downgrade to ``output.finish_reason`` so
                # the client doesn't wait for a tool call that never
                # materialised (bug R11-V1, ~50% reproducible on Qwen3
                # + ``tool_choice="required"`` pre-fix).
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason=self._compute_finish_reason(output),
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Filter empty
        if content is not None and content == "":
            content = None

        finish_reason = self._compute_finish_reason(output)

        if not content and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        # Nemotron thinking prefix - injected AFTER sanitize_output so the
        # ``<think>`` opener survives to the wire. sanitize_output strips
        # ``<think>`` tags as special tokens; injecting pre-sanitize (the
        # former location) made the prefix dead code - sanitize ate it
        # before emission (test_thinking_prefix_injected).
        if self._is_thinking_model and not self._think_prefix_sent and content:
            content = "<think>" + content
            self._think_prefix_sent = True
            logger.debug(
                "nemotron thinking prefix injected (model=%s)",
                getattr(self, "_thinking_model_name", "nemotron"),
            )

        # When finish_reason is set, emit ONE finish event with content merged in.
        # Never enable separate content + finish events — that would cause
        # double-emission of the same content and duplicate logprobs.
        if finish_reason:
            # R10-C7: latch on real plain-content emission so a later
            # ``<think>`` token in this same request can't be
            # misclassified as a head-of-buffer reasoning opener by
            # the router (see ``_should_route_through_reasoning``).
            # Codex r10-F HIGH: gate on a non-whitespace byte — the
            # router's head check uses ``probe.lstrip()`` so leading
            # whitespace alone must NOT block a subsequent legitimate
            # ``<think>`` promotion. Empty content (finish-only) also
            # MUST NOT latch.
            if content and content.strip():
                self._standard_content_observed = True
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        if content:
            # R10-C7: see comment on the finish branch above — only
            # non-whitespace bytes are evidence that we're past the
            # head of the buffer.
            if content.strip():
                self._standard_content_observed = True
            return [StreamEvent(type="content", content=content)]
        return []

    def finalize(self) -> list[StreamEvent]:
        """Finalize stream — flush remaining tool calls, emit corrections.

        Call after the engine stream ends.
        """
        events = []

        # Codex round-3 BLOCKING #1: when the per-request reasoning cap
        # latches on the LAST reasoning chunk of the stream (model stops
        # immediately at the budget, or stops within the exact-boundary
        # chunk), no subsequent ``process_chunk`` call ever runs the
        # ``</think>`` injection — the parser is left mid-think, any
        # held content past the cap stays buffered, and the client sees
        # a reasoning-only response with no visible answer. Force the
        # injection here so a terminal cap-hit still flips the parser
        # to content and any held bytes are flushed as a final content
        # delta. Idempotent via ``_reasoning_close_injected`` so a
        # mid-stream injection on a normal chunk doesn't double-fire.
        if (
            self.reasoning_parser is not None
            and self._reasoning_cap_hit
            and not self._reasoning_close_injected
        ):
            self._reasoning_close_injected = True
            previous_text = self.accumulated_text
            injected_delta = "</think>"
            # Codex round-5 BLOCKING #1: build the parser's view of
            # ``current`` LOCALLY rather than mutating
            # ``self.accumulated_text``. Downstream (routes/chat.py
            # post-finalize usage assembly) reads ``accumulated_text``
            # to compute the chars-÷4 reasoning split for the usage
            # block. Appending the forged ``</think>`` to the shared
            # buffer would (a) make the usage tokens differ by 2 from
            # what was actually streamed, and (b) — more importantly —
            # if any future code path runs the parser's non-stream
            # ``finalize_streaming`` over ``accumulated_text``, it
            # would re-emit the same buffered bytes the in-finalize
            # injection just released. Keep the mutation scoped.
            local_current = previous_text + injected_delta
            delta_msg = None
            try:
                delta_msg = self.reasoning_parser.extract_reasoning_streaming(
                    previous_text, local_current, injected_delta
                )
            except Exception as e:
                # Codex round-5 BLOCKING #2 / #3: a parser exception on
                # the forced close path is an INTERNAL server failure,
                # not a model answer. The earlier draft emitted a
                # diagnostic string ``"[reasoning cap hit — parser
                # flush failed]"`` directly into ``content``, which
                # leaks server implementation details into the
                # assistant message. Drop the fabrication and log only;
                # the client sees an empty completion if the cap path
                # fails (the route's existing error semantics handle
                # truly catastrophic failures via 5xx).
                logger.warning(
                    "finalize close-marker injection raised on %r: %s — "
                    "trailing reasoning content (if any) will not be "
                    "promoted to content for this request",
                    type(self.reasoning_parser).__name__,
                    e,
                )
            if delta_msg is not None:
                trailing_content = getattr(delta_msg, "content", None)
                if isinstance(trailing_content, str) and trailing_content:
                    trailing_content = sanitize_output(
                        strip_special_tokens(trailing_content)
                    )
                    if trailing_content:
                        events.append(
                            StreamEvent(type="content", content=trailing_content)
                        )

        # Fallback tool call detection: streaming parser missed a tool call
        # that the non-stream parser can recover. The streaming code path of
        # each parser is necessarily simpler than ``extract_tool_calls`` —
        # it can't backtrack and typically only handles the canonical
        # wrapper format. ``extract_tool_calls`` has the full set of fallback
        # patterns (bare JSON, alternate XML forms, text-format degradation).
        # Running it here gives streaming the same tolerance as non-stream.
        #
        # Previously gated on ``has_pending_tool_call`` — but that gate
        # uses the SAME canonical-wrapper check as the streaming parser, so
        # by construction it can never catch what the streaming parser
        # missed. The 2026-05-20 ≥20B onboarding sweep caught gemma-4-26b-4bit
        # producing structured tool_calls in non-stream mode that the
        # streaming parser dropped on the floor; the only difference between
        # the two modes was this gate. See knowledge/guided_generation_gaps_2026-05-20.md
        # "Bug A — Streaming tool-parser coverage gap is family-wide".
        #
        # Cheap pre-check: every known tool-call format carries at least
        # one structural marker — ``<`` (XML wrappers: ``<tool_call>``,
        # ``<function=>``, ``<|tool_call>``), ``{`` (bare JSON, parameter
        # blocks), or ``[Calling`` (text-format degradation). Skipping the
        # full regex scan when none of these markers is present keeps
        # end-of-stream cost flat on plain-text responses that happened to
        # have ``tools=...`` in the request (DeepSeek pr_validate finding
        # on PR #424 — high-throughput servers with tool-enabled
        # endpoints would otherwise pay the parser cost on every reply
        # that didn't actually call a tool).
        _fallback_text = self.tool_accumulated_text or self.accumulated_text
        _has_plausible_markup = bool(_fallback_text) and (
            "<" in _fallback_text
            or "{" in _fallback_text
            or "[Calling" in _fallback_text
        )
        if (
            self.tool_parser
            and _fallback_text
            and not self.tool_calls_detected
            and _has_plausible_markup
        ):
            result = self.tool_parser.extract_tool_calls(
                _fallback_text, request=self.request
            )
            if result.tools_called:
                # F-200: forced ``tool_choice`` filter on the finalize
                # ``extract_tool_calls`` recovery path. The parser
                # may return multiple calls (a scratch
                # ``<tool_call>`` inside ``<think>`` with bare int /
                # string ``arguments`` PLUS the real call) — drop
                # any whose name does not match the forced choice
                # OR whose ``arguments`` parses as a JSON non-object
                # (codex r1 BLOCKING: filtering by name alone leaks
                # a same-name scratch call with primitive args).
                _forced_name = self._forced_tool_choice_name()
                _required_mode = self._is_tool_choice_required()
                if _forced_name:
                    _filtered_calls = [
                        tc
                        for tc in result.tool_calls
                        if tc.get("name") == _forced_name
                        and not self._forced_tool_choice_arguments_violate_object_root(
                            tc.get("arguments")
                        )
                    ]
                elif _required_mode:
                    # r10-J round-4 (codex r4 HIGH #1): for
                    # ``tool_choice="required"`` ``_forced_tool_choice_name``
                    # returns None (no specific function pinned), so the
                    # pre-fix branch passed every recovered call through.
                    # That reopened the malformed-args leak the streaming
                    # ``_apply_forced_tool_choice_filter`` was meant to
                    # close — finalize-recovered calls with
                    # ``arguments="20230805"`` (bare-string root) reached
                    # the client despite ``required`` semantics. Apply
                    # the same object-root gate here. No name constraint
                    # in required mode (multi-tool parallel is legal),
                    # only the schema-shape gate.
                    _filtered_calls = [
                        tc
                        for tc in result.tool_calls
                        if not self._forced_tool_choice_arguments_violate_object_root(
                            tc.get("arguments")
                        )
                    ]
                else:
                    _filtered_calls = list(result.tool_calls)
                if _filtered_calls:
                    events.append(
                        self._build_tool_call_event(
                            {
                                "id": tc["id"],
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            }
                            for tc in _filtered_calls
                        )
                    )
                    self.tool_calls_detected = True
            else:
                # Cross-format fallback. The configured streaming parser is bound to
                # ONE wire format; ``parse_tool_calls`` in ``api/tool_calling.py``
                # scans every known format and recovers calls the per-parser path
                # misses (e.g. ``qwen3_xml`` is registered to ``QwenToolParser``
                # which expects JSON inside ``<tool_call>``, but Qwen3.6-35B-A3B
                # emits the ``<function=name><parameter=...>`` XML body). The
                # non-stream path at ``service/helpers.py:604`` already falls back;
                # this mirrors it on streaming. Wrapped defensively to match the
                # non-stream try/except — a parser bug must not abort the stream.
                # See #425.
                try:
                    _, fb_tcs = parse_tool_calls(_fallback_text, self.request)
                except Exception as e:
                    logger.warning(
                        "finalize cross-format fallback parser raised: %s", e
                    )
                    fb_tcs = None
                if fb_tcs:
                    # F-200: forced ``tool_choice`` filter on the
                    # cross-format fallback recovery path. Apply BOTH
                    # name AND arguments-root-object validation —
                    # codex r1 BLOCKING: name-only filtering let
                    # same-name scratch calls with primitive / list
                    # ``arguments`` leak through.
                    _forced_name = self._forced_tool_choice_name()
                    _required_mode = self._is_tool_choice_required()
                    if _forced_name:
                        fb_tcs = [
                            tc
                            for tc in fb_tcs
                            if tc.function.name == _forced_name
                            and not self._forced_tool_choice_arguments_violate_object_root(
                                tc.function.arguments
                            )
                        ]
                    elif _required_mode:
                        # r10-J round-4 — twin of the
                        # ``extract_tool_calls`` recovery branch above:
                        # cross-format fallback under ``required`` must
                        # also drop primitive-args recovered calls so
                        # the contract is symmetric across the two
                        # finalize recovery paths.
                        fb_tcs = [
                            tc
                            for tc in fb_tcs
                            if not self._forced_tool_choice_arguments_violate_object_root(
                                tc.function.arguments
                            )
                        ]
                if fb_tcs:
                    logger.info(
                        "[finalize] cross-format fallback recovered %d tool_call(s); "
                        "configured parser=%r returned tools_called=False — "
                        "consider whether --tool-call-parser matches the model's wire format",
                        len(fb_tcs),
                        getattr(self.cfg, "tool_call_parser", None),
                    )
                    events.append(
                        self._build_tool_call_event(
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in fb_tcs
                        )
                    )
                    self.tool_calls_detected = True

        # Dogfood F-R1-04 (codex r5 BLOCKING): UI-TARS reasoning
        # parser-specific EOF flush. The opener-prefix hold-back
        # logic returns ``None`` (no event) while the buffer is a
        # strict prefix of a known opener — ``"Thought"`` waiting
        # for the colon, ``"Reflection"`` waiting, etc. If the
        # stream ends mid-prefix (e.g. ``max_tokens`` truncation
        # mid-token, or the model genuinely produced bare
        # ``"Thought"`` text), those bytes are otherwise silently
        # dropped at EOF. Mirror the ``tool_parser.flush_held_content``
        # pattern below but scope it to the UI-TARS reasoning
        # parser specifically — other reasoning parsers
        # (``qwen3`` / ``deepseek_r1`` / ``gemma4``) have their own
        # ``finalize_streaming`` semantics tied to specific call
        # sites that this generic hook would clash with.
        if (
            self.reasoning_parser is not None
            and self.accumulated_text
            and type(self.reasoning_parser).__name__ == "UiTarsReasoningParser"
        ):
            try:
                final_msg = self.reasoning_parser.finalize_streaming(
                    self.accumulated_text
                )
            except Exception as e:
                logger.warning(
                    "UI-TARS finalize_streaming raised: %s — any held "
                    "trailing bytes will not be flushed for this request",
                    e,
                )
                final_msg = None
            if final_msg is not None:
                final_reasoning = getattr(final_msg, "reasoning", None)
                if isinstance(final_reasoning, str) and final_reasoning:
                    events.append(
                        StreamEvent(type="reasoning", reasoning=final_reasoning)
                    )
                final_content = getattr(final_msg, "content", None)
                if isinstance(final_content, str) and final_content:
                    events.append(StreamEvent(type="content", content=final_content))

        # Release any prefix-held content trailing the stream. Hermes
        # and harmony streaming parsers hold back partial sentinel
        # suffixes (``<``, ``<|``, ``<func``...) so per-char streaming
        # doesn't leak them before the full sentinel arrives. If the
        # stream ends with bytes still held AND no tool call ever
        # fired, those bytes are ordinary content and would otherwise
        # be silently dropped (codex round-3 CRITICAL on the streaming-
        # parser cluster PR). When a tool call DID fire, the held
        # bytes are part of the tool-call body and stay suppressed.
        if (
            self.tool_parser
            and self.tool_accumulated_text
            and not self.tool_calls_detected
        ):
            held = self.tool_parser.flush_held_content(self.tool_accumulated_text)
            # Strict-string check: ``flush_held_content`` is part of the
            # parser interface and must return a real ``str``. Defending
            # against accidental ``None`` / non-string returns avoids a
            # buggy override surfacing as a malformed StreamEvent
            # downstream.
            if isinstance(held, str) and held:
                events.append(StreamEvent(type="content", content=held))

        # H-07: ```json fence-strip on the finalize event list AND
        # drain any held tail in the SAME pass. The route's
        # "buffered_finish" merge path concatenates
        # ``finalize()``-produced content events into the terminal
        # SSE chunk; without the strip here the closing fence would
        # survive on the last few bytes of the stream. The single-
        # pass drain (``drain_tail=True``) merges any held tail into
        # the LAST content/finish event in this batch so JSON bytes
        # never sit after a terminal marker (codex r4 BLOCKING #2).
        events = self._filter_events_for_json_fence(events, drain_tail=True)

        # R10-C8: run the tool-prose-prefix filter on the finalize
        # events too — the route's mid-stream emitter has already
        # decided whether to ship structured tool_calls, so the
        # buffer's discard/release predicate is now fully informed.
        # ``_flush_tool_prose_buffer`` drops the buffer when
        # ``tool_calls_detected`` (the buffer was a dispatch preamble)
        # and otherwise releases it as legitimate trailing prose.
        events = self._filter_events_for_tool_prose(events)
        tail = self._flush_tool_prose_buffer()
        if tail:
            # Merge into the last finish/content event so the prose
            # tail doesn't sit AFTER a terminal marker. Same shape
            # as the json-fence drain merge above.
            from dataclasses import replace as _dc_replace

            merged = False
            for i in range(len(events) - 1, -1, -1):
                if events[i].type in ("finish", "content"):
                    prev = events[i].content or ""
                    events[i] = _dc_replace(events[i], content=prev + tail)
                    merged = True
                    break
            if not merged:
                events.append(StreamEvent(type="content", content=tail))

        return events
