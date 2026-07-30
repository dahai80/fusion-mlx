# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processor — unified reasoning + tool call + sanitization pipeline.

Replaces 500+ lines of duplicated logic across stream_chat_completion,
_stream_anthropic_messages, and stream_completion. NOT a filter chain —
one cohesive orchestrator, because reasoning/tool/sanitize are tightly coupled.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import uuid
from typing import TYPE_CHECKING
from unittest.mock import Mock

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


def _create_reasoning_parser(cfg: ServerConfig):
    """Create a per-request reasoning parser instance."""
    if not cfg.reasoning_parser_name:
        return None
    try:
        from ...reasoning import get_parser

        parser_cls = get_parser(cfg.reasoning_parser_name)
        return parser_cls()
    except Exception as e:
        logger.warning(f"Failed to create reasoning parser: {e}")
        return None

def _create_tool_parser(cfg: ServerConfig, tools_requested: bool):
    """Create a per-request tool parser instance."""
    from ...tool_parsers import ToolParserManager

    tokenizer = None
    if cfg.engine is not None and hasattr(cfg.engine, "_tokenizer"):
        tokenizer = cfg.engine._tokenizer

    if cfg.enable_auto_tool_choice and cfg.tool_call_parser:
        try:
            parser_cls = ToolParserManager.get_tool_parser(cfg.tool_call_parser)
            return parser_cls(tokenizer)
        except Exception as e:
            logger.warning(f"Failed to create tool parser for streaming: {e}")

    if tools_requested and cfg.reasoning_parser_name:
        _PARSER_MAP = {"minimax": "minimax"}
        inferred = _PARSER_MAP.get(cfg.reasoning_parser_name)
        if inferred:
            try:
                parser_cls = ToolParserManager.get_tool_parser(inferred)
                return parser_cls(tokenizer)
            except Exception as e:
                logger.debug(f"Auto-infer tool parser for streaming failed: {e}")

    return None

def _clone_injected_tool_parser(parser):
    if parser is None:
        return None
    if isinstance(parser, Mock) and os.environ.get("PYTEST_CURRENT_TEST"):
        return parser
    try:
        return copy.deepcopy(parser)
    except Exception:
        try:
            return copy.copy(parser)
        except Exception:
            raise RuntimeError(
                "Injected tool parser instance could not be cloned safely "
                "for a request-local stream"
            )

def _forced_tool_choice_arguments_violate_object_root(args_str: str | None) -> bool:
    """Return True when a finalized anchor's arguments value
    violates the OpenAI spec — not a JSON-object-encoded string."""
    if not args_str or not args_str.strip():
        return False
    try:
        parsed = json.loads(args_str)
    except (ValueError, TypeError):
        open_braces = args_str.count("{")
        close_braces = args_str.count("}")
        if open_braces > close_braces:
            return False
        return True
    return not isinstance(parsed, dict)

def _continuation_arguments_definitively_non_object(args_str: str | None) -> bool:
    """Return True when a continuation fragment's accumulated arguments
    are definitively not a JSON object."""
    if not args_str or not args_str.strip():
        return False
    try:
        parsed = json.loads(args_str)
    except (ValueError, TypeError):
        return False
    return not isinstance(parsed, dict)


class StreamingPostProcessorParserMixin:
    """Parser methods for StreamingPostProcessor — tool call / reasoning / thinking block parsing."""

    _THINK_OPEN_TOKEN = "<think>"
    def _should_route_through_reasoning(self, delta_text: str = "") -> bool:
        """Decide whether the current chunk should go through the reasoning
        parser.

        Default policy: route when ``enable_thinking is not False``. The
        ``False`` bypass exists because Qwen3's implicit-think heuristic
        otherwise misroutes a plain answer to ``reasoning_content`` when
        the chat template skipped the ``<think>`` injection.

        R8-M2 override (2026-06-22): even when ``enable_thinking=False``,
        if the model has ALREADY emitted an explicit ``<think>`` token
        (or the buffer head is consistent with the leading bytes of one
        — e.g. ``<th`` waiting for ``ink>``), re-enter the reasoning
        lane so the gate splits BEFORE the wrapper text leaks into
        ``delta.content``. The pre-fix bypass shipped the literal
        wrapper to the client whenever ``tool_choice="auto"`` +
        thinking-capable model + the model decided to think anyway
        despite the off-flag (Sven r8 evidence; Qwen3 with two tools
        defined).

        Signal: ``self.accumulated_text + delta_text`` — checked
        BEFORE this chunk's bytes have been folded into the
        accumulator (the per-chunk fold happens inside the reasoning
        path itself). Two cases enable the promotion:
          1. The complete ``<think>`` opener is in the probe → latch.
          2. The probe's HEAD (after leading whitespace) is a strict
             prefix of ``<think>`` → tentative re-route this chunk so
             a split-SSE tag (``<th`` then ``ink>``) doesn't leak.
             The latch stays off until the full opener resolves; if
             the prefix turns out NOT to be a tag (e.g. the model
             emitted ``<thanks for asking!``), the parser falls back
             to its Case-3 content path on the next chunk via the
             same mechanism the default path uses.

        Once promoted, the latch (``_explicit_think_seen``) makes the
        decision sticky for the rest of the request so it doesn't
        oscillate as the accumulator grows past the opener.

        ``delta_text`` is optional to keep the helper callable as a
        property-style predicate from other call sites that don't have
        the live delta handy; those sites use the strict
        accumulated-buffer signal (sufficient post-first-chunk).
        """
        if self.enable_thinking is not False:
            return True
        if self._explicit_think_seen:
            return True
        # R10-C7 (2026-06-23): if ``_process_standard`` has already
        # emitted plain content for this request, refuse to promote
        # any subsequent ``<think>`` token into the reasoning lane.
        # ``self.accumulated_text`` is NOT updated by the standard
        # path (only ``_process_with_reasoning`` mutates it at lines
        # 2158/2163), so the r8-C head-of-buffer anchor below would
        # otherwise evaluate against an empty accumulator and treat
        # mid-content ``<think>`` as a fresh reasoning opener. Mira
        # r10-R1 root cause. The latch is one-way and cleared at
        # ``reset_for_new_request`` time.
        if self._standard_content_observed:
            return False
        probe = self.accumulated_text + (delta_text or "")
        # Codex r8-C round-2 MED: anchor the complete-token branch at
        # the FIRST non-whitespace bytes, matching the split-prefix
        # branch below. Without the anchor, a stream that produces a
        # plain answer like ``You asked about <think> tags in HTML.``
        # — literal ``<think>`` mid-content, NOT the model entering
        # reasoning — would latch ``_explicit_think_seen`` and route
        # all subsequent chunks through the reasoning parser, hiding
        # the answer body. The intent is "the model started an
        # explicit reasoning wrapper", which only happens at the head
        # of the buffer (Qwen3 templates inject ``<think>`` first).
        head = probe.lstrip()
        if head.startswith(self._THINK_OPEN_TOKEN):
            self._explicit_think_seen = True
            return True
        # Tentative re-route: the head of the probe (post-leading-ws)
        # might be the start of a ``<think>`` opener whose tail hasn't
        # arrived yet. Route this chunk through the reasoning parser
        # so a split-SSE tag doesn't pre-leak as content. Don't latch
        # — we'll re-evaluate next chunk once more bytes arrive.
        #
        # #447 (2026-06-26): suppress the tentative re-route when a
        # tool-call envelope is already in flight (the tool parser is
        # mid-block accumulating an unclosed ``<tool_call>`` body).
        # Reproduction: ``enable_thinking=False`` + ``tool_choice="auto"``
        # on hermes + Qwen3 streams the Nemotron-shape envelope
        # ``<tool_call>\n<function=NAME>\n<parameter=K>V</parameter>\n
        # </function>\n</tool_call>``. After the opening ``<tool_call>``
        # chunk lands in the tool parser, the standalone ``<`` of the
        # inner ``<function=`` tag matches the split-prefix branch (``<``
        # is a strict prefix of ``<think>``), gets re-routed to the
        # reasoning lane, and is held back by the reasoning parser's
        # partial-tag withhold. The byte is then NEVER fed to the tool
        # parser's ``tool_accumulated_text``, so the assembled body
        # reads ``<tool_call>\nfunction=...\n<parameter=...`` — the
        # outer ``<function=`` opener is corrupted, the Nemotron regex
        # fails to match, and the whole envelope is suppressed as an
        # in-flight tool block until end-of-stream finishes with a
        # bare ``finish_reason="stop"`` and zero tool_calls. The
        # tool-parser path has its own held-back machinery for partial
        # sentinels (``hermes._safe_content_prefix``) so deferring
        # routing here is safe — the ``<`` lands in the tool parser,
        # is held until enough bytes arrive to commit, and the
        # downstream tool-envelope detection completes normally.
        if head and self._THINK_OPEN_TOKEN.startswith(head):
            if self._tool_envelope_in_flight():
                return False
            return True
        return False
    # Common ``<tool_call>``-style envelope openers shared across the
    # text-parser families (hermes / qwen3_xml / glm47 / minimax / nemotron).
    # Used by ``_tool_envelope_in_flight`` to detect that the tool parser
    # is mid-accumulation so the split-prefix ``<think>`` rescue
    # (``_should_route_through_reasoning``) does not eat the next ``<``
    # byte into the reasoning lane (#447).
    _TOOL_ENVELOPE_OPENERS: tuple[tuple[str, str], ...] = (
        ("<tool_call>", "</tool_call>"),
        ("<minimax:tool_call>", "</minimax:tool_call>"),
    )
    def _tool_envelope_in_flight(self) -> bool:
        """Return True iff the tool parser is mid-block on an unclosed
        ``<tool_call>``-style envelope.

        Consulted by the ``_should_route_through_reasoning`` split-prefix
        branch to decide whether a bare ``<`` (ambiguous between the
        start of ``<think>`` and the start of an inner Nemotron-shape
        ``<function=...>`` / ``<parameter=...>`` tag) should be allowed
        to re-route into the reasoning lane. The default-true behaviour
        is correct for plain prose; the false override here only fires
        once the tool parser has already accepted an unclosed envelope
        opener and the next ``<`` is overwhelmingly an inner tag (see
        comment at the call site for the failure mode).

        Cheap O(envelope_count) scan over ``tool_accumulated_text``;
        ``_TOOL_ENVELOPE_OPENERS`` keeps the openers/closers paired so
        a future wire format can be added in one place.
        """
        buf = self.tool_accumulated_text
        if not buf:
            return False
        for opener, closer in self._TOOL_ENVELOPE_OPENERS:
            if buf.count(opener) > buf.count(closer):
                return True
        return False
    def _consume_reasoning_budget(self, reasoning_text: str) -> tuple[str, str]:
        """Account for ``reasoning_text`` against the per-request cap.

        Returns ``(reasoning_kept, content_overflow)``:

        * ``reasoning_kept`` — the portion that fits under the cap; this
          is emitted as ``reasoning_content`` to the client.
        * ``content_overflow`` — the portion past the cap; the caller
          re-routes it to the CONTENT channel so no model output is
          silently dropped.

        Codex round-12 BLOCKING #1: cumulative-CHARACTER accounting
        (not per-chunk ceiling). The earlier draft converted each
        chunk to ``max(1, ceil(len/4))`` tokens, which made 4
        one-character reasoning deltas consume 4 "tokens" while the
        SAME 4 characters consume 1 token when chunked together. The
        cap then depended on engine chunking — a transient SSE flush
        could fire the cap pages earlier than expected. Fix: track
        cumulative reasoning chars and compare against ``cap * 4``
        (same character ceiling the non-stream
        ``_apply_reasoning_cap`` uses). All chunking patterns yield
        identical cap-firing positions, matching the non-stream path.

        Sets ``_reasoning_cap_hit`` to True the moment the running
        char count meets-or-exceeds ``cap * 4``.
        ``_reasoning_max_tokens=None`` short-circuits to "no cap".
        """
        if self._reasoning_max_tokens is None or not reasoning_text:
            return reasoning_text, ""
        if self._reasoning_cap_hit:
            # Cap already fired — anything still arriving on the
            # reasoning channel is overflow content.
            return "", reasoning_text
        max_chars = self._reasoning_max_tokens * 4
        # ``_reasoning_tokens_emitted`` actually stores CHARACTERS
        # post-round-12 (the field name is kept for backward-compat
        # with downstream usage-block consumers that grep for the
        # symbol — the value is still divided by 4 for the
        # ``completion_tokens_details.reasoning_tokens`` derivation).
        new_total_chars = self._reasoning_tokens_emitted + len(reasoning_text)
        if new_total_chars < max_chars:
            self._reasoning_tokens_emitted = new_total_chars
            return reasoning_text, ""
        if new_total_chars == max_chars:
            # Exact-boundary fit: the current chunk uses up the budget
            # but doesn't overflow. Keep it as reasoning AND latch the
            # cap so the NEXT incoming chunk is rerouted / triggers the
            # ``</think>`` injection. Codex round-2 BLOCKING #1.
            self._reasoning_tokens_emitted = new_total_chars
            self._reasoning_cap_hit = True
            return reasoning_text, ""
        # Cap crosses inside this chunk. Split at the remaining char
        # budget so the kept prefix stays under the ceiling and the
        # rest spills to content.
        remaining_chars = max_chars - self._reasoning_tokens_emitted
        keep_chars = max(0, remaining_chars)
        kept = reasoning_text[:keep_chars]
        overflow = reasoning_text[keep_chars:]
        self._reasoning_tokens_emitted = max_chars
        self._reasoning_cap_hit = True
        return kept, overflow
    def _maybe_inject_reasoning_close(self, delta_text: str) -> str:
        """Inject ``</think>`` once into the next model-text chunk when
        the cap fires on a text-parser engine.

        Text-parser engines (hermes / qwen3 / glm47) emit
        ``<think>...</think>`` themselves and rely on the streaming
        reasoning parser to split content from reasoning. Once the cap
        fires, we forge the close marker so the parser flips to content
        on the very next call to ``extract_reasoning_streaming`` —
        mirrors the channel-routed engines' force-close behavior so the
        client-visible semantic is identical across parser families.

        Codex round-10 BLOCKING #1: the latch
        (``_reasoning_close_injected = True``) used to flip HERE,
        BEFORE the parser call. If the parser then raised on the
        injected chunk, the next chunk would still see the latch set
        and skip injection — leaving the parser permanently mid-think.
        The latch is now flipped in the CALLER
        (``_process_with_reasoning``) AFTER the parser call succeeds.
        This function still gates on the latch (idempotency) and
        prepends the marker, but no longer mutates state.
        """
        if not self._reasoning_cap_hit or self._reasoning_close_injected:
            return delta_text
        if self.reasoning_parser is None:
            # Standard / channel-routed path doesn't need the injection.
            return delta_text
        # Prepend the marker so the parser sees ``</think>`` BEFORE the
        # next body bytes. The caller flips ``_reasoning_close_injected``
        # only after the parser call succeeds.
        return "</think>" + delta_text
    def _forced_tool_choice_name(self) -> str | None:
        """Return the forced ``tool_choice`` function name, if any.

        OpenAI spec: ``tool_choice={"type":"function","function":
        {"name":"X"}}`` forces the model to call exactly the named
        function — no other tool may appear in ``tool_calls[*]``.

        F-200 (2026-06-20): reasoning models that share the hermes
        tool parser (qwen3-thinking, phi-4-mini-reasoning, …)
        speculatively emit scratch ``<tool_call>...</tool_call>``
        blocks INSIDE ``<think>`` while planning. The MiniMax tool-
        markup redirect (load-bearing for the forced-prefix-in-think
        path) promotes those scratch blocks to content + tool_call
        detection, which then ship as schema-violating tool_calls
        with non-JSON ``arguments`` (e.g. bare ``"1234567890"``).
        Filtering on the forced name at delta-emission time keeps
        ONLY the spec-compliant call on the wire.

        Returns ``None`` when ``tool_choice`` is unset, ``"auto"`` /
        ``"none"`` / ``"required"``, or a non-string-named function
        shape — i.e. only the unambiguous named-function form gates
        the filter. ``"required"`` (no name) is intentionally NOT
        gated here: the model may legitimately choose any of the
        submitted tools.
        """
        req = self.request
        if req is None:
            return None
        if isinstance(req, dict):
            tc = req.get("tool_choice")
        else:
            tc = getattr(req, "tool_choice", None)
        if tc is None:
            return None

        # Production routes call ``request.model_dump(exclude_none=True)``
        # before constructing the postprocessor so ``tool_choice`` is a
        # plain dict here. Codex r4 BLOCKING: a typed-request callpath
        # (test fixtures, future refactors that thread the model
        # object directly) would leave ``tc`` as a Pydantic model with
        # ``.type`` / ``.function.name`` attributes — the dict-only
        # gate silently disabled the filter on that path. Read both
        # shapes via a tiny shape-agnostic accessor so future drift
        # cannot reopen the leak.
        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        if _get(tc, "type") != "function":
            return None
        fn = _get(tc, "function")
        if fn is None:
            return None
        name = _get(fn, "name")
        return name if isinstance(name, str) and name else None
    def _is_tool_choice_required(self) -> bool:
        """Return True iff the request set ``tool_choice="required"``.

        R10-H3 (Sven r10-R1): under ``tool_choice="required"`` the model
        is forced to emit at least one tool_call, but the OpenAI spec
        still requires every emitted call's ``arguments`` to be a JSON-
        object-encoded string. Pre-fix the streaming postprocessor
        applied no argument validation in this mode, so reasoning
        models occasionally streamed raw token sequences (``"20230805"``,
        ``"☉ Paris"``) into the ``arguments`` field. Clients running
        ``json.loads(arguments)`` then bailed mid-agent-loop.

        Helper kept SEPARATE from ``_forced_tool_choice_name`` so the
        forced-named-choice path (single-call enforcement, name match)
        and the ``required``-but-flexible path (any tool, args must
        still be objects) don't conflate.
        """
        req = self.request
        if req is None:
            return False
        if isinstance(req, dict):
            tc = req.get("tool_choice")
        else:
            tc = getattr(req, "tool_choice", None)
        return tc == "required"
    def _apply_forced_tool_choice_filter(self, tool_calls: list[dict]) -> list[dict]:
        """Suppress streaming tool_calls deltas that violate the
        ``tool_choice`` contract (forced named, or ``"required"``).

        Drop conditions enforced per the OpenAI spec:

        1. **Wrong function** (forced-name mode only): an anchor delta
           naming a function other than the forced choice. This catches
           harmony / gemma4 channel-routed calls to other tools the
           model speculated on but the client never requested.

        2. **Schema-violating arguments** (forced-name AND required): an
           anchor whose ``arguments`` is non-empty AND does not parse
           as a JSON OBJECT. The OpenAI spec mandates ``arguments`` be
           a JSON-encoded string and tool schemas are object-shaped
           (``{"type":"object","properties":{…}}``); a bare integer
           ``"1234567890"`` / ``"20230805"`` or string ``"☉ Paris output"``
           is the model's scratch-pad — not a real call. Captures the
           F-200 reasoning-model scratch leak that the MiniMax tool-
           markup redirect promoted into structured deltas, AND the
           R10-H3 (Sven r10-R1) ``tool_choice="required"`` token-string
           leak on qwen3-bf16.

        3. **Single-call cap** (forced-name mode only): once an anchor
           has been admitted for the forced choice, every subsequent
           anchor is dropped. R10-H2 (Sven r10-R1) — pre-fix, qwen3
           streaming sometimes emitted two anchor deltas with the same
           function name and DIFFERENT ``call_id`` values; openai-agents
           and claude-agents loops then executed the tool twice. The
           OpenAI spec for forced named-function mode is "exactly one
           tool_call for the named function". ``tool_choice="required"``
           is NOT subject to this cap (parallel multi-tool dispatch is
           legal there); the argument-shape gate above still applies.

        Argument-fragment continuation deltas (no name, no id) pass
        through unconditionally — the parallel-cap layer already
        tracks ``_no_index_last_dropped`` so fragments routed to a
        dropped anchor are suppressed.

        No-op when ``tool_choice`` is unset / ``"auto"`` / ``"none"``:
        the auto-mode flows must keep working without per-anchor
        post-filtering.
        """
        forced_name = self._forced_tool_choice_name()
        required_mode = self._is_tool_choice_required()
        if not forced_name and not required_mode:
            return tool_calls
        filtered: list[dict] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                filtered.append(tc)
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
            wrapped_name = (
                fn.get("name") if fn and isinstance(fn.get("name"), str) else None
            )
            flat_name = tc.get("name") if isinstance(tc.get("name"), str) else None
            anchor_name = wrapped_name or flat_name
            if anchor_name is None:
                # Continuation fragment — usually defer to cap-layer
                # routing. When the prior anchor for this forced choice
                # was dropped (wrong name, schema-violating args, or
                # single-call cap hit), ``_no_index_last_dropped`` is
                # set and the cap layer suppresses these fragments too.
                #
                # r10-J round-3 (codex r3 HIGH #1): the deferral was
                # incomplete. Some streaming parsers admit a valid
                # name-only anchor first and then send the ARGUMENTS
                # in a follow-up continuation that carries
                # ``{"function": {"arguments": "20230805"}}`` — a
                # schema-violating non-object root. The prior anchor
                # was admitted (so ``_no_index_last_dropped`` is False)
                # and the cap layer happily passes the continuation
                # through, leaking the malformed-args contract the
                # finalized-anchor branch below was meant to close.
                #
                # Mirror the object-root gate here: if the continuation
                # carries args AND those args parse to a non-object
                # root, drop it and tell the cap layer to drop the
                # rest. Partial-fragment JSON (unbalanced braces) is
                # passed through unchanged — the helper already returns
                # False for those, so the legitimate
                # name-then-fragmented-args streaming pattern keeps
                # working.
                wrapped_args = (
                    fn.get("arguments")
                    if fn and isinstance(fn.get("arguments"), str)
                    else None
                )
                flat_args = (
                    tc.get("arguments")
                    if isinstance(tc.get("arguments"), str)
                    else None
                )
                cont_args = wrapped_args if wrapped_args is not None else flat_args
                # r10-J round-5 (codex r5 HIGH #1): use the narrower
                # continuation-specific predicate, NOT the finalized-
                # anchor one. The finalized-anchor predicate treats
                # "balanced-but-broken" / "more } than {" as garbage
                # to drop, which over-rotates a legitimate split-JSON
                # closing fragment (``ris"}`` half of ``{"city":"Pa``
                # / ``ris"}``) into "drop". The continuation helper
                # only drops when the fragment ALONE parses as a
                # confirmed JSON non-object root.
                if self._continuation_arguments_definitively_non_object(cont_args):
                    self._no_index_last_dropped = True
                    continue
                filtered.append(tc)
                continue
            if forced_name and anchor_name != forced_name:
                # Wrong function — suppress this anchor and tell the
                # cap layer to drop its fragment continuations.
                self._no_index_last_dropped = True
                continue
            # Right function: validate the (so-far complete) arguments
            # field. ``arguments`` can be absent on an anchor that
            # only carries name (the JSON body streams in later
            # fragments); pass those through. When arguments IS
            # present and is non-empty, require it to parse as a
            # JSON object.
            wrapped_args = (
                fn.get("arguments")
                if fn and isinstance(fn.get("arguments"), str)
                else None
            )
            flat_args = (
                tc.get("arguments") if isinstance(tc.get("arguments"), str) else None
            )
            args_str = wrapped_args if wrapped_args is not None else flat_args
            if self._forced_tool_choice_arguments_violate_object_root(args_str):
                # F-200 root case + R10-H3 (Sven r10-R1): ``arguments``
                # parsed as JSON but the root type is not ``object``
                # (raw token sequence ``"20230805"``, bare integer,
                # bare string). Schema-violating — drop the anchor and
                # route fragments to drop. Equally important under
                # forced ``tool_choice``: surfacing a malformed call
                # to clients causes their ``json.loads(arguments)`` to
                # break the agent loop. Silent drop here is preferred
                # over surfacing a broken contract.
                self._no_index_last_dropped = True
                continue
            # R10-H2 (Sven r10-R1): single-call enforcement under
            # forced ``tool_choice={"type":"function","function":
            # {"name":X}}``. Once an anchor for the forced choice has
            # been admitted, every NEW anchor (different ``id``) is
            # dropped — the OpenAI spec mandates exactly one tool_call
            # for a forced named choice. Pre-fix qwen3 / hermes
            # streaming sometimes emitted the same call shape twice
            # (scratch + final) with two distinct ``call_id`` values
            # and agent loops (openai-agents, claude-agents) executed
            # the tool twice.
            #
            # The latch keys on the admitted ``id``: cumulative
            # argument-update parsers re-emit the same anchor
            # (same ``id``, growing arguments JSON) on every delta;
            # those re-emissions MUST pass through so the parallel-
            # cap layer can route them as continuations of the
            # admitted call. Round-10 of the parallel-cap codex review
            # already wired the no-index re-emission path; the
            # forced-choice latch mirrors that contract by id-match.
            #
            # ``tool_choice="required"`` is NOT subject to the
            # single-call latch: the spec only mandates "at least one
            # tool call", so multi-tool parallel dispatch is legal in
            # that mode. The argument-shape gate above still fires for
            # required, but the count gate stays open.
            if forced_name and self._forced_anchor_admitted_id is not None:
                anchor_id = (
                    tc.get("id")
                    if isinstance(tc.get("id"), str) and tc.get("id")
                    else ""
                )
                if anchor_id != self._forced_anchor_admitted_id:
                    # Duplicate call — different id from the admitted
                    # one. Drop the anchor and route its fragment
                    # continuations to drop via the cap layer.
                    self._no_index_last_dropped = True
                    continue
                # Same id — cumulative re-emission of the admitted
                # call's growing arguments JSON. Pass through.
            elif forced_name:
                anchor_id = (
                    tc.get("id")
                    if isinstance(tc.get("id"), str) and tc.get("id")
                    else ""
                )
                self._forced_anchor_admitted_id = anchor_id
            filtered.append(tc)
        return filtered
    def _parallel_tool_calls_allowed(self) -> bool:
        """Return False iff the request explicitly opted out of
        parallel tool calls via ``parallel_tool_calls=false``.

        OpenAI spec: ``True`` and unset both mean "no cap". Only the
        explicit ``false`` triggers single-call enforcement (matches
        the non-streaming trim in ``routes/chat.py`` post-parse). The
        request may arrive as a pydantic model (production) or a dict
        (test fixtures, lifted bench scaffolds); accept both.
        """
        req = self.request
        if req is None:
            return True
        if isinstance(req, dict):
            val = req.get("parallel_tool_calls")
        else:
            val = getattr(req, "parallel_tool_calls", None)
        return val is not False
    def _apply_parallel_cap(self, tool_calls: list[dict]) -> list[dict]:
        """Filter a streaming tool_calls delta list under the
        ``parallel_tool_calls=false`` cap, distinguishing NEW tool
        calls (unseen ``index``) from CONTINUATION deltas (seen
        ``index`` — name + incremental argument fragments for an
        already-admitted call).

        Text-parser streaming paths (hermes, qwen3_coder, etc.) emit
        many deltas per logical call: a header carrying ``{index, id,
        function: {name}}``, then a sequence of deltas carrying only
        ``{index, function: {arguments: "<fragment>"}}``. PR #518 round-1
        codex BLOCKING: the prior implementation consumed a cap slot
        per delta, so the first argument fragment for index 0 took the
        only slot and every subsequent fragment of THE SAME CALL was
        dropped — silently truncating the JSON arguments mid-string.

        New rule:
          - Uncapped (parallel=true / unset): pass everything; ONLY
            track ``index`` admits (for the channel-routed branch's
            monotonic-counter math). Do NOT touch
            ``_no_index_call_admitted`` here — that field is cap-only
            state, and mutating it in the uncapped path could
            pollute cap accounting if request flags change mid-stream
            (PR #518 round-3 codex NIT).
          - Capped (parallel=false): for each delta, if its ``index``
            is already admitted, pass through (continuation). If
            ``index`` is absent AND the delta carries ONLY argument
            fragments (no new ``id`` / ``name``), treat it as a
            continuation of the in-flight no-index call. A no-index
            delta carrying a fresh ``id`` or function ``name`` is a
            NEW call — admit only if the cap allows. PR #518 round-3
            codex BLOCKING: previously, every subsequent no-index
            delta was treated as a continuation, leaking a second
            full call past the cap.
          - Cap-full new calls are dropped, AND their later
            continuations are dropped too (no admit ever fired, so
            the index/no-index slot was never taken).

        Returns the filtered list (possibly empty if every delta in
        the batch is a new call past the cap).
        """
        if self._parallel_tool_calls_allowed():
            # Still track admitted indices so the channel-routed branch
            # can use the same set when assigning its own monotonic
            # ``index`` values from the count.
            for tc in tool_calls:
                idx = tc.get("index") if isinstance(tc, dict) else None
                if isinstance(idx, int):
                    self._admitted_tool_call_indices.add(idx)
            self._structured_tool_call_count = max(
                self._structured_tool_call_count,
                len(self._admitted_tool_call_indices),
            )
            return list(tool_calls)

        allowed: list[dict] = []
        for tc in tool_calls:
            idx = tc.get("index") if isinstance(tc, dict) else None
            fn = tc.get("function") if isinstance(tc, dict) else None
            has_wrapped_name = (
                isinstance(fn, dict)
                and isinstance(fn.get("name"), str)
                and fn.get("name")
            )
            # Round-8 codex BLOCKING #2: parsers can emit FLAT-shape
            # tool calls (``{"name": "X", "arguments": ...}`` — no
            # ``function`` wrapper, mirrored from raw engine output
            # via ``_tool_call_name`` shape #3 in chat.py). Without
            # the top-level ``name`` check, a flat-shape second call
            # was misclassified as a continuation and leaked past
            # the ``parallel_tool_calls=false`` cap.
            has_flat_name = (
                isinstance(tc, dict)
                and isinstance(tc.get("name"), str)
                and tc.get("name")
            )
            has_id = (
                isinstance(tc, dict) and isinstance(tc.get("id"), str) and tc.get("id")
            )
            is_anchor = bool(has_wrapped_name or has_flat_name or has_id)

            if isinstance(idx, int) and idx in self._admitted_tool_call_indices:
                # Continuation of an already-admitted indexed call —
                # always forward so the client's arguments JSON is
                # complete. Round-9 codex BLOCKING #2: seeing a fresh
                # continuation of an admitted indexed call signals
                # that the in-flight call is still alive, so reset
                # the dropped-anchor flag — otherwise a NO-INDEX
                # argument fragment immediately following this
                # indexed continuation would be wrongly dropped as
                # "belongs to a dropped call" when it really belongs
                # to THIS admitted call.
                self._no_index_last_dropped = False
                allowed.append(tc)
                continue

            # No-index anchor matching the admitted no-index call's
            # identity: cumulative argument-update parsers re-emit
            # ``{"id": "<same>", "function": {"name": "<same>",
            # "arguments": "<grew>"}}`` on every delta rather than
            # emitting a single anchor and bare-argument continuations.
            # Without this branch, every such re-emission would be
            # mis-classified as a new call and dropped under
            # ``parallel_tool_calls=false`` (round-10 codex BLOCKING #2).
            # Match if BOTH the delta and the admitted call carry id
            # AND ids match, OR if id is absent on the delta and the
            # function names match — never silently accept a different
            # call identity as continuation.
            if idx is None and is_anchor and self._no_index_call_admitted:
                delta_id = tc.get("id") if has_id else None
                delta_name = (
                    fn.get("name")
                    if has_wrapped_name
                    else (tc.get("name") if has_flat_name else None)
                )
                id_matches = (
                    delta_id is not None
                    and self._no_index_admitted_id is not None
                    and delta_id == self._no_index_admitted_id
                )
                name_matches_no_id_conflict = (
                    delta_id is None
                    and delta_name is not None
                    and self._no_index_admitted_name is not None
                    and delta_name == self._no_index_admitted_name
                )
                if id_matches or name_matches_no_id_conflict:
                    self._no_index_last_dropped = False
                    allowed.append(tc)
                    continue

            # Argument-only no-index fragment: routes to whichever
            # anchor was most recently seen. Any admitted call (indexed
            # OR no-index slot) keeps the fragment unless the most
            # recent anchor was dropped.
            #
            # Round-5 codex BLOCKING #2: previously this branch only
            # fired when ``_no_index_call_admitted`` was True. An
            # indexed FIRST delta (e.g. ``{"index": 0, "id": "a",
            # "function": {"name": "a", "arguments": "{"}}``) followed
            # by argument-only no-index deltas (``{"function":
            # {"arguments": "}"}}``) routed the fragments to the
            # new-call cap-check and dropped them as cap-full —
            # truncating the JSON. Now any admitted call (indexed
            # or no-index) absorbs no-index argument fragments.
            if idx is None and not is_anchor:
                has_admitted_call = bool(self._admitted_tool_call_indices) or (
                    self._no_index_call_admitted
                )
                if has_admitted_call:
                    if self._no_index_last_dropped:
                        # Most recent anchor was dropped; suppress so
                        # the dropped call's args don't leak into the
                        # admitted call's payload.
                        continue
                    allowed.append(tc)
                    continue
                # Falls through to new-call branch (first delta of the
                # stream has no index AND no anchor — treat as new).

            # New call: unseen index, fresh no-index call with id/name,
            # or first no-index delta with no admitted call yet.
            already_admitted = len(self._admitted_tool_call_indices) + (
                1 if self._no_index_call_admitted else 0
            )
            if already_admitted >= 1:
                # Cap full — drop this new call AND any further
                # continuations of its index, since we never admit it.
                # Mark so subsequent no-index argument-only fragments
                # are routed to "dropped" rather than silently
                # appended to the admitted call. Round-6 codex
                # BLOCKING: previously this flag was only set when
                # the dropped anchor was no-index, so an INDEXED
                # dropped anchor would leave the flag clear and the
                # next no-index argument fragment would leak into
                # the admitted call's payload.
                self._no_index_last_dropped = True
                continue
            if isinstance(idx, int):
                self._admitted_tool_call_indices.add(idx)
                # Indexed admit: subsequent no-index argument fragments
                # belong to the in-flight admitted call. Reset the
                # dropped-anchor flag (the cap-full branch above is
                # the only writer).
                self._no_index_last_dropped = False
            else:
                # Mark the no-index slot as taken; subsequent no-index
                # deltas hit the continuation branch above. Reset the
                # dropped-anchor flag — this delta is the most recent
                # anchor and it was admitted, so its fragments belong
                # here. Capture the admitted identity (id + name) so a
                # later anchor delta carrying the SAME id/name (parsers
                # that re-emit the anchor with cumulative arguments) is
                # matched as a continuation rather than misclassified
                # as a new call. PR #518 round-10 codex BLOCKING #2.
                self._no_index_call_admitted = True
                self._no_index_last_dropped = False
                if has_id:
                    self._no_index_admitted_id = tc.get("id")
                if has_wrapped_name:
                    self._no_index_admitted_name = fn.get("name")
                elif has_flat_name:
                    self._no_index_admitted_name = tc.get("name")
            self._structured_tool_call_count = max(
                self._structured_tool_call_count,
                len(self._admitted_tool_call_indices)
                + (1 if self._no_index_call_admitted else 0),
            )
            allowed.append(tc)
        return allowed
    def _detect_tool_calls(self, content: str) -> dict | None:
        """Run incremental tool call detection.

        Returns None if content is suppressed (inside tool markup).
        Returns {"tool_calls": [...]} if tool calls detected.
        Returns {"content": "..."} for normal content pass-through.
        """
        if not self.tool_markup_possible and "<" not in content and "[" not in content:
            # The hardcoded ``<``/``[`` heuristic catches every parser
            # whose wire markers open with one of those chars. The
            # gemma4 stripped wire form is the exception: on
            # DiffusionGemma, HF's ``tokenizer.decode(skip_special_
            # tokens=True)`` removes the ``<|tool_call>``/``<tool_call|>``
            # outer wrappers, so what reaches the postprocessor is the
            # bare body ``call:NAME{...}`` — no ``<``, no ``[``. Without
            # the parser-level fallback below, those deltas would slip
            # straight through this fast-path as plain ``content`` and
            # leak ``call:calculator{expression:432+1}``-style raw wire
            # text to the SSE client (regression reported via vnsh.dev
            # share probe 2026-06-11, PR #558).
            candidate = self.tool_accumulated_text + content
            pending = False
            if self.tool_parser is not None:
                _check = getattr(self.tool_parser, "has_pending_tool_call", None)
                if callable(_check):
                    try:
                        pending = bool(_check(candidate))
                    except Exception:
                        pending = False
            if not pending:
                self.tool_accumulated_text += content
                return {"content": content}
            # Parser sees in-flight markup with non-``<``/``[`` opener
            # (the gemma4 stripped form). Fall through to the full
            # streaming path so it can suppress / emit structured
            # tool_calls instead of leaking the body as content.
            self.tool_markup_possible = True

        if not self.tool_markup_possible:
            self.tool_markup_possible = True

        tool_previous = self.tool_accumulated_text
        self.tool_accumulated_text += content
        tool_result = self.tool_parser.extract_tool_calls_streaming(
            tool_previous,
            self.tool_accumulated_text,
            content,
            request=self.request,
        )

        if tool_result is None:
            return None  # inside tool markup

        if "tool_calls" in tool_result:
            self.tool_calls_detected = True
            return tool_result

        return {"content": tool_result.get("content", "")}

    @staticmethod
    def _create_reasoning_parser(cfg):
        return _create_reasoning_parser(cfg)

    @staticmethod
    def _create_tool_parser(cfg, tools_requested):
        return _create_tool_parser(cfg, tools_requested)

    @staticmethod
    def _clone_injected_tool_parser(parser):
        return _clone_injected_tool_parser(parser)

    @staticmethod
    def _forced_tool_choice_arguments_violate_object_root(args_str):
        return _forced_tool_choice_arguments_violate_object_root(args_str)

    @staticmethod
    def _continuation_arguments_definitively_non_object(args_str):
        return _continuation_arguments_definitively_non_object(args_str)
