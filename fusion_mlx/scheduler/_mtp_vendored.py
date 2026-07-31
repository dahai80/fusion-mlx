# SPDX-License-Identifier: Apache-2.0
"""Vendored MTP hot-loop installer.

Migrated from Rapid-MLX ``vllm_mlx.scheduler._install_mtp_vendored`` (PR #990).
Installs the vendored ``mtp_generate_step`` hot loop into
``GenerationBatch._step`` for ``--spec-decode mtp`` (Gemma 4 external
assistant + Qwen3.5 baked-in MTP). Re-exports as
``fusion_mlx.scheduler._install_mtp_vendored``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_lm.generate import BatchGenerator

logger = logging.getLogger(__name__)


def _install_mtp_vendored(
    batch_gen: BatchGenerator,
    model: Any,
    requests: dict[str, Any] | None = None,
    uid_to_request_id: dict[int, str] | None = None,
    max_k: int = 3,
    disable_auto_k: bool = False,
    controller_key: str | None = None,
) -> bool:
    """Install the vendored PR #990 ``mtp_generate_step`` hot loop into
    ``GenerationBatch._step``.

    This is the SERVER-SIDE wiring for ``--spec-decode mtp`` (Gemma 4
    external assistant + Qwen3.5 baked-in MTP). It supplants the legacy
    :func:`_install_mtp` (which targets a stale mlx-lm 0.30 shape whose
    ``BatchGenerator._step`` no longer exists in 0.31+).

    Gate (all required):
      * ``model`` exposes the ``mtp_generate_step`` protocol:
        ``mtp_forward``, ``make_mtp_cache`` (installed by
        :func:`~vllm_mlx.spec_decode.mtp.dispatch.dispatch_mtp_inject`).
      * ``batch_gen._generation_batch`` exists (mlx-lm 0.31+).

    On a gate miss, logs a WARN and returns ``False`` — the request
    continues on plain autoregressive decode.

    Hook shape: replaces ``GenerationBatch._step`` (mlx-lm 0.31+ shape:
    ``() -> (List[int], List[mx.array])``). Per-step, exactly one primary
    token is returned to keep the mlx-lm ``next()`` contract intact.
    Multi-token gains come from the generator's internal batched
    backbone+MTP passes (up to 2 tokens per pass), not from returning
    multiple tokens per ``_step`` call. Extra tokens produced by the
    generator are queued and drained on the following ``_step`` calls.

    K=1 chain-of-1 scope (PR-A of 0.9.13 stack):

    * Single-request only (``len(gb.uids) == 1``). Multi-request batches
      fall through to ``_orig_step`` — Gemma 4's MTP fast-path is
      batch=1-only (``mtp_forward`` raises on B>1) and the vendored
      generator maintains its own per-request state. Auto-K controller
      lives in PR-B; batched residual+bonus sync lives in PR-C.

    * Greedy sampling only (temperature == 0). Non-greedy falls through
      to ``_orig_step`` — the byte-lossless verify contract lives in the
      generator's residual-distribution sampling on reject, which the
      MVP does not exercise. Non-greedy support is a follow-up.

    * No logits processors. If any position of ``gb.logits_processors``
      is truthy we fall through — the generator has its own logits-
      processor plumbing but wiring the mlx-lm per-uid processor list
      through to the generator is out of MVP scope.

    * On the very first ``_step`` call we short-circuit and return the
      token that mlx-lm's fresh ``GenerationBatch.__init__._step()``
      already sampled and stashed in ``_next_tokens``. This preserves
      byte-equal output vs. baseline: the FIRST generated token is the
      argmax(prefill-final-logits), identical to plain decode. We seed
      the generator with that same token so its first backbone step
      produces the SECOND generated token.
    """
    gb = getattr(batch_gen, "_generation_batch", None)
    if gb is None:
        logger.warning(
            "[MTP-vendored] disabled: BatchGenerator has no _generation_batch "
            "attribute (mlx-lm version mismatch — expected >=0.31)."
        )
        return False

    if not (
        hasattr(model, "mtp_forward")
        and hasattr(model, "make_mtp_cache")
        and hasattr(model, "mtp")
    ):
        logger.warning(
            "[MTP-vendored] disabled: model lacks mtp_forward / make_mtp_cache / "
            "mtp attributes — dispatch_mtp_inject did not run or returned False. "
            "--spec-decode mtp will be a no-op; requests continue on plain "
            "autoregressive decode."
        )
        return False

    # Lazy import — the generator module pulls in mlx-lm's sample_utils and
    # patches ArraysCache; keep the import off the scheduler boot path so a
    # non-MTP build has zero cost.
    from ..speculative.mtp.generator import mtp_generate_step

    _orig_step = gb._step

    # Per-uid MTP state. Each entry:
    #   {
    #     "gen": the mtp_generate_step generator instance (or None on FIRST call),
    #     "queue": deque of pending (tok_int, lp_array, from_draft_bool),
    #     "primed": True after we emit the vanilla-sampled first token,
    #     "request_id": the request_id captured at construction time —
    #       codex round-K BLOCKING #1. mlx-lm reuses uid ints when a
    #       request completes; without tracking the owning request
    #       here, a new request that draws the same uid would resume
    #       the OLD generator (built for the old prompt/prompt_cache)
    #       and emit stale tokens from the previous request — a data
    #       corruption bug. On every ``_mtp_step`` call, we compare
    #       ``_state[uid]["request_id"]`` against the current
    #       ``uid_to_request_id[uid]``; on mismatch we treat the state
    #       as stale and reset to the FIRST-call branch.
    #   }
    # Only one uid is ever active at a time under the batch=1 gate.
    _state: dict[int, dict[str, Any]] = {}

    # Codex round-D blocker #2 + round-E blocker #1: permanent-skip
    # map, keyed by uid with the request_id at the time of disabling
    # as the value. Used to:
    #
    # 1. Skip retrying MTP construction on a uid whose first-call
    #    construction failed (round-D — otherwise a bad sidecar or
    #    weight-shape mismatch would DoS the request with one failed
    #    construction attempt per token).
    #
    # 2. Detect uid reuse across requests and re-enable MTP for the
    #    new request. mlx-lm reuses uid ints when a request completes;
    #    keying only by uid (round-D's initial fix) let a bad sidecar
    #    state from request N permanently disable MTP for request
    #    N+1, N+2, … that happened to draw the same uid.
    #    (round-E BLOCKER #1). Storing the request_id lets us
    #    distinguish "same request, still disabled" from "uid was
    #    reused, forget the stale disable."
    #
    # The value can be None (as a placeholder) when the outer install
    # was called with ``uid_to_request_id=None`` — that case is
    # unavoidable and we accept the pre-round-E uid-lifetime scope
    # (this only happens under bench harness callers, where uids are
    # not reused across requests anyway).
    _disabled_uids: dict[int, str | None] = {}

    _stats = {
        "vendored_steps": 0,
        "fallthrough_steps": 0,
        "ft_batch_size": 0,
        "ft_non_greedy": 0,
        "ft_logits_processors": 0,
        "ft_disabled": 0,
        "gen_exhausted": 0,
        "gen_raised": 0,
        # Codex round-L BLOCKING #2-4: track uids that have been
        # handed off from MTP to plain decode mid-stream so subsequent
        # fallback branches can log ONCE per uid rather than once per
        # step. Silent degradation is per Ollama's depth-0 park
        # behavior; but we still want the operator to see the
        # degradation happened, without log spam if the batch stays
        # B>1 (or non-greedy, or has an lp) for many tokens.
        "ft_mid_stream_handoff": 0,
    }

    # Codex round-L BLOCKING #2-4: log-once bookkeeping for mid-stream
    # MTP → plain-decode handoffs. Keyed by (uid, reason) so the same
    # uid can log both "B>1" and "non-greedy" if it hits both, but
    # each reason surfaces at most once per uid lifetime.
    _handoff_logged: set[tuple[int, str]] = set()

    def _log_mtp_mid_stream_handoff_once(uid: int, reason: str, detail: str) -> None:
        """Emit a WARN log for a mid-stream MTP → plain-decode handoff,
        at most once per (uid, reason).

        Codex round-L BLOCKING #2-4: the fallback design matches
        Ollama's depth-0 park behavior — MTP silently degrades to
        plain decode when the current step is incompatible (B>1,
        non-greedy, or has a logits processor) instead of aborting
        the request with a RuntimeError. But the tradeoff is real:
        ``gb._next_tokens`` currently holds the last-MTP-emitted
        token (see ``_sync_next_tokens_after_emit``) rather than a
        fresh baseline sample, so ``_orig_step`` may emit a
        duplicated token or sample from a slightly stale cache
        position for one step before the request continues on plain
        decode. Log the handoff so the operator can correlate the
        potential stream artifact with the load-balancing event.
        """
        key = (uid, reason)
        if key in _handoff_logged:
            return
        _handoff_logged.add(key)
        logger.warning(
            "[MTP-vendored] uid=%s handoff to plain decode (%s): %s. "
            "The MTP generator was closed; the request continues on "
            "baseline mlx-lm _step. gb._next_tokens still holds the "
            "last-MTP-emitted token so the next _orig_step call may "
            "produce a duplicated token or a token sampled from a "
            "slightly stale cache position for one step — a bounded, "
            "known tradeoff for not killing the request "
            "(Ollama-style depth-0 park behavior).",
            uid,
            reason,
            detail,
        )

    def _cleanup_uid(uid: int) -> None:
        # Codex round-G BLOCKING #1: DO NOT clear _disabled_uids here.
        # This helper runs on every fallthrough branch (B>1, non-greedy,
        # logits-processors, mid-stream failure), so unconditionally
        # popping _disabled_uids would silently "un-disable" a uid the
        # very next step. That would re-enable retry of MTP construction
        # (or of the vendored generator) on a request whose earlier
        # ``_mtp_step`` call has already proven the path is broken —
        # a slow-loss loop that codex round-G rightly called out.
        #
        # _disabled_uids has exactly TWO valid clear paths:
        #   1. Reuse detection in the ``uid in _disabled_uids`` gate
        #      inside _mtp_step (the round-E fix): a NEW request_id for
        #      the same uid means mlx-lm reused the uid; clear and let
        #      MTP re-arm for the new request.
        #   2. Never for the current request. The disable is a permanent
        #      marker for the request's lifetime.
        #
        # State (the per-uid MTP generator + queue) is cleaned here as
        # usual — that's per-generator lifecycle, not per-request.
        state = _state.pop(uid, None)
        if state is None:
            return
        gen = state.get("gen")
        if gen is not None:
            try:
                gen.close()
            except Exception:  # noqa: BLE001
                pass

    def _is_greedy_for_uid(uid: int) -> bool:
        """Return True when the request behind ``uid`` sampled at temp=0.

        K=1 MVP: matches the greedy contract that
        ``vllm_mlx/spec_decode/mtp/generator.py::mtp_generate_step``
        implements with ``temp=0.0``. Under temp>0, the vendored
        generator can still preserve the lossless marginal via its
        residual-distribution sample on reject — but the MVP install
        hard-codes ``temp=0.0`` into the generator constructor, so any
        request with temperature>0 would silently receive a
        different sampled marginal.

        Codex round-A blocker #1: fail closed on unresolvable metadata.
        Prior revision returned ``True`` when ``uid_to_request_id`` or
        ``requests`` were ``None`` (or the request lookup failed) —
        that would silently apply greedy sampling to a temp>0 request
        whose bookkeeping had just been evicted. Return ``False`` here
        so the caller falls through to ``_orig_step()`` (which reads
        the real sampler from ``gb.samplers[0]``) instead of applying
        the MTP-hardcoded greedy path.

        Codex round-B blocker: also fail closed when ``temperature is
        None``. ``vllm_mlx.request.SamplingParams`` defaults
        ``temperature=0.7`` (not zero) and ``None`` is not a normal
        value — it typically signals "use the server / OpenAI-route
        default," which is likewise nonzero. Treating a bare ``None``
        as greedy would silently apply the MTP-hardcoded ``temp=0.0``
        marginal to a request the operator meant to sample stochast-
        ically. Only an EXPLICIT ``0.0`` passes the gate; every other
        shape falls through to plain decode.
        """
        if uid_to_request_id is None or requests is None:
            return False
        req_id = uid_to_request_id.get(uid)
        req = requests.get(req_id) if req_id else None
        if req is None or getattr(req, "sampling_params", None) is None:
            return False
        temp = getattr(req.sampling_params, "temperature", None)
        return temp == 0.0

    def _mtp_step():
        """Wrapped ``GenerationBatch._step`` for --spec-decode mtp.

        See :func:`_install_mtp_vendored` docstring for the gate matrix
        and MVP caveats.
        """

        # --- Gate matrix ---
        # Batch=1 only. mlx-lm's ``PromptProcessingBatch.generate``
        # constructs a fresh ``GenerationBatch`` with size 1 per request
        # split; the persistent ``_generation_batch`` then extends
        # in-place. Under the smoke script's single-request load this
        # stays at 1 throughout.
        #
        # Codex round-A blocker #3 (initial cleanup requirement)
        # + codex round-H BLOCKING #1-3 (fallthrough safety):
        #
        # When B>1 (or non-greedy / logits-proc appears MID-stream):
        # if MTP has already emitted tokens for the affected uid,
        # falling through to ``_orig_step()`` is UNSAFE. The
        # wrapper never updates ``gb._next_tokens`` — it still
        # holds ``first_gen_tok`` from the priming ``_step`` in
        # ``__init__`` — so ``_orig_step()`` would emit
        # ``first_gen_tok`` AGAIN, duplicating the stream.
        #
        # Two-way split for every fallthrough branch:
        #   * ``_state`` empty for the affected uid → soft-fall-
        #     through to ``_orig_step()``. MTP hasn't primed
        #     anything, ``gb._next_tokens`` is the fresh sample
        #     baseline ``_step`` needs. Also mark the uid as
        #     disabled so subsequent steps in this request skip the
        #     wrapper entirely.
        #   * ``_state`` non-empty for the affected uid → TERMINAL.
        #     Record the disable marker (so any retry short-
        #     circuits) and raise ``RuntimeError``. Recovering to
        #     plain decode would require synthesising
        #     ``gb._next_tokens`` from the last MTP-emitted token,
        #     which we don't stage anywhere.
        def _record_terminal_disable(u: int) -> None:
            """Record a terminal disable marker for uid ``u`` and
            drop any per-generator state. Used on the "MTP already
            emitted, fallthrough is unsafe" path."""
            _term_req_id = None
            if uid_to_request_id is not None:
                _term_req_id = uid_to_request_id.get(u)
            _disabled_uids[u] = _term_req_id
            _state_entry = _state.pop(u, None)
            if _state_entry is not None:
                _gen = _state_entry.get("gen")
                if _gen is not None:
                    try:
                        _gen.close()
                    except Exception:  # noqa: BLE001
                        pass

        def _mark_disabled(u: int) -> None:
            """Mark uid ``u`` as disabled (for pre-MTP soft-fall-
            through paths). No state to clean up because state was
            empty at this branch."""
            _term_req_id = None
            if uid_to_request_id is not None:
                _term_req_id = uid_to_request_id.get(u)
            _disabled_uids[u] = _term_req_id

        def _sync_next_tokens_after_emit(
            gb_ref: Any,
            emitted_tok: int,
            emitted_lp: Any,
        ) -> None:
            """Sync ``gb._next_tokens`` / ``gb._next_logprobs`` shape
            with the token the wrapper just emitted.

            Codex round-I BLOCKING #2 / round-J BLOCKING #2+#3: mlx-lm's
            ``GenerationBatch._step`` contract maintains
            ``_next_tokens`` in a canonical shape so ``.filter(keep)``
            slicing and ``.extend(batch)`` concatenation see a live
            tensor at every step (initialized from ``inputs`` in
            ``__init__``, sliced by ``keep`` on request completion).
            The vendored wrapper's queue-driven emission path never
            touched those fields, leaving them frozen at the
            ``first_gen_tok`` staged by ``__init__``'s priming
            ``_step`` — a rank-1 uint32 that gets increasingly stale
            across the whole request.

            Round-J review: a prior revision drove the MTP generator
            one step ahead (a "prefetch") to publish the NEXT
            to-be-emitted token here, but that changed
            ``gb.prompt_cache`` state behind ``GenerationBatch``'s
            bookkeeping and swallowed generator exceptions (delaying
            the terminal-raise). Both were correctly flagged as
            unsafe.

            Simpler contract that satisfies round-I without the
            round-J side effects: stash the JUST-EMITTED token as the
            placeholder. Shape / dtype match mlx-lm's expected
            ``mx.array([tok], dtype=uint32)`` invariant so
            ``.filter`` / ``.extend`` slicing succeeds; the VALUE is
            semantically stale ("last emitted" rather than "next to
            feed"), but that's tolerated:

            * ``.filter(keep)`` / ``.extend`` don't forward through
              the model — they mutate the tensor in place. No
              downstream cache interaction.
            * Codex round-L BLOCKING #2-4 relaxed the round-H
              terminal-raise contract: the B>1 / non-greedy /
              logits-processor fallthrough branches now delegate to
              ``_orig_step()`` instead of aborting the request. In
              that handoff path, ``_orig_step()`` will read the
              stale ``_next_tokens`` and may emit a duplicated token
              or sample from a slightly stale cache position for one
              step. The wrapper logs a WARN (once per uid+reason)
              on the handoff so the operator can correlate the
              artifact with the load-balancing event. This is the
              accepted tradeoff for not killing the request — see
              :func:`_log_mtp_mid_stream_handoff_once` and the
              round-L rationale comments in the three fallthrough
              branches.

            Cache state stays under the MTP generator's control — the
            wrapper never advances ``prompt_cache`` outside a
            ``next(gen)`` call driven by an actual mlx-lm ``_step``
            request.
            """
            gb_ref._next_tokens = mx.array([int(emitted_tok)], dtype=mx.uint32)
            gb_ref._next_logprobs = [emitted_lp]

        if not gb.uids or len(gb.uids) != 1:
            _stats["fallthrough_steps"] += 1
            _stats["ft_batch_size"] += 1
            # Codex round-L BLOCKING #2: prior round-H revision raised
            # ``RuntimeError`` here when any uid in ``_state`` had in-
            # flight MTP emissions. That killed the request whenever
            # normal continuous-batching load added a second uid to
            # the batch — hostile behavior for a multi-request server
            # where B>1 is the norm, not the exception.
            #
            # Round-L fix: hand off to ``_orig_step`` regardless of
            # whether MTP has emitted. The MTP generator is closed and
            # the affected uid(s) are marked disabled so we don't
            # retry MTP on subsequent steps. The stream may briefly
            # exhibit a duplicated token or a token sampled from a
            # slightly stale cache position (``gb._next_tokens`` still
            # holds the last-MTP-emitted token) — a bounded, known
            # tradeoff that matches Ollama's ``depth=0`` park behavior
            # when speculation cannot proceed. See
            # :func:`_log_mtp_mid_stream_handoff_once` for the operator-
            # facing warning contract.
            if _state:
                terminal_uids = list(_state)
                _stats["ft_mid_stream_handoff"] += len(terminal_uids)
                for stale_uid in terminal_uids:
                    _log_mtp_mid_stream_handoff_once(
                        stale_uid,
                        "b_gt_1",
                        f"batch grew to size {len(gb.uids)}",
                    )
                    _record_terminal_disable(stale_uid)
            return _orig_step()

        uid = gb.uids[0]

        # Codex round-D blocker #2 + round-E blocker #1: honour the
        # permanent-skip map BEFORE re-entering FIRST-call
        # construction, but detect uid reuse across requests. mlx-lm
        # can recycle uid ints once a request completes; without the
        # request-id cross-check a bad sidecar state from a completed
        # request could silently disable MTP for every subsequent
        # request that happened to draw the same uid.
        if uid in _disabled_uids:
            disabled_req_id = _disabled_uids[uid]
            current_req_id = None
            if uid_to_request_id is not None:
                current_req_id = uid_to_request_id.get(uid)
            # Same request: still disabled — skip MTP for the rest of
            # its lifetime.
            #
            # Different request (uid reused): the disable state is
            # stale; drop it and re-enter normal MTP path. The new
            # request may be pointed at a working sidecar even if the
            # previous one wasn't.
            #
            # Missing bookkeeping (both sides None or the map itself
            # is None): can't distinguish. Fall back to the round-D
            # behaviour of honouring the disable — under bench-harness
            # callers uids aren't reused anyway, and treating this as
            # "still disabled" is the safe default.
            if (
                disabled_req_id is not None
                and current_req_id is not None
                and disabled_req_id != current_req_id
            ):
                # uid was reused for a new request — forget the stale
                # disable and fall through to normal MTP path.
                del _disabled_uids[uid]
            else:
                _stats["fallthrough_steps"] += 1
                _stats["ft_disabled"] += 1
                return _orig_step()

        if not _is_greedy_for_uid(uid):
            _stats["fallthrough_steps"] += 1
            _stats["ft_non_greedy"] += 1
            # Codex round-L BLOCKING #3: prior round-H revision raised
            # ``RuntimeError`` here when sampling switched to non-
            # greedy after MTP had already emitted. That killed the
            # request on a legitimate runtime sampling-param change.
            #
            # Round-L fix: hand off to ``_orig_step`` regardless of
            # state. The MTP generator is closed and the uid is
            # marked disabled so subsequent steps skip MTP entirely.
            # Same bounded stream-artifact tradeoff as the B>1 handoff
            # above; see :func:`_log_mtp_mid_stream_handoff_once` for
            # the operator-facing WARN contract.
            if uid in _state:
                _stats["ft_mid_stream_handoff"] += 1
                _log_mtp_mid_stream_handoff_once(
                    uid,
                    "non_greedy",
                    "sampling switched to temperature > 0 mid-stream",
                )
                _record_terminal_disable(uid)
            else:
                _mark_disabled(uid)
            return _orig_step()

        _lp = getattr(gb, "logits_processors", None)
        if _lp and any(p for p in _lp if p):
            _stats["fallthrough_steps"] += 1
            _stats["ft_logits_processors"] += 1
            # Codex round-L BLOCKING #4: prior round-H revision raised
            # ``RuntimeError`` here when a logits processor was added
            # mid-stream after MTP had already emitted. That killed
            # the request whenever an operator toggled a per-request
            # processor (e.g., a guided-decoding grammar) after the
            # first tokens streamed.
            #
            # Round-L fix: hand off to ``_orig_step`` regardless of
            # state. Same handoff pattern as B>1 and non-greedy: log
            # once per uid, mark disabled, delegate.
            if uid in _state:
                _stats["ft_mid_stream_handoff"] += 1
                _log_mtp_mid_stream_handoff_once(
                    uid,
                    "logits_processor",
                    "logits processor appeared mid-stream",
                )
                _record_terminal_disable(uid)
            else:
                _mark_disabled(uid)
            return _orig_step()

        state = _state.get(uid)

        # Codex round-K BLOCKING #1: uid reuse detection for the
        # ACTIVE (non-disabled) state map. mlx-lm reuses uid ints
        # when a request completes and a new one joins the batch.
        # Without this check the wrapper would resume the OLD
        # request's generator (built for a different prompt +
        # prompt_cache state) on the NEW request's next _step call
        # — a data corruption bug because the SUBSEQUENT branch
        # pulls tokens from the stale generator and appends them
        # to gb.tokens[0]. The round-E fix wired this exact
        # detection into ``_disabled_uids``; codex round-K
        # correctly notes the same treatment is missing here.
        #
        # If ``uid_to_request_id`` is not plumbed (bench harness)
        # we can't distinguish reuse from continuation and fall
        # back to the pre-round-K behaviour; this only matters
        # for harnesses that DON'T reuse uids anyway.
        if state is not None and uid_to_request_id is not None:
            stashed_req_id = state.get("request_id")
            current_req_id = uid_to_request_id.get(uid)
            if (
                stashed_req_id is not None
                and current_req_id is not None
                and stashed_req_id != current_req_id
            ):
                # uid was reused for a NEW request. Close the OLD
                # generator + drop the queue, then fall through to
                # FIRST-call construction so the new request gets a
                # fresh MTP path.
                _cleanup_uid(uid)
                state = None

        if state is None:
            # --- FIRST call for this uid ---
            # mlx-lm's fresh ``GenerationBatch.__init__`` ran its
            # ORIGINAL ``_step`` once (before our patch took effect on
            # the persistent gb), which fed ``last_prompt_token``
            # through the model, advanced ``prompt_cache`` by 1
            # position, and stashed the sampled FIRST generated token
            # in ``_next_tokens``. Emit that token now to preserve
            # byte-equality with plain-decode baseline: the argmax at
            # the prompt-end hidden state is deterministic.
            #
            # Then set up the vendored generator seeded with that same
            # token as the "prompt" — the generator's first backbone
            # step feeds it, advances the cache to +1, and samples the
            # SECOND generated token.
            first_tok_arr = gb._next_tokens
            first_lp_list = gb._next_logprobs
            if first_tok_arr is None or not first_lp_list:
                # Shouldn't happen — the fresh __init__ always calls _step.
                # But fall back defensively rather than crashing.
                _stats["fallthrough_steps"] += 1
                return _orig_step()
            first_tok = int(first_tok_arr[0].item())
            first_lp = first_lp_list[0]

            # Compute a generous max_tokens for the generator. Even
            # when the request's max_tokens is small (e.g. 80), the
            # generator uses this as an internal upper bound. Overshoot
            # is fine — mlx-lm's ``next()`` enforces the true max via
            # ``_num_tokens[i] >= self.max_tokens[i]``.
            gen_max = int(gb.max_tokens[0]) if gb.max_tokens else 4096

            # Codex round-A blocker #2: construct the generator BEFORE
            # mutating ``gb.tokens[0]``. Prior revision appended the
            # first token first, then constructed the generator; on
            # construction failure the fallthrough path called
            # ``_orig_step()`` which appended the SAME token again,
            # double-booking bookkeeping and emitting a duplicated
            # token to the stream.
            #
            # Codex round-D blocker #2: on failure here the invariant
            # is that we have NOT yet advanced any state — ``_next_
            # tokens`` still contains ``first_gen_tok`` (staged by
            # the ``GenerationBatch.__init__._step()`` priming call),
            # ``prompt_cache`` is still at position ``prompt_len``,
            # and ``gb.tokens[0]`` still ends at the last prompt
            # token. Delegating to ``_orig_step()`` is byte-equal to
            # plain decode because ``_orig_step`` will read
            # ``_next_tokens = first_gen_tok``, feed it through the
            # target (advancing cache to ``prompt_len+1``), sample
            # ``second_gen_tok``, stage it into ``_next_tokens``, and
            # append ``first_gen_tok`` to ``gb.tokens[0]``. The
            # request-visible first output is ``first_gen_tok``,
            # exactly as it would be under baseline. Mark the uid as
            # permanently disabled so we don't retry construction on
            # every subsequent step.
            try:
                gen = mtp_generate_step(
                    prompt=first_tok_arr.astype(mx.uint32),
                    model=model,
                    max_tokens=gen_max,
                    prompt_cache=gb.prompt_cache,
                    temp=0.0,
                    # 0.9.13 PR-B: EV depth controller.
                    model_id=controller_key or f"mtp-model-{id(model)}",
                    max_k=max_k,
                    disable_auto_k=disable_auto_k,
                    # 0.9.13 PR-C: EOS holdout — feed the
                    # BatchGenerator's assembled stop set to the
                    # controller so positions past EOS are not
                    # logged as (nonexistent) rejections. Emitted
                    # tokens are unchanged; only the acceptance
                    # model's training window shrinks.
                    stop_tokens=getattr(batch_gen, "stop_tokens", None),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[MTP-vendored] mtp_generate_step construction failed "
                    "(%s); disabling MTP for uid=%s and falling back to "
                    "plain decode for the rest of the request. "
                    "_next_tokens is untouched, so the baseline _step "
                    "will correctly emit the first generated token.",
                    e,
                    uid,
                )
                # Codex round-E blocker #1: record the request_id at
                # disable time so uid reuse across requests re-enables
                # MTP for the new request. Store ``None`` if the outer
                # bookkeeping map is None (bench-harness path) — the
                # gate above treats that as "keep disabled" which is
                # the safe default for callers without request IDs.
                _disabled_req_id = None
                if uid_to_request_id is not None:
                    _disabled_req_id = uid_to_request_id.get(uid)
                _disabled_uids[uid] = _disabled_req_id
                _stats["fallthrough_steps"] += 1
                return _orig_step()

            # Generator built successfully — safe to record the first
            # token now. Match the bookkeeping mlx-lm's original _step
            # performs on the ``self.tokens`` list per emitted token.
            gb.tokens[0].append(first_tok)

            # Codex round-K BLOCKING #1: capture the owning
            # request_id so the uid-reuse gate at wrapper entry can
            # detect when mlx-lm reassigns this uid to a different
            # request. ``None`` when ``uid_to_request_id`` isn't
            # plumbed (bench harness); the reuse gate treats
            # ``None`` as "cannot distinguish, keep existing
            # state" — safe because harnesses that don't plumb
            # uid_to_request_id also don't reuse uids.
            _first_call_req_id: str | None = None
            if uid_to_request_id is not None:
                _first_call_req_id = uid_to_request_id.get(uid)

            _state[uid] = {
                "gen": gen,
                "queue": [],
                "primed": True,
                "request_id": _first_call_req_id,
            }
            _stats["vendored_steps"] += 1
            # Codex round-I BLOCKING #2 / round-J BLOCKING #2+#3:
            # keep ``gb._next_tokens`` / ``gb._next_logprobs`` in a
            # coherent shape for downstream ``.filter`` /
            # ``.extend`` slicing. Uses the just-emitted token as
            # the placeholder (round-J review: a prior revision
            # prefetched the next generator token here, which
            # changed ``gb.prompt_cache`` state behind mlx-lm's
            # bookkeeping and swallowed generator exceptions —
            # both correctly flagged as unsafe). See
            # ``_sync_next_tokens_after_emit`` docstring for the
            # full "stale value is safe" argument (short version:
            # round-H terminal-raise fires before any
            # ``_orig_step`` can consume the stale value).
            _sync_next_tokens_after_emit(gb, first_tok, first_lp)
            return [first_tok], [first_lp]

        # --- SUBSEQUENT calls: drain queue, else pull from generator ---
        queue = state["queue"]
        if not queue:
            gen = state["gen"]
            try:
                tok_int, lp_arr, _from_draft = next(gen)
                queue.append((int(tok_int), lp_arr))
            except StopIteration:
                _stats["gen_exhausted"] += 1
                # Codex round-G BLOCKING #2: preserve the terminal
                # disabled marker for the current request BEFORE
                # dropping any state. If mlx-lm somehow re-enters
                # ``_mtp_step`` for this uid after the raise (e.g.,
                # the caller decides to retry on a scheduler tick
                # instead of failing the request), the disabled-uid
                # gate MUST fire; otherwise the wrapper would try
                # to construct a fresh generator and hit the same
                # bug again. Record the request_id so uid reuse
                # for a NEW request still re-enables MTP.
                _terminal_req_id = None
                if uid_to_request_id is not None:
                    _terminal_req_id = uid_to_request_id.get(uid)
                _disabled_uids[uid] = _terminal_req_id
                # Generator's own state can go — nothing to close
                # here (StopIteration means it already tore down).
                _state.pop(uid, None)
                # Codex round-D blocker #3: falling back to
                # ``_orig_step()`` mid-stream is UNSAFE — see the
                # comment on the ``Exception`` branch below.
                # ``StopIteration`` before mlx-lm hits max_tokens is
                # a plumbing bug; the generator's internal
                # ``max_tokens`` should always overshoot the
                # request's ``max_tokens``. Terminating the request
                # is safer than emitting a duplicate token.
                raise RuntimeError(
                    "[MTP-vendored] internal generator exhausted "
                    f"for uid={uid} before mlx-lm hit max_tokens. "
                    "This is a plumbing bug — the generator's "
                    "internal max_tokens should always overshoot "
                    "the request's max_tokens. Failing request to "
                    "avoid duplicate-token stream corruption on "
                    "fallback."
                )
            except Exception as e:  # noqa: BLE001
                # Codex round-D blocker #3: mid-stream generator
                # failure. Baseline ``_orig_step()`` here would
                # dutifully read ``gb._next_tokens`` (STALE — still
                # ``first_gen_tok`` from the priming ``_step``),
                # feed it through the model, and emit
                # ``first_gen_tok`` again — a duplicate. ``gb.tokens
                # [0]`` would also gain a duplicated ``first_gen_tok``
                # entry, corrupting the KV/token log invariant.
                #
                # The safe options are (a) terminate the request or
                # (b) rebuild the baseline state before delegating.
                # (b) is impossible without the next-token — we
                # never staged one — so (a) is the only clean path.
                _stats["gen_raised"] += 1
                logger.exception(
                    "[MTP-vendored] generator raised on uid=%s mid-stream: "
                    "%s. Terminating request — cannot fall back to plain "
                    "decode because gb._next_tokens is stale relative to "
                    "the tokens already emitted by the vendored path.",
                    uid,
                    e,
                )
                # Codex round-G BLOCKING #2: same terminal-marker
                # treatment as the StopIteration branch. Ensures a
                # retry on this uid+request_id hits the disable
                # gate and short-circuits to plain decode instead
                # of re-arming the vendored path.
                _terminal_req_id = None
                if uid_to_request_id is not None:
                    _terminal_req_id = uid_to_request_id.get(uid)
                _disabled_uids[uid] = _terminal_req_id
                # Close the (broken) generator; don't touch
                # _disabled_uids from inside _cleanup_uid.
                _state_entry = _state.pop(uid, None)
                if _state_entry is not None:
                    _gen = _state_entry.get("gen")
                    if _gen is not None:
                        try:
                            _gen.close()
                        except Exception:  # noqa: BLE001
                            pass
                raise RuntimeError(
                    f"[MTP-vendored] uid={uid} generator raised mid-"
                    f"stream ({type(e).__name__}: {e}); cannot fall "
                    "back to plain decode without corrupting the "
                    "output stream. Original exception logged above."
                ) from e

        tok_int, lp_arr = queue.pop(0)
        gb.tokens[0].append(tok_int)
        _stats["vendored_steps"] += 1
        # Codex round-I BLOCKING #2 / round-J BLOCKING #2+#3:
        # mirror the FIRST-call branch — sync ``gb._next_tokens`` /
        # ``gb._next_logprobs`` with the just-emitted token so
        # ``.filter`` / ``.extend`` see a coherent shape. No
        # generator prefetch here — that would advance
        # ``gb.prompt_cache`` behind mlx-lm's bookkeeping (round-J
        # BLOCKING #2) and could swallow generator exceptions
        # (round-J BLOCKING #3). See ``_sync_next_tokens_after_emit``
        # docstring for why the stale placeholder is safe under
        # round-H's terminal-raise regime.
        _sync_next_tokens_after_emit(gb, tok_int, lp_arr)
        return [tok_int], [lp_arr]

    # Patch onto the persistent _generation_batch. New GenerationBatch
    # instances created inside PromptProcessingBatch.generate() use the
    # CLASS _step for their priming call (which is exactly what we want:
    # the first sampled token comes from mlx-lm's plain argmax path so
    # it matches baseline byte-for-byte). The state transfer to the
    # persistent gb happens via .extend(), after which our patched
    # _step takes over.
    gb._step = _mtp_step
    batch_gen._mtp_vendored_stats = _stats

    logger.info(
        "[MTP-vendored] installed on GenerationBatch._step "
        "(single-request greedy K=1 chain-of-1; falls through on B>1 / "
        "non-greedy / logits-processors)."
    )
    return True
