# SPDX-License-Identifier: Apache-2.0
"""Speculative decode integration for the pure-decode fast path.

Uses DraftModelDecoder (small LM like Qwen3-0.6B-4bit) to draft K tokens,
then verifies them in a single forward pass of the target model.

Token flow:
  1. Regular _step() produces token T and samples T_next
  2. spec_decode_step checks if draft model has draft tokens for T
  3. If yes, feeds [D1, D2, ..., DK] to the target model (caches past T)
     - logits[i] = target prediction AFTER D_i (matches D_{i+1})
     - D1 is verified against T_next (from the regular step)
  4. Accepted drafts + resampled token are emitted as RequestOutputs
     - The resampled token goes into gen._next_tokens for the next step
"""

import copy
import logging
import time

import mlx.core as mx

from ..request import RequestOutput

logger = logging.getLogger(__name__)

SPEC_NUM_DRAFT_TOKENS = int(
    __import__("os").environ.get("FUSION_SPEC_DRAFT_TOKENS", "3")
)
SPEC_WARMUP_STEPS = int(__import__("os").environ.get("FUSION_SPEC_WARMUP_STEPS", "2"))
SPEC_DRAFT_MODEL_ENABLED = (
    __import__("os").environ.get("FUSION_DRAFT_MODEL_ENABLED", "1") == "1"
)
# Phase-2 item 3: lower threshold 0.10 -> 0.05 so a marginal draft model
# is not permanently disabled; the adaptive pause/resume below re-probes
# periodically instead of giving up after the first low window.
SPEC_MIN_ACCEPT_RATE = float(
    __import__("os").environ.get("FUSION_SPEC_MIN_ACCEPT_RATE", "0.05")
)
SPEC_ADAPTIVE_WINDOW = int(
    __import__("os").environ.get("FUSION_SPEC_ADAPTIVE_WINDOW", "20")
)
# Phase-2 item 3: while paused, re-probe acceptance every N steps so a
# transient low-acceptance window does not disable spec decode for the
# rest of the request. Without this the resume branch in record_accepted
# is dead code — once paused, should_speculate() is always False, so no
# drafts run, so record_accepted() never fires to un-pause.
SPEC_RESUME_CHECK_INTERVAL = int(
    __import__("os").environ.get("FUSION_SPEC_RESUME_CHECK_INTERVAL", "10")
)


class SpecDecodeState:
    """Per-scheduler speculative decode state."""

    def __init__(self, draft_model_decoder=None, hidden_capture=None):
        self.draft_model = draft_model_decoder
        self.hidden_capture = hidden_capture
        self.steps_since_start = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        self._recent_accepted = []
        self._spec_paused = False
        # Phase-2 item 3: steps elapsed while paused, for periodic re-probe.
        self._paused_steps = 0
        # Phase-2 item 3: acceptance records collected only during the
        # paused re-probe phase, so resume fires on probe recovery rather
        # than waiting for the full adaptive window to flush stale records.
        self._probe_records = []

    def reset(self):
        if self.draft_model:
            self.draft_model.reset()
        self.steps_since_start = 0
        self._last_request_id = None

    def on_new_request(self, request_id: str, prompt_tokens: list[int]):
        if self._last_request_id != request_id:
            if self.draft_model:
                self.draft_model.on_new_request(request_id, prompt_tokens)
            self._last_request_id = request_id
            self.steps_since_start = 0
            self._spec_paused = False
            self._recent_accepted.clear()
            self._paused_steps = 0
            self._probe_records.clear()

    def add_token(self, token: int):
        self.steps_since_start += 1

    def should_speculate(self) -> bool:
        if self.steps_since_start < SPEC_WARMUP_STEPS:
            return False
        if self._spec_paused:
            # Phase-2 item 3: periodic re-probe. Let one spec step through
            # every SPEC_RESUME_CHECK_INTERVAL steps while paused so
            # record_accepted() can collect a fresh acceptance sample and
            # un-pause if the draft model recovered. Without this the
            # resume branch below is dead code (paused -> never speculates
            # -> never records -> never resumes).
            self._paused_steps += 1
            if self._paused_steps % SPEC_RESUME_CHECK_INTERVAL == 0:
                logger.debug(
                    "spec_decode: paused re-probe at step=%d (every %d)",
                    self._paused_steps,
                    SPEC_RESUME_CHECK_INTERVAL,
                )
                return True
            return False
        return True

    def get_drafts(self, current_token: int | None = None) -> list[int]:
        if self.draft_model and current_token is not None:
            return self.draft_model.generate_draft_tokens(current_token)
        return []

    def record_accepted(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted
        if self.draft_model:
            self.draft_model.record_accepted(n_accepted)

        # Phase-2 item 3: while paused, this record comes from a re-probe
        # step (should_speculate let one through). Accumulate probe samples
        # and decide resume on a small burst rather than the full adaptive
        # window — the window still holds the stale low-acceptance records
        # that triggered the pause, so it cannot drive resume.
        if self._spec_paused:
            self._probe_records.append((n_accepted, n_total))
            probe_a = sum(a for a, t in self._probe_records)
            probe_t = sum(t for a, t in self._probe_records)
            probe_rate = probe_a / probe_t if probe_t > 0 else 0
            # Evaluate after a few probe samples to avoid resuming on a
            # single lucky draft.
            if len(self._probe_records) >= 3:
                probe_count = len(self._probe_records)
                if probe_rate >= SPEC_MIN_ACCEPT_RATE:
                    self._spec_paused = False
                    self._paused_steps = 0
                    self._probe_records.clear()
                    self._recent_accepted.clear()
                    logger.info(
                        "spec_decode: resuming — probe acceptance %.1f%% "
                        "recovered over %d probes",
                        probe_rate * 100,
                        probe_count,
                    )
                else:
                    logger.info(
                        "spec_decode: staying paused — probe acceptance "
                        "%.1f%% still < %.1f%% over %d probes",
                        probe_rate * 100,
                        SPEC_MIN_ACCEPT_RATE * 100,
                        len(self._probe_records),
                    )
                    self._probe_records.clear()
            return

        self._recent_accepted.append((n_accepted, n_total))
        if len(self._recent_accepted) > SPEC_ADAPTIVE_WINDOW:
            self._recent_accepted.pop(0)

        if len(self._recent_accepted) >= SPEC_ADAPTIVE_WINDOW:
            total_a = sum(a for a, t in self._recent_accepted)
            total_t = sum(t for a, t in self._recent_accepted)
            recent_rate = total_a / total_t if total_t > 0 else 0
            if recent_rate < SPEC_MIN_ACCEPT_RATE and not self._spec_paused:
                self._spec_paused = True
                self._paused_steps = 0
                self._probe_records.clear()
                logger.info(
                    "spec_decode: pausing — acceptance %.1f%% < %.1f%%",
                    recent_rate * 100,
                    SPEC_MIN_ACCEPT_RATE * 100,
                )

    def get_stats(self) -> dict:
        rate = (
            self.total_draft_accepted / self.total_draft_proposed
            if self.total_draft_proposed > 0
            else 0.0
        )
        stats = {
            "spec_steps": self.total_spec_steps,
            "draft_proposed": self.total_draft_proposed,
            "draft_accepted": self.total_draft_accepted,
            "acceptance_rate": rate,
            "paused": self._spec_paused,
        }
        if self.draft_model:
            stats["draft_model"] = self.draft_model.get_stats()
        return stats


def _snapshot_non_trimmable_caches(prompt_cache: list) -> list | None:
    """Deep-copy non-trimmable cache entries before verify."""
    snapshots = []
    for i, c in enumerate(prompt_cache):
        if hasattr(c, "is_trimmable") and not c.is_trimmable():
            snapshots.append((i, copy.deepcopy(c)))
    return snapshots if snapshots else None


def _restore_non_trimmable_caches(prompt_cache: list, snapshots: list):
    for i, snapshot in snapshots:
        prompt_cache[i] = snapshot


def _trim_trimmable(prompt_cache: list, num_tokens: int):
    # Trim each trimmable cache (KVCache) by num_tokens in place.
    # mlx_cache.trim_prompt_cache is a no-op when any layer is
    # non-trimmable (hybrid GDN models carry ArraysCache layers whose
    # is_trimmable() is False), which would leave the rejected drafts in
    # KVCache; this trims the trimmable layers and skips the rest.
    for c in prompt_cache:
        if getattr(c, "is_trimmable", None) and c.is_trimmable():
            c.trim(num_tokens)


def _run_spec_verify(
    model,
    current_token: int,
    draft_tokens: list[int],
    prompt_cache: list,
    sampled_from_regular: int | None = None,
) -> tuple[list[int], int, int]:
    """Verify draft tokens against the target model.

    Caches are positioned after current_token (from the regular step).
    We feed [D1, D2, ..., DK] to the model:
      - logits[0] = prediction after D1 → should match D2
      - logits[1] = prediction after D2 → should match D3
      - logits[K-1] = prediction after DK → resampled token

    D1 is verified against sampled_from_regular (the regular step's
    sampled token), since we can't get logits at D1's position without
    re-processing current_token.

    Returns (verified_tokens, n_accepted, cache_tokens_processed).
    """
    K = len(draft_tokens)

    # Verify D1 against the regular step's prediction
    if sampled_from_regular is not None and draft_tokens[0] != sampled_from_regular:
        logger.debug(
            "spec_verify: D1 rejected — draft=%d sampled=%d",
            draft_tokens[0],
            sampled_from_regular,
        )
        return [sampled_from_regular], 0, 0

    # D1 accepted
    n_accepted = 1 if sampled_from_regular is not None else 0

    if K == 1:
        return draft_tokens[:1], n_accepted, 0

    # Feed [D1, ..., DK] — caches are past current_token
    verify_input = mx.array(draft_tokens, mx.uint32)
    logits = model(verify_input[None], cache=prompt_cache)
    logits = logits.squeeze(0)  # [K, vocab]

    sampled = mx.argmax(logits, axis=-1)
    mx.eval(sampled)
    sampled_list = sampled.tolist()

    # Compare logits[i] (after D_i) with D_{i+1}
    for i in range(K - 1):
        if sampled_list[i] == draft_tokens[i + 1]:
            n_accepted += 1
        else:
            break
    else:
        n_accepted = K

    # Build verified list: accepted drafts + resampled token. The bonus is
    # the model's prediction AFTER the last ACCEPTED draft (sampled[n_accepted-1]),
    # NOT after the first rejected draft (sampled[n_accepted]). Using the
    # latter emits a token conditioned on a rejected draft — a correctness
    # bug on any rejection, pure-attention or hybrid. Mirrors ngram_spec.
    resample_idx = max(0, n_accepted - 1)
    verified = draft_tokens[:n_accepted]
    verified.append(sampled_list[resample_idx])

    return verified, n_accepted, K


def spec_decode_step(
    scheduler,
    output,
    current_token: int,
    request_id: str,
) -> list[RequestOutput]:
    """Run speculative decode after a regular decode step."""
    spec_state = scheduler._spec_decode_state
    if spec_state is None:
        return []

    if not spec_state.draft_model:
        return []

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    if spec_state._last_request_id != request_id:
        spec_state.on_new_request(request_id, request.prompt_token_ids or [])

    spec_state.add_token(current_token)

    if not spec_state.should_speculate():
        if spec_state.steps_since_start <= 3:
            logger.info(
                "spec_decode: warming up step=%d/%d",
                spec_state.steps_since_start,
                SPEC_WARMUP_STEPS,
            )
        return []

    draft_tokens = spec_state.get_drafts(current_token)
    if not draft_tokens:
        if spec_state.total_spec_steps == 0:
            logger.info(
                "spec_decode: no draft tokens generated for token=%d", current_token
            )
        return []

    K = len(draft_tokens)

    bg = scheduler.batch_generator
    if bg is None:
        return []

    gen = bg._generation_batch
    if gen is None:
        return []

    prompt_cache = gen.prompt_cache
    model = gen.model

    # Capture regular step's sampled token for D1 verification.
    # mx arrays expose ``.item()`` (not ``.flat``).
    sampled_from_regular = None
    if gen._next_tokens is not None:
        try:
            sampled_from_regular = int(gen._next_tokens.item())
        except Exception:
            pass

    non_trimmable_snapshots = _snapshot_non_trimmable_caches(prompt_cache)

    with mx.stream(scheduler._stream):
        t0 = time.perf_counter()
        verified, n_accepted, cache_tokens_processed = _run_spec_verify(
            model,
            current_token,
            draft_tokens,
            prompt_cache,
            sampled_from_regular=sampled_from_regular,
        )
    dt = time.perf_counter() - t0

    # Cache rollback for rejected tokens
    if cache_tokens_processed > 0 and n_accepted < K:
        n_rejected = cache_tokens_processed - n_accepted
        if non_trimmable_snapshots is not None:
            _restore_non_trimmable_caches(prompt_cache, non_trimmable_snapshots)
            if n_accepted > 0:
                accepted_prefix = draft_tokens[:n_accepted]
                replay_input = mx.array(accepted_prefix, mx.uint32)
                with mx.stream(scheduler._stream):
                    replay_logits = model(replay_input[None], cache=prompt_cache)
                    mx.eval(replay_logits)
            logger.debug(
                "spec_decode: restored %d non-trimmable caches, %d/%d rejected",
                len(non_trimmable_snapshots),
                n_rejected,
                K,
            )
        # Trim trimmable KVCache layers. trim_prompt_cache is a no-op on
        # hybrid caches (ArraysCache is non-trimmable); _trim_trimmable
        # skips non-trimmable layers instead. The replay above appended
        # n_accepted duplicates to KVCache, so a hybrid cache holds
        # K+n_accepted -> trim K (==cache_tokens_processed) to leave
        # n_accepted. A pure KVCache (no snapshot/replay) still holds K
        # -> trim only the n_rejected rejected drafts.
        trim_count = (
            cache_tokens_processed
            if non_trimmable_snapshots is not None
            else n_rejected
        )
        if trim_count > 0:
            _trim_trimmable(prompt_cache, trim_count)

    spec_state.record_accepted(n_accepted, K)

    if spec_state.total_spec_steps % 50 == 1:
        stats = spec_state.get_stats()
        logger.info(
            "spec_decode: step=%d, K=%d, accepted=%d/%d (%.1f%%), "
            "verify=%.1fms, rate=%.1f%%",
            spec_state.total_spec_steps,
            K,
            n_accepted,
            K,
            100.0 * n_accepted / K if K else 0,
            dt * 1000,
            stats["acceptance_rate"] * 100,
        )

    if not verified:
        return []

    # The resampled token (verified[-1]) will be returned by the next
    # regular _step() via gen._next_tokens. _step() returns the INPUT
    # token as the Response, so emitting it here would double-count.
    accepted_only = verified[:-1]

    last_token = verified[-1]
    gen._next_tokens = mx.array([last_token], mx.uint32)

    if gen.tokens and len(gen.tokens) > 0:
        for t in accepted_only:
            gen.tokens[0].append(t)

    outputs = []
    step_now = time.monotonic()

    for i, token in enumerate(accepted_only):
        request.append_output_token(token)
        request.last_activity_at = step_now

        detokenizer = scheduler._get_detokenizer(request_id)
        if detokenizer is not None:
            detokenizer.add_token(token)
            new_text = detokenizer.last_segment
        else:
            new_text = scheduler.tokenizer.decode([token])

        eos_ids = (
            scheduler.tokenizer.eos_token_id
            if hasattr(scheduler.tokenizer, "eos_token_id")
            else []
        )
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        is_eos = token in eos_ids
        is_length = request.num_output_tokens >= request.max_tokens
        is_finished = is_eos or is_length

        out = RequestOutput(
            request_id=request_id,
            new_token_ids=[token],
            new_text="" if is_eos else new_text,
            completion_tokens=request.num_output_tokens,
            prompt_tokens=request.num_prompt_tokens,
            cached_tokens=request.cached_tokens,
            finished=is_finished,
            finish_reason="stop" if is_eos else ("length" if is_length else None),
        )

        if is_finished:
            from ..request import RequestStatus

            request.set_finished(
                RequestStatus.FINISHED_STOPPED
                if is_eos
                else RequestStatus.FINISHED_LENGTH_CAPPED
            )
            out.output_token_ids = list(request.output_token_ids)
            # Non-streaming responses read ``output_text`` off the final
            # RequestOutput (``engine_core.generate`` returns the last
            # output). The regular finalize path in ``sched_response``
            # that normally sets ``output_text`` is bypassed when the
            # spec path finishes the request, so mirror it here — decode
            # the full token list, leaving special-token scrubbing to
            # the engine layer (``clean_special_tokens``). Without this
            # the non-streaming ``content`` comes back empty while
            # streaming (which uses ``new_text``) stays correct (#364).
            out.output_text = scheduler.tokenizer.decode(request.output_token_ids)
            request.output_text = out.output_text

        outputs.append(out)
        if is_finished:
            break

    return outputs


DFLASH_SPEC_WARMUP_STEPS = 2
DFLASH_SPEC_LOG_INTERVAL = 50


class DFlashSpecState:
    """Per-scheduler DFlash speculative decode state."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.steps_since_start = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None

    def on_new_request(self, request_id: str):
        if self._last_request_id != request_id:
            self._last_request_id = request_id
            self.steps_since_start = 0
            drafter = getattr(self.runtime, "drafter", None)
            if drafter is not None:
                drafter.reset()

    def add_token(self, token: int):
        self.steps_since_start += 1

    def should_speculate(self) -> bool:
        return self.steps_since_start >= DFLASH_SPEC_WARMUP_STEPS

    def record_result(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted

    def get_stats(self) -> dict:
        rate = (
            self.total_draft_accepted / self.total_draft_proposed
            if self.total_draft_proposed > 0
            else 0.0
        )
        return {
            "spec_steps": self.total_spec_steps,
            "draft_proposed": self.total_draft_proposed,
            "draft_accepted": self.total_draft_accepted,
            "acceptance_rate": rate,
            "block_size": self.runtime.drafter.block_size,
        }


def dflash_spec_step(
    scheduler,
    output,
    current_token: int,
    request_id: str,
) -> list[RequestOutput]:
    """DFlash block-diffusion speculative decode after a regular decode step.

    Uses the DFlash drafter to produce a draft block (block_size tokens),
    then verifies against the target model using _run_spec_verify.
    The cache is already past current_token from the regular step, so
    we feed [D1, D2, ..., DK] (not [current_token, D1, ...]).
    """
    dflash_state = getattr(scheduler, "_dflash_spec_state", None)
    if dflash_state is None:
        dflash_runtime = scheduler._dflash_runtime
        if dflash_runtime is None:
            return []
        dflash_state = DFlashSpecState(dflash_runtime)
        scheduler._dflash_spec_state = dflash_state

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    if dflash_state._last_request_id != request_id:
        dflash_state.on_new_request(request_id)

    dflash_state.add_token(current_token)

    if not dflash_state.should_speculate():
        if dflash_state.steps_since_start <= 3:
            logger.info(
                "dflash_spec: warming up step=%d/%d",
                dflash_state.steps_since_start,
                DFLASH_SPEC_WARMUP_STEPS,
            )
        return []

    bg = scheduler.batch_generator
    if bg is None:
        return []

    gen = bg._generation_batch
    if gen is None:
        return []

    prompt_cache = gen.prompt_cache
    model = gen.model
    drafter = dflash_state.runtime.drafter

    # Get current offset from cache
    current_offset = 0
    for c in prompt_cache:
        if hasattr(c, "offset"):
            current_offset = max(current_offset, c.offset)

    # Draft a block using the DFlash drafter
    prefix_tokens = mx.array([current_token], dtype=mx.uint32)
    try:
        draft_block = drafter.draft_block(prefix_tokens, current_offset)
    except (IndexError, RuntimeError) as e:
        logger.debug("dflash_spec: draft_block failed: %s", e)
        return []

    block_size = drafter.block_size
    draft_tokens = [int(x) for x in draft_block.tolist()]
    K = len(draft_tokens)
    if K == 0:
        return []

    # Capture regular step's sampled token for D1 verification
    sampled_from_regular = None
    if gen._next_tokens is not None:
        try:
            sampled_from_regular = int(gen._next_tokens.item())
        except Exception:
            pass

    # D1 gate (CPU-only, no GPU sync needed for the gate itself)
    if sampled_from_regular is not None and draft_tokens[0] != sampled_from_regular:
        dflash_state.record_result(0, K)
        return []

    non_trimmable_snapshots = _snapshot_non_trimmable_caches(prompt_cache)

    with mx.stream(scheduler._stream):
        t0 = time.perf_counter()
        verified, n_accepted, cache_tokens_processed = _run_spec_verify(
            model,
            current_token,
            draft_tokens,
            prompt_cache,
            sampled_from_regular=sampled_from_regular,
        )
    dt = time.perf_counter() - t0

    # Cache rollback for rejected tokens
    if cache_tokens_processed > 0 and n_accepted < K:
        n_rejected = cache_tokens_processed - n_accepted
        if non_trimmable_snapshots is not None:
            _restore_non_trimmable_caches(prompt_cache, non_trimmable_snapshots)
            if n_accepted > 0:
                accepted_prefix = draft_tokens[:n_accepted]
                replay_input = mx.array(accepted_prefix, mx.uint32)
                with mx.stream(scheduler._stream):
                    replay_logits = model(replay_input[None], cache=prompt_cache)
                    mx.eval(replay_logits)
            logger.debug(
                "dflash_spec: restored %d non-trimmable caches, %d/%d rejected",
                len(non_trimmable_snapshots),
                n_rejected,
                K,
            )
        # Same trim logic as the spec path: _trim_trimmable (not the
        # no-op trim_prompt_cache), and trim K when a replay ran (hybrid)
        # vs only n_rejected for a pure KVCache. See spec path above.
        trim_count = (
            cache_tokens_processed
            if non_trimmable_snapshots is not None
            else n_rejected
        )
        if trim_count > 0:
            _trim_trimmable(prompt_cache, trim_count)

    dflash_state.record_result(n_accepted, K)

    if dflash_state.total_spec_steps % DFLASH_SPEC_LOG_INTERVAL == 1:
        stats = dflash_state.get_stats()
        logger.info(
            "dflash_spec: step=%d, block=%d, accepted=%d/%d (%.1f%%), "
            "verify=%.1fms, rate=%.1f%%",
            dflash_state.total_spec_steps,
            block_size,
            n_accepted,
            K,
            100.0 * n_accepted / K if K else 0,
            dt * 1000,
            stats["acceptance_rate"] * 100,
        )

    if not verified:
        return []

    accepted_only = verified[:-1]
    last_token = verified[-1]
    gen._next_tokens = mx.array([last_token], mx.uint32)

    if gen.tokens and len(gen.tokens) > 0:
        for t in accepted_only:
            gen.tokens[0].append(t)

    outputs = []
    step_now = time.monotonic()

    for i, token in enumerate(accepted_only):
        request.append_output_token(token)
        request.last_activity_at = step_now

        detokenizer = scheduler._get_detokenizer(request_id)
        if detokenizer is not None:
            detokenizer.add_token(token)
            new_text = detokenizer.last_segment
        else:
            new_text = scheduler.tokenizer.decode([token])

        eos_ids = (
            scheduler.tokenizer.eos_token_id
            if hasattr(scheduler.tokenizer, "eos_token_id")
            else []
        )
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        is_eos = token in eos_ids
        is_length = request.num_output_tokens >= request.max_tokens
        is_finished = is_eos or is_length

        out = RequestOutput(
            request_id=request_id,
            new_token_ids=[token],
            new_text="" if is_eos else new_text,
            completion_tokens=request.num_output_tokens,
            prompt_tokens=request.num_prompt_tokens,
            cached_tokens=request.cached_tokens,
            finished=is_finished,
            finish_reason="stop" if is_eos else ("length" if is_length else None),
        )

        if is_finished:
            from ..request import RequestStatus

            request.set_finished(
                RequestStatus.FINISHED_STOPPED
                if is_eos
                else RequestStatus.FINISHED_LENGTH_CAPPED
            )
            out.output_token_ids = list(request.output_token_ids)
            out.output_text = scheduler.tokenizer.decode(request.output_token_ids)
            request.output_text = out.output_text

        outputs.append(out)
        if is_finished:
            break

    return outputs


DSPARK_SPEC_LOG_INTERVAL = 50


class DSparkSpecState:
    """Per-scheduler DSpark speculative decode state.

    DSparkGenerator is self-contained (loads its own target + draft),
    so the per-step integration pulls tokens from its internal
    propose-verify loop rather than running our own verify.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        # Active generator sessions: request_id -> token iterator
        self._sessions: dict = {}

    def on_new_request(self, request_id: str):
        if self._last_request_id != request_id:
            self._last_request_id = request_id
            self.total_spec_steps = 0

    def get_session(self, request_id: str):
        return self._sessions.get(request_id)

    def set_session(self, request_id: str, session):
        self._sessions[request_id] = session

    def remove_session(self, request_id: str):
        self._sessions.pop(request_id, None)

    def record_result(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted

    def get_stats(self) -> dict:
        rate = (
            self.total_draft_accepted / self.total_draft_proposed
            if self.total_draft_proposed > 0
            else 0.0
        )
        return {
            "spec_steps": self.total_spec_steps,
            "draft_proposed": self.total_draft_proposed,
            "draft_accepted": self.total_draft_accepted,
            "acceptance_rate": rate,
            "active_sessions": len(self._sessions),
        }


def _emit_spec_tokens(
    scheduler,
    request_id: str,
    tokens: list[int],
) -> list[RequestOutput]:
    """Build RequestOutputs for spec-decode accepted tokens.

    Batch-optimized: appends all tokens to the request, detokenizes in one
    pass, and emits a single RequestOutput for the whole batch when no
    finishing token is present. Falls back to per-token split only when
    EOS or length cap falls mid-batch.
    """
    if not tokens:
        return []

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    bg = scheduler.batch_generator
    gen = bg._generation_batch if bg else None

    eos_ids = (
        scheduler.tokenizer.eos_token_id
        if hasattr(scheduler.tokenizer, "eos_token_id")
        else []
    )
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    step_now = time.monotonic()
    detokenizer = scheduler._get_detokenizer(request_id)

    finish_idx = -1
    finish_reason = None
    for i, token in enumerate(tokens):
        request.append_output_token(token)
        if token in eos_ids:
            finish_idx = i
            finish_reason = "stop"
            break
        if request.num_output_tokens >= request.max_tokens:
            finish_idx = i
            finish_reason = "length"
            break

    request.last_activity_at = step_now

    if detokenizer is not None:
        for token in tokens[: finish_idx + 1 if finish_idx >= 0 else len(tokens)]:
            detokenizer.add_token(token)
        if finish_idx >= 0 and finish_reason == "stop":
            detokenizer.finalize()
        batch_text = detokenizer.last_segment if detokenizer else ""
    else:
        batch_text = scheduler.tokenizer.decode(
            tokens[: finish_idx + 1 if finish_idx >= 0 else len(tokens)]
        )

    if finish_idx >= 0:
        pre_tokens = tokens[:finish_idx]
        fin_token = tokens[finish_idx]
        outputs = []

        if pre_tokens:
            out = RequestOutput(
                request_id=request_id,
                new_token_ids=pre_tokens,
                new_text=batch_text,
                completion_tokens=request.num_output_tokens,
                prompt_tokens=request.num_prompt_tokens,
                cached_tokens=request.cached_tokens,
                finished=False,
                finish_reason=None,
            )
            outputs.append(out)

        from ..request import RequestStatus

        request.set_finished(
            RequestStatus.FINISHED_STOPPED
            if finish_reason == "stop"
            else RequestStatus.FINISHED_LENGTH_CAPPED
        )
        fin_out = RequestOutput(
            request_id=request_id,
            new_token_ids=[fin_token],
            new_text="",
            completion_tokens=request.num_output_tokens,
            prompt_tokens=request.num_prompt_tokens,
            cached_tokens=request.cached_tokens,
            finished=True,
            finish_reason=finish_reason,
        )
        fin_out.output_token_ids = list(request.output_token_ids)
        fin_out.output_text = scheduler.tokenizer.decode(request.output_token_ids)
        request.output_text = fin_out.output_text
        outputs.append(fin_out)
    else:
        outputs = [
            RequestOutput(
                request_id=request_id,
                new_token_ids=list(tokens),
                new_text=batch_text,
                completion_tokens=request.num_output_tokens,
                prompt_tokens=request.num_prompt_tokens,
                cached_tokens=request.cached_tokens,
                finished=False,
                finish_reason=None,
            )
        ]

    if gen is not None and tokens:
        gen._next_tokens = mx.array([tokens[-1]], mx.uint32)
        if gen.tokens and len(gen.tokens) > 0:
            for t in tokens[:-1]:
                gen.tokens[0].append(t)

    return outputs


def dspark_spec_step(
    scheduler,
    output,
    current_token: int,
    request_id: str,
) -> list[RequestOutput]:
    """DSpark speculative decode step.

    DSparkGenerator runs its own propose-verify loop internally.
    This step pulls accepted tokens from the DSpark session and
    emits them as RequestOutputs. DSpark is self-contained (loads
    its own target + draft model), so this does NOT use the
    scheduler's model or cache for the propose-verify loop.
    """
    dspark_state = getattr(scheduler, "_dspark_spec_state", None)
    if dspark_state is None:
        dspark_runtime = scheduler._dspark_runtime
        if dspark_runtime is None:
            return []
        dspark_state = DSparkSpecState(dspark_runtime)
        scheduler._dspark_spec_state = dspark_state

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    if dspark_state._last_request_id != request_id:
        dspark_state.on_new_request(request_id)

    # Check for an active DSpark generation session
    session = dspark_state.get_session(request_id)
    if session is None:
        prompt_tokens = getattr(request, "prompt_token_ids", None)
        if not prompt_tokens:
            return []

        generator = dspark_state.runtime.generator
        if generator is None:
            return []

        try:
            max_tokens = request.max_tokens or 4096
            token_iter = generator.stream_from_tokens(
                prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            dspark_state.set_session(request_id, token_iter)
            session = token_iter
            logger.info(
                "dspark_spec: started session for request=%s, "
                "prompt_len=%d, max_tokens=%d",
                request_id[:8],
                len(prompt_tokens),
                max_tokens,
            )
        except Exception as e:
            logger.warning("dspark_spec: failed to start session: %s", e)
            return []

    # Pull the next batch of tokens from the DSpark generator
    try:
        accepted_tokens = []
        block_size = getattr(dspark_state.runtime.generator, "block_size", 7)
        for _ in range(block_size):
            try:
                tok = next(session)
                accepted_tokens.append(int(tok))
            except StopIteration:
                break

        if not accepted_tokens:
            dspark_state.remove_session(request_id)
            return []

        n_accepted = len(accepted_tokens)
        dspark_state.record_result(n_accepted, n_accepted)

        if dspark_state.total_spec_steps % DSPARK_SPEC_LOG_INTERVAL == 1:
            stats = dspark_state.get_stats()
            logger.info(
                "dspark_spec: step=%d, accepted=%d, rate=%.1f%%, sessions=%d",
                dspark_state.total_spec_steps,
                n_accepted,
                stats["acceptance_rate"] * 100,
                stats["active_sessions"],
            )

        return _emit_spec_tokens(scheduler, request_id, accepted_tokens)

    except Exception as e:
        logger.warning("dspark_spec: session error: %s", e)
        dspark_state.remove_session(request_id)
        return []


DFLASH2_SPEC_LOG_INTERVAL = 50
DFLASH2_SPEC_WARMUP_STEPS = 3
DFLASH2_CIRCUIT_BREAKER_WINDOW = 10
DFLASH2_CIRCUIT_BREAKER_THRESHOLD = 0.20


class DFlash2SpecState:
    """Per-scheduler DFlash2 in-target speculative decode state.

    In-target pattern: the drafter binds to the scheduler's already-loaded
    target model. The per-step integration runs propose->verify->rollback
    using gen.model + gen.prompt_cache — no duplicate target load, no
    prefill replay. Mirrors DFlashSpecState (v1).
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.steps_since_start = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        self._recent_rates: list[float] = []
        self._circuit_tripped = False

    def on_new_request(self, request_id: str):
        if self._last_request_id != request_id:
            self._last_request_id = request_id
            self.steps_since_start = 0
            self.total_spec_steps = 0
            self._recent_rates = []
            self._circuit_tripped = False
            drafter = getattr(self.runtime, "drafter", None)
            if drafter is not None:
                drafter.reset()

    def add_token(self, token: int):
        self.steps_since_start += 1

    def should_speculate(self) -> bool:
        if self._circuit_tripped:
            return False
        return self.steps_since_start >= DFLASH2_SPEC_WARMUP_STEPS

    def record_result(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted
        self.runtime.record_accept(n_accepted)
        if n_total > 0:
            rate = n_accepted / n_total
            self._recent_rates.append(rate)
            if len(self._recent_rates) > DFLASH2_CIRCUIT_BREAKER_WINDOW:
                self._recent_rates.pop(0)
            if len(self._recent_rates) >= DFLASH2_CIRCUIT_BREAKER_WINDOW:
                avg = sum(self._recent_rates) / len(self._recent_rates)
                if avg < DFLASH2_CIRCUIT_BREAKER_THRESHOLD:
                    self._circuit_tripped = True
                    logger.warning(
                        "dflash2_spec: circuit breaker tripped "
                        "(avg accept %.1f%% < %.1f%% over %d steps) — "
                        "disabling spec for rest of request",
                        avg * 100,
                        DFLASH2_CIRCUIT_BREAKER_THRESHOLD * 100,
                        len(self._recent_rates),
                    )

    def get_stats(self) -> dict:
        rate = (
            self.total_draft_accepted / self.total_draft_proposed
            if self.total_draft_proposed > 0
            else 0.0
        )
        return {
            "spec_steps": self.total_spec_steps,
            "draft_proposed": self.total_draft_proposed,
            "draft_accepted": self.total_draft_accepted,
            "acceptance_rate": rate,
            "circuit_tripped": self._circuit_tripped,
        }


def dflash2_spec_step(
    scheduler,
    output,
    current_token: int,
    request_id: str,
) -> list[RequestOutput]:
    """DFlash2 in-target speculative decode step.

    Uses the DFlash2 drafter to propose a draft block, then verifies
    against the scheduler's already-loaded target model using the
    scheduler's prompt_cache. Mirrors dflash_spec_step (v1) pattern:
    drafter.propose_block -> model verify -> cache rollback -> emit.
    No duplicate target load, no prefill replay.
    """
    dflash2_state = getattr(scheduler, "_dflash2_spec_state", None)
    if dflash2_state is None:
        dflash2_runtime = scheduler._dflash2_runtime
        if dflash2_runtime is None:
            return []
        dflash2_state = DFlash2SpecState(dflash2_runtime)
        scheduler._dflash2_spec_state = dflash2_state

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    if dflash2_state._last_request_id != request_id:
        dflash2_state.on_new_request(request_id)

    dflash2_state.add_token(current_token)

    if not dflash2_state.should_speculate():
        if dflash2_state.steps_since_start <= DFLASH2_SPEC_WARMUP_STEPS:
            logger.debug(
                "dflash2_spec: warming up step=%d/%d",
                dflash2_state.steps_since_start,
                DFLASH2_SPEC_WARMUP_STEPS,
            )
        return []

    bg = scheduler.batch_generator
    if bg is None:
        return []
    gen = bg._generation_batch
    if gen is None:
        return []

    prompt_cache = gen.prompt_cache
    model = gen.model
    # VLM: gen.model is VLMModelAdapter — _hidden_states installed on inner
    # _language_model. Unwrap for get_hidden / store_verify_hidden.
    inner_model = getattr(model, "_language_model", None) or model
    drafter = dflash2_state.runtime.drafter
    if drafter is None or not drafter._bound:
        return []

    temperature = getattr(request, "temperature", 0.0) or 0.0

    current_offset = 0
    for c in prompt_cache:
        if hasattr(c, "offset"):
            current_offset = max(current_offset, c.offset)

    current_offset = int(current_offset)

    first_step = drafter._last_hidden is None
    # Clear stale verify hidden — the regular step's forward refreshed
    # model._hidden_states to the current 1-token position. Without this,
    # get_hidden returns the 5-token hidden from the previous verify
    # forward, causing the draft to propose with wrong context.
    drafter._last_hidden = None
    if first_step:
        drafter.align_draft_cache(current_offset)

    hidden = drafter.get_hidden(inner_model)

    try:
        with mx.stream(scheduler._stream):
            draft_tokens = drafter.propose_block(current_token, hidden, temperature)
        mx.async_eval(draft_tokens)
    except (IndexError, RuntimeError, ValueError, TypeError) as e:
        logger.warning(
            "dflash2_spec: propose failed: %s (type=%s)", e, type(e).__name__
        )
        return []

    draft_list = draft_tokens[0].tolist()
    K = len(draft_list)
    if K == 0:
        return []

    sampled_from_regular = None
    if gen._next_tokens is not None:
        try:
            sampled_from_regular = int(gen._next_tokens.item())
        except Exception:
            pass

    # D1 gate: draft[0] must match the regular step's sampled prediction.
    # No GPU sync needed — sampled_from_regular is already on device.
    if sampled_from_regular is not None and draft_list[0] != sampled_from_regular:
        dflash2_state.record_result(0, K)
        return []

    non_trimmable_snapshots = _snapshot_non_trimmable_caches(prompt_cache)

    # Feed [D1, ..., DK] — NOT [current_token, D1, ...]. The cache is
    # already past current_token; re-feeding it would double-count and
    # corrupt the KV state. logits[0]=pred after D1 → verify D2, etc.
    # D1 is verified above via sampled_from_regular (no GPU sync needed).
    with mx.stream(scheduler._stream):
        t0 = time.perf_counter()
        verified, n_accepted, cache_tokens_processed = _run_spec_verify(
            model,
            current_token,
            draft_list,
            prompt_cache,
            sampled_from_regular=sampled_from_regular,
        )
    dt = time.perf_counter() - t0

    # Cache rollback for rejected tokens. Mirrors dflash_spec_step line 639+.
    if cache_tokens_processed > 0 and n_accepted < K:
        n_rejected = cache_tokens_processed - n_accepted
        if non_trimmable_snapshots is not None:
            _restore_non_trimmable_caches(prompt_cache, non_trimmable_snapshots)
            if n_accepted > 0:
                replay = mx.array(draft_list[:n_accepted], dtype=mx.uint32)
                with mx.stream(scheduler._stream):
                    replay_logits = model(replay[None], cache=prompt_cache)
                    mx.eval(replay_logits)
            trim_count = cache_tokens_processed
        else:
            trim_count = n_rejected
        if trim_count > 0:
            _trim_trimmable(prompt_cache, trim_count)

    # Draft cache: after propose the draft advanced by block_size. Trim
    # to match the target's post-step offset (current_offset + n_accepted).
    expected_draft_offset = current_offset + n_accepted
    drafter.trim_draft_cache(expected_draft_offset)

    dflash2_state.record_result(n_accepted, K)

    # verified = [D1..D_n_accepted, bonus]. The bonus (verified[-1]) will
    # be returned by the next regular _step() via gen._next_tokens —
    # _step() returns the INPUT token as the Response, so emitting it here
    # would double-count. Emit only accepted drafts. Mirrors eagle3 line 417.
    bonus = verified[-1]
    accepted_drafts = verified[:-1]

    if dflash2_state.total_spec_steps % DFLASH2_SPEC_LOG_INTERVAL == 1:
        stats = dflash2_state.get_stats()
        logger.info(
            "dflash2_spec: step=%d, block=%d, accepted=%d/%d (%.1f%%), "
            "verify=%.1fms, rate=%.1f%%, circuit=%s",
            dflash2_state.total_spec_steps,
            drafter.block_size,
            n_accepted,
            K,
            100.0 * n_accepted / K if K else 0,
            dt * 1000,
            stats["acceptance_rate"] * 100,
            stats["circuit_tripped"],
        )

    outputs = _emit_spec_tokens(scheduler, request_id, accepted_drafts)

    # _emit_spec_tokens set gen._next_tokens = accepted_drafts[-1] and
    # appended accepted_drafts[:-1] to gen.tokens[0] (it assumes tokens[-1]
    # is the next feed token). Override: the BONUS is the next feed token,
    # and ALL accepted drafts belong in gen.tokens. Append the last
    # accepted draft that _emit_spec_tokens skipped, then set _next_tokens
    # to the bonus. Mirrors eagle3/ngram manual emission + gen state.
    gen._next_tokens = mx.array([bonus], dtype=mx.uint32)
    if gen.tokens and len(gen.tokens) > 0 and accepted_drafts:
        gen.tokens[0].append(accepted_drafts[-1])

    return outputs
