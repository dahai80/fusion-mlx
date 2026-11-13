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
import zlib

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
# Adaptive pause: lowered threshold 0.05 -> 0.10 and window 20 -> 8 so
# low-acceptance scenarios (free generation) pause FAST instead of
# burning 20 wasted draft forwards. At 0% acceptance, 8 steps = 8 wasted
# drafts vs 20 before — 60% less overhead before pause kicks in.
SPEC_MIN_ACCEPT_RATE = float(
    __import__("os").environ.get("FUSION_SPEC_MIN_ACCEPT_RATE", "0.10")
)
SPEC_ADAPTIVE_WINDOW = int(
    __import__("os").environ.get("FUSION_SPEC_ADAPTIVE_WINDOW", "8")
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
        self._req_draft_proposed = 0
        self._req_draft_accepted = 0
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
            self._req_draft_proposed = 0
            self._req_draft_accepted = 0
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
        self._req_draft_proposed += n_total
        self._req_draft_accepted += n_accepted
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

    def pop_request_stats(self, request_id: str) -> dict:
        """Per-request draft accept totals for the request-end telemetry
        record; resets the counters (total_* fields stay cumulative).
        Returns {} when the request never speculated (no record noise)."""
        if self._last_request_id != request_id:
            return {}
        if not self._req_draft_proposed and not self._req_draft_accepted:
            return {}
        stats = {
            "spec_draft_proposed": self._req_draft_proposed,
            "spec_draft_accepted": self._req_draft_accepted,
        }
        self._req_draft_proposed = 0
        self._req_draft_accepted = 0
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


def _run_spec_verify_sampled(
    model,
    draft_tokens: list[int],
    q_rows: mx.array,
    cand_ids: mx.array,
    p0_logprobs: mx.array,
    prompt_cache: list,
    *,
    inv_temp: float,
    top_k: int,
    top_p: float,
    min_p: float,
    accept_uniforms: mx.array,
    final_uniform: float,
) -> tuple[list[int], int, int]:
    """Rejection-sampling verify for the DSpine T>0 path.

    Feeds [D1..DK] to the target (caches past the anchor token), builds
    K+1 filtered target rows (p_0 from the regular step + K verify rows),
    and runs the capped-chain rejection sampler (t0_verify). q_rows are
    the per-position distributions over the K candidate ids the drafts
    were actually drawn from (serial_sample_chain output); they are
    scattered to full-vocab rows here. Returns
    (verified_tokens, n_accepted, cache_tokens_processed) with the same
    layout contract as _run_spec_verify: verified = accepted drafts +
    replacement/bonus token.
    """
    from ..speculative.dspine.propose import filter_target_probs
    from ..speculative.dspine.t0_verify import sampled_capped_reject

    K = len(draft_tokens)
    verify_input = mx.array(draft_tokens, mx.uint32)
    logits = model(verify_input[None], cache=prompt_cache)
    logits = logits.squeeze(0)  # [K, vocab]
    mx.eval(logits)

    # Target rows: p_0 from the regular step's post-processor logprobs
    # (the row the AR path would sample from), then the K verify-forward
    # rows. filter_target_probs renormalizes over the kept set, matching
    # mlx-lm top-p/top-k sampler semantics — the rejection sampler's
    # emitted marginal then equals plain AR with the same request params.
    p_rest = mx.softmax(logits.astype(mx.float32) * float(inv_temp), axis=-1)
    p0 = mx.softmax(
        mx.asarray(p0_logprobs).reshape(-1).astype(mx.float32) * float(inv_temp)
    )
    target_probs = mx.concatenate([p0[None, :], p_rest], axis=0)  # [K+1, V]
    V = int(target_probs.shape[1])
    top_k_eff = int(top_k) if int(top_k) > 0 else V
    target_probs = filter_target_probs(target_probs, top_k_eff, top_p, min_p)

    # Scatter the candidate-space q rows into full-vocab rows: position j
    # of the chain (draft_tokens[j]) was drawn from q_rows[j] over
    # cand_ids[j]. Mass outside the candidates is 0, so the residual
    # (p - q)+ can still resample non-drafted tokens.
    q_vocab = []
    for j in range(K):
        ids_j = cand_ids[j].astype(mx.int32).reshape(-1)
        one_hot = (mx.arange(V)[None, :] == ids_j[:, None]).astype(mx.float32)
        q_vocab.append(one_hot.T @ q_rows[j].astype(mx.float32))
    q_full = mx.stack(q_vocab)  # [K, V]

    cand_chain = mx.array([0] + list(draft_tokens), dtype=mx.int32)
    accepted, bonus, emitted = sampled_capped_reject(
        cand_chain,
        target_probs,
        q_full,
        verify_len=K + 1,
        uniforms=accept_uniforms,
        uniform_final=final_uniform,
    )
    mx.eval(target_probs, q_full)
    verified = (
        [int(x) for x in emitted.tolist()]
        if hasattr(emitted, "tolist")
        else [int(x) for x in emitted]
    )
    return verified, int(accepted), K


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

    # #1055: verify forward writes up to K draft KV tokens to prompt_cache.
    # If _run_spec_verify or the replay below raises, that draft KV was
    # never trimmed/restored, permanently polluting the request's
    # prompt_cache (repeated/garbled output downstream). Wrap verify +
    # rollback so the exception path restores non-trimmable snapshots and
    # trims the K draft tokens from trimmable layers before re-raising.
    try:
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
    except Exception:
        # Verify/replay raised after writing draft KV. Restore to the
        # pre-verify snapshot point: non-trimmable layers via the deepcopy,
        # trimmable KVCache layers by trimming the K draft tokens verify
        # would have appended.
        if non_trimmable_snapshots is not None:
            _restore_non_trimmable_caches(prompt_cache, non_trimmable_snapshots)
        _trim_trimmable(prompt_cache, K)
        logger.warning(
            "spec_decode: verify/replay raised for %s; restored prompt_cache "
            "to pre-verify state (trimmed %d draft tokens)",
            request_id,
            K,
            exc_info=True,
        )
        raise

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

    _MAX_SESSIONS = 32

    def __init__(self, runtime):
        self.runtime = runtime
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        self._sessions: dict = {}
        self._session_order: list[str] = []

    def on_new_request(self, request_id: str):
        # #1074: total_spec_steps / total_draft_proposed / total_draft_accepted
        # are lifecycle (cumulative) counters — they feed Prometheus _total
        # gauges and must accumulate across requests, not reset per request.
        # Only the per-request session state resets here.
        if self._last_request_id != request_id:
            self._last_request_id = request_id

    def get_session(self, request_id: str):
        return self._sessions.get(request_id)

    def set_session(self, request_id: str, session):
        if request_id in self._sessions:
            self._close_session(request_id)
        while len(self._sessions) >= self._MAX_SESSIONS:
            oldest = self._session_order[0]
            self._close_session(oldest)
            logger.warning(
                "dspark_spec: session pool full (%d), evicted oldest=%s",
                self._MAX_SESSIONS,
                oldest[:8],
            )
        self._sessions[request_id] = session
        self._session_order.append(request_id)

    def _close_session(self, request_id: str):
        session = self._sessions.pop(request_id, None)
        if request_id in self._session_order:
            self._session_order.remove(request_id)
        if session is not None:
            close_fn = getattr(session, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.debug(
                        "dspark_spec: session close failed for %s",
                        request_id[:8],
                    )

    def remove_session(self, request_id: str):
        self._close_session(request_id)

    def cleanup_request(self, request_id: str):
        self._close_session(request_id)

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
# Re-probe interval: while circuit-tripped, let one spec step through
# every N steps to check if acceptance recovered. Without this the
# circuit breaker hard-trips and never resumes within a request —
# a transient low-acceptance window permanently disables DFlash2.
DFLASH2_REPROBE_INTERVAL = int(
    __import__("os").environ.get("FUSION_DFLASH2_REPROBE_INTERVAL", "16")
)


class DFlash2SpecState:
    """Per-scheduler DFlash2 in-target speculative decode state.

    In-target pattern: the drafter binds to the scheduler's already-loaded
    target model. The per-step integration runs propose->verify->rollback
    using gen.model + gen.prompt_cache — no duplicate target load, no
    prefill replay. Mirrors DFlashSpecState (v1).
    """

    def __init__(self, runtime, preset=None):
        self.runtime = runtime
        self._preset = preset
        self._warmup = (
            getattr(preset, "warmup_steps", DFLASH2_SPEC_WARMUP_STEPS)
            if preset
            else DFLASH2_SPEC_WARMUP_STEPS
        )
        self._cb_threshold = (
            getattr(
                preset, "circuit_breaker_threshold", DFLASH2_CIRCUIT_BREAKER_THRESHOLD
            )
            if preset
            else DFLASH2_CIRCUIT_BREAKER_THRESHOLD
        )
        self._cb_window = (
            getattr(preset, "circuit_breaker_window", DFLASH2_CIRCUIT_BREAKER_WINDOW)
            if preset
            else DFLASH2_CIRCUIT_BREAKER_WINDOW
        )
        self.steps_since_start = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        self._recent_rates: list[float] = []
        self._circuit_tripped = False
        self._paused_steps = 0
        self._probe_rates: list[float] = []

    def on_new_request(self, request_id: str):
        # #1074: total_spec_steps is a lifecycle (cumulative) counter matching
        # eagle3/DSpark/dflash — accumulates across requests for Prometheus
        # _total gauges. Only per-request adaptive state resets here.
        if self._last_request_id != request_id:
            self._last_request_id = request_id
            self.steps_since_start = 0
            self._recent_rates = []
            self._circuit_tripped = False
            self._paused_steps = 0
            self._probe_rates = []
            drafter = getattr(self.runtime, "drafter", None)
            if drafter is not None:
                drafter.reset()

    def add_token(self, token: int):
        self.steps_since_start += 1

    def should_speculate(self) -> bool:
        if self._circuit_tripped:
            # Re-probe: let one spec step through every DFLASH2_REPROBE_INTERVAL
            # steps so record_result can collect a fresh acceptance sample
            # and un-trip if the drafter recovered. Without this the circuit
            # breaker is a permanent kill switch for the rest of the request.
            self._paused_steps += 1
            if self._paused_steps % DFLASH2_REPROBE_INTERVAL == 0:
                logger.debug(
                    "dflash2_spec: paused re-probe at step=%d",
                    self._paused_steps,
                )
                return True
            return False
        return self.steps_since_start >= self._warmup

    def record_result(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted
        self.runtime.record_accept(n_accepted)
        if n_total <= 0:
            return
        rate = n_accepted / n_total
        # While tripped, this record comes from a re-probe step. Accumulate
        # probe samples and decide un-trip on a small burst rather than the
        # full window (which still holds the stale low-acceptance records).
        if self._circuit_tripped:
            self._probe_rates.append(rate)
            if len(self._probe_rates) >= 3:
                probe_avg = sum(self._probe_rates) / len(self._probe_rates)
                if probe_avg >= self._cb_threshold:
                    self._circuit_tripped = False
                    self._paused_steps = 0
                    self._recent_rates = list(self._probe_rates)
                    self._probe_rates = []
                    logger.info(
                        "dflash2_spec: circuit breaker reset — probe "
                        "acceptance %.1f%% recovered over %d probes",
                        probe_avg * 100,
                        3,
                    )
                else:
                    self._probe_rates = []
            return
        self._recent_rates.append(rate)
        if len(self._recent_rates) > self._cb_window:
            self._recent_rates.pop(0)
        if len(self._recent_rates) >= self._cb_window:
            avg = sum(self._recent_rates) / len(self._recent_rates)
            if avg < self._cb_threshold:
                self._circuit_tripped = True
                self._paused_steps = 0
                self._probe_rates = []
                logger.info(
                    "dflash2_spec: circuit breaker paused "
                    "(avg accept %.1f%% < %.1f%% over %d steps) — "
                    "re-probing every %d steps",
                    avg * 100,
                    self._cb_threshold * 100,
                    len(self._recent_rates),
                    DFLASH2_REPROBE_INTERVAL,
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
            "paused_steps": self._paused_steps,
        }


def _request_temperature(request) -> float:
    """Temperature for spec-decode gates, read from SamplingParams.

    Request carries sampling_params (dataclass field), NOT a temperature
    attribute — getattr(request, "temperature", 0.0) always returned the
    0.0 default, so the dflash2 sampled path saw every request as greedy
    and T>0 requests were misgated.
    """
    params = getattr(request, "sampling_params", None)
    temperature = getattr(params, "temperature", 0.0)
    return float(temperature) if temperature else 0.0


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
        _preset = getattr(dflash2_runtime, "_preset", None)
        dflash2_state = DFlash2SpecState(dflash2_runtime, preset=_preset)
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

    temperature = _request_temperature(request)

    current_offset = 0
    for c in prompt_cache:
        if hasattr(c, "offset"):
            current_offset = max(current_offset, c.offset)

    current_offset = int(current_offset)

    # Build propose context hidden: stored verify hidden (previous spec
    # step) + the current step's forward hidden (1-token regular decode,
    # or the accepted-replay after rollback). Mirrors standalone dflash,
    # which proposes from the previous verify forward's hidden. Do NOT
    # clear _last_hidden here — clearing it left only the 1-token decode
    # hidden, which both crashes RotatingKVCache (negative
    # max_size - offset in _update_in_place) and strips the drafter of
    # all context beyond its cache history.
    hidden = drafter.get_hidden(inner_model)
    if hidden is None or int(hidden.shape[1]) < 2:
        # Bootstrap: before the first verify there is no stored hidden and
        # only the 1-token regular decode hidden. Seed _last_hidden with it
        # so the next step has a 2-token context; until then skip (regular
        # decode already emitted this step's token).
        if drafter._last_hidden is None:
            _hs = getattr(inner_model, "_hidden_states", None)
            if _hs and all(h is not None for h in _hs):
                drafter._last_hidden = mx.concatenate(_hs, axis=-1)
        return []

    # Rope-align draft caches so the pushed chunk ends at
    # current_offset - 1 (the newest token). Replaces the old
    # first-step-only align_draft_cache, which drifted by len(hidden)
    # every step.
    drafter.sync_draft_cache(current_offset, int(hidden.shape[1]))

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
    _rb0 = time.perf_counter()
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
    _rollback_ms = (time.perf_counter() - _rb0) * 1000

    # Store the verify/accepted-replay hidden for the NEXT propose. On the
    # hybrid path the accepted-replay forward has just refreshed
    # _hidden_states to the accepted region only — exactly the context the
    # next chunk needs. Without this the next step falls back to the
    # 1-token regular decode hidden and the drafter goes context-blind.
    drafter.store_verify_hidden(inner_model)

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
            "verify=%.1fms rollback=%.1fms, rate=%.1f%%, circuit=%s",
            dflash2_state.total_spec_steps,
            drafter.block_size,
            n_accepted,
            K,
            100.0 * n_accepted / K if K else 0,
            dt * 1000,
            _rollback_ms,
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


# --- DSpine (#1549): parallel-drafter spec decode over target intermediate
# --- layer hiddens. T>0 requests run the sampled rejection-verify path (P2-2).

DSPINE_SPEC_WARMUP_STEPS = 2
DSPINE_SPEC_LOG_INTERVAL = 50


class DSpineSpecState:
    """Per-scheduler DSpine speculative decode state.

    Holds the per-request draft KV caches and the pending hidden window
    (target hidden rows for committed tokens, oldest first, concat of the
    configured target layers along the last axis). The window mirrors the
    reference DSpine loop: round R+1 drafts from rows
    [anchor_h, D1_h..D_a_h] where anchor = the token the previous round's
    regular step consumed and D1..D_a are the drafts accepted in round R.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.steps_since_start = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._req_draft_proposed = 0
        self._req_draft_accepted = 0
        self._last_request_id = None
        self._caches = None
        self._window = None

    def on_new_request(self, request_id: str):
        if self._last_request_id != request_id:
            self._last_request_id = request_id
            self.steps_since_start = 0
            self._req_draft_proposed = 0
            self._req_draft_accepted = 0
            self._caches = None
            self._window = None
            drafter = getattr(self.runtime, "drafter", None)
            if drafter is not None:
                drafter.reset_cache()

    def add_token(self, token: int):
        self.steps_since_start += 1

    def should_speculate(self) -> bool:
        return self.steps_since_start >= DSPINE_SPEC_WARMUP_STEPS

    def record_result(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted
        self._req_draft_proposed += n_total
        self._req_draft_accepted += n_accepted
        self.runtime.record_accept(n_accepted / n_total if n_total else None)

    def pop_request_stats(self, request_id: str) -> dict:
        """Per-request draft accept totals for the request-end telemetry
        record; resets the counters (total_* fields stay cumulative).
        Returns {} when the request never speculated (no record noise)."""
        if self._last_request_id != request_id:
            return {}
        if not self._req_draft_proposed and not self._req_draft_accepted:
            return {}
        stats = {
            "spec_draft_proposed": self._req_draft_proposed,
            "spec_draft_accepted": self._req_draft_accepted,
        }
        self._req_draft_proposed = 0
        self._req_draft_accepted = 0
        return stats

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
            "block_size": self.runtime.block_size,
        }


def _dspine_concat_captured(captured: dict, layer_ids: list[int], take_rows: int):
    """Concat captured per-layer hiddens [1,K,H] -> [1,take_rows,concat]."""
    if not captured:
        return None
    states = []
    for lid in layer_ids:
        s = captured.get(lid)
        if s is None or int(s.shape[1]) < take_rows:
            return None
        states.append(s[:, :take_rows, :])
    return mx.concatenate(states, axis=-1)


_PROMPT_CLASS_GATE_ENV = "FUSION_MLX_SPEC_PROMPT_CLASS_GATE"


def _prompt_class_gate_on() -> bool:
    import os

    return os.environ.get(_PROMPT_CLASS_GATE_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


_PROMPT_CLASS_GATE_ON = _prompt_class_gate_on()


def _classify_request_prompt(request) -> str:
    # Deterministic classification of the request prompt for the spec gate.
    # Text prompts classify the LAST USER TURN (the scheduler sees the
    # rendered chat template — system text and special-token markers like
    # `<|im_end|>` would poison marker-based classification). Token-id
    # prompts decode best-effort via the tokenizer captured on the runtime
    # (fall back to "other" = ungated).
    from ..speculative.prompt_class import classify_prompt, extract_last_user_turn

    prompt = getattr(request, "prompt", None)
    if isinstance(prompt, list) and prompt:
        prompt = None
    if isinstance(prompt, str) and prompt:
        prompt = extract_last_user_turn(prompt)
    return classify_prompt(prompt)


def spec_gate_allows(prompt_class: str, temperature: float) -> bool:
    from ..speculative.prompt_class import spec_gate_allows as _allows

    return _allows(prompt_class, temperature)


def dspine_spec_step(
    scheduler,
    output,
    current_token: int,
    request_id: str,
) -> list[RequestOutput]:
    """DSpine spec step after a regular decode step.

    Window/bookkeeping layout (reference: dspine generate loop):
      - The regular step consumed token X and sampled current_token; the
        capture store holds X's hidden row ([1,1,concat]).
      - Draft round: draft(window, [current_token + masks], caches).
      - Verify feeds [D1..DK] to the target (cache already past X);
        capture rows become D1_h..DK_h.
      - Next window = [X_h] + D1_h..D_a_h (a = accepted drafts).
    """
    dspine_state = getattr(scheduler, "_dspine_spec_state", None)
    if dspine_state is None:
        dspine_runtime = getattr(scheduler, "_dspine_runtime", None)
        if dspine_runtime is None:
            return []
        dspine_state = DSpineSpecState(dspine_runtime)
        scheduler._dspine_spec_state = dspine_state

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    if dspine_state._last_request_id != request_id:
        dspine_state.on_new_request(request_id)
        # P2-2 prompt-class gate: classify once per request, then gate
        # speculation for classes that historically lose (open chat ~15%
        # acceptance ≈ parity at T>0 — draft overhead cancels the accept
        # wins). Greedy chat still speculates; math/code/other always do.
        # FUSION_MLX_SPEC_PROMPT_CLASS_GATE=0 restores ungated behavior.
        _sp0 = getattr(request, "sampling_params", None)
        _temp0 = float(getattr(_sp0, "temperature", 0.0) or 0.0)
        dspine_state.req_prompt_class = _classify_request_prompt(request)
        if _PROMPT_CLASS_GATE_ON and not spec_gate_allows(
            dspine_state.req_prompt_class, _temp0
        ):
            logger.info(
                "dspine_spec: prompt class %s gated off speculation for "
                "request %s (T>0 losing class)",
                dspine_state.req_prompt_class,
                request_id,
            )
            dspine_state.req_gated = True
        else:
            dspine_state.req_gated = False
    if getattr(dspine_state, "req_gated", False):
        return []

    dspine_state.add_token(current_token)

    if not dspine_state.should_speculate():
        return []

    # T>0 runs the sampled rejection-verify path (P2-2): draft via the
    # sampled refinement tail, verify via t0_verify's capped-chain
    # rejection sampler. Only a missing p_0 row (below) forces the
    # greedy-only skip. Sampling params live on request.sampling_params —
    # reading them off the request itself always yielded the defaults and
    # silently ran the greedy path for sampled requests.
    _sp = getattr(request, "sampling_params", None)
    temperature = float(getattr(_sp, "temperature", 0.0) or 0.0)
    top_k = int(getattr(_sp, "top_k", 0) or 0)
    top_p = float(getattr(_sp, "top_p", 1.0) or 1.0)
    min_p = float(getattr(_sp, "min_p", 0.0) or 0.0)
    sampled_path = temperature > 1e-6

    bg = scheduler.batch_generator
    if bg is None:
        return []
    gen = bg._generation_batch
    if gen is None:
        return []

    prompt_cache = gen.prompt_cache
    model = gen.model
    runtime = dspine_state.runtime
    drafter = runtime.drafter
    capture = runtime.capture
    layer_ids = drafter.target_layer_ids

    # Anchor row: the regular step's captured hidden for its input token X.
    regular_row = _dspine_concat_captured(capture.get_captured(), layer_ids, 1)
    if regular_row is None:
        if dspine_state.total_spec_steps == 0:
            logger.debug(
                "dspine_spec: no regular-step capture rows (layers=%s captured=%s)",
                layer_ids,
                sorted(capture.get_captured().keys()),
            )
        return []

    # Bootstrap: first spec round of the request. The eagle3-style
    # prefill-capture seed (window = last prefill chunk's hidden rows) is
    # DISABLED by default: measured on Qwen3-8B-4bit + DSpine-b16 it cuts
    # draft acceptance roughly in half on every request after the first
    # (window=68 poisoned trace ~20% per-round vs window=1 clean ~35-100%,
    # T=0 math 91 vs 157 tok/s; the captured rows misalign with the draft
    # cache's rope offsets once prefix-cache/partial prefill forwards
    # replace the full prefill). A window of just the regular row lets the
    # draft KV cache build clean context from decode rows — acceptance
    # climbs to 15/15 within a few rounds. Opt back in with
    # FUSION_DSPINE_PREFILL_BOOTSTRAP=1.
    if dspine_state._window is None:
        if __import__("os").environ.get("FUSION_DSPINE_PREFILL_BOOTSTRAP") == "1":
            prefill_states = []
            pc = capture.get_prefill_captured()
            for lid in layer_ids:
                s = pc.get(lid)
                if s is None:
                    prefill_states = []
                    break
                prefill_states.append(s)
            if prefill_states:
                min_sl = min(int(s.shape[1]) for s in prefill_states)
                dspine_state._window = mx.concatenate(
                    [s[:, -min_sl:, :] for s in prefill_states], axis=-1
                )
            capture.clear_prefill_captured()
        if dspine_state._window is None:
            dspine_state._window = regular_row

    if dspine_state._caches is None:
        dspine_state._caches = drafter.make_caches()

    # p_0 row: the regular step's post-processor logprobs at D1's position.
    # Sampled verify needs it as the target row for D1; without it the
    # request falls back to the greedy-only skip.
    p0_logprobs = None
    if sampled_path:
        nlp = getattr(gen, "_next_logprobs", None)
        if nlp:
            p0_logprobs = nlp[0]
        else:
            sampled_path = False
            logger.debug("dspine_spec: no _next_logprobs row, skipping sampled verify")

    draft_q_rows = None
    draft_cand = None
    try:
        with mx.stream(scheduler._stream):
            if sampled_path:
                n_draft = max(1, drafter.block_size - 1)
                step_seed = (
                    zlib.crc32(request_id.encode()) ^ dspine_state.total_spec_steps
                ) & 0xFFFFFFFF
                draft_key = mx.random.key(step_seed)
                draft_uniforms = mx.random.uniform(0.0, 1.0, (n_draft,), key=draft_key)
                draft_tokens, draft_q_rows, draft_cand = drafter.draft_block_sampled(
                    dspine_state._window,
                    current_token,
                    dspine_state._caches,
                    inv_temp=1.0 / temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    uniforms=draft_uniforms,
                )
            else:
                draft_tokens = drafter.draft_block(
                    dspine_state._window, current_token, dspine_state._caches
                )
    except (IndexError, RuntimeError, ValueError, TypeError) as e:
        logger.warning(
            "dspine_spec: draft_block failed: %s (type=%s)", e, type(e).__name__
        )
        return []

    K = len(draft_tokens)
    if K == 0:
        return []

    if dspine_state.total_spec_steps == 0:
        logger.info(
            "dspine_spec: first draft round K=%d window=%d anchor=%d sampled=%s",
            K,
            int(dspine_state._window.shape[1]),
            current_token,
            sampled_path,
        )

    sampled_from_regular = None
    if not sampled_path and gen._next_tokens is not None:
        try:
            sampled_from_regular = int(gen._next_tokens.item())
        except Exception:
            pass

    # D1 gate (greedy path only): draft[0] must match the regular step's
    # sampled prediction. On the sampled path D1 is accepted
    # probabilistically inside the rejection sampler against p_0.
    if (
        not sampled_path
        and sampled_from_regular is not None
        and draft_tokens[0] != sampled_from_regular
    ):
        dspine_state.record_result(0, K)
        # Verify forward never ran; capture still holds the regular row.
        dspine_state._window = regular_row
        return []

    non_trimmable_snapshots = _snapshot_non_trimmable_caches(prompt_cache)

    with mx.stream(scheduler._stream):
        t0 = time.perf_counter()
        if sampled_path:
            verify_key = mx.random.key(step_seed ^ 0x9E3779B9)
            accept_uniforms = mx.random.uniform(0.0, 1.0, (K,), key=verify_key)
            final_uniform = float(
                mx.random.uniform(
                    0.0, 1.0, (1,), key=mx.random.key(step_seed ^ 0x85EB)
                ).item()
            )
            verified, n_accepted, cache_tokens_processed = _run_spec_verify_sampled(
                model,
                draft_tokens,
                draft_q_rows,
                draft_cand,
                p0_logprobs,
                prompt_cache,
                inv_temp=1.0 / temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                accept_uniforms=accept_uniforms,
                final_uniform=final_uniform,
            )
        else:
            verified, n_accepted, cache_tokens_processed = _run_spec_verify(
                model,
                current_token,
                draft_tokens,
                prompt_cache,
                sampled_from_regular=sampled_from_regular,
            )
    dt = time.perf_counter() - t0

    # Cache rollback for rejected tokens (mirrors dflash_spec_step).
    if cache_tokens_processed > 0 and n_accepted < K:
        n_rejected = cache_tokens_processed - n_accepted
        if non_trimmable_snapshots is not None:
            _restore_non_trimmable_caches(prompt_cache, non_trimmable_snapshots)
            if n_accepted > 0:
                replay = mx.array(draft_tokens[:n_accepted], dtype=mx.uint32)
                with mx.stream(scheduler._stream):
                    replay_logits = model(replay[None], cache=prompt_cache)
                    mx.eval(replay_logits)
            trim_count = cache_tokens_processed
        else:
            trim_count = n_rejected
        if trim_count > 0:
            _trim_trimmable(prompt_cache, trim_count)

    # Next hidden window: [X_h] + D1_h..D_a_h. On the accepted-replay path
    # the capture holds exactly the a accepted rows; on the pure-verify path
    # it holds all K rows and we slice. Rows beyond a belong to rejected
    # drafts and must never enter the window.
    a = min(n_accepted, len(verified) - 1)
    window = None
    if a > 0:
        verify_rows = _dspine_concat_captured(capture.get_captured(), layer_ids, a)
        if verify_rows is not None:
            window = mx.concatenate([regular_row, verify_rows], axis=1)
    if window is None:
        window = regular_row
    dspine_state._window = window

    dspine_state.record_result(n_accepted, K)

    if __import__("os").environ.get("DSPINE_SPEC_DEBUG_ROUNDS"):
        _cache = dspine_state._caches[0] if dspine_state._caches else None
        logger.info(
            "dspine_round_dbg: step=%d acc=%d/%d win=%d cache_off=%s anchor=%d",
            dspine_state.total_spec_steps,
            n_accepted,
            K,
            int(window.shape[1]),
            int(_cache.offset) if _cache is not None else -1,
            current_token,
        )

    if dspine_state.total_spec_steps % DSPINE_SPEC_LOG_INTERVAL == 1:
        stats = dspine_state.get_stats()
        logger.info(
            "dspine_spec: step=%d, block=%d, accepted=%d/%d (%.1f%%), "
            "verify=%.1fms, rate=%.1f%%, window=%d",
            dspine_state.total_spec_steps,
            runtime.block_size,
            n_accepted,
            K,
            100.0 * n_accepted / K if K else 0,
            dt * 1000,
            stats["acceptance_rate"] * 100,
            int(window.shape[1]),
        )

    if not verified:
        return []

    # verified = [D1..D_a, bonus]. Emit accepted drafts; bonus becomes the
    # next feed token (mirrors dflash2 emission override).
    bonus = verified[-1]
    accepted_drafts = verified[:-1]

    outputs = _emit_spec_tokens(scheduler, request_id, accepted_drafts)
    gen._next_tokens = mx.array([bonus], dtype=mx.uint32)
    if gen.tokens and len(gen.tokens) > 0 and accepted_drafts:
        gen.tokens[0].append(accepted_drafts[-1])

    return outputs
