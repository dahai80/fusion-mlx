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
SPEC_MIN_ACCEPT_RATE = float(
    __import__("os").environ.get("FUSION_SPEC_MIN_ACCEPT_RATE", "0.10")
)
SPEC_ADAPTIVE_WINDOW = int(
    __import__("os").environ.get("FUSION_SPEC_ADAPTIVE_WINDOW", "20")
)


class SpecDecodeState:
    """Per-scheduler speculative decode state."""

    def __init__(self, draft_model_decoder=None):
        self.draft_model = draft_model_decoder
        self.steps_since_start = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        self._recent_accepted = []
        self._spec_paused = False

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

    def add_token(self, token: int):
        self.steps_since_start += 1

    def should_speculate(self) -> bool:
        if self.steps_since_start < SPEC_WARMUP_STEPS:
            return False
        if self._spec_paused:
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

        self._recent_accepted.append((n_accepted, n_total))
        if len(self._recent_accepted) > SPEC_ADAPTIVE_WINDOW:
            self._recent_accepted.pop(0)

        if len(self._recent_accepted) >= SPEC_ADAPTIVE_WINDOW:
            total_a = sum(a for a, t in self._recent_accepted)
            total_t = sum(t for a, t in self._recent_accepted)
            recent_rate = total_a / total_t if total_t > 0 else 0
            if recent_rate < SPEC_MIN_ACCEPT_RATE and not self._spec_paused:
                self._spec_paused = True
                logger.info(
                    "spec_decode: pausing — acceptance %.1f%% < %.1f%%",
                    recent_rate * 100,
                    SPEC_MIN_ACCEPT_RATE * 100,
                )
            elif recent_rate >= SPEC_MIN_ACCEPT_RATE and self._spec_paused:
                self._spec_paused = False
                logger.info(
                    "spec_decode: resuming — acceptance %.1f%% recovered",
                    recent_rate * 100,
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
    """Build RequestOutputs for spec-decode accepted tokens."""
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

    outputs = []
    step_now = time.monotonic()

    for token in tokens:
        request.append_output_token(token)
        request.last_activity_at = step_now

        detokenizer = scheduler._get_detokenizer(request_id)
        if detokenizer is not None:
            detokenizer.add_token(token)
            new_text = detokenizer.last_segment
        else:
            new_text = scheduler.tokenizer.decode([token])

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

    # Update gen._next_tokens so the regular step picks up the last token
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
                "dspark_spec: step=%d, accepted=%d, rate=%.1f%%, " "sessions=%d",
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
