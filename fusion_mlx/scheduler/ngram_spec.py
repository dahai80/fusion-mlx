# SPDX-License-Identifier: Apache-2.0
"""N-gram speculative decode integration.

Uses NGramPredictor (CPU-side) to draft K tokens from n-gram patterns,
then verifies them in a single forward pass of the target model.

Unlike draft-model spec decode, this has ZERO GPU overhead for drafting —
predictions come from a CPU hash table. The only GPU cost is the verify
pass, which processes K tokens in one batch.

Token flow:
  1. Regular _step() produces token T
  2. ngram_spec_step looks up n-gram predictions [D1, D2, ..., DK]
  3. Feed [D1, D2, ..., DK] to target model for verification
  4. Accepted drafts + resampled token are emitted as RequestOutputs
"""

import copy
import logging
import time
from typing import Any

import mlx.core as mx

from ..request import RequestOutput
from ..speculative.ngram_predictor import NGramPredictor

logger = logging.getLogger(__name__)

NGRAM_SPEC_ENABLED = __import__("os").environ.get("FUSION_NGRAM_SPEC_ENABLED", "1") == "1"
NGRAM_SPEC_WARMUP = int(__import__("os").environ.get("FUSION_NGRAM_SPEC_WARMUP", "5"))
NGRAM_SPEC_MIN_ACCEPT = float(__import__("os").environ.get("FUSION_NGRAM_SPEC_MIN_ACCEPT", "0.05"))


class NGramSpecState:
    """Per-scheduler n-gram speculative decode state."""

    def __init__(self, predictor: NGramPredictor | None = None):
        self.predictor = predictor or NGramPredictor()
        self.steps = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        self._recent_rates = []
        self._paused = False
        self._window = 20

    def reset(self):
        self.predictor.reset()
        self.steps = 0
        self._last_request_id = None

    def on_new_request(self, request_id: str):
        if self._last_request_id != request_id:
            self.predictor.reset()
            self._last_request_id = request_id
            self.steps = 0
            self._paused = False
            self._recent_rates.clear()

    def add_token(self, token: int):
        self.steps += 1
        self.predictor.add_token(token)

    def should_speculate(self) -> bool:
        if self.steps < NGRAM_SPEC_WARMUP:
            return False
        if self._paused:
            return False
        return True

    def get_drafts(self) -> list[int]:
        return self.predictor.predict()

    def record_result(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted
        self.predictor.record_accepted(n_accepted)

        if n_total > 0:
            self._recent_rates.append(n_accepted / n_total)
        if len(self._recent_rates) > self._window:
            self._recent_rates.pop(0)

        if len(self._recent_rates) >= self._window:
            avg_rate = sum(self._recent_rates) / len(self._recent_rates)
            if avg_rate < NGRAM_SPEC_MIN_ACCEPT and not self._paused:
                self._paused = True
                logger.info("ngram_spec: pausing — avg rate %.1f%% < %.1f%%", avg_rate * 100, NGRAM_SPEC_MIN_ACCEPT * 100)
            elif avg_rate >= NGRAM_SPEC_MIN_ACCEPT and self._paused:
                self._paused = False
                logger.info("ngram_spec: resuming — avg rate %.1f%% recovered", avg_rate * 100)

    def get_stats(self) -> dict:
        rate = self.total_draft_accepted / self.total_draft_proposed if self.total_draft_proposed > 0 else 0.0
        stats = {
            "spec_steps": self.total_spec_steps,
            "draft_proposed": self.total_draft_proposed,
            "draft_accepted": self.total_draft_accepted,
            "acceptance_rate": rate,
            "paused": self._paused,
        }
        stats["ngram"] = self.predictor.get_stats()
        return stats


def _snapshot_non_trimmable(prompt_cache: list) -> list | None:
    snapshots = []
    for i, c in enumerate(prompt_cache):
        if hasattr(c, "is_trimmable") and not c.is_trimmable():
            snapshots.append((i, copy.deepcopy(c)))
    return snapshots if snapshots else None


def _restore_non_trimmable(prompt_cache: list, snapshots: list):
    for i, snapshot in snapshots:
        prompt_cache[i] = snapshot


def _verify_drafts(
    model,
    draft_tokens: list[int],
    prompt_cache: list,
    sampled_from_regular: int | None = None,
) -> tuple[list[int], int, int]:
    """Verify n-gram draft tokens against target model.

    Returns (verified_tokens, n_accepted, cache_tokens_processed).
    """
    K = len(draft_tokens)

    # Verify D1 against regular step's prediction
    if sampled_from_regular is not None and draft_tokens[0] != sampled_from_regular:
        return [sampled_from_regular], 0, 0

    n_accepted = 1 if sampled_from_regular is not None else 0

    if K == 1:
        return draft_tokens[:1], n_accepted, 0

    # Feed [D1, ..., DK] through model
    verify_input = mx.array(draft_tokens, mx.uint32)
    logits = model(verify_input[None], cache=prompt_cache)
    logits = logits.squeeze(0)

    sampled = mx.argmax(logits, axis=-1)
    mx.eval(sampled)
    sampled_list = sampled.tolist()

    # Compare logits[i] with D_{i+1}
    for i in range(K - 1):
        if sampled_list[i] == draft_tokens[i + 1]:
            n_accepted += 1
        else:
            break
    else:
        n_accepted = K

    # Build verified list
    resample_idx = min(n_accepted, K - 1)
    verified = draft_tokens[:n_accepted]
    verified.append(sampled_list[resample_idx])

    return verified, n_accepted, K


def ngram_spec_step(
    scheduler,
    output,
    current_token: int,
    request_id: str,
) -> list[RequestOutput]:
    """Run n-gram speculative decode after a regular decode step."""
    spec_state = scheduler._ngram_spec_state
    if spec_state is None:
        return []

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    if spec_state._last_request_id != request_id:
        spec_state.on_new_request(request_id)

    spec_state.add_token(current_token)

    if not spec_state.should_speculate():
        return []

    draft_tokens = spec_state.get_drafts()
    if not draft_tokens:
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

    sampled_from_regular = None
    if gen._next_tokens is not None:
        try:
            sampled_from_regular = int(gen._next_tokens.flat[0])
        except Exception:
            pass

    non_trimmable_snapshots = _snapshot_non_trimmable(prompt_cache)

    with mx.stream(scheduler._stream):
        t0 = time.perf_counter()
        verified, n_accepted, cache_tokens_processed = _verify_drafts(
            model, draft_tokens, prompt_cache,
            sampled_from_regular=sampled_from_regular,
        )
    dt = time.perf_counter() - t0

    # Cache rollback for rejected tokens
    if cache_tokens_processed > 0 and n_accepted < K:
        n_rejected = cache_tokens_processed - n_accepted
        if non_trimmable_snapshots is not None:
            _restore_non_trimmable(prompt_cache, non_trimmable_snapshots)
            if n_accepted > 0:
                accepted_prefix = draft_tokens[:n_accepted]
                replay_input = mx.array(accepted_prefix, mx.uint32)
                with mx.stream(scheduler._stream):
                    replay_logits = model(replay_input[None], cache=prompt_cache)
                    mx.eval(replay_logits)
        if n_rejected > 0:
            from mlx_lm.models import cache as mlx_cache
            mlx_cache.trim_prompt_cache(prompt_cache, n_rejected)

    spec_state.record_result(n_accepted, K)

    if spec_state.total_spec_steps % 50 == 1:
        stats = spec_state.get_stats()
        logger.info(
            "ngram_spec: step=%d, K=%d, accepted=%d/%d (%.1f%%), "
            "verify=%.1fms, rate=%.1f%%",
            spec_state.total_spec_steps, K,
            n_accepted, K, 100.0 * n_accepted / K if K else 0,
            dt * 1000, stats["acceptance_rate"] * 100,
        )

    if not verified:
        return []

    # Add accepted draft tokens to n-gram predictor
    accepted_only = verified[:-1]
    for t in accepted_only:
        spec_state.predictor.add_token(t)

    # Set up next token for regular decode
    last_token = verified[-1]
    gen._next_tokens = mx.array([last_token], mx.uint32)

    if gen.tokens and len(gen.tokens) > 0:
        for t in accepted_only:
            gen.tokens[0].append(t)

    # Build RequestOutputs
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

        eos_ids = scheduler.tokenizer.eos_token_id if hasattr(scheduler.tokenizer, "eos_token_id") else []
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
                RequestStatus.FINISHED_STOPPED if is_eos
                else RequestStatus.FINISHED_LENGTH_CAPPED
            )
            out.output_token_ids = list(request.output_token_ids)

        outputs.append(out)
        if is_finished:
            break

    return outputs
