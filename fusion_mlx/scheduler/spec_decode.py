# SPDX-License-Identifier: Apache-2.0
"""Speculative decode integration for the pure-decode fast path.

Uses PromptLookupDecoder (n-gram matching) to draft tokens, then verifies
them in a single forward pass. When the model is memory-bandwidth-bound
(single-request decode), the verify pass costs only ~2x a single-token step
but can produce up to K+1 tokens, yielding significant throughput gains for
repetitive or structured text.
"""

import copy
import logging
import time
from typing import Any

import mlx.core as mx

from ..speculative.prompt_lookup import PromptLookupDecoder
from ..request import RequestOutput

logger = logging.getLogger(__name__)

SPEC_NUM_DRAFT_TOKENS = int(__import__("os").environ.get("FUSION_SPEC_DRAFT_TOKENS", "4"))
SPEC_NGRAM_SIZE = int(__import__("os").environ.get("FUSION_SPEC_NGRAM_SIZE", "3"))
SPEC_MIN_MATCHES = int(__import__("os").environ.get("FUSION_SPEC_MIN_MATCHES", "2"))
SPEC_WARMUP_STEPS = int(__import__("os").environ.get("FUSION_SPEC_WARMUP_STEPS", "8"))


class SpecDecodeState:
    """Per-scheduler speculative decode state for single-request decode."""

    def __init__(self):
        self.decoder = PromptLookupDecoder(
            num_draft_tokens=SPEC_NUM_DRAFT_TOKENS,
            ngram_size=SPEC_NGRAM_SIZE,
            min_matches=SPEC_MIN_MATCHES,
        )
        self.steps_since_start = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None

    def reset(self):
        self.decoder.reset()
        self.steps_since_start = 0
        self._last_request_id = None

    def on_new_request(self, request_id: str, prompt_tokens: list[int]):
        if self._last_request_id != request_id:
            self.decoder.reset()
            self.decoder.add_prompt_tokens(prompt_tokens)
            self._last_request_id = request_id
            self.steps_since_start = 0

    def add_token(self, token: int):
        self.decoder.add_generated_token(token)
        self.steps_since_start += 1

    def should_speculate(self) -> bool:
        return self.steps_since_start >= SPEC_WARMUP_STEPS

    def get_drafts(self) -> list[int]:
        return self.decoder.get_draft_tokens()

    def record_accepted(self, n_accepted: int, n_total: int):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted
        self.decoder.record_accepted(n_accepted)

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
        }


def _snapshot_non_trimmable_caches(prompt_cache: list) -> list | None:
    """Deep-copy non-trimmable cache entries (ArraysCache, etc).

    Returns list of (index, deep_copy) for each non-trimmable cache entry,
    or None if all caches are trimmable (fast path: no snapshot needed).
    """
    snapshots = []
    for i, c in enumerate(prompt_cache):
        if hasattr(c, "is_trimmable") and not c.is_trimmable():
            snapshots.append((i, copy.deepcopy(c)))
    return snapshots if snapshots else None


def _restore_non_trimmable_caches(prompt_cache: list, snapshots: list):
    """Restore non-trimmable caches from snapshot after rejected drafts."""
    for i, snapshot in snapshots:
        prompt_cache[i] = snapshot


def _run_spec_verify(
    model,
    current_token: int,
    draft_tokens: list[int],
    prompt_cache: list,
    sampler: Any = None,
) -> tuple[list[int], int]:
    """Run speculative verification forward pass.

    Feeds [current_token, D1, D2, ..., DK] to the model and compares
    each draft token against the model's prediction at that position.

    Returns (verified_tokens, n_accepted) where:
      - verified_tokens: list of all tokens to yield (accepted drafts +
        the resampled token at the first rejection point)
      - n_accepted: number of draft tokens that matched
    """
    K = len(draft_tokens)
    verify_input = mx.array([current_token] + draft_tokens, mx.uint32)

    logits = model(verify_input[None], cache=prompt_cache)
    logits = logits.squeeze(0)  # [K+1, vocab]

    if sampler is not None:
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = sampler(logprobs)
    else:
        sampled = mx.argmax(logits, axis=-1)

    mx.eval(sampled)
    sampled_list = sampled.tolist()

    n_accepted = 0
    for i in range(K):
        model_pred = sampled_list[i]
        if model_pred == draft_tokens[i]:
            n_accepted += 1
        else:
            break

    resample_idx = n_accepted
    verified = draft_tokens[:n_accepted]
    if resample_idx < len(sampled_list):
        verified.append(sampled_list[resample_idx])

    return verified, n_accepted


def _trim_cache_for_rejected(prompt_cache: list, n_total: int, n_accepted: int):
    """Trim KV cache to remove entries for rejected draft tokens.

    After a verify pass with K+1 tokens, if only n_accepted < K draft tokens
    were accepted, we need to remove K - n_accepted entries from the cache.
    This only works for trimmable caches (KVCache, RotatingKVCache).
    """
    n_rejected = n_total - 1 - n_accepted  # -1 for the current_token position
    if n_rejected > 0:
        from mlx_lm.models import cache as mlx_cache
        trimmed = mlx_cache.trim_prompt_cache(prompt_cache, n_rejected)
        logger.debug(
            "spec_decode: trimmed %d tokens from trimmable caches", trimmed or 0
        )


def spec_decode_step(
    scheduler,
    output,
    current_token: int,
    request_id: str,
) -> list[RequestOutput]:
    """Run speculative decode after a regular decode step.

    Called from _step_pure_decode after the regular step produces one token.
    If PromptLookup has draft tokens, runs a verify pass and returns
    additional RequestOutput objects for accepted drafts.

    For hybrid models with non-trimmable caches (ArraysCache in GatedDeltaNet),
    we snapshot these caches before the verify pass and restore them if any
    draft tokens are rejected (since ArraysCache cannot be trimmed).
    """
    spec_state = scheduler._spec_decode_state
    if spec_state is None:
        spec_state = SpecDecodeState()
        scheduler._spec_decode_state = spec_state

    request = scheduler.running.get(request_id)
    if request is None:
        return []

    if spec_state._last_request_id != request_id:
        spec_state.on_new_request(request_id, request.prompt_token_ids or [])

    spec_state.add_token(current_token)

    if not spec_state.should_speculate():
        return []

    draft_tokens = spec_state.get_drafts()
    if not draft_tokens or len(draft_tokens) < SPEC_MIN_MATCHES:
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

    # Snapshot non-trimmable caches (ArraysCache) before verify pass.
    # These caches store RNN state that cannot be trimmed — if draft tokens
    # are rejected, we must restore the original state.
    non_trimmable_snapshots = _snapshot_non_trimmable_caches(prompt_cache)

    with mx.stream(scheduler._stream):
        t0 = time.perf_counter()
        verified, n_accepted = _run_spec_verify(
            model, current_token, draft_tokens, prompt_cache
        )
    dt = time.perf_counter() - t0

    # Handle cache rollback for rejected tokens
    if n_accepted < K:
        n_rejected = K - n_accepted
        # For non-trimmable caches (ArraysCache): restore from snapshot
        if non_trimmable_snapshots is not None:
            _restore_non_trimmable_caches(prompt_cache, non_trimmable_snapshots)
            # Re-run the verify pass with ONLY accepted tokens + resample
            # This re-populates caches correctly for the accepted prefix
            if n_accepted > 0:
                # Re-run with accepted tokens to update non-trimmable caches
                # correctly. We only feed up to n_accepted+1 tokens (the
                # accepted prefix + the resampled token).
                accepted_prefix = draft_tokens[:n_accepted]
                resample_token = verified[-1] if verified else current_token
                replay_input = mx.array(
                    [current_token] + accepted_prefix, mx.uint32
                )
                with mx.stream(scheduler._stream):
                    replay_logits = model(replay_input[None], cache=prompt_cache)
                    mx.eval(replay_logits)
            else:
                # All drafts rejected — caches already restored to pre-verify
                # state. No re-run needed; the regular step's cache is intact.
                pass
            logger.debug(
                "spec_decode: restored %d non-trimmable caches, %d/%d rejected",
                len(non_trimmable_snapshots), n_rejected, K,
            )
        # For trimmable caches (KVCache): just trim
        _trim_cache_for_rejected(prompt_cache, K + 1, n_accepted)

    spec_state.record_accepted(n_accepted, K)

    if spec_state.total_spec_steps % 50 == 1:
        stats = spec_state.get_stats()
        logger.info(
            "spec_decode: step=%d, drafts=%d, accepted=%d/%d (%.1f%%), "
            "verify_time=%.1fms, acceptance_rate=%.1f%%",
            spec_state.total_spec_steps,
            K, n_accepted, K,
            100.0 * n_accepted / K if K > 0 else 0,
            dt * 1000,
            stats["acceptance_rate"] * 100,
        )

    if not verified:
        return []

    last_token = verified[-1]
    gen._next_tokens = mx.array([last_token], mx.uint32)

    if gen.tokens and len(gen.tokens) > 0:
        for t in verified[:-1]:
            gen.tokens[0].append(t)

    outputs = []
    step_now = time.monotonic()

    for i, token in enumerate(verified):
        request.append_output_token(token)
        request.last_activity_at = step_now

        detokenizer = scheduler._get_detokenizer(request_id)
        if detokenizer is not None:
            detokenizer.add_token(token)
            new_text = detokenizer.last_segment
        else:
            new_text = scheduler.tokenizer.decode([token])

        is_eos = token in (
            scheduler.tokenizer.eos_token_id
            if hasattr(scheduler.tokenizer, "eos_token_id")
            else []
        )
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
