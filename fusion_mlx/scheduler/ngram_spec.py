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

import mlx.core as mx

from ..request import RequestOutput
from ..speculative.ngram_predictor import NGramPredictor

logger = logging.getLogger(__name__)

NGRAM_SPEC_ENABLED = (
    __import__("os").environ.get("FUSION_NGRAM_SPEC_ENABLED", "1") == "1"
)
NGRAM_SPEC_WARMUP = int(__import__("os").environ.get("FUSION_NGRAM_SPEC_WARMUP", "5"))
NGRAM_SPEC_MIN_ACCEPT = float(
    __import__("os").environ.get("FUSION_NGRAM_SPEC_MIN_ACCEPT", "0.05")
)
# Conservative break-even used until real T_verify/T_decode is measured.
# GDN verify forwards are expensive (recurrent layers don't batch), so a
# model-specific V/K is computed online; this default just keeps spec off
# until V is known on workloads where D1 match rate is mediocre.
NGRAM_SPEC_DEFAULT_BREAK_EVEN = float(
    __import__("os").environ.get("FUSION_NGRAM_SPEC_BREAK_EVEN", "0.5")
)
# Predictor geometry. order=5 needs a 5-token exact repeat before firing,
# which is too strict for short repetitive patterns (code, lists) where spec
# otherwise wins big. order=3 fires ~4x more often with no false-accept cost
# (rejected drafts are bounded by K). Tunable without code changes.
NGRAM_SPEC_ORDER = int(__import__("os").environ.get("FUSION_NGRAM_SPEC_ORDER", "5"))
NGRAM_SPEC_NUM_DRAFT = int(__import__("os").environ.get("FUSION_NGRAM_SPEC_NUM_DRAFT", "3"))


class NGramSpecState:
    """Per-scheduler n-gram speculative decode state.

    Adaptive gating: the verify forward costs Vx a regular decode (V varies
    by model — GDN recurrent layers don't batch, so V is high). Spec only
    wins when per-step acceptance exceeds V/K. We measure V from real
    verify/decode timings and pause spec when acceptance falls below that
    break-even, so n-gram spec never regresses throughput on hostile
    workloads (diverse text) while still accelerating repetitive ones.
    """

    def __init__(self, predictor: NGramPredictor | None = None):
        self.predictor = predictor or NGramPredictor(
            order=NGRAM_SPEC_ORDER, num_draft=NGRAM_SPEC_NUM_DRAFT
        )
        self.steps = 0
        self.total_spec_steps = 0
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self._last_request_id = None
        self._recent_rates = []
        self._paused = False
        self._window = 12
        # D1 match rate (cheap, no GPU): how often the n-gram's top-1
        # prediction agrees with the model's own next token. Upper-bounds
        # K-draft acceptance, so D1 rate < break-even means spec would
        # certainly regress — gate spec on it BEFORE spending a GPU verify.
        self._d1_rates = []
        self._d1_min_samples = 6
        # EMA of GPU timings (seconds) for dynamic break-even. verify is the
        # spec verify forward; decode is the regular 1-token forward.
        self._verify_dt_ema = None
        self._decode_dt_ema = None
        self._last_K = 3
        self._alpha = 0.3
        # Probe-based resume: while paused, run one spec step every
        # _probe_interval regular steps to re-check acceptance. A probe that
        # beats break-even resumes spec; otherwise spec stays paused.
        self._probe_interval = 64
        self._probe_counter = 0
        self._probing = False

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
            self._d1_rates.clear()
            self._probe_counter = 0
            self._probing = False

    def add_token(self, token: int):
        # Reset per-step; should_speculate sets it True when allowing a probe.
        self._probing = False
        self.steps += 1
        self.predictor.add_token(token)

    def track_d1_match(self, model_token: int | None):
        """Record whether the n-gram's top-1 matches the model's last token.

        Cheap (CPU-only): one hash lookup, no GPU. ``model_token`` is the
        token the model just produced (already a Python int from
        ``resp.token`` — zero sync). Call this BEFORE ``add_token`` so
        ``predict_top1`` predicts exactly ``model_token`` from the prior
        history. The resulting D1 match rate upper-bounds K-draft
        acceptance, so D1 rate < break-even means spec would certainly
        regress — gate spec on it BEFORE spending a GPU verify.
        """
        if model_token is None:
            return
        d1 = self.predictor.predict_top1()
        # No n-gram hit -> spec wouldn't emit anyway; count as a miss so a
        # sparse table keeps spec disabled.
        matched = 1 if (d1 is not None and d1 == model_token) else 0
        self._d1_rates.append(matched)
        if len(self._d1_rates) > self._window:
            self._d1_rates.pop(0)

    def should_speculate(self) -> bool:
        if self.steps < NGRAM_SPEC_WARMUP:
            return False
        # D1 gate: until we have enough cheap D1 samples, stay off (no GPU
        # spent probing). Once we do, require D1 rate >= break-even.
        if len(self._d1_rates) >= self._d1_min_samples:
            d1_rate = sum(self._d1_rates) / len(self._d1_rates)
            if d1_rate < self._break_even():
                return False
        if self._paused:
            if self._probe_counter <= 0:
                self._probe_counter = self._probe_interval
                self._probing = True
                return True
            self._probe_counter -= 1
            return False
        return True

    def get_drafts(self) -> list[int]:
        return self.predictor.predict()

    def _break_even(self) -> float:
        """Acceptance threshold below which spec regresses.

        Spec wins iff n_accepted > V where V = T_verify / T_decode. Per-step
        acceptance = n_accepted / K, so the break-even rate = V / K. Until
        both timings are measured, fall back to a conservative default so
        spec stays off on mediocre-D1 workloads instead of probing blindly.
        """
        if self._verify_dt_ema is None or self._decode_dt_ema is None:
            return NGRAM_SPEC_DEFAULT_BREAK_EVEN
        v = self._verify_dt_ema / max(self._decode_dt_ema, 1e-6)
        return min(0.9, max(NGRAM_SPEC_MIN_ACCEPT, v / max(self._last_K, 1)))

    def record_result(
        self,
        n_accepted: int,
        n_total: int,
        verify_dt: float | None = None,
        decode_dt: float | None = None,
    ):
        self.total_spec_steps += 1
        self.total_draft_proposed += n_total
        self.total_draft_accepted += n_accepted
        self.predictor.record_accepted(n_accepted)

        if n_total > 0:
            self._last_K = n_total
        if verify_dt is not None and verify_dt > 0:
            if self._verify_dt_ema is None:
                self._verify_dt_ema = verify_dt
            else:
                self._verify_dt_ema = (
                    1 - self._alpha
                ) * self._verify_dt_ema + self._alpha * verify_dt
        if decode_dt is not None and decode_dt > 0:
            if self._decode_dt_ema is None:
                self._decode_dt_ema = decode_dt
            else:
                self._decode_dt_ema = (
                    1 - self._alpha
                ) * self._decode_dt_ema + self._alpha * decode_dt

        rate = n_accepted / n_total if n_total > 0 else 0.0

        # A probe step resumes only if its own acceptance beats break-even.
        # Probe samples are sparse by design (1 per _probe_interval) so they
        # are not folded into the rolling window — they would skew the mean.
        if self._probing:
            threshold = self._break_even()
            if rate >= threshold:
                self._paused = False
                self._recent_rates.clear()
                logger.info(
                    "ngram_spec: resuming after probe rate %.0f%% >= %.0f%%",
                    rate * 100,
                    threshold * 100,
                )
            return

        if n_total > 0:
            self._recent_rates.append(rate)
        if len(self._recent_rates) > self._window:
            self._recent_rates.pop(0)

        if len(self._recent_rates) >= self._window:
            avg_rate = sum(self._recent_rates) / len(self._recent_rates)
            threshold = self._break_even()
            if avg_rate < threshold and not self._paused:
                self._paused = True
                v_ratio = (self._verify_dt_ema or 0) / max(
                    self._decode_dt_ema or 1e-6, 1e-6
                )
                logger.info(
                    "ngram_spec: pausing — avg rate %.1f%% < break-even %.1f%% "
                    "(V=%.2fx, K=%d)",
                    avg_rate * 100,
                    threshold * 100,
                    v_ratio,
                    self._last_K,
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


def _trim_trimmable(prompt_cache: list, num_tokens: int):
    """Trim each trimmable cache (KVCache) by ``num_tokens`` in place.

    Unlike ``mlx_cache.trim_prompt_cache`` this is NOT a no-op when the cache
    list contains non-trimmable layers (hybrid GDN models have ArraysCache
    layers): it trims the trimmable caches and skips the rest. Used by the
    n-gram rejection path to drop the rejected drafts (and the duplicates the
    replay wrote) from KVCache layers.
    """
    for c in prompt_cache:
        if getattr(c, "is_trimmable", None) and c.is_trimmable():
            c.trim(num_tokens)


def _rollback_spec_cache(
    prompt_cache: list,
    snapshots,
    draft_tokens: list[int],
    n_accepted: int,
    cache_tokens_processed: int,
    model,
    stream,
):
    """Roll back cache state after a rejected n-gram verify pass.

    Verify wrote all ``cache_tokens_processed`` (= K) drafts to every cache.
    On rejection only ``n_accepted`` of them should be kept. Non-trimmable
    recurrent caches (ArraysCache) were snapshotted before verify and are
    restored, then the accepted prefix is replayed to re-derive their state.
    Trimmable caches (KVCache) end up with K + n_accepted entries (the K
    drafts plus ``n_accepted`` duplicates from the replay); trimming by K
    leaves exactly the n_accepted accepted drafts.
    """
    if snapshots is not None:
        _restore_non_trimmable(prompt_cache, snapshots)
        if n_accepted > 0:
            accepted_prefix = draft_tokens[:n_accepted]
            replay_input = mx.array(accepted_prefix, mx.uint32)
            with mx.stream(stream):
                replay_logits = model(replay_input[None], cache=prompt_cache)
                mx.eval(replay_logits)
    if cache_tokens_processed > 0:
        _trim_trimmable(prompt_cache, cache_tokens_processed)


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

    # Build verified list. The bonus token is the model's prediction AFTER the
    # last accepted draft, i.e. sampled[n_accepted - 1] (sampled[i] is the pred
    # after draft_tokens[i]). Using min(n_accepted, K-1) picks the pred after
    # the first REJECTED draft on rejection paths, which corrupts the bonus.
    resample_idx = max(0, n_accepted - 1)
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

    # Free D1 gate (no GPU sync): current_token is already a Python int
    # (resp.token). predict_top1 reads history as it stood BEFORE this
    # token, so it predicts exactly current_token — the right comparison.
    # Must run BEFORE add_token advances the history.
    spec_state.track_d1_match(current_token)
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

    # ``gen._next_tokens`` is the regular step's freshly-sampled token (the
    # one the next _step would process). Reading it lets us validate D1
    # WITHOUT a GPU forward: if the n-gram's first draft already disagrees
    # with the model's own prediction, we skip verify entirely and just
    # emit the regular token. mx arrays expose ``.item()`` (not ``.flat``).
    # This is a GPU sync, so it ONLY runs when should_speculate already
    # approved spec (rare on hostile workloads) — never per-step.
    sampled_from_regular = None
    if gen._next_tokens is not None:
        try:
            sampled_from_regular = int(gen._next_tokens.item())
        except Exception:
            pass

    # Only the K>=2 verify path mutates the cache (it feeds all K drafts
    # through the model, appending to every cache). The K=1 path and the
    # D1-mismatch path both return WITHOUT a GPU forward and WITHOUT cache
    # mutation, so they need no snapshot. Snapshots deepcopy all ~48
    # ArraysCache recurrent states — doing that unconditionally on every
    # step is the dominant overhead at low acceptance, so gate it here.
    d1 = draft_tokens[0]
    d1_matches = (sampled_from_regular is None) or (d1 == sampled_from_regular)
    needs_snapshot = (K >= 2) and d1_matches
    non_trimmable_snapshots = (
        _snapshot_non_trimmable(prompt_cache) if needs_snapshot else None
    )

    with mx.stream(scheduler._stream):
        t0 = time.perf_counter()
        verified, n_accepted, cache_tokens_processed = _verify_drafts(
            model,
            draft_tokens,
            prompt_cache,
            sampled_from_regular=sampled_from_regular,
        )
    dt = time.perf_counter() - t0

    # Cache rollback for rejected tokens
    if cache_tokens_processed > 0 and n_accepted < K:
        _rollback_spec_cache(
            prompt_cache,
            non_trimmable_snapshots,
            draft_tokens,
            n_accepted,
            cache_tokens_processed,
            model,
            scheduler._stream,
        )

    spec_state.record_result(
        n_accepted,
        K,
        # Only real GPU verifies (K>=2, D1 match) contribute to the verify
        # EMA. K=1 / D1-mismatch paths skip the GPU (dt ~ 0) and would
        # falsely deflate V if folded in.
        verify_dt=dt if cache_tokens_processed > 0 else None,
        decode_dt=scheduler._last_decode_dt,
    )

    if spec_state.total_spec_steps % 50 == 1:
        stats = spec_state.get_stats()
        logger.info(
            "ngram_spec: step=%d, K=%d, accepted=%d/%d (%.1f%%), "
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
            # streaming (which uses ``new_text``) stays correct.
            out.output_text = scheduler.tokenizer.decode(request.output_token_ids)
            request.output_text = out.output_text

        outputs.append(out)
        if is_finished:
            break

    return outputs
