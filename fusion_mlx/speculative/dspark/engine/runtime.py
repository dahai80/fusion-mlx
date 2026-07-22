"""DSpark decode loop: draft -> confidence-pruned verify -> rejection
sampling -> KV rollback.

Mirrors the PyTorch reference:
- refs/deepspec/deepspec/eval/base_evaluator.py ``generate_decoding_sample``
  and ``verify_draft_tokens`` (loop structure, acceptance math, stop-token
  truncation, target-cache crop-to-committed).
- refs/deepspec/deepspec/eval/dspark/draft_ops.py (parallel block forward,
  serial Markov sampling, ``_confident_prefix_length`` static threshold,
  draft-cache crop that retains ctx K/V and drops only block positions).

Invariants (batch=1 throughout):
- ``start`` counts committed tokens including the prompt; ``output_tokens``
  always holds ``start + 1`` tokens, the last being the current anchor whose
  K/V is *not* yet in the target cache.
- After every round the target cache offset equals ``start`` (per layer) and
  the draft cache offset equals ``start`` as well (ctx K/V of the committed
  prefix; block K/V cropped each round).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .adapters import LoadedTargetModel
from .draft import DSparkDraftModel
from .sampling import (
    DRAFT_PROB_CLAMP,
    GREEDY_TEMP_EPS,
    KeySequence,
    gather_token_probs,
    logits_to_probs,
    sample_from_logits,
    sample_from_probs,
    sample_residual,
    speculative_accept,
)

VERIFY_MODES = ("full", "lazy-logits")

# Optional per-round trace callback; receives one dict per decode round with
# the fields U5's golden-trace comparison needs (see _emit_trace).
TraceHook = Callable[[dict[str, Any]], None]


@dataclass
class DSparkRuntimeEvent:
    token_ids: list[int]
    output_tokens: list[int]
    metrics: dict[str, Any] | None = None
    finished: bool = False


def sample_tokens(logits: mx.array, temperature: float) -> mx.array:
    """Legacy helper (kept for the public API surface): sample [..., V] logits."""
    vocab_size = logits.shape[-1]
    flat = logits.reshape(-1, vocab_size)
    sampled = sample_from_logits(flat, temperature)
    return sampled.reshape(logits.shape[:-1]).astype(mx.uint32)


def trim_draft_cache(cache: list[Any], num_tokens: int) -> None:
    for layer_cache in cache:
        layer_cache.trim(num_tokens)


def generated_token_count(output_tokens: list[int], prompt_len: int) -> int:
    return max(len(output_tokens) - prompt_len, 0)


def longest_prefix_match(draft_tokens: list[int], verifier_tokens: list[int]) -> int:
    matched = 0
    for draft_token, verifier_token in zip(draft_tokens, verifier_tokens):
        if draft_token != verifier_token:
            break
        matched += 1
    return matched


def peak_memory_gb() -> float:
    return mx.get_peak_memory() / 1e9


def profile_start(profile: dict[str, float] | None) -> float:
    return time.perf_counter() if profile is not None else 0.0


def add_profile_elapsed(
    profile: dict[str, float] | None,
    key: str,
    start: float,
) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + time.perf_counter() - start


def confident_prefix_length(
    confidence_logits: mx.array | None,
    block_size: int,
    threshold: float,
    sts_temperatures: list[float] | None = None,
) -> int:
    """Reference ``_confident_prefix_length``: length of the prefix whose
    sigmoid(confidence) >= threshold. threshold <= 0 (or no confidence head)
    -> the full block; the FIRST sub-threshold position truncates there, so a
    sub-threshold position 0 yields an empty proposal.

    When STS temperatures are present (U6 calibration, paper §3.2.1) the
    confidence logits are divided per-position by them before the sigmoid,
    so the threshold compares against *calibrated* survival confidences.
    Temperature scaling is monotone per position, hence order-preserving;
    with ``sts_temperatures=None`` behavior is byte-identical to before.
    """
    if confidence_logits is None or threshold <= 0.0:
        return int(block_size)
    logits_row = confidence_logits[0]
    if sts_temperatures is not None:
        length = int(logits_row.shape[-1])
        if length > len(sts_temperatures):
            raise ValueError(
                f"confidence logits have {length} positions but only "
                f"{len(sts_temperatures)} STS temperatures are available"
            )
        logits_row = logits_row / mx.array(
            sts_temperatures[:length], dtype=logits_row.dtype
        )
    below = (mx.sigmoid(logits_row) < threshold).tolist()
    for idx, flag in enumerate(below):
        if flag:
            return idx
    return int(block_size)


def draft_block(
    draft: DSparkDraftModel,
    anchor_token: int,
    ctx_taps: mx.array,
    draft_cache: list[Any],
    temperature: float,
    keys: KeySequence,
) -> tuple[list[int], mx.array, mx.array | None]:
    """One parallel block forward + serial Markov sampling round.

    Reference: ``forward_dspark_draft_block`` + ``sample_block_tokens`` +
    ``_predict_confidence_logits``. The draft cache receives ctx+block K/V;
    the block positions are cropped here so the cache retains exactly the
    rotated ctx K/V of the committed prefix across rounds.

    Returns ``(drafted_tokens, corrected_logits [1, B, V], confidence_logits
    [1, B] | None)`` where corrected logits are post-Markov-bias (the sampled
    distribution the acceptance ratio must use).
    """
    block_size = draft.block_size
    _, noise_embedding = draft.block_inputs(mx.array([anchor_token], dtype=mx.uint32))
    hidden = draft(noise_embedding, ctx_taps, cache=draft_cache)
    base_logits = draft.compute_logits(hidden)

    step_keys = keys.split(block_size)
    markov_head = getattr(draft, "markov_head", None)
    prev_tokens = mx.array([anchor_token], dtype=mx.uint32)
    sampled_steps: list[mx.array] = []
    corrected_steps: list[mx.array] = []
    for step in range(block_size):
        step_logits = base_logits[:, step, :]
        if markov_head is not None:
            step_logits = step_logits + markov_head.step_bias(prev_tokens)
        corrected_steps.append(step_logits[:, None, :])
        sampled = sample_from_logits(step_logits, temperature, key=step_keys[step])
        sampled_steps.append(sampled[:, None])
        prev_tokens = sampled

    sampled_tokens = mx.concatenate(sampled_steps, axis=1)  # [1, B]
    corrected_logits = mx.concatenate(corrected_steps, axis=1)  # [1, B, V]

    # Confidence features use the token *preceding* each position:
    # [anchor, x_1 .. x_{B-1}] (reference _predict_confidence_logits).
    confidence_prev = mx.concatenate(
        [mx.array([[anchor_token]], dtype=mx.uint32), sampled_tokens[:, :-1]],
        axis=1,
    )
    confidence_logits = draft.confidence_logits(hidden, confidence_prev)

    if confidence_logits is not None:
        mx.eval(sampled_tokens, confidence_logits)
    else:
        mx.eval(sampled_tokens)
    trim_draft_cache(draft_cache, block_size)
    drafted = [int(token) for token in sampled_tokens[0].tolist()]
    return drafted, corrected_logits, confidence_logits


def verify_greedy_from_posterior(
    drafted: list[int],
    posterior: list[int],
) -> tuple[int, int]:
    """temperature==0 fast path: with one-hot probs the acceptance math
    degenerates to exact argmax matching, and both the residual bonus (on
    rejection) and the p_t bonus (all accepted) are the target argmax at the
    first divergent / final position — i.e. ``posterior[n]``.
    """
    num_draft = len(drafted)
    matched = longest_prefix_match(drafted, posterior[:num_draft])
    return matched, posterior[matched]


def verify_lazy_greedy(
    target: LoadedTargetModel,
    norm_hidden: mx.array,
    drafted: list[int],
    chunk_size: int,
) -> tuple[int, int]:
    """Chunked lm_head greedy verify with early exit past the first mismatch."""
    num_draft = len(drafted)
    chunk = max(1, chunk_size)
    accepted = 0
    for chunk_start in range(0, num_draft + 1, chunk):
        chunk_end = min(chunk_start + chunk, num_draft + 1)
        posterior = mx.argmax(
            target.lm_head_logits(norm_hidden[:, chunk_start:chunk_end, :]),
            axis=-1,
        )
        mx.eval(posterior)
        for local_idx, token in enumerate(posterior[0].tolist()):
            pos = chunk_start + local_idx
            if pos == num_draft or int(token) != drafted[pos]:
                return accepted, int(token)
            accepted += 1
    raise RuntimeError("Lazy greedy verifier reached an impossible state.")


def verify_lazy_accept(
    target: LoadedTargetModel,
    norm_hidden: mx.array,
    drafted: list[int],
    p_draft: mx.array,
    temperature: float,
    key: mx.array | None,
    chunk_size: int,
) -> tuple[int, int]:
    """Chunked lm_head rejection sampling with early exit.

    Consumes RNG keys identically to ``speculative_accept`` (one split into
    (u_key, bonus_key); the full-length acceptance uniform vector is drawn up
    front since it does not depend on target logits), so for a given key this
    makes bit-identical decisions to the full-logits path.
    """
    num_draft = len(drafted)
    if key is not None:
        u_key, bonus_key = mx.random.split(key)
    else:
        u_key = bonus_key = None
    uniforms = mx.random.uniform(shape=(1, num_draft), key=u_key)
    token_arr = mx.array(drafted, dtype=mx.uint32)[None]
    selected_draft = mx.maximum(
        gather_token_probs(p_draft, token_arr), DRAFT_PROB_CLAMP
    )

    chunk = max(1, chunk_size)
    accepted = 0
    reject_target_row: mx.array | None = None
    for chunk_start in range(0, num_draft, chunk):
        chunk_end = min(chunk_start + chunk, num_draft)
        probs_chunk = logits_to_probs(
            target.lm_head_logits(norm_hidden[:, chunk_start:chunk_end, :]),
            temperature,
        )
        selected_target = gather_token_probs(
            probs_chunk, token_arr[:, chunk_start:chunk_end]
        )
        accept_prob = mx.minimum(
            selected_target / selected_draft[:, chunk_start:chunk_end], 1.0
        )
        accept = uniforms[:, chunk_start:chunk_end] < accept_prob
        mx.eval(accept)
        local_accepted = 0
        for flag in accept[0].tolist():
            if not flag:
                break
            local_accepted += 1
        accepted = chunk_start + local_accepted
        if local_accepted < (chunk_end - chunk_start):
            reject_target_row = probs_chunk[:, local_accepted, :]
            break

    if reject_target_row is not None:
        bonus = sample_residual(
            reject_target_row,
            p_draft[:, accepted, :],
            key=bonus_key,
        )
    else:
        last_probs = logits_to_probs(
            target.lm_head_logits(norm_hidden[:, num_draft : num_draft + 1, :]),
            temperature,
        )
        bonus = sample_from_probs(last_probs[:, -1, :], key=bonus_key)
    mx.eval(bonus)
    return accepted, int(bonus.item())


def dspark_generate_stream(
    target: LoadedTargetModel,
    draft: DSparkDraftModel,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: set[int],
    layer_ids: list[int],
    speculative_tokens: int | None = None,
    confidence_threshold: float = 0.0,
    verify_mode: str = "full",
    verify_chunk_size: int = 4,
    seed: int | None = None,
    profile: bool = False,
    trace_hook: TraceHook | None = None,
) -> Iterator[DSparkRuntimeEvent]:
    if verify_mode not in VERIFY_MODES:
        raise ValueError(
            f"verify_mode must be one of {VERIFY_MODES}, got {verify_mode!r}."
        )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
        )

    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    keys = KeySequence(seed)
    profile_times: dict[str, float] | None = {} if profile else None
    prompt_len = int(prompt_tokens.shape[0])
    total_max_tokens = prompt_len + max_new_tokens
    block_size = draft.block_size
    if speculative_tokens is None:
        proposal_cap = block_size
    else:
        proposal_cap = max(1, min(speculative_tokens, block_size))
    greedy = temperature < GREEDY_TEMP_EPS

    # ---- prefill: one target forward over the prompt; sample the first token
    prefill_start = time.perf_counter()
    norm_hidden, ctx_taps = target.forward_verifier_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_logits = target.lm_head_logits(norm_hidden[:, -1:, :])
    first_token_arr = sample_from_logits(
        first_logits[:, -1, :], temperature, key=keys.next()
    )
    mx.eval(first_token_arr)
    first_token = int(first_token_arr.item())
    prefill_time = time.perf_counter() - prefill_start

    output_tokens = prompt_tokens.tolist() + [first_token]
    start = prompt_len
    streamed_len = prompt_len
    acceptance_lengths: list[int] = []
    proposal_lengths: list[int] = []
    accepted_draft_lengths: list[int] = []
    finished = first_token in stop_token_ids

    if max_new_tokens > 0:
        yield DSparkRuntimeEvent(
            token_ids=output_tokens[streamed_len:],
            output_tokens=list(output_tokens),
        )
        streamed_len = len(output_tokens)

    # STS calibration temps (dspark_config.sts_temperatures, fitted by
    # scripts/calibrate.py); absent on fakes/uncalibrated drafts -> None.
    sts_temperatures = getattr(draft, "sts_temperatures", None)

    decode_start = time.perf_counter()
    round_idx = 0
    while not finished and start < total_max_tokens:
        anchor_token = output_tokens[start]

        # ---- draft: parallel block forward + serial Markov sampling ----
        draft_start = profile_start(profile_times)
        drafted, corrected_logits, confidence_logits = draft_block(
            draft=draft,
            anchor_token=anchor_token,
            ctx_taps=ctx_taps,
            draft_cache=draft_cache,
            temperature=temperature,
            keys=keys,
        )
        proposal_len = confident_prefix_length(
            confidence_logits,
            block_size,
            confidence_threshold,
            sts_temperatures=sts_temperatures,
        )
        proposal_len = min(proposal_len, proposal_cap)
        add_profile_elapsed(profile_times, "draft_time_s", draft_start)

        # ---- verify: one target forward over [anchor, x_1..x_l] ----
        verify_start = profile_start(profile_times)
        verify_tokens = [anchor_token] + drafted[:proposal_len]
        verify_inputs = mx.array(verify_tokens, dtype=mx.uint32)[None]
        accept_key = keys.next()
        verify_logits: mx.array | None = None
        verify_norm_hidden: mx.array | None = None
        p_target: mx.array | None = None
        p_draft: mx.array | None = None

        if verify_mode == "lazy-logits" and proposal_len > 0:
            verify_norm_hidden, verifier_taps = target.forward_verifier_states(
                verify_inputs, target_cache, layer_ids
            )
            if greedy:
                accepted, bonus_token = verify_lazy_greedy(
                    target,
                    verify_norm_hidden,
                    drafted[:proposal_len],
                    verify_chunk_size,
                )
            else:
                p_draft = logits_to_probs(
                    corrected_logits[:, :proposal_len, :], temperature
                )
                accepted, bonus_token = verify_lazy_accept(
                    target,
                    verify_norm_hidden,
                    drafted[:proposal_len],
                    p_draft,
                    temperature,
                    accept_key,
                    verify_chunk_size,
                )
        else:
            verify_logits, verifier_taps = target.forward_with_hidden_states(
                verify_inputs, target_cache, layer_ids
            )
            if greedy:
                posterior = mx.argmax(verify_logits[0], axis=-1)
                mx.eval(posterior)
                accepted, bonus_token = verify_greedy_from_posterior(
                    drafted[:proposal_len],
                    [int(token) for token in posterior.tolist()],
                )
            else:
                p_target = logits_to_probs(verify_logits, temperature)
                if proposal_len > 0:
                    p_draft = logits_to_probs(
                        corrected_logits[:, :proposal_len, :], temperature
                    )
                accepted, bonus_token = speculative_accept(
                    drafted[:proposal_len],
                    p_draft,
                    p_target,
                    temperature,
                    key=accept_key,
                )

        # ---- stop-token truncation inside the accepted block ----
        # (reference verify_draft_tokens: the first stop token among the
        # accepted drafts truncates the commit there and drops the bonus).
        accepted_effective = accepted
        terminated_by_stop_token = False
        if stop_token_ids and accepted > 0:
            for idx, token in enumerate(verify_tokens[1 : accepted + 1]):
                if token in stop_token_ids:
                    accepted_effective = idx + 1
                    terminated_by_stop_token = True
                    break

        # ---- commit + target-cache rollback to the committed length ----
        if terminated_by_stop_token:
            committed = verify_tokens[1 : accepted_effective + 1]
            trim = (proposal_len + 1) - accepted_effective
            acceptance_lengths.append(accepted_effective)
            proposal_lengths.append(accepted_effective)
            accepted_draft_lengths.append(accepted_effective)
            finished = True
        else:
            committed = verify_tokens[1 : accepted + 1] + [bonus_token]
            trim = proposal_len - accepted
            acceptance_lengths.append(accepted + 1)
            proposal_lengths.append(proposal_len)
            accepted_draft_lengths.append(accepted)
            # Next-round ctx: verify-pass taps sliced to accepted + 1.
            ctx_taps = verifier_taps[:, : accepted + 1, :]
            if bonus_token in stop_token_ids:
                finished = True
        if trim > 0:
            target.rewind_kv_caches(target_cache, trim)

        output_tokens.extend(committed)
        start += len(committed)
        add_profile_elapsed(profile_times, "verify_time_s", verify_start)

        if trace_hook is not None:
            _emit_trace(
                trace_hook=trace_hook,
                target=target,
                round_idx=round_idx,
                anchor_token=anchor_token,
                drafted=drafted,
                proposal_len=proposal_len,
                corrected_logits=corrected_logits,
                confidence_logits=confidence_logits,
                verify_logits=verify_logits,
                verify_norm_hidden=verify_norm_hidden,
                p_target=p_target,
                p_draft=p_draft,
                temperature=temperature,
                accepted=accepted,
                accepted_effective=accepted_effective,
                bonus_token=None if terminated_by_stop_token else bonus_token,
                committed=committed,
                terminated_by_stop_token=terminated_by_stop_token,
            )
        round_idx += 1

        bookkeeping_start = profile_start(profile_times)
        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            finished = True
        token_ids = output_tokens[streamed_len:]
        if token_ids:
            yield DSparkRuntimeEvent(
                token_ids=token_ids,
                output_tokens=list(output_tokens),
            )
            streamed_len = len(output_tokens)
        add_profile_elapsed(profile_times, "bookkeeping_time_s", bookkeeping_start)

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    generated_tokens = generated_token_count(output_tokens, prompt_len)
    total_time = prefill_time + decode_time

    metrics = {
        "num_input_tokens": prompt_len,
        "num_output_tokens": generated_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "prompt_tps": prompt_len / max(prefill_time, 1e-9),
        "generation_tps": generated_tokens / max(decode_time, 1e-9),
        "end_to_end_tps": generated_tokens / max(total_time, 1e-9),
        # Reference semantics: tokens committed per verify round
        # (accepted drafts + bonus); tau = mean of these.
        "avg_acceptance_length": sum(acceptance_lengths)
        / max(len(acceptance_lengths), 1),
        "acceptance_lengths": acceptance_lengths,
        "proposal_lengths": proposal_lengths,
        "accepted_draft_lengths": accepted_draft_lengths,
        "confidence_threshold": confidence_threshold,
        "verify_mode": verify_mode,
        "peak_memory_gb": peak_memory_gb(),
        "target_cache_summary": target.cache_summary(target_cache),
        "speculative_tokens": proposal_cap,
    }
    if profile_times is not None:
        profiled_time = sum(
            profile_times.get(key, 0.0)
            for key in ("draft_time_s", "verify_time_s", "bookkeeping_time_s")
        )
        metrics["profile"] = {
            **profile_times,
            "unattributed_decode_time_s": decode_time - profiled_time,
            "steps": len(acceptance_lengths),
        }
    yield DSparkRuntimeEvent(
        token_ids=[],
        output_tokens=list(output_tokens),
        metrics=metrics,
        finished=True,
    )


def _emit_trace(
    *,
    trace_hook: TraceHook,
    target: LoadedTargetModel,
    round_idx: int,
    anchor_token: int,
    drafted: list[int],
    proposal_len: int,
    corrected_logits: mx.array,
    confidence_logits: mx.array | None,
    verify_logits: mx.array | None,
    verify_norm_hidden: mx.array | None,
    p_target: mx.array | None,
    p_draft: mx.array | None,
    temperature: float,
    accepted: int,
    accepted_effective: int,
    bonus_token: int | None,
    committed: list[int],
    terminated_by_stop_token: bool,
) -> None:
    """Per-round trace record for the U5 golden-trace comparison.

    Fields: draft tokens (full block), proposal length after confidence
    pruning, draft/target probabilities at the drafted positions, confidence
    logits, accepted length, bonus token, committed tokens. Probability
    gathers force evaluation, so tracing costs a little extra sync per round.
    """
    draft_probs_at: list[float] = []
    target_probs_at: list[float] = []
    if verify_logits is None:
        assert verify_norm_hidden is not None
        verify_logits = target.lm_head_logits(verify_norm_hidden)
    if proposal_len > 0:
        token_arr = mx.array(drafted[:proposal_len], dtype=mx.uint32)[None]
        if p_draft is None:
            p_draft = logits_to_probs(
                corrected_logits[:, :proposal_len, :], temperature
            )
        draft_probs_at = gather_token_probs(p_draft, token_arr)[0].tolist()
        if p_target is None:
            p_target = logits_to_probs(verify_logits, temperature)
        target_probs_at = gather_token_probs(p_target[:, :-1, :], token_arr)[0].tolist()
    # Top-k target logits per verify position (divergence gap analysis in the
    # U5 harness: a near-tie top-2 gap at a greedy divergence indicates a
    # dispatch-divergence tie-flip rather than a real bug).
    top_k = min(4, int(verify_logits.shape[-1]))
    round_logits = verify_logits[0].astype(mx.float32)
    part_ids = mx.argpartition(-round_logits, kth=top_k - 1, axis=-1)[:, :top_k]
    part_vals = mx.take_along_axis(round_logits, part_ids, axis=-1)
    order = mx.argsort(-part_vals, axis=-1)
    verify_top_ids = mx.take_along_axis(part_ids, order, axis=-1)
    verify_top_logits = mx.take_along_axis(part_vals, order, axis=-1)
    mx.eval(verify_top_ids, verify_top_logits)
    trace_hook(
        {
            "round": round_idx,
            "anchor_token": anchor_token,
            "draft_tokens": list(drafted),
            "proposal_len": proposal_len,
            "draft_probs_at_tokens": draft_probs_at,
            "target_probs_at_tokens": target_probs_at,
            "confidence_logits": (
                confidence_logits[0].tolist() if confidence_logits is not None else None
            ),
            "accepted_draft_tokens": (
                accepted_effective if terminated_by_stop_token else accepted
            ),
            "bonus_token": bonus_token,
            "committed_tokens": list(committed),
            "terminated_by_stop_token": terminated_by_stop_token,
            "verify_top_ids": verify_top_ids.tolist(),
            "verify_top_logits": verify_top_logits.tolist(),
        }
    )


def dspark_generate(
    target: LoadedTargetModel,
    draft: DSparkDraftModel,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: set[int],
    layer_ids: list[int],
    speculative_tokens: int | None = None,
    confidence_threshold: float = 0.0,
    verify_mode: str = "full",
    verify_chunk_size: int = 4,
    seed: int | None = None,
    profile: bool = False,
    trace_hook: TraceHook | None = None,
) -> tuple[list[int], dict[str, Any]]:
    final_event: DSparkRuntimeEvent | None = None
    for event in dspark_generate_stream(
        target=target,
        draft=draft,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
        layer_ids=layer_ids,
        speculative_tokens=speculative_tokens,
        confidence_threshold=confidence_threshold,
        verify_mode=verify_mode,
        verify_chunk_size=verify_chunk_size,
        seed=seed,
        profile=profile,
        trace_hook=trace_hook,
    ):
        if event.finished:
            final_event = event
    if final_event is None or final_event.metrics is None:
        raise RuntimeError("DSpark generation did not produce a final event.")
    return final_event.output_tokens, final_event.metrics
