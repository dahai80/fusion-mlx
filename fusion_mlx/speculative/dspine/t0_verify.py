# P2-2 (catch-up G1): DSpine T>0 sampled verification — capped-chain
# rejection sampling, MLX port of sglang_dspine/sampling/rejection.py.
# Offline math core (scoping doc P1): pure mx ops, no scheduler plumbing.
# Property: emitted token marginal equals the target distribution exactly
# (classical speculative rejection), given q rows are the same rows the
# draft chain sampled from.
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def _clean_probs(row: mx.array) -> mx.array:
    return mx.where((row >= 0.0) & (row <= 1.0), row, 0.0)


def _sample_from_dist(dist: mx.array, threshold: float) -> int:
    cdf = mx.cumsum(dist)
    total = float(cdf[-1])
    if total <= 0.0:
        return int(mx.argmax(dist))
    hit = int(mx.sum((cdf <= threshold * total).astype(mx.int32)))
    return min(hit, dist.shape[-1] - 1)


def sampled_capped_reject(
    candidates: mx.array,
    target_probs: mx.array,
    draft_probs: mx.array,
    verify_len: int,
    uniforms: mx.array,
    uniform_final: float,
) -> tuple[int, int, list[int]]:
    """One request's capped-chain rejection round.

    candidates: [T] draft token ids (candidates[0] is the pre-draft anchor).
    target_probs: [T, V] row i = target distribution at position i (the
        distribution that predicts candidates[i + 1]).
    draft_probs: [T - 1, V] row i = q row the draft chain sampled
        candidates[i + 1] from (coupled — carry from draft, never recompute).
    verify_len: cap (number of verified positions, <= T).
    uniforms: [T - 1] acceptance coins.
    uniform_final: bonus-sampling coin.

    Returns (accepted, bonus, emitted) where emitted is the committed
    token list [candidates[1], ..., candidates[accepted], bonus] of
    length accepted + 1.
    """
    T = int(candidates.shape[0])
    cap = min(int(verify_len), T, int(draft_probs.shape[0]) + 1)
    accepted = 0
    for step in range(1, cap):
        token = int(candidates[step])
        p = float(target_probs[step - 1, token])
        q = float(draft_probs[step - 1, token])
        coin = float(uniforms[step - 1])
        if (q > 0.0) and (q <= 1.0) and (coin * q < p):
            accepted += 1
        else:
            break

    p_row = target_probs[accepted]
    all_accepted = accepted == cap - 1
    if all_accepted:
        dist = _clean_probs(p_row)
    else:
        q_row = _clean_probs(draft_probs[accepted])
        dist = mx.maximum(p_row - q_row, 0.0)
        if float(mx.sum(dist)) <= 0.0:
            dist = _clean_probs(p_row)

    bonus = _sample_from_dist(dist, uniform_final)
    emitted = [int(candidates[i]) for i in range(1, accepted + 1)]
    emitted.append(bonus)
    return accepted, bonus, emitted


def acceptance_probability(p: float, q: float) -> float:
    """Analytic accept probability min(1, p/q) with q=0 semantics."""
    if q <= 0.0:
        return 1.0 if p > 0.0 else 0.0
    return min(1.0, p / q)


def emitted_marginal_single_position(
    target_probs: mx.array, draft_probs: mx.array
) -> mx.array:
    """Analytic single-position emitted distribution (V-vector).

    P(x) = q(x) * min(1, p(x)/q(x)) + P_reject * residual(x), where
    residual = (p - q)+ / mass, falling back to p when mass = 0.
    Used by tests to validate the sampler without sampling noise.
    """
    p = _clean_probs(target_probs)
    q = _clean_probs(draft_probs)
    accept = q * mx.minimum(1.0, p / mx.maximum(q, 1e-30))
    accept = mx.where(q > 0.0, accept, 0.0)
    residual = mx.maximum(p - q, 0.0)
    mass = float(mx.sum(residual))
    p_reject = 1.0 - float(mx.sum(accept))
    if mass > 0.0:
        return accept + p_reject * residual / mass
    return accept + p_reject * p / max(float(mx.sum(p)), 1e-30)
