"""Distribution-preserving speculative sampling primitives.

Mirrors the PyTorch reference exactly:
- refs/deepspec/deepspec/utils/sampling.py (``logits_to_probs``,
  ``sample_from_probs``, ``gather_token_probs``, ``sample_residual``)
- refs/deepspec/deepspec/eval/base_evaluator.py ``verify_draft_tokens``
  (acceptance math: accept x_k w.p. min(1, p_t(x_k)/p_d(x_k)) with draft
  probs clamped at 1e-8; on rejection sample the bonus token from
  normalize(clamp(p_t - p_d, 0)) with fallback to p_t when the residual
  mass is <= 1e-8; when everything is accepted, sample the bonus from p_t
  at the last verified position).

Temperature semantics follow the reference: ``temperature < 1e-5`` means
greedy — probabilities collapse to a float32 one-hot at the argmax, so the
acceptance math degenerates to exact argmax matching with no randomness.

Key-splitting scheme (reproducibility)
--------------------------------------
Callers derive a root key from an integer seed via :class:`KeySequence`.
Every random *event* consumes exactly one subkey, split off the rolling
root key in a fixed order:

1. the first-token sample after prefill,
2. per decode round: ``block_size`` draft-token samples (one key per
   serial Markov step, pre-split before the loop),
3. per decode round: one key passed to :func:`speculative_accept`, which
   internally splits it once into ``(u_key, bonus_key)`` — ``u_key`` draws
   the ``[1, L]`` acceptance uniforms in a single call and ``bonus_key``
   draws the bonus token (residual or target sample).

Because the acceptance uniforms are drawn from ``u_key`` in one shot and do
not depend on target logits, the chunked lazy-logits verifier consumes keys
identically to the full-logits verifier and makes bit-identical decisions.
With ``seed=None`` every key is ``None`` and MLX's global RNG stream is
used instead (set ``mx.random.seed`` for coarse reproducibility).
"""

from __future__ import annotations

import mlx.core as mx

# Reference constants (verify_draft_tokens / sample_residual / sampling.py).
DRAFT_PROB_CLAMP = 1e-8
RESIDUAL_EPS = 1e-8
GREEDY_TEMP_EPS = 1e-5


class KeySequence:
    """Deterministic stream of mx.random subkeys derived from a seed.

    ``next()`` splits the rolling root key and returns a fresh subkey.
    ``split(n)`` returns ``n`` subkeys via repeated ``next()`` so the
    consumption order is stable regardless of how callers batch requests.
    With ``seed=None`` all methods return ``None`` (global RNG stream).
    """

    def __init__(self, seed: int | None = None):
        self._key = None if seed is None else mx.random.key(seed)

    def next(self) -> mx.array | None:
        if self._key is None:
            return None
        self._key, subkey = mx.random.split(self._key)
        return subkey

    def split(self, num: int) -> list[mx.array | None]:
        return [self.next() for _ in range(num)]


def logits_to_probs(logits: mx.array, temperature: float) -> mx.array:
    """float32 probabilities; temp < 1e-5 -> one-hot at the argmax.

    Mirrors ``deepspec.utils.sampling.logits_to_probs``.
    """
    if temperature < GREEDY_TEMP_EPS:
        indices = mx.argmax(logits, axis=-1, keepdims=True)
        return mx.put_along_axis(
            mx.zeros(logits.shape, dtype=mx.float32),
            indices,
            mx.array(1.0, dtype=mx.float32),
            axis=-1,
        )
    return mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)


def sample_from_logits(
    logits: mx.array,
    temperature: float,
    key: mx.array | None = None,
) -> mx.array:
    """Sample token ids from logits ``[..., V]`` -> ``[...]`` uint32.

    temp < 1e-5 -> argmax; otherwise categorical over
    ``softmax(logits / temperature)`` (Gumbel trick on scaled logits, which
    is distributionally identical to the reference's
    ``multinomial(softmax(logits / temperature))``).
    """
    if temperature < GREEDY_TEMP_EPS:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    scaled = logits.astype(mx.float32) / temperature
    return mx.random.categorical(scaled, key=key).astype(mx.uint32)


def sample_from_probs(probs: mx.array, key: mx.array | None = None) -> mx.array:
    """Sample token ids from probabilities ``[..., V]`` -> ``[...]`` uint32.

    Zero-probability entries map to ``log(0) = -inf`` which the Gumbel
    argmax never selects.
    """
    return mx.random.categorical(mx.log(probs), key=key).astype(mx.uint32)


def gather_token_probs(probs: mx.array, token_ids: mx.array) -> mx.array:
    """probs ``[..., L, V]`` gathered at token_ids ``[..., L]`` -> ``[..., L]``."""
    return mx.take_along_axis(probs, token_ids[..., None], axis=-1).squeeze(-1)


def sample_residual(
    target_probs: mx.array,
    draft_probs: mx.array,
    key: mx.array | None = None,
) -> mx.array:
    """Bonus-token distribution on rejection (reference ``sample_residual``).

    residual = clamp(p_t - p_d, min=0); rows whose residual mass is
    <= 1e-8 fall back to p_t; the result is normalized with the mass
    clamped at 1e-8 before division. Shapes: ``[..., V]`` -> ``[...]``.
    """
    target_probs = target_probs.astype(mx.float32)
    draft_probs = draft_probs.astype(mx.float32)
    residual = mx.maximum(target_probs - draft_probs, 0.0)
    residual_mass = residual.sum(axis=-1, keepdims=True)
    residual = mx.where(residual_mass <= RESIDUAL_EPS, target_probs, residual)
    residual_mass = residual.sum(axis=-1, keepdims=True)
    residual = residual / mx.maximum(residual_mass, DRAFT_PROB_CLAMP)
    return sample_from_probs(residual, key=key)


def speculative_accept(
    draft_tokens: list[int],
    p_draft: mx.array | None,
    p_target: mx.array,
    temperature: float,
    key: mx.array | None = None,
) -> tuple[int, int]:
    """Reference rejection sampling over one drafted block.

    Args:
        draft_tokens: the L drafted token ids (may be empty — the reference
            ``draft_token_count=0`` degenerate path).
        p_draft: ``[1, L, V]`` draft probabilities at the drafted positions
            (post-Markov-bias, post-temperature). ``None`` iff ``L == 0``.
        p_target: ``[1, L+1, V]`` target probabilities over the verify block
            ``[anchor, x_1..x_L]``.
        temperature: sampling temperature; < 1e-5 skips the uniform draws
            (one-hot probs make every accept probability exactly 0 or 1).
        key: RNG key, split once into ``(u_key, bonus_key)``; see the module
            docstring for the key scheme.

    Returns:
        ``(accepted_len, bonus_token)`` — ``accepted_len`` drafted tokens are
        accepted and ``bonus_token`` is the extra committed token (residual
        sample on rejection, target sample at the last position otherwise).
    """
    draft_tokens = [int(token) for token in draft_tokens]
    num_draft = len(draft_tokens)
    if key is not None:
        u_key, bonus_key = mx.random.split(key)
    else:
        u_key = bonus_key = None

    if num_draft == 0:
        bonus = sample_from_probs(p_target[:, -1, :], key=bonus_key)
        mx.eval(bonus)
        return 0, int(bonus.item())

    assert p_draft is not None, "p_draft is required when draft_tokens is non-empty"
    token_arr = mx.array(draft_tokens, dtype=mx.uint32)[None]
    selected_target = gather_token_probs(p_target[:, :-1, :], token_arr)
    selected_draft = mx.maximum(
        gather_token_probs(p_draft, token_arr), DRAFT_PROB_CLAMP
    )
    accept_prob = mx.minimum(selected_target / selected_draft, 1.0)
    if temperature < GREEDY_TEMP_EPS:
        # One-hot probs: accept probability is exactly 0 or 1; skip the draws.
        accept = accept_prob >= 1.0
    else:
        uniforms = mx.random.uniform(shape=(1, num_draft), key=u_key)
        accept = uniforms < accept_prob
    mx.eval(accept)

    accepted_len = 0
    for flag in accept[0].tolist():
        if not flag:
            break
        accepted_len += 1

    if accepted_len < num_draft:
        bonus = sample_residual(
            p_target[:, accepted_len, :],
            p_draft[:, accepted_len, :],
            key=bonus_key,
        )
    else:
        bonus = sample_from_probs(p_target[:, -1, :], key=bonus_key)
    mx.eval(bonus)
    return accepted_len, int(bonus.item())
