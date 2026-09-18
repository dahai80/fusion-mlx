# SPDX-License-Identifier: Apache-2.0
"""fusion-mlx sampling utilities — mx.compile-free re-implementation of mlx-lm samplers.

mlx-lm 0.31.x decorates ``categorical_sampling`` and the apply_* helpers with
``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)``. In
the fusion-mlx server environment the decorator stops advancing the RNG state after
the first call: all subsequent samples reuse the same state, so identical
prompts produce character-identical output even at temperature > 1. Direct
calls to the underlying primitives advance the state correctly.

This module mirrors the mlx-lm implementation but drops the ``mx.compile``
wrappers, keeping behavior identical otherwise. ``make_sampler`` matches
``mlx_lm.sample_utils.make_sampler`` so it can replace the import in scheduler
without further changes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx


def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """Top-p (nucleus) filtering — keep the smallest set of tokens whose
    cumulative probability mass is at least ``top_p``."""
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


def apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """Min-p filtering — drop tokens with probability below
    ``max(p) * min_p``, while always keeping ``min_tokens_to_keep`` tokens."""
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )

    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p

    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove,
            top_indices,
            False,
            axis=-1,
        )

    return mx.where(tokens_to_remove, -float("inf"), logprobs)


def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Top-k filtering — keep only the ``top_k`` highest-probability tokens."""
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs


def apply_xtc(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: list[int],
) -> mx.array:
    """XTC sampling — with ``xtc_probability``, mask out all but the lowest
    above-threshold token to encourage diversity."""
    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(
            f"`threshold` has to be a float in the [0, 0.5] interval, but is {xtc_threshold}"
        )
    if not (0 <= xtc_probability <= 1.0):
        raise ValueError(
            f"`probability` has to be a float in the [0, 1] interval, but is {xtc_probability}"
        )

    probs = mx.softmax(logits, -1)
    mask = probs > mx.where(probs > xtc_threshold, probs, mx.inf).min()
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False

    return mx.where(
        mx.random.uniform(0, 1) > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


def categorical_sampling(logits: mx.array, temp: float) -> mx.array:
    """Sample a token id from the categorical distribution defined by
    ``logits / temp``. RNG state is advanced through ``mx.random.categorical``."""
    return mx.random.categorical(logits * (1 / temp))


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
) -> Callable[[mx.array], mx.array]:
    """Build a sampler callable matching ``mlx_lm.sample_utils.make_sampler``.

    Returns ``argmax`` when ``temp == 0``; otherwise composes optional
    top-p / min-p / xtc / top-k filters and finishes with categorical sampling.
    """
    # Guard against None values from callers that don't set defaults
    if temp is None:
        temp = 0.0
    if top_p is None:
        top_p = 1.0  # disable top-p filtering
    if min_p is None:
        min_p = 0.0
    if top_k is None:
        top_k = 0
    if xtc_probability is None:
        xtc_probability = 0.0
    if xtc_threshold is None:
        xtc_threshold = 0.1
    if temp == 0:
        sampler = lambda x: mx.argmax(x, axis=-1)
    else:
        sampling_methods = []
        if top_p > 0 and top_p < 1.0:
            sampling_methods.append(lambda x: apply_top_p(x, top_p))
        if min_p != 0.0:
            sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
        if xtc_probability > 0.0:
            sampling_methods.append(
                lambda x: apply_xtc(
                    x, xtc_probability, xtc_threshold, xtc_special_tokens
                )
            )
        if top_k > 0:
            sampling_methods.append(lambda x: apply_top_k(x, top_k))

        def sampler(logprobs: mx.array) -> mx.array:
            for method in sampling_methods:
                logprobs = method(logprobs)
            return categorical_sampling(logprobs, temp)

    # Expose sampling params on the returned callable so downstream code
    # (e.g. MTP acceptance check) can rebuild the filtered distribution
    # without re-plumbing the params through the BatchGenerator contract.
    # Lambda functions accept attribute assignment in CPython.
    sampler.temp = temp
    sampler.top_p = top_p
    sampler.min_p = min_p
    sampler.top_k = top_k
    sampler.min_tokens_to_keep = min_tokens_to_keep
    return sampler


# ---------------------------------------------------------------------------
# Stateful logits processors — DRY + Mirostat v2 (v2 doc §3.5 SamplerChain).
#
# These are CPU-state samplers: they track per-request state across decode
# steps (Mirostat's running mu; DRY's recent-token window). They plug into
# mlx_lm's BatchGenerator logits_processors list, signature:
#   processor(token_ids: list[int], logits: mx.array) -> mx.array
# The GPU-side transforms (top_k/top_p/min_p/temp/categorical) stay in the
# sampler closure built by make_sampler — the §3.5 CPU/GPU split.
# ---------------------------------------------------------------------------


def make_mirostat_v2_processor(
    tau: float, eta: float, vocab_size: int
) -> Callable[[list[int], mx.array], mx.array]:
    """Build a Mirostat v2 logits processor.

    Maintains a running ``mu`` (starts at 2*tau). Each step:
      1. Update mu from the previous token's surprise (err = surprise - mu;
         mu -= eta * err) using the logits cached from the prior step.
      2. Mask current logits to tokens whose surprise <= mu (i.e.
         p >= 2**-mu), always keeping the argmax.

    The base temperature / categorical sampling still runs in the downstream
    sampler closure; Mirostat only narrows the candidate set dynamically.

    When Mirostat is active the fused fast path is disabled (sched_token.py)
    so the non-fused make_sampler chain drives sampling.
    """
    if not (tau > 0.0):
        raise ValueError(f"mirostat_tau must be > 0, got {tau}")
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"mirostat_eta must be in (0, 1], got {eta}")

    state = {"mu": 2.0 * tau, "prev_logits": None}

    log2 = math.log(2.0)

    def processor(token_ids: list[int], logits: mx.array) -> mx.array:
        # Step 1: update mu from the previous step's chosen token.
        if state["prev_logits"] is not None and len(token_ids) > 0:
            prev_logits = state["prev_logits"]
            # The tail of token_ids is the token produced after the logits
            # cached last step — i.e. the token sampled from prev_logits
            # (or a forced stop token, which llama.cpp also feeds back).
            # Do NOT resync against a stored prev_token: the stored value
            # is one step older than token_ids[-1] and forcing it back
            # computes the surprise of a token that predates prev_logits,
            # corrupting mu on every step.
            chosen = token_ids[-1]
            prev_f = prev_logits.astype(mx.float32)
            logp = prev_f - mx.logsumexp(prev_f, axis=-1, keepdims=True)
            p_chosen = logp[..., chosen]
            surprise = float((-p_chosen / log2).item())
            err = surprise - state["mu"]
            state["mu"] -= eta * err
            state["mu"] = max(state["mu"], 0.0)

        # Step 2: mask current logits to the Mirostat candidate set.
        logits_f = logits.astype(mx.float32)
        logp = logits_f - mx.logsumexp(logits_f, axis=-1, keepdims=True)
        # surprise = -log2(p) = -logp / log2 ; keep where surprise <= mu
        threshold = -state["mu"] * log2  # keep logp >= threshold
        keep = logp >= threshold
        # Always keep the argmax (avoid empty candidate set).
        max_idx = mx.argmax(logits_f, axis=-1, keepdims=True)
        keep = mx.put_along_axis(keep, max_idx, mx.array(True), axis=-1)
        masked = mx.where(keep, logits, mx.array(-float("inf"), logits.dtype))

        # Cache for next step's mu update.
        state["prev_logits"] = logits
        return masked

    processor.mirostat_tau = tau  # type: ignore[attr-defined]
    processor.mirostat_eta = eta  # type: ignore[attr-defined]
    return processor


def make_dry_processor(
    multiplier: float,
    base: float = 1.75,
    allowed_length: int = 2,
    penalty_last_n: int = -1,
    breaker_ids: list[int] | None = None,
) -> Callable[[list[int], mx.array], mx.array]:
    """Build a DRY (Don't Repeat Yourself) logits processor.

    Penalizes tokens that would extend a recently-seen token sequence beyond
    ``allowed_length``. For each candidate next token, if appending it to the
    recent suffix produces a sequence that already appears in the trailing
    window, an additive penalty is applied:

        penalty = multiplier * base ** (match_len - allowed_length)

    where match_len is the length of the longest matching recent suffix.
    Matches do not cross breaker tokens (e.g. newline) — a breaker resets the
    sequence. ``penalty_last_n`` bounds the window (-1 = all, capped to 256
    for cost).

    Ported from llama.cpp dry_sampler semantics (v2 doc §3.5).
    """
    if multiplier <= 0.0:
        raise ValueError(f"dry_multiplier must be > 0, got {multiplier}")
    if base <= 1.0:
        raise ValueError(f"dry_base must be > 1.0, got {base}")
    if allowed_length < 1:
        raise ValueError(f"dry_allowed_length must be >= 1, got {allowed_length}")

    breakers = set(breaker_ids or [])
    window_cap = 256

    def processor(token_ids: list[int], logits: mx.array) -> mx.array:
        n = len(token_ids)
        if n < allowed_length:
            return logits
        if penalty_last_n >= 0:
            start = max(0, n - penalty_last_n)
        else:
            start = 0
        # Cap the window to bound the O(n * L) scan.
        if n - start > window_cap:
            start = n - window_cap
        window = token_ids[start:]

        # Build a per-position penalty for each candidate token.
        # For each suffix anchor j (the start of a candidate match in the
        # window), find the longest prefix of window[j:] that equals the
        # tail of window[:j] (i.e. a repeat). If that length >= allowed_length,
        # the token right after the match in the tail is penalized.
        penalties: dict[int, float] = {}
        wlen = len(window)
        # Compare suffix of window against earlier substrings.
        for match_len in range(min(wlen, 32), allowed_length - 1, -1):
            suffix = window[-match_len:]
            # Search for this suffix occurring earlier in the window
            # (not at the very end). If found, the token that would extend
            # the repeat is window[-match_len + match_len] == window[0] of
            # an extension — i.e. penalize the token following the matched
            # suffix at the earlier occurrence.
            for i in range(wlen - match_len):
                if window[i : i + match_len] == suffix:
                    # The extension token at the earlier occurrence:
                    ext_idx = i + match_len
                    if ext_idx < wlen:
                        ext_tok = window[ext_idx]
                    else:
                        continue
                    # No breaker inside the matched suffix.
                    if any(t in breakers for t in window[i:ext_idx]):
                        continue
                    penalty = multiplier * (base ** (match_len - allowed_length))
                    prev = penalties.get(ext_tok, 0.0)
                    if penalty > prev:
                        penalties[ext_tok] = penalty
                    # No break: llama.cpp penalizes the extension token
                    # after EVERY earlier occurrence, not just the first.
                    # penalties dict keeps the max per token.

        if not penalties:
            return logits

        # Apply additive penalties (logits are pre-softmax log-probs).
        idx = mx.array(list(penalties.keys()))
        vals = mx.array(list(penalties.values()), dtype=mx.float32)
        # mlx_lm passes logits as (1, vocab) per active sequence; handle
        # both 1D and 2D by shaping the index to match the last axis.
        if logits.ndim > 1:
            idx_shaped = idx.reshape(1, -1)
            cur = mx.take_along_axis(logits, idx_shaped, axis=-1)
            new_vals = cur.astype(mx.float32) - vals
            out = mx.put_along_axis(logits, idx_shaped, new_vals, axis=-1)
        else:
            cur = logits[idx]
            new_vals = cur.astype(mx.float32) - vals
            out = mx.put_along_axis(logits, idx, new_vals, axis=-1)
        return out

    processor.dry_multiplier = multiplier  # type: ignore[attr-defined]
    return processor
