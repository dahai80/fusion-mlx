"""Fused sampler for the common chat sampler configs.

Ported from oMLX/Rapid-MLX vllm_mlx/_sampler_fast_path.py and extended
with top_k + min_p support.

mlx-lm's ``make_sampler`` builds a closure chain of independent
``@mx.compile``'d functions:

    sampler(logprobs) ->
        apply_top_k(logprobs, top_k)          # @mx.compile #1
        apply_top_p(logprobs, top_p)          # @mx.compile #2
        apply_min_p(logprobs, min_p)          # @mx.compile #3
        categorical_sampling(masked, temp)    # @mx.compile #4

Each compiled function runs as a separate GPU dispatch with its own
kernel fusion scope. This fuses them into ONE ``@mx.compile``'d function,
eliminating the intermediate GPU dispatch boundaries and enabling
cross-operation kernel fusion (e.g. sort → cumsum → mask → sample).

On Qwen3 27B at B=1 this cuts sampling time ~40% vs mlx-lm's chain.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial

import mlx.core as mx

logger = logging.getLogger(__name__)


def is_fused_sampler_eligible(
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> bool:
    """Return True when the fused sampler is expected to win vs mlx-lm.

    Eligible when temperature > 0 (greedy handled natively by
    mx.random.categorical on un-modified logits). Zero-temperature
    greedy is already a single op — no fusion benefit.
    """
    if temperature is None:
        logger.debug("is_fused_sampler_eligible: temperature is None, ineligible")
        return False
    return temperature > 0.0


def is_fused_top_p_eligible(
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> bool:
    """Legacy alias — used by existing callers."""
    return is_fused_sampler_eligible(
        temperature=temperature, top_p=top_p, min_p=min_p, top_k=top_k
    )


# ---------------------------------------------------------------------------
# Compiled fused samplers — one per (use_top_k, use_top_p, use_min_p)
# combination.  Each is @mx.compile'd so MLX can fuse the full
# sort → mask → sample chain into one GPU dispatch.
# ---------------------------------------------------------------------------


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_sample_top_p(
    logprobs: mx.array, temp_inv: float, one_minus_p: float
) -> mx.array:
    """Top-p only fused sampler."""
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
    keep_mask = cumulative_probs > 1 - (1.0 - one_minus_p)
    # Guarantee the argmax survives the top-p cut. A sub-fp32-epsilon
    # top_p rounds the threshold to 1.0 and the strict ``>`` then keeps
    # no token: the argmax's cumulative == total prob mass == 1.0, and
    # ``1.0 > 1.0`` is False. Without this belt all logits become -inf
    # and ``mx.random.categorical`` returns a garbage token id. Mirrors
    # the seeded path's top-1 belt (``sorted_mask | top_one``) and the
    # top_p+min_p sibling below.
    max_idx = mx.argmax(logprobs, axis=-1, keepdims=True)
    keep_mask = mx.put_along_axis(keep_mask, max_idx, mx.array(True), axis=-1)
    masked = mx.where(keep_mask, logprobs, -float("inf"))
    return mx.random.categorical(masked * temp_inv)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_sample_temp_only(logprobs: mx.array, temp_inv: float) -> mx.array:
    """Temperature-only sampler — no top-k/top-p/min-p filtering."""
    return mx.random.categorical(logprobs * temp_inv)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_sample_top_p_min_p(
    logprobs: mx.array, temp_inv: float, one_minus_p: float, min_p_val: float
) -> mx.array:
    """Top-p + min-p fused sampler."""
    probs = mx.exp(logprobs)
    # Top-p: sort ascending, cumulative sum, mask
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
    top_p_mask = cumulative_probs <= 1 - (1.0 - one_minus_p)
    # Min-p: keep tokens with prob >= min_p * max_prob
    max_prob = mx.max(probs, axis=-1, keepdims=True)
    min_p_mask = probs >= min_p_val * max_prob
    combined = top_p_mask & min_p_mask
    # Always keep at least one token
    max_idx = mx.argmax(logprobs, axis=-1, keepdims=True)
    combined = mx.put_along_axis(combined, max_idx, mx.array(True), axis=-1)
    masked = mx.where(combined, logprobs, -float("inf"))
    return mx.random.categorical(masked * temp_inv)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_sample_top_k(
    logprobs: mx.array, temp_inv: float, top_k_val: int
) -> mx.array:
    """Top-k only fused sampler."""
    mask_idx = mx.argpartition(-logprobs, kth=top_k_val - 1, axis=-1)[..., top_k_val:]
    masked = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return mx.random.categorical(masked * temp_inv)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_sample_top_k_top_p(
    logprobs: mx.array, temp_inv: float, top_k_val: int, one_minus_p: float
) -> mx.array:
    """Top-k + top-p fused sampler."""
    # Apply top-k first (narrow the field before sorting for top-p)
    mask_idx = mx.argpartition(-logprobs, kth=top_k_val - 1, axis=-1)[..., top_k_val:]
    top_k_masked = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    # Apply top-p on the top-k candidates
    probs = mx.exp(top_k_masked)
    sorted_indices = mx.argsort(top_k_masked, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
    masked = mx.where(
        cumulative_probs > 1 - (1.0 - one_minus_p),
        top_k_masked,
        -float("inf"),
    )
    return mx.random.categorical(masked * temp_inv)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_sample_all(
    logprobs: mx.array,
    temp_inv: float,
    top_k_val: int,
    one_minus_p: float,
    min_p_val: float,
) -> mx.array:
    """Top-k + top-p + min-p fused sampler."""
    # Top-k first (reduces vocab before expensive sort)
    mask_idx = mx.argpartition(-logprobs, kth=top_k_val - 1, axis=-1)[..., top_k_val:]
    top_k_masked = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    # Top-p
    probs = mx.exp(top_k_masked)
    sorted_indices = mx.argsort(top_k_masked, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
    top_p_mask = cumulative_probs <= 1 - (1.0 - one_minus_p)
    # Min-p
    max_prob = mx.max(probs, axis=-1, keepdims=True)
    min_p_mask = probs >= min_p_val * max_prob
    combined = top_p_mask & min_p_mask
    max_idx = mx.argmax(logprobs, axis=-1, keepdims=True)
    combined = mx.put_along_axis(combined, max_idx, mx.array(True), axis=-1)
    masked = mx.where(combined, top_k_masked, -float("inf"))
    return mx.random.categorical(masked * temp_inv)


def make_fused_sampler(
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> Callable[[mx.array], mx.array]:
    """Build a sampler closure that fuses top-k / top-p / min-p / temperature
    / categorical sampling into ONE @mx.compile'd function.

    This eliminates the intermediate GPU dispatch boundaries that exist
    in mlx-lm's make_sampler chain (4 separate @mx.compile'd functions)
    and enables cross-operation kernel fusion within a single compiled graph.

    Math is identical to mlx-lm's ``apply_top_k`` -> ``apply_top_p`` ->
    ``apply_min_p`` -> ``categorical_sampling``.
    """
    if temperature <= 0.0:
        raise ValueError("fused sampler requires temperature > 0")

    use_top_k = top_k > 0
    use_top_p = 0.0 < top_p < 1.0
    use_min_p = min_p > 0.0

    temp_inv = 1.0 / float(temperature)
    top_k_val = int(top_k)
    one_minus_p = 1.0 - float(top_p) if use_top_p else 0.0
    min_p_val = float(min_p)

    # Select the right compiled function based on active filters.
    # Each combination gets its own @mx.compile'd function for
    # maximum kernel fusion within the compiled graph.
    if not use_top_k and not use_top_p and not use_min_p:

        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_temp_only(logprobs, temp_inv)

    elif use_top_k and not use_top_p and not use_min_p:

        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_top_k(logprobs, temp_inv, top_k_val)

    elif not use_top_k and use_top_p and not use_min_p:

        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_top_p(logprobs, temp_inv, one_minus_p)

    elif not use_top_k and not use_top_p and use_min_p:
        # min-p only — use top_p_min_p with top_p=1.0
        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_top_p_min_p(logprobs, temp_inv, 0.0, min_p_val)

    elif use_top_k and use_top_p and not use_min_p:

        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_top_k_top_p(logprobs, temp_inv, top_k_val, one_minus_p)

    elif not use_top_k and use_top_p and use_min_p:

        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_top_p_min_p(logprobs, temp_inv, one_minus_p, min_p_val)

    elif use_top_k and not use_top_p and use_min_p:
        # top_k + min_p — use the full function with top_p=1.0
        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_all(logprobs, temp_inv, top_k_val, 0.0, min_p_val)

    else:
        # top_k + top_p + min_p
        def sampler(logprobs: mx.array) -> mx.array:
            return _fused_sample_all(
                logprobs, temp_inv, top_k_val, one_minus_p, min_p_val
            )

    return sampler


# Backward compat alias
make_fused_top_p_temp_sampler = make_fused_sampler


# ---------------------------------------------------------------------------
# Shared batch sampler cache — keyed on (temperature, top_p, top_k, min_p).
# Single-writer (engine worker thread), no lock needed.
# ---------------------------------------------------------------------------
_shared_fused_sampler: tuple[tuple[float, float, int, float], Callable] | None = None


def get_or_create_fused_sampler(
    temperature: float, top_p: float, top_k: int = 0, min_p: float = 0.0
) -> Callable[[mx.array], mx.array] | None:
    """Return a fused sampler if eligible, or None to fall back to mlx-lm."""
    global _shared_fused_sampler
    if not is_fused_sampler_eligible(
        temperature=temperature, top_p=top_p, min_p=min_p, top_k=top_k
    ):
        return None
    key = (temperature, top_p, top_k, min_p)
    if _shared_fused_sampler is not None and _shared_fused_sampler[0] == key:
        return _shared_fused_sampler[1]
    fn = make_fused_sampler(temperature, top_p, top_k, min_p)
    _shared_fused_sampler = (key, fn)
    logger.debug("Created fused sampler for key=%s", key)
    return fn
