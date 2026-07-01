"""Fused top-p + temperature sampler for the common chat sampler config.

Ported from oMLX/Rapid-MLX vllm_mlx/_sampler_fast_path.py.

mlx-lm's ``make_sampler`` builds a closure chain of independent
``@mx.compile``'d functions:

    sampler(logprobs) ->
        apply_top_p(logprobs, top_p)         # @mx.compile #1
        categorical_sampling(masked, temp)   # @mx.compile #2

This fuses them into one Python call and one lazy-graph segment,
avoiding the two @mx.compile boundaries that break lazy-graph fusion.
On Qwen 3.6 35B 4-bit @ B=1 on M3 Ultra this cuts step time ~30%.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

import logging

logger = logging.getLogger(__name__)


def is_fused_top_p_eligible(
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> bool:
    """Return True when the fused sampler is expected to win vs mlx-lm."""
    return temperature > 0.0 and min_p == 0.0 and 0.0 < top_p < 1.0


def make_fused_top_p_temp_sampler(
    temperature: float, top_p: float, top_k: int = 0
) -> Callable[[mx.array], mx.array]:
    """Build a sampler closure that fuses top-p / top-k / temperature /
    categorical sampling into one Python call and one lazy-graph segment.

    Math is identical to mlx-lm's ``apply_top_p`` -> ``apply_top_k`` ->
    ``categorical_sampling``. Index space differs: we sample in sorted
    space and map back via ``take_along_axis`` instead of building
    ``inverse_indices``.
    """
    if temperature <= 0.0:
        raise ValueError("fused sampler requires temperature > 0")
    if not (0.0 < top_p < 1.0):
        raise ValueError("fused sampler requires top_p in (0, 1)")
    use_top_k = top_k > 0

    temp_inv = 1.0 / float(temperature)
    one_minus_p = 1.0 - float(top_p)
    top_k_val = int(top_k)

    def sampler(logprobs: mx.array) -> mx.array:
        work = logprobs.astype(mx.float32) if logprobs.dtype != mx.float32 else logprobs
        probs = mx.exp(work)
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_logits = mx.take_along_axis(work, sorted_indices, axis=-1)

        vocab = sorted_logits.shape[-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        top_one_mask = mx.arange(vocab) == (vocab - 1)
        mask = (cumulative > one_minus_p) | top_one_mask
        if use_top_k:
            top_k_mask = mx.arange(vocab) >= (vocab - top_k_val)
            mask = (mask & top_k_mask) | top_one_mask

        masked_sorted = mx.where(
            mask,
            sorted_logits * temp_inv,
            -float("inf"),
        )
        sampled_pos = mx.random.categorical(masked_sorted)
        return mx.take_along_axis(
            sorted_indices, sampled_pos[..., None], axis=-1
        ).squeeze(-1)

    return sampler


# ---------------------------------------------------------------------------
# Shared batch sampler cache — keyed on (temperature, top_p, top_k, min_p).
# Single-writer (MLLMScheduler worker thread), no lock needed.
# ---------------------------------------------------------------------------
_shared_fused_sampler: tuple[tuple[float, float, int, float], Callable] | None = None


def get_or_create_fused_sampler(
    temperature: float, top_p: float, top_k: int = 0, min_p: float = 0.0
) -> Callable[[mx.array], mx.array] | None:
    """Return a fused sampler if eligible, or None to fall back to mlx-lm."""
    global _shared_fused_sampler
    if not is_fused_top_p_eligible(
        temperature=temperature, top_p=top_p, min_p=min_p, top_k=top_k
    ):
        return None
    key = (temperature, top_p, top_k, min_p)
    if _shared_fused_sampler is not None and _shared_fused_sampler[0] == key:
        return _shared_fused_sampler[1]
    fn = make_fused_top_p_temp_sampler(temperature, top_p, top_k)
    _shared_fused_sampler = (key, fn)
    logger.debug("Created fused sampler for key=%s", key)
    return fn
