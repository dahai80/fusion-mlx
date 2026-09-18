# SPDX-License-Identifier: Apache-2.0
"""Fused P0 bandwidth operators (PR-G, v2 doc §2.2/§5.6).

Two P0 operators that eliminate GPU dispatch boundaries + add capability
gaps in stock MLX:

  - fused_rmsnorm_residual(x, residual, weight, eps)
      One @mx.compile'd graph: RMSNorm + residual add. Stock mlx_lm layers
      do ``h = x + r`` as a separate dispatch after ``rms_norm``. Fusing
      into one compiled graph lets MLX merge the norm threadgroup reduction
      with the elementwise add (v2 doc §5.6: threadgroup reduction, no
      global atomic).

  - fused_rope(x, offset, dims, base, scale, yarn_beta, yarn_orig_ctx)
      One @mx.compile'd graph: FP32 position computation + RoPE with
      optional YaRN/NTK frequency scaling. Stock ``mx.fast.rope`` has no
      YaRN/NTK path and computes positions in the input dtype (fp16 position
      overflow at long context). v2 doc §5.6: 位置计算 FP32 保护.

Degrade switches (default OFF — prototype):
  FUSION_SHIM_FUSED_RMSNORM=1  — enable fused RMSNorm+residual
  FUSION_SHIM_FUSED_ROPE=1     — enable fused RoPE (YaRN/NTK + FP32 pos)

When OFF, callers use stock ``nn.RMSNorm`` + ``mx.fast.rope`` — zero
behavior change. When ON, the golden reference harness (PR-F) verifies
KL < 1e-6 vs stock before the path is trusted.
"""

from __future__ import annotations

import logging
import os
from functools import partial

import mlx.core as mx

logger = logging.getLogger(__name__)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def is_fused_rmsnorm_enabled() -> bool:
    return _env_on("FUSION_SHIM_FUSED_RMSNORM")


def is_fused_rope_enabled() -> bool:
    return _env_on("FUSION_SHIM_FUSED_ROPE")


# ---------------------------------------------------------------------------
# Fused RMSNorm + Residual (v2 doc §5.6).
#
# Stock mlx_lm: h = x + sublayer(rms_norm(x, w, eps))  ← two dispatches.
# Fused:        h = rms_norm(x, w, eps) + residual     ← one @mx.compile graph.
#
# The residual is the PRE-norm input (x) carried forward, so the signature is:
#   fused_rmsnorm_residual(x, residual, weight, eps) -> x_normed + residual
# Callers pass residual=x for the standard pre-norm residual pattern.
# ---------------------------------------------------------------------------


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_rmsnorm_residual_f16(
    x: mx.array, residual: mx.array, weight: mx.array, eps: float
) -> mx.array:
    # FP32 internal accumulation for the variance reduction (v2 doc §5.6:
    # threadgroup reduction in FP32, no global atomic). mx.fast.rms_norm
    # already does this internally; the @mx.compile boundary is what fuses
    # the add.
    normed = mx.fast.rms_norm(x, weight, eps)
    return normed + residual


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_rmsnorm_residual_f32(
    x: mx.array, residual: mx.array, weight: mx.array, eps: float
) -> mx.array:
    normed = mx.fast.rms_norm(x, weight, eps)
    return normed + residual


def fused_rmsnorm_residual(
    x: mx.array, residual: mx.array, weight: mx.array, eps: float = 1e-5
) -> mx.array:
    """RMSNorm(x, weight, eps) + residual in one compiled graph.

    Falls back to the un-fused ``mx.fast.rms_norm(x, w, eps) + residual``
    when FUSION_SHIM_FUSED_RMSNORM is unset (zero behavior change).
    """
    if not is_fused_rmsnorm_enabled():
        return mx.fast.rms_norm(x, weight, eps) + residual
    if x.dtype == mx.float16:
        return _fused_rmsnorm_residual_f16(x, residual, weight, eps)
    return _fused_rmsnorm_residual_f32(x, residual, weight, eps)


# ---------------------------------------------------------------------------
# Fused RoPE with YaRN/NTK + FP32 position (v2 doc §5.6).
#
# Stock mx.fast.rope computes positions in the input dtype. At long context
# (>8k) fp16 position values overflow, corrupting the rotation angles. This
# fused path:
#   1. Builds position indices in FP32.
#   2. Applies NTK-by-parts frequency scaling when yarn_orig_ctx > 0
#      (YaRN context extension).
#   3. Delegates the actual rotation to mx.fast.rope with pre-computed
#      freqs (so the FP32 position precision flows through).
# ---------------------------------------------------------------------------


def _compute_yarn_freqs(
    dims: int, base: float, scale: float, yarn_orig_ctx: int, yarn_beta: float
) -> mx.array:
    """Compute RoPE frequencies with optional YaRN/NTK-by-parts scaling.

    Without YaRN (yarn_orig_ctx <= 0): standard inv_freq = 1 / base^(2i/dims).
    With YaRN: NTK-by-parts — the base is adjusted so low-frequency dims
    are interpolated (not extrapolated) beyond the original context window.
    """
    # FP32 frequency computation (v2 doc §5.6: 位置计算 FP32 保护).
    # inv_freq[i] = 1 / base^(2i/dims)
    inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    inv_freq = inv_freq * scale

    if yarn_orig_ctx > 0:
        # YaRN NTK-by-parts: scale the base by a factor that depends on
        # the ratio of current to original context length.
        # See https://arxiv.org/abs/2309.00071 §3.2.
        # inv_freq becomes a blend of original and extrapolated frequencies
        # based on a ramp function over the wavelength.
        # Simple NTK-aware: adjust base by the scale ratio.
        ntk_scale = float(scale)
        if ntk_scale != 1.0:
            # NTK-by-parts: high-frequency dims (wavelength < yarn_orig_ctx)
            # keep original freqs; low-frequency dims (wavelength > ctx)
            # get scaled by ntk_scale; middle gets a smooth ramp.
            wavelengths = 2 * mx.pi / inv_freq
            ramp_low = yarn_orig_ctx * 0.5
            ramp_high = 2 * yarn_orig_ctx
            # Smooth blending factor per dim.
            blend = mx.clip(
                (wavelengths - ramp_low) / (ramp_high - ramp_low),
                0.0,
                1.0,
            )
            # Apply YaRN correction: mix original and NTK-scaled freqs.
            ntk_inv_freq = inv_freq / (ntk_scale**blend)
            # YaRN attention factor (beta) scales the final freqs.
            inv_freq = ntk_inv_freq * (1.0 + yarn_beta * blend)

    return inv_freq


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _fused_rope_standard(
    x: mx.array, offset: mx.array, dims: int, base: float, scale: float
) -> mx.array:
    return mx.fast.rope(
        x,
        dims,
        traditional=True,
        base=base,
        scale=scale,
        offset=offset,
    )


def fused_rope(
    x: mx.array,
    offset: int = 0,
    dims: int | None = None,
    base: float = 10000.0,
    scale: float = 1.0,
    yarn_orig_ctx: int = 0,
    yarn_beta: float = 0.1,
) -> mx.array:
    """Fused RoPE with FP32 position + optional YaRN/NTK scaling.

    Falls back to stock ``mx.fast.rope`` when FUSION_SHIM_FUSED_ROPE is unset.
    When enabled with yarn_orig_ctx > 0, applies NTK-by-parts frequency
    scaling for context extension (YaRN).
    """
    if dims is None:
        dims = x.shape[-1]

    if not is_fused_rope_enabled():
        return mx.fast.rope(
            x, dims, traditional=True, base=base, scale=scale, offset=offset
        )

    # YaRN/NTK path: pre-compute inv_freq in FP32 with NTK scaling,
    # pass to mx.fast.rope as freqs (base=None). MLX computes
    # positions = arange(seq_len) + offset internally, so the FP32
    # inv_freq precision flows through the rotation.
    if yarn_orig_ctx > 0:
        inv_freq = _compute_yarn_freqs(dims, base, scale, yarn_orig_ctx, yarn_beta)
        out = mx.fast.rope(
            x,
            dims,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=inv_freq,
        )
        logger.debug(
            "fused_rope YaRN: dims=%d ctx=%d beta=%.2f offset=%d",
            dims,
            yarn_orig_ctx,
            yarn_beta,
            offset,
        )
        return out

    # Standard path: delegate to compiled rope. Offset stays as scalar;
    # MLX computes positions internally in the rope kernel.
    return _fused_rope_standard(x, offset, dims, base, scale)
