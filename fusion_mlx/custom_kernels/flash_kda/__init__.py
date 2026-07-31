# SPDX-License-Identifier: Apache-2.0
"""FlashKDA — Kimi Delta Attention for MLX (Apple Silicon).

Metal port of FlashKDA (CUDA SM90+). Implements gated linear attention
recurrence: h_t = g_t * h_{t-1} + beta_t * (k_t ⊗ v_t), o_t = q_t^T * h_t

Constraint: K = V = 128 (matches CUDA original).

Two backends:
- Python reference (always available, correct but slow)
- Metal kernel (auto-selected when available, ~8-12x faster)

User instruction: "出了P2剩余工作，P3的工作也要全部落地，最后~/model/FlashKDA要移植到fusion-mlx"
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .metal_kernels import fwd as fwd_metal
from .metal_kernels import metal_available
from .reference import fwd as fwd_reference

logger = logging.getLogger(__name__)

__all__ = ["fwd", "metal_available"]

_CHUNK = 16


def fwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float = 1.0,
    A_log: mx.array | None = None,
    dt_bias: mx.array | None = None,
    lower_bound: float = -5.0,
    initial_state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """FlashKDA forward pass.

    Args:
        q: Query, bf16, shape [B, T, H, K].
        k: Key, bf16, shape [B, T, H, K].
        v: Value, bf16, shape [B, T, H, V].
        g: Gate logits (pre-sigmoid), bf16, shape [B, T, H, K].
        beta: Beta logits (pre-sigmoid), bf16, shape [B, T, H].
        scale: Scaling factor for qk.
        A_log: Log-gate parameter, fp32, shape [H].
        dt_bias: Gate bias, fp32, shape [H, K].
        lower_bound: Gate lower bound in [-5.0, 0].
        initial_state: Initial recurrent state, shape [B, H, V, K].

    Returns:
        Tuple of (output, final_state).
        output: bf16, shape [B, T, H, V].
        final_state: bf16, shape [B, H, V, K].
    """
    if metal_available():
        logger.debug("FlashKDA: using Metal kernel")
        return fwd_metal(
            q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, initial_state
        )
    logger.debug("FlashKDA: using Python reference")
    return fwd_reference(
        q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, initial_state
    )
