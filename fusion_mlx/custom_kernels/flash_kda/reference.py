# SPDX-License-Identifier: Apache-2.0
"""FlashKDA Python reference implementation using MLX ops.

Implements the KDA recurrence in pure MLX:
  h_t = g_t * h_{t-1} + beta_t * (k_t ⊗ v_t)
  o_t = q_t^T * h_t

This is the correctness reference; the Metal kernel must match this
exactly. CHUNK=16 chunking mirrors the CUDA original's K1/K2 split.
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_CHUNK = 16


def _sigmoid(x: mx.array) -> mx.array:
    return mx.sigmoid(x)


def _compute_gate(
    g: mx.array,
    A_log: Optional[mx.array],
    dt_bias: Optional[mx.array],
    lower_bound: float,
) -> mx.array:
    """Compute gate values from g logits, A_log, dt_bias.

    g: [B, T, H, K] gate logits (bf16)
    A_log: [H] log-gate (fp32) or None
    dt_bias: [H, K] gate bias (fp32) or None
    lower_bound: clamp floor for gate values

    Returns: [B, T, H, K] gate values in (0, 1] after sigmoid + exp + clamp.
    """
    g_f = g.astype(mx.float32)

    if A_log is not None:
        A_log_expanded = A_log.reshape(1, 1, -1, 1)
        g_f = g_f + A_log_expanded

    if dt_bias is not None:
        g_f = g_f + dt_bias.reshape(1, 1, *dt_bias.shape)

    gate_sigmoid = _sigmoid(g_f)
    gate = mx.exp(-mx.exp(gate_sigmoid) * lower_bound)
    gate = mx.clip(gate, 0.0, 1.0)

    return gate


def fwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float = 1.0,
    A_log: Optional[mx.array] = None,
    dt_bias: Optional[mx.array] = None,
    lower_bound: float = -5.0,
    initial_state: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """FlashKDA forward — Python reference.

    Args:
        q: [B, T, H, K] query (bf16)
        k: [B, T, H, K] key (bf16)
        v: [B, T, H, V] value (bf16)
        g: [B, T, H, K] gate logits (bf16)
        beta: [B, T, H] beta logits (bf16)
        scale: qk scale factor
        A_log: [H] log-gate (fp32) or None
        dt_bias: [H, K] gate bias (fp32) or None
        lower_bound: gate lower bound
        initial_state: [B, H, V, K] or None

    Returns:
        (output, final_state)
        output: [B, T, H, V] bf16
        final_state: [B, H, V, K] bf16
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    gate = _compute_gate(g, A_log, dt_bias, lower_bound)
    beta_val = _sigmoid(beta.astype(mx.float32))

    q_f = q.astype(mx.float32) * scale
    k_f = k.astype(mx.float32)
    v_f = v.astype(mx.float32)

    if initial_state is not None:
        h = initial_state.astype(mx.float32)
    else:
        h = mx.zeros((B, H, V, K), dtype=mx.float32)

    outputs = []

    for t in range(T):
        g_t = gate[:, t, :, :]
        beta_t = beta_val[:, t, :]
        k_t = k_f[:, t, :, :]
        v_t = v_f[:, t, :, :]
        q_t = q_f[:, t, :, :]

        h = mx.expand_dims(g_t, 2) * h

        kv_t = mx.einsum("bhv,bhk->bhvk", v_t, k_t)
        beta_t_4d = beta_t.reshape(B, H, 1, 1)
        h = h + beta_t_4d * kv_t

        o_t = mx.einsum("bhk,bhvk->bhv", q_t, h)
        outputs.append(o_t)

    output = mx.stack(outputs, axis=1)
    final_state = h.astype(q.dtype)

    output = output.astype(q.dtype)
    return output, final_state
