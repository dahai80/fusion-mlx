# SPDX-License-Identifier: Apache-2.0
"""PR-O: Mamba SSM_SCAN parallel prefix-scan prototype (v2 doc §2.8).

mlx_lm's `ssm_attn` (ssm.py) runs the SSD-form scan chunk-serially: each
256-token chunk produces its output AND carries the running state into
the next chunk, so chunk j's work cannot start before chunk j-1's state
exists. The Mamba-2 SSD form allows a 3-pass decomposition:

  pass 1 — every chunk's local output and local state depend only on
           in-chunk data: computable independently (large matmuls)
  pass 2 — a tiny prefix scan over the per-chunk states:
           S'_j = a_j * S'_{j-1} + S_j  (n_chunks iterations, cheap)
  pass 3 — inter-chunk outputs y_inter = decay * (C_j @ S'_{j-1})

This module implements that decomposition with pure mx ops. Semantics
match `ssm_attn` exactly for the mask-free prefill path: with matching
chunk shapes the outputs agree with `ssm_attn` to 1-2 ulp (GPU GEMM
last-ulp varies with buffer context; verified in tests); against a
float64 sequential reference the residual is bounded by MLX GPU fp32
matmul precision (Metal GEMM computes in fp16, ~1e-3 relative error;
CPU matmul is exact — reference tests therefore run on the CPU device).
mask/lengths are not supported — raise loudly rather than silently
disagreeing with ssm_attn.

Tier-2 conditional — only meaningful for Mamba/linear-attention models;
no default wiring. Degrade switch FUSION_SHIM_SSM (default OFF).
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_STEP = 256


def is_ssm_scan_enabled() -> bool:
    """FUSION_SHIM_SSM=1 enables the shim parallel scan where wired."""
    return os.environ.get("FUSION_SHIM_SSM", "0") == "1"


def compute_dt_shim(dt, dt_bias, time_step_limit):
    """Same dt transform as mlx_lm ssm.compute_dt (kept local for tests)."""
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def _chunk_segsum_decay(dtA_chunk):
    """exp(segsum) within a chunk: (b, h, s, s) lower-triangular decay."""
    s = dtA_chunk.shape[-1]
    x = mx.repeat(dtA_chunk[..., None], s, axis=-1)
    x = mx.tril(x, -1)
    return mx.exp(mx.cumsum(x, axis=-2))


def ssm_scan_parallel(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    time_step_limit: tuple[float, float] = (0.001, 100.0),
    step: int = DEFAULT_STEP,
    mask: mx.array | None = None,
    lengths: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Parallel 3-pass SSD scan. Signature mirrors mlx_lm.ssm.ssm_attn.

    x: (b, l, h, dh); dt: (b, l, h); A_log, D: (h,); B, C: (b, l, g, ds);
    state: (b, h, dh, ds) or None. Returns (y, final_state) with
    y (b, l, h, dh) and final_state (b, h, dh, ds).
    """
    if mask is not None or lengths is not None:
        raise ValueError(
            "ssm_scan_parallel: mask/lengths not supported in shim prototype"
        )
    if x.ndim != 4:
        raise ValueError(f"ssm_scan_parallel: x must be 4D, got {x.shape}")
    if B.ndim != 4 or C.ndim != 4:
        raise ValueError(
            f"ssm_scan_parallel: B/C must be 4D (b,l,g,ds), got {B.shape}/{C.shape}"
        )
    if step < 1:
        raise ValueError(f"ssm_scan_parallel: step must be >= 1, got {step}")
    b, l, h, dh = x.shape
    _, _, g, ds = B.shape
    if h % g != 0:
        raise ValueError(f"ssm_scan_parallel: heads {h} not divisible by groups {g}")

    dt = compute_dt_shim(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -mx.exp(A_log).astype(dt.dtype)
    dtA = dt * A.reshape(1, 1, -1)
    dtx = dt.reshape(b, l, h, 1) * x

    n_chunks = (l + step - 1) // step
    pad = n_chunks * step - l
    if pad:
        dtx = mx.concatenate([dtx, mx.zeros((b, pad, h, dh), dtx.dtype)], axis=1)
        # Pad dtA with 0 (identity decay), not -inf: exp(cumsum) then
        # carries prior contributions through the pad unchanged, so the
        # padded chunk's local state stays the state after the last real
        # token. -inf would zero every real contribution (exp(-inf)=0).
        dtA = mx.concatenate([dtA, mx.zeros((b, pad, h), dtA.dtype)], axis=1)
        B = mx.concatenate([B, mx.zeros((b, pad, g, ds), B.dtype)], axis=1)
        C = mx.concatenate([C, mx.zeros((b, pad, g, ds), C.dtype)], axis=1)

    # Pass 1: per-chunk local outputs and local states (independent).
    y_intra = []
    states = []
    chunk_decay_out = []
    in_chunk_cum = []
    for j in range(n_chunks):
        sl = slice(j * step, (j + 1) * step)
        dtA_c = dtA[:, sl].swapaxes(1, 2)
        B_c = mx.transpose(B[:, sl], (0, 2, 3, 1))
        C_c = mx.swapaxes(C[:, sl], 1, 2)

        CB = C_c @ B_c
        CB = mx.repeat(CB, repeats, axis=1)
        decay = _chunk_segsum_decay(dtA_c)
        y_intra.append(
            (mx.tril(CB * decay, 0) @ dtx[:, sl].swapaxes(1, 2)).swapaxes(1, 2)
        )
        # local state: only in-chunk contributions
        dtxdecay = (
            (dtx[:, sl] * decay[:, :, -1:, :].transpose(0, 3, 1, 2))
            .swapaxes(1, 2)
            .swapaxes(2, 3)
        )
        B_rep = mx.repeat(B_c, repeats, axis=1).swapaxes(2, 3)
        states.append(dtxdecay @ B_rep)
        cum = mx.exp(mx.cumsum(dtA[:, sl], axis=1))
        in_chunk_cum.append(cum)
        chunk_decay_out.append(cum[:, -1][:, :, None, None])

    # Pass 2: prefix scan over chunk states. states[j] is the state at
    # chunk end counting only in-chunk tokens; S'_j = a_j*S'_{j-1} + S_j.
    scanned = []
    carry = state
    for j in range(n_chunks):
        if carry is None:
            carry = states[j]
        else:
            carry = states[j] + chunk_decay_out[j] * carry
        scanned.append(carry)

    # Pass 3: inter-chunk contribution from S'_{j-1}.
    ys = []
    for j in range(n_chunks):
        C_c = C[:, j * step : (j + 1) * step]
        prev = scanned[j - 1] if j > 0 else state
        cum = in_chunk_cum[j]
        if prev is None:
            ys.append(y_intra[j])
            continue
        y_prev = (
            (prev.reshape(b, 1, g, repeats, dh, ds) @ C_c.reshape(b, -1, g, 1, ds, 1))
            .squeeze(-1)
            .flatten(2, 3)
        )
        ys.append(y_intra[j] + cum[..., None] * y_prev)

    y = mx.concatenate(ys, axis=1)
    if pad:
        y = y[:, :l]
    else:
        y = y
    final_state = scanned[-1]
    y = y + x * D.reshape(1, 1, h, 1)
    y = y.astype(x.dtype)
    logger.debug(
        "ssm_scan_parallel: b=%d l=%d h=%d dh=%d chunks=%d", b, l, h, dh, n_chunks
    )
    return y, final_state
