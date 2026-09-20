# SPDX-License-Identifier: Apache-2.0
"""Fused Conv3x3 + SafeGroupNorm + SiLU MSL kernel (#924).

Two-stage fused kernel via ``mx.fast.metal_kernel``. Replaces the 3-kernel
sequence (Conv -> GroupNorm -> SiLU) that ``mx.compile`` cannot fuse across
the conv/norm boundary. Stage 1 computes the 3x3 conv output and accumulates
FP32 GroupNorm sum/sum_sq per (batch, group) in one pass over the input;
stage 2 reads the conv output once, applies the FP32-derived affine normalize
+ SiLU, and writes the final fp16 output.

Design + correctness rationale: see issue #924. The candidate single-kernel
design in ``fusion-mlx-optimize-0920.md`` is infeasible (GroupNorm needs the
full H*W spatial extent, which exceeds threadgroup shared memory at 256^2)
and incorrect (hardcoded output channel 0, dropped affine gamma/beta, wrong
bias read). This two-stage structure is the smallest correct kernel that
removes the dominant conv-output global round trip.

Constraints: Conv kernel_size=3, stride=1, padding=1 (covers every Conv in
MuseTalk UNet + SD-VAE). NHWC layout. fp16 compute, fp32 GN stats.
"""

from __future__ import annotations

import logging

import mlx.core as mx

log = logging.getLogger(__name__)

_AVAILABLE: bool | None = None
_CONV_STATS_KERNEL = None
_APPLY_KERNEL = None


def _check_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        mk = getattr(mx.fast, "metal_kernel", None)
        _AVAILABLE = mk is not None
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


# Stage 1: conv + FP32 GN stats. grid=(B*num_groups), tg=(ch_per_group).
# Each thread owns one output channel in its (batch,group); loops H*W, computes
# the 3x3 conv for that channel, writes fp16 conv_out, accumulates fp32
# sum/sumsq. Tree-reduce across ch_per_group threads -> per-(batch,group) stats.
_CONV_STATS_SRC = """
const uint tg = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint b = tg / NUM_GROUPS;
const uint g = tg % NUM_GROUPS;
const uint c = g * CH_PER_GROUP + tid;
threadgroup float s_sum[256];
threadgroup float s_sq[256];
float sum = 0.0f;
float sumsq = 0.0f;
if (c < Cout) {
    float bc = float(bias[c]);
    for (uint hw = 0; hw < H * W; hw++) {
        uint h = hw / W;
        uint w = hw % W;
        float acc = bc;
        for (int kh = -1; kh <= 1; kh++) {
            int ih = int(h) + kh;
            if (ih < 0 || ih >= int(H)) continue;
            for (int kw = -1; kw <= 1; kw++) {
                int iw = int(w) + kw;
                if (iw < 0 || iw >= int(W)) continue;
                uint in_base = uint((b * H + ih) * W + iw) * Cin;
                uint w_base = uint((c * 3 + (kh + 1)) * 3 + (kw + 1)) * Cin;
                for (uint ic = 0; ic < Cin; ic++)
                    acc += float(x[in_base + ic]) * float(wt[w_base + ic]);
            }
        }
        conv_out[uint((b * H + h) * W + w) * Cout + c] = half(acc);
        sum += acc;
        sumsq += acc * acc;
    }
}
s_sum[tid] = sum;
s_sq[tid] = sumsq;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float gs = 0.0f, gq = 0.0f;
    for (uint i = 0; i < CH_PER_GROUP; i++) { gs += s_sum[i]; gq += s_sq[i]; }
    sums[b * NUM_GROUPS + g] = gs;
    sumsqs[b * NUM_GROUPS + g] = gq;
}
"""


# Stage 2: normalize + affine + SiLU. grid=(B*H*W), tg=(Cout).
# Each threadgroup = one pixel (b,h,w); thread tid = output channel c.
# Reads conv_out, looks up mean/inv_std for (b, c's group), gamma/beta, writes
# silu((v-mean)*inv_std*gamma+beta) as fp16.
_APPLY_SRC = """
const uint tg = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint b = tg / (H * W);
const uint hw = tg % (H * W);
const uint c = tid;
if (c < Cout) {
    uint g = c / CH_PER_GROUP;
    float v = float(conv_out[uint(b * H * W + hw) * Cout + c]);
    float m = means[b * NUM_GROUPS + g];
    float s = inv_stds[b * NUM_GROUPS + g];
    float xn = (v - m) * s;
    float y = xn * float(gamma[c]) + float(beta[c]);
    float sig = 1.0f / (1.0f + exp(-y));
    out[uint(b * H * W + hw) * Cout + c] = half(y * sig);
}
"""


def _kernel_cache_key(B, H, W, Cin, Cout, num_groups):
    ch_per_group = Cout // num_groups
    return (B, H, W, Cin, Cout, num_groups, ch_per_group)


def _get_kernels(B, H, W, Cin, Cout, num_groups):
    global _CONV_STATS_KERNEL, _APPLY_KERNEL
    ch_per_group = Cout // num_groups
    tmpl_stats = [
        ("CH_PER_GROUP", ch_per_group),
        ("H", H),
        ("W", W),
        ("Cin", Cin),
        ("Cout", Cout),
        ("NUM_GROUPS", num_groups),
    ]
    tmpl_apply = [
        ("Cout", Cout),
        ("NUM_GROUPS", num_groups),
        ("CH_PER_GROUP", ch_per_group),
        ("H", H),
        ("W", W),
    ]
    if _CONV_STATS_KERNEL is None:
        _CONV_STATS_KERNEL = mx.fast.metal_kernel(
            name="conv_stats",
            input_names=["x", "wt", "bias"],
            output_names=["conv_out", "sums", "sumsqs"],
            source=_CONV_STATS_SRC,
        )
        _APPLY_KERNEL = mx.fast.metal_kernel(
            name="gn_affine_silu",
            input_names=["conv_out", "means", "inv_stds", "gamma", "beta"],
            output_names=["out"],
            source=_APPLY_SRC,
        )
        log.info("[fused_cgs] compiled conv_stats + gn_affine_silu kernels (#924)")
    return _CONV_STATS_KERNEL, _APPLY_KERNEL, tmpl_stats, tmpl_apply, ch_per_group


def fused_conv_gn_silu(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    gamma: mx.array,
    beta: mx.array,
    num_groups: int,
    eps: float = 1e-6,
) -> mx.array | None:
    # Fused Conv3x3(stride=1,pad=1) + SafeGroupNorm(FP32 stats) + SiLU.
    # Returns None if unavailable / shape unsupported (caller falls back).
    if not _check_available():
        return None
    B, H, W, Cin = x.shape
    Cout = weight.shape[0]
    if weight.shape[1:] != (3, 3, Cin):
        return None
    if Cout % num_groups != 0:
        return None
    if bias is None:
        bias = mx.zeros((Cout,), dtype=weight.dtype)
    k_stats, k_apply, tmpl_stats, tmpl_apply, ch_per_group = _get_kernels(
        B, H, W, Cin, Cout, num_groups
    )
    try:
        conv_out, sums, sumsqs = k_stats(
            inputs=[x, weight, bias],
            template=tmpl_stats,
            grid=(B * num_groups * ch_per_group, 1, 1),
            threadgroup=(ch_per_group, 1, 1),
            output_shapes=[(B, H, W, Cout), (B, num_groups), (B, num_groups)],
            output_dtypes=[mx.float16, mx.float32, mx.float32],
        )
    except Exception as e:
        log.warning("[fused_cgs] conv_stats dispatch failed (%s); fallback", e)
        return None
    N = float(H * W * ch_per_group)
    means = (sums / N).astype(mx.float32)
    var = sumsqs / N - means * means
    inv_stds = mx.rsqrt(var + eps)
    try:
        (out,) = k_apply(
            inputs=[conv_out, means, inv_stds, gamma, beta],
            template=tmpl_apply,
            grid=(B * H * W * Cout, 1, 1),
            threadgroup=(Cout, 1, 1),
            output_shapes=[(B, H, W, Cout)],
            output_dtypes=[mx.float16],
        )
    except Exception as e:
        log.warning("[fused_cgs] gn_affine_silu dispatch failed (%s); fallback", e)
        return None
    return out
