# SPDX-License-Identifier: Apache-2.0
# Parity test for the fused Conv+GroupNorm+SiLU MSL kernel (#924).
# Reference: Conv -> SafeGroupNorm(FP32 stats) -> SiLU (3-op chain).
# Fused: mx.fast.metal_kernel two-stage. Gate: cosine >= 0.98 (PRD V2 tier-2).

import os

import mlx.core as mx
import numpy as np
import pytest

os.environ.setdefault("FUSION_FUSED_CONV_GN_SILU", "1")

from fusion_mlx.custom_kernels.fused_conv_gn_silu import fused_conv_gn_silu
from fusion_mlx.nn_ext.safe_group_norm import SafeGroupNorm


def _reference(x, w, b, gamma, beta, num_groups, eps):
    import mlx.nn as nn

    Cout = w.shape[0]
    conv = nn.Conv2d(
        in_channels=x.shape[-1],
        out_channels=Cout,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    conv.weight = mx.array(w)
    conv.bias = mx.array(b)
    gn = SafeGroupNorm(num_groups, Cout, eps=eps, pytorch_compatible=True)
    gn.weight = mx.array(gamma)
    gn.bias = mx.array(beta)
    y = conv(x)
    y = gn(y)
    return nn.silu(y)


def _csim(ref, fused):
    ref = np.array(ref).astype(np.float32).ravel()
    fused = np.array(fused).astype(np.float32).ravel()
    return float(
        np.dot(ref, fused) / (np.linalg.norm(ref) * np.linalg.norm(fused) + 1e-8)
    )


@pytest.mark.parametrize(
    "B,H,W,Cin,Cout,ng", [(1, 32, 32, 128, 320, 32), (1, 16, 16, 512, 512, 32)]
)
def test_fused_parity(B, H, W, Cin, Cout, ng):
    rng = np.random.default_rng(42)
    x = mx.array(rng.normal(size=(B, H, W, Cin)).astype(np.float16) * 0.5)
    w = mx.array(rng.normal(size=(Cout, 3, 3, Cin)).astype(np.float16) * 0.1)
    b = mx.array(rng.normal(size=(Cout,)).astype(np.float16) * 0.05)
    gamma = mx.array(rng.normal(size=(Cout,)).astype(np.float16) * 0.3 + 1.0)
    beta = mx.array(rng.normal(size=(Cout,)).astype(np.float16) * 0.1)
    eps = 1e-6
    ref = _reference(x, w, b, gamma, beta, ng, eps)
    fused = fused_conv_gn_silu(x, w, b, gamma, beta, ng, eps=eps)
    if fused is None:
        pytest.skip("fused kernel unavailable on this platform")
    mx.eval([ref, fused])
    cs = _csim(ref, fused)
    assert cs >= 0.98, f"cosine {cs:.4f} < 0.98"


def test_fused_affine_applied():
    # If gamma/beta were dropped (defect #2 in the candidate kernel), a
    # non-trivial gamma/beta would diverge from the reference. Force large
    # affine so a missing affine is obvious.
    B, H, W, Cin, Cout, ng = 1, 8, 8, 16, 32, 8
    rng = np.random.default_rng(7)
    x = mx.array(rng.normal(size=(B, H, W, Cin)).astype(np.float16) * 0.5)
    w = mx.array(rng.normal(size=(Cout, 3, 3, Cin)).astype(np.float16) * 0.1)
    b = mx.array(np.zeros((Cout,), dtype=np.float16))
    gamma = mx.array(np.full((Cout,), 3.0, dtype=np.float16))
    beta = mx.array(np.full((Cout,), 2.0, dtype=np.float16))
    ref = _reference(x, w, b, gamma, beta, ng, 1e-6)
    fused = fused_conv_gn_silu(x, w, b, gamma, beta, ng, eps=1e-6)
    if fused is None:
        pytest.skip("fused kernel unavailable")
    mx.eval([ref, fused])
    assert _csim(ref, fused) >= 0.98
