# SPDX-License-Identifier: Apache-2.0
# Perf record for the fused Conv+GroupNorm+SiLU kernel (#924).
# The kernel is CORRECT (parity >= 0.98, see test_fused_conv_gn_silu.py) but
# the conv stage is a naive scalar loop — NOT yet faster than MLX's optimized
# im2col+matmul conv2d compiled with the 3-op chain. This test records the
# current gap so a future tiled-conv rewrite can prove a speedup.
# Marked slow; not in the default CI fast path.

import os

import pytest

os.environ.setdefault("FUSION_FUSED_CONV_GN_SILU", "1")

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from fusion_mlx.custom_kernels.fused_conv_gn_silu import fused_conv_gn_silu
from fusion_mlx.nn_ext.safe_group_norm import SafeGroupNorm


def _bench(fused_fn, ref_fn, x, n=30):
    for _ in range(5):
        mx.eval(fused_fn())
        mx.eval(ref_fn(x))
    import time

    ts_f, ts_r = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fused_fn())
        ts_f.append((time.perf_counter() - t0) * 1000)
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(ref_fn(x))
        ts_r.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts_f)), float(np.mean(ts_r))


@pytest.mark.slow
def test_fused_perf_records_gap():
    # Records the current perf state. Does NOT assert a speedup — the naive
    # conv loop is slower than MLX's im2col conv today. A tiled-conv rewrite
    # must flip this ratio to merge.
    B, H, W, Cin, Cout, ng = 1, 32, 32, 320, 320, 32
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(B, H, W, Cin)).astype(np.float16) * 0.5)
    wt = mx.array(rng.normal(size=(Cout, 3, 3, Cin)).astype(np.float16) * 0.1)
    b = mx.array(rng.normal(size=(Cout,)).astype(np.float16) * 0.05)
    g = mx.array(np.ones((Cout,), dtype=np.float16))
    be = mx.array(np.zeros((Cout,), dtype=np.float16))
    conv = nn.Conv2d(Cin, Cout, 3, 1, 1, bias=True)
    conv.weight = mx.array(wt)
    conv.bias = mx.array(b)
    gn = SafeGroupNorm(ng, Cout, eps=1e-6, pytorch_compatible=True)
    gn.weight = mx.array(g)
    gn.bias = mx.array(be)
    ref_fn = mx.compile(lambda x: nn.silu(gn(conv(x))))
    fused_fn = lambda: fused_conv_gn_silu(x, wt, b, g, be, ng, eps=1e-6)
    if fused_fn() is None:
        pytest.skip("fused kernel unavailable")
    fm, rm = _bench(fused_fn, ref_fn, x)
    # Record: today the fused path is SLOWER. This assertion documents that
    # fact and will fail (correctly) once a tiled rewrite makes it faster.
    assert fm > rm, (
        f"fused {fm:.2f}ms should be > ref {rm:.2f}ms today (naive conv); "
        "if fused is now faster, flip this assertion to a speedup gate"
    )
