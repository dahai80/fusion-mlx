# SPDX-License-Identifier: Apache-2.0
"""PR-P: shim full-chain tests (v2 doc §7 L2/L5).

Two gates over the landed shim ops (PR-C..PR-O):
- composition: MoE dispatch/matmul/combine output feeds the SSM parallel
  scan, compared against a float64 sequential recurrence on the CPU
  device (GPU fp32 matmul computes in fp16 — CPU is exact).
- memory: repeated shim-op iterations must not grow RSS after warmup
  (MemoryGrowthTracker from the PR-F harness) and must return active
  allocator memory after cache clear.
"""

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.shim.moe_dispatch import (
    gather_combine,
    gather_rows,
    mul_mat_id,
    route_dispatch,
)
from fusion_mlx.shim.ssm_scan import ssm_scan_parallel

_rng = np.random.default_rng(20260919)


def _run_cpu(fn, *args, **kwargs):
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        out = fn(*args, **kwargs)
        if isinstance(out, tuple):
            return tuple(mx.eval(o) or o for o in out)
        return mx.eval(out) or out
    finally:
        mx.set_default_device(prev)


def _moe_stage(x, weights, inds, scores):
    plan = route_dispatch(inds, weights.shape[0])
    xs = gather_rows(x, plan)
    ys = mul_mat_id(xs, weights, plan)
    return gather_combine(ys, scores, plan)


def _ssm_reference(x, A_log, B, C, D, dt_raw, dt_bias, state, limits):
    """Token-by-token Mamba recurrence in float64 (mirrors PR-O tests)."""
    dt = np.clip(np.logaddexp(0.0, dt_raw.astype(np.float64) + dt_bias), *limits)
    A = -np.exp(A_log.astype(np.float64))
    b, l, h, dh = x.shape
    g = B.shape[2]
    repeats = h // g
    S = np.zeros((b, h, dh, B.shape[3]), dtype=np.float64)
    y = np.zeros_like(x, dtype=np.float64)
    for t in range(l):
        dA = np.exp(A * dt[:, t])
        dBx = dt[:, t].astype(np.float64)[..., None] * x[:, t].astype(np.float64)
        for hh in range(h):
            gg = hh // repeats
            S[:, hh] = (
                dA[:, hh, None, None] * S[:, hh]
                + dBx[:, hh][..., None] * B[:, t, gg].astype(np.float64)[:, None, :]
            )
            y[:, t, hh] = (S[:, hh] @ C[:, t, gg].astype(np.float64)[..., None])[..., 0]
    y += x.astype(np.float64) * D.reshape(1, 1, h, 1)
    return y, S


class TestMoeSsmChain:
    def test_chain_matches_reference(self):
        n_tokens, top_k, num_experts, d = 4, 2, 3, 8
        x = _rng.normal(size=(n_tokens, d)).astype(np.float32)
        weights = _rng.normal(size=(num_experts, d, d)).astype(np.float32) * 0.3
        inds = _rng.integers(0, num_experts, size=(n_tokens, top_k)).astype(np.int32)
        scores = _rng.normal(size=(n_tokens, top_k)).astype(np.float32)

        combined = _run_cpu(
            _moe_stage, mx.array(x), mx.array(weights), mx.array(inds), mx.array(scores)
        )
        assert combined.dtype == mx.float32
        assert combined.shape == (n_tokens, d)

        # feed the MoE output into the SSM scan as a 1-batch sequence
        h, dh, g, ds = 2, 4, 2, 3
        seq = np.asarray(combined).reshape(1, n_tokens, h, dh)
        A_log = _rng.normal(size=(h,)).astype(np.float32) * 0.5
        D = _rng.normal(size=(h,)).astype(np.float32)
        dt_raw = _rng.normal(size=(1, n_tokens, h)).astype(np.float32)
        dt_bias = _rng.normal(size=(h,)).astype(np.float32)
        B = _rng.normal(size=(1, n_tokens, g, ds)).astype(np.float32)
        C = _rng.normal(size=(1, n_tokens, g, ds)).astype(np.float32)
        limits = (0.001, 100.0)

        y_shim, s_shim = _run_cpu(
            ssm_scan_parallel,
            mx.array(seq),
            mx.array(A_log),
            mx.array(B),
            mx.array(C),
            mx.array(D),
            mx.array(dt_raw),
            mx.array(dt_bias),
            None,
            limits,
            step=n_tokens,
        )
        y_ref, s_ref = _ssm_reference(
            seq, A_log, B, C, D, dt_raw, dt_bias, None, limits
        )
        np.testing.assert_allclose(np.array(y_shim), y_ref, rtol=2e-3, atol=1e-5)
        np.testing.assert_allclose(np.array(s_shim), s_ref, rtol=2e-3, atol=1e-5)


def _metal_ok():
    metal = getattr(mx, "metal", None)
    try:
        return bool(metal.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _metal_ok(), reason="memory gates need the real Metal device")
class TestMemoryNoLeak:
    def test_active_memory_returns_after_clear(self):
        x = _rng.normal(size=(16, 8)).astype(np.float32)
        weights = _rng.normal(size=(4, 8, 8)).astype(np.float32) * 0.3
        inds = _rng.integers(0, 4, size=(16, 2)).astype(np.int32)
        scores = _rng.normal(size=(16, 2)).astype(np.float32)
        seq = _rng.normal(size=(1, 16, 2, 4)).astype(np.float32)
        ssm_args = (
            mx.array(seq),
            mx.array(_rng.normal(size=(2,)).astype(np.float32) * 0.5),
            mx.array(_rng.normal(size=(1, 16, 2, 3)).astype(np.float32)),
            mx.array(_rng.normal(size=(1, 16, 2, 3)).astype(np.float32)),
            mx.array(_rng.normal(size=(2,)).astype(np.float32)),
            mx.array(_rng.normal(size=(1, 16, 2)).astype(np.float32)),
            mx.array(_rng.normal(size=(2,)).astype(np.float32)),
            None,
            (0.001, 100.0),
        )

        for _ in range(8):
            out = _moe_stage(
                mx.array(x), mx.array(weights), mx.array(inds), mx.array(scores)
            )
            mx.eval(out)
        mx.synchronize()
        mx.clear_cache()
        baseline = mx.get_active_memory()

        for _ in range(300):
            out = _moe_stage(
                mx.array(x), mx.array(weights), mx.array(inds), mx.array(scores)
            )
            mx.eval(out)
            y, s = ssm_scan_parallel(*ssm_args)
            mx.eval(y)
            mx.eval(s)
        mx.synchronize()
        mx.clear_cache()
        after = mx.get_active_memory()
        assert (
            after <= baseline * 1.05 + 1024
        ), f"active memory grew {baseline} -> {after} over 300 shim iterations"

    def test_rss_slope_bounded(self):
        from fusion_mlx.eval.golden_reference import MemoryGrowthTracker

        x = _rng.normal(size=(16, 8)).astype(np.float32)
        weights = _rng.normal(size=(4, 8, 8)).astype(np.float32) * 0.3
        inds = _rng.integers(0, 4, size=(16, 2)).astype(np.int32)
        scores = _rng.normal(size=(16, 2)).astype(np.float32)

        tracker = MemoryGrowthTracker(warmup_tokens=100, max_post_warmup_slope=4096.0)
        for i in range(400):
            out = _moe_stage(
                mx.array(x), mx.array(weights), mx.array(inds), mx.array(scores)
            )
            mx.eval(out)
            tracker.sample(i)
        report = tracker.report()
        assert report.post_warmup_slope_bytes_per_token < 4096.0, report.to_dict()
