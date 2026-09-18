# SPDX-License-Identifier: Apache-2.0
"""PR-O tests: shim parallel SSM scan vs sequential numpy reference + ssm_attn parity."""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")
mx = mlx

from fusion_mlx.shim.ssm_scan import (  # noqa: E402
    compute_dt_shim,
    is_ssm_scan_enabled,
    ssm_scan_parallel,
)

RNG = np.random.default_rng(20260918)
LIMITS = (0.001, 100.0)


def _inputs(b=2, l=10, h=6, dh=8, g=3, ds=5, with_state=False, step=4):
    x = RNG.normal(size=(b, l, h, dh)).astype(np.float32)
    dt_raw = RNG.normal(size=(b, l, h)).astype(np.float32)
    A_log = RNG.normal(size=(h,)).astype(np.float32)
    D = RNG.normal(size=(h,)).astype(np.float32)
    dt_bias = RNG.normal(size=(h,)).astype(np.float32)
    B = RNG.normal(size=(b, l, g, ds)).astype(np.float32)
    C = RNG.normal(size=(b, l, g, ds)).astype(np.float32)
    state = None
    if with_state:
        state = RNG.normal(size=(b, h, dh, ds)).astype(np.float32) * 0.1
    return x, A_log, B, C, D, dt_raw, dt_bias, state, step


def _sequential_reference(x, A_log, B, C, D, dt_raw, dt_bias, state):
    """Token-by-token Mamba SSM recurrence in float64 (ground truth)."""
    dt = np.clip(np.logaddexp(0.0, dt_raw.astype(np.float64) + dt_bias), *LIMITS)
    A = -np.exp(A_log.astype(np.float64))
    b, l, h, dh = x.shape
    g = B.shape[2]
    ds = B.shape[3]
    repeats = h // g
    S = (
        np.zeros((b, h, dh, ds), dtype=np.float64)
        if state is None
        else state.astype(np.float64).copy()
    )
    y = np.zeros((b, l, h, dh), dtype=np.float64)
    for t in range(l):
        dA = np.exp(A * dt[:, t])  # (b, h)
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


def _run_shim(inputs):
    x, A_log, B, C, D, dt_raw, dt_bias, state, step = inputs
    return ssm_scan_parallel(
        mx.array(x),
        mx.array(A_log),
        mx.array(B),
        mx.array(C),
        mx.array(D),
        mx.array(dt_raw),
        mx.array(dt_bias),
        None if state is None else mx.array(state),
        LIMITS,
        step=step,
    )


def _run_shim_cpu(inputs):
    """Run the shim on the CPU device: MLX GPU fp32 matmul computes in
    fp16 (~1e-3 relative error); CPU matmul is exact, so float64
    reference comparisons stay tight."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return _run_shim(inputs)
    finally:
        mx.set_default_device(prev)


class TestSequentialReference:
    def test_matches_sequential_no_state(self):
        inputs = _inputs(l=10, step=4)  # 3 chunks, pad=2, h=6 != chunk s=4
        y_shim, s_shim = _run_shim_cpu(inputs)
        y_ref, s_ref = _sequential_reference(*inputs[:7], inputs[6 + 1])
        np.testing.assert_allclose(np.array(y_shim), y_ref, rtol=2e-3, atol=1e-5)
        np.testing.assert_allclose(np.array(s_shim), s_ref, rtol=2e-3, atol=1e-5)

    def test_matches_sequential_with_state(self):
        inputs = _inputs(l=10, step=4, with_state=True)
        y_shim, s_shim = _run_shim_cpu(inputs)
        y_ref, s_ref = _sequential_reference(*inputs[:7], inputs[7])
        np.testing.assert_allclose(np.array(y_shim), y_ref, rtol=2e-3, atol=1e-5)
        np.testing.assert_allclose(np.array(s_shim), s_ref, rtol=2e-3, atol=1e-5)

    def test_single_chunk_matches_sequential(self):
        inputs = _inputs(l=6, step=256)  # no chunking, no padding
        y_shim, s_shim = _run_shim_cpu(inputs)
        y_ref, s_ref = _sequential_reference(*inputs[:7], inputs[7])
        np.testing.assert_allclose(np.array(y_shim), y_ref, rtol=2e-3, atol=1e-5)
        np.testing.assert_allclose(np.array(s_shim), s_ref, rtol=2e-3, atol=1e-5)

    def test_state_not_mutated(self):
        inputs = _inputs(l=10, step=4, with_state=True)
        state_before = inputs[7].copy()
        _run_shim(inputs)
        np.testing.assert_array_equal(inputs[7], state_before)


class TestSsmAttnParity:
    def test_ulp_parity_no_state(self):
        """Matching chunk shapes, no padding: 1-ulp parity with ssm_attn."""
        from mlx_lm.models.ssm import ssm_attn

        inputs = _inputs(l=8, step=4)  # pad=0, chunks s=4 on both sides
        x, A_log, B, C, D, dt_raw, dt_bias, _, _ = inputs
        y_attn, s_attn = ssm_attn(
            mx.array(x),
            mx.array(A_log),
            mx.array(B),
            mx.array(C),
            mx.array(D),
            mx.array(dt_raw),
            mx.array(dt_bias),
            None,
            LIMITS,
            step=4,
        )
        y_shim, s_shim = _run_shim(inputs)
        np.testing.assert_allclose(
            np.array(y_shim), np.array(y_attn), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(s_shim), np.array(s_attn), rtol=1e-5, atol=1e-6
        )

    def test_ulp_parity_single_chunk(self):
        from mlx_lm.models.ssm import ssm_attn

        inputs = _inputs(l=6, step=256)  # pad=0, single chunk
        x, A_log, B, C, D, dt_raw, dt_bias, _, _ = inputs
        y_attn, s_attn = ssm_attn(
            mx.array(x),
            mx.array(A_log),
            mx.array(B),
            mx.array(C),
            mx.array(D),
            mx.array(dt_raw),
            mx.array(dt_bias),
            None,
            LIMITS,
            step=256,
        )
        y_shim, s_shim = _run_shim(inputs)
        np.testing.assert_allclose(
            np.array(y_shim), np.array(y_attn), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(s_shim), np.array(s_attn), rtol=1e-5, atol=1e-6
        )

    def test_parity_no_state(self):
        from mlx_lm.models.ssm import ssm_attn

        inputs = _inputs(l=10, step=4)
        x, A_log, B, C, D, dt_raw, dt_bias, _, _ = inputs
        y_attn, s_attn = ssm_attn(
            mx.array(x),
            mx.array(A_log),
            mx.array(B),
            mx.array(C),
            mx.array(D),
            mx.array(dt_raw),
            mx.array(dt_bias),
            None,
            LIMITS,
            step=4,
        )
        y_shim, s_shim = _run_shim(inputs)
        np.testing.assert_allclose(
            np.array(y_shim), np.array(y_attn), rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(s_shim), np.array(s_attn), rtol=1e-3, atol=1e-5
        )

    def test_parity_with_state(self):
        from mlx_lm.models.ssm import ssm_attn

        inputs = _inputs(l=10, step=4, with_state=True)
        x, A_log, B, C, D, dt_raw, dt_bias, state, _ = inputs
        y_attn, s_attn = ssm_attn(
            mx.array(x),
            mx.array(A_log),
            mx.array(B),
            mx.array(C),
            mx.array(D),
            mx.array(dt_raw),
            mx.array(dt_bias),
            mx.array(state),
            LIMITS,
            step=4,
        )
        y_shim, s_shim = _run_shim(inputs)
        np.testing.assert_allclose(
            np.array(y_shim), np.array(y_attn), rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(s_shim), np.array(s_attn), rtol=1e-3, atol=1e-5
        )


class TestValidation:
    def test_x_not_4d_raises(self):
        with pytest.raises(ValueError):
            ssm_scan_parallel(
                mx.zeros((2, 4, 8)),
                mx.zeros(2),
                mx.zeros((2, 4, 3, 4)),
                mx.zeros((2, 4, 3, 4)),
                mx.zeros(2),
                mx.zeros((2, 4, 2)),
                mx.zeros(2),
                None,
                LIMITS,
            )

    def test_b_not_4d_raises(self):
        with pytest.raises(ValueError):
            ssm_scan_parallel(
                mx.zeros((1, 4, 2, 8)),
                mx.zeros(2),
                mx.zeros((1, 4, 3)),
                mx.zeros((1, 4, 3, 4)),
                mx.zeros(2),
                mx.zeros((1, 4, 2)),
                mx.zeros(2),
                None,
                LIMITS,
            )

    def test_step_below_one_raises(self):
        inputs = _inputs(l=4, step=0)
        with pytest.raises(ValueError):
            _run_shim(inputs)

    def test_heads_not_divisible_raises(self):
        x = RNG.normal(size=(1, 4, 5, 8)).astype(np.float32)
        B = RNG.normal(size=(1, 4, 2, 4)).astype(np.float32)
        with pytest.raises(ValueError):
            ssm_scan_parallel(
                mx.array(x),
                mx.zeros(5),
                mx.array(B),
                mx.array(B),
                mx.zeros(5),
                mx.zeros((1, 4, 5)),
                mx.zeros(5),
                None,
                LIMITS,
            )

    def test_mask_or_lengths_raise(self):
        x = mx.zeros((1, 4, 2, 8))
        b4 = mx.zeros((1, 4, 2, 4))
        with pytest.raises(ValueError):
            ssm_scan_parallel(
                x,
                mx.zeros(2),
                b4,
                b4,
                mx.zeros(2),
                mx.zeros((1, 4, 2)),
                mx.zeros(2),
                None,
                LIMITS,
                mask=mx.ones((1, 4)),
            )
        with pytest.raises(ValueError):
            ssm_scan_parallel(
                x,
                mx.zeros(2),
                b4,
                b4,
                mx.zeros(2),
                mx.zeros((1, 4, 2)),
                mx.zeros(2),
                None,
                LIMITS,
                lengths=mx.ones((1,), dtype=mx.int32),
            )


class TestSwitch:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("FUSION_SHIM_SSM", raising=False)
        assert is_ssm_scan_enabled() is False

    def test_on(self, monkeypatch):
        monkeypatch.setenv("FUSION_SHIM_SSM", "1")
        assert is_ssm_scan_enabled() is True

    def test_dt_transform_matches_mlx_lm(self):
        from mlx_lm.models.ssm import compute_dt

        dt_raw = RNG.normal(size=(2, 6, 4)).astype(np.float32)
        dt_bias = RNG.normal(size=(4,)).astype(np.float32)
        got = np.array(compute_dt_shim(mx.array(dt_raw), mx.array(dt_bias), LIMITS))
        want = np.array(compute_dt(mx.array(dt_raw), mx.array(dt_bias), LIMITS))
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
