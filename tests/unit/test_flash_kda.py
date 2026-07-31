# SPDX-License-Identifier: Apache-2.0
"""Tests for FlashKDA MLX port."""

from __future__ import annotations

import mlx.core as mx

from fusion_mlx.custom_kernels.flash_kda import fwd, metal_available
from fusion_mlx.custom_kernels.flash_kda.reference import fwd as fwd_ref


def _make_inputs(B=1, T=4, H=2, D=128):
    q = mx.random.normal((B, T, H, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, H, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, H, D)).astype(mx.bfloat16)
    g = mx.random.normal((B, T, H, D)).astype(mx.bfloat16) * 0.1
    beta = mx.random.normal((B, T, H)).astype(mx.bfloat16) * 0.1
    A_log = mx.zeros((H,), dtype=mx.float32)
    dt_bias = mx.zeros((H, D), dtype=mx.float32)
    return q, k, v, g, beta, A_log, dt_bias


class TestFlashKDAReference:
    def test_output_shape(self):
        q, k, v, g, beta, A_log, dt_bias = _make_inputs()
        out, state = fwd_ref(q, k, v, g, beta, scale=1.0, A_log=A_log, dt_bias=dt_bias)
        assert out.shape == (1, 4, 2, 128)
        assert state.shape == (1, 2, 128, 128)

    def test_dtypes(self):
        q, k, v, g, beta, A_log, dt_bias = _make_inputs()
        out, state = fwd_ref(q, k, v, g, beta, scale=1.0, A_log=A_log, dt_bias=dt_bias)
        assert out.dtype == mx.bfloat16
        assert state.dtype == mx.bfloat16

    def test_initial_state_propagated(self):
        q, k, v, g, beta, A_log, dt_bias = _make_inputs(T=1)
        init_state = mx.ones((1, 2, 128, 128), dtype=mx.bfloat16) * 0.5
        out_with, state_with = fwd_ref(
            q,
            k,
            v,
            g,
            beta,
            scale=1.0,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=init_state,
        )
        out_without, state_without = fwd_ref(
            q, k, v, g, beta, scale=1.0, A_log=A_log, dt_bias=dt_bias
        )
        mx.eval(out_with, state_with, out_without, state_without)
        assert not mx.allclose(out_with, out_without)

    def test_deterministic(self):
        q, k, v, g, beta, A_log, dt_bias = _make_inputs()
        out1, state1 = fwd_ref(
            q, k, v, g, beta, scale=1.0, A_log=A_log, dt_bias=dt_bias
        )
        out2, state2 = fwd_ref(
            q, k, v, g, beta, scale=1.0, A_log=A_log, dt_bias=dt_bias
        )
        mx.eval(out1, state1, out2, state2)
        assert mx.allclose(out1, out2)
        assert mx.allclose(state1, state2)


class TestFlashKDABridge:
    def test_bridge_returns_same_as_reference(self):
        q, k, v, g, beta, A_log, dt_bias = _make_inputs(T=2, H=1)
        out_bridge, state_bridge = fwd(
            q, k, v, g, beta, scale=1.0, A_log=A_log, dt_bias=dt_bias
        )
        out_ref, state_ref = fwd_ref(
            q, k, v, g, beta, scale=1.0, A_log=A_log, dt_bias=dt_bias
        )
        mx.eval(out_bridge, state_bridge, out_ref, state_ref)
        assert mx.allclose(out_bridge, out_ref)
        assert mx.allclose(state_bridge, state_ref)

    def test_metal_available_is_bool(self):
        assert isinstance(metal_available(), bool)
