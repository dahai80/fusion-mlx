# SPDX-License-Identifier: Apache-2.0
"""Tests for fused INT4 dequant+GEMV Metal kernel (V2 simdgroup kernel).

Parity verified vs mx.quantized_matmul (max diff 4.8e-7, exact 0.0 across
seeds). V2 adopts native qmv_fast_impl's 3 optimizations (shift-elimination,
affine factoring, simdgroup layout) — closed the gap from prior NSX kernel
(+89% pure-GPU) to +11.4% eager per-op (production path) and PARITY in
compiled 32-layer chain at large K. Native still wins eager by ~11%
(instruction-scheduling edge on latency-bound op; tensor cores irrelevant
for batch=1 — native qmv_fast is scalar, not MMA). Default OFF, opt-in.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.custom_kernels.fused_quant_gemv import (
    fused_dequant_gemv_int4,
    install_fused_quant_gemv_patch,
    is_fused_quant_gemv_enabled,
    uninstall_fused_quant_gemv_patch,
)


@pytest.fixture
def custom_on(monkeypatch):
    monkeypatch.setenv("FUSION_FUSED_QUANT_GEMV", "1")
    yield
    monkeypatch.delenv("FUSION_FUSED_QUANT_GEMV", raising=False)


@pytest.fixture
def custom_off(monkeypatch):
    monkeypatch.delenv("FUSION_FUSED_QUANT_GEMV", raising=False)


def test_default_off(custom_off):
    assert not is_fused_quant_gemv_enabled()


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_custom_parity_small(custom_on):
    mx.random.seed(0)
    K, M, gs = 8192, 2048, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((1, K))
    uninstall_fused_quant_gemv_patch()
    y_ref = lin(x)
    y_kern = fused_dequant_gemv_int4(
        lin["weight"], lin["scales"], lin["biases"], x, gs, 4
    )
    mx.eval(y_kern)
    diff = float(mx.max(mx.abs(y_ref - y_kern)))
    assert diff < 1e-4, f"parity diff {diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_custom_parity_qwen_shape(custom_on):
    mx.random.seed(1)
    K, M, gs = 18944, 3584, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((1, K))
    uninstall_fused_quant_gemv_patch()
    y_ref = lin(x)
    y_kern = fused_dequant_gemv_int4(
        lin["weight"], lin["scales"], lin["biases"], x, gs, 4
    )
    mx.eval(y_kern)
    diff = float(mx.max(mx.abs(y_ref - y_kern)))
    assert diff < 1e-4, f"parity diff {diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_custom_parity_with_bias(custom_on):
    mx.random.seed(5)
    K, M, gs = 8192, 128, 128
    lin = nn.QuantizedLinear(
        input_dims=K, output_dims=M, bits=4, group_size=gs, bias=True
    )
    x = mx.random.normal((1, K))
    uninstall_fused_quant_gemv_patch()
    y_ref = lin(x)
    y_kern = (
        fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, gs, 4)
        + lin["bias"]
    )
    mx.eval(y_kern)
    diff = float(mx.max(mx.abs(y_ref - y_kern)))
    assert diff < 1e-4, f"parity diff {diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_small_k_delegates_to_native(custom_on):
    assert is_fused_quant_gemv_enabled()
    mx.random.seed(2)
    K, M, gs = 4096, 2048, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((1, K))
    uninstall_fused_quant_gemv_patch()
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, gs, 4)
    mx.eval(y)
    y_ref = lin(x)
    diff = float(mx.max(mx.abs(y_ref - y)))
    assert diff < 1e-6, f"small-K should be native passthrough (exact), diff {diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_batch_gt1_delegates(custom_on):
    mx.random.seed(3)
    K, M, gs = 8192, 2048, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((4, K))
    uninstall_fused_quant_gemv_patch()
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, gs, 4)
    mx.eval(y)
    y_ref = lin(x)
    diff = float(mx.max(mx.abs(y_ref - y)))
    assert diff < 1e-6, f"batch>1 should be native passthrough (exact), diff {diff}"


def test_bits_not_4_delegates(custom_off):
    mx.random.seed(4)
    K, M, gs = 8192, 2048, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=8, group_size=gs)
    x = mx.random.normal((1, K))
    uninstall_fused_quant_gemv_patch()
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, gs, 8)
    mx.eval(y)
    y_ref = lin(x)
    diff = float(mx.max(mx.abs(y_ref - y)))
    assert diff < 1e-6


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_patch_routes_large_k_decode(custom_on):
    install_fused_quant_gemv_patch()
    try:
        mx.random.seed(6)
        K, M, gs = 12288, 4096, 128
        lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
        x = mx.random.normal((1, K))
        y_patched = lin(x)
        uninstall_fused_quant_gemv_patch()
        y_native = lin(x)
        install_fused_quant_gemv_patch()
        mx.eval([y_patched, y_native])
        diff = float(mx.max(mx.abs(y_patched - y_native)))
        assert diff < 1e-4, f"patched large-K parity diff {diff}"
    finally:
        uninstall_fused_quant_gemv_patch()


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_patch_small_k_passthrough(custom_on):
    install_fused_quant_gemv_patch()
    try:
        mx.random.seed(7)
        K, M, gs = 2048, 2048, 128
        lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
        x = mx.random.normal((1, K))
        y_patched = lin(x)
        uninstall_fused_quant_gemv_patch()
        y_native = lin(x)
        install_fused_quant_gemv_patch()
        mx.eval([y_patched, y_native])
        diff = float(mx.max(mx.abs(y_patched - y_native)))
        assert diff == 0.0, f"small-K should be exact passthrough, diff {diff}"
    finally:
        uninstall_fused_quant_gemv_patch()
