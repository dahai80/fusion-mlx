# SPDX-License-Identifier: Apache-2.0
"""Tests for fused INT4 dequant+GEMV Metal kernel.

Parity verified vs mx.quantized_matmul. Custom kernel is opt-in
(FUSION_FUSED_QUANT_GEMV=1); default delegates to native.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.custom_kernels.fused_quant_gemv import (
    fused_dequant_gemv_int4,
    is_fused_quant_gemv_enabled,
)


@pytest.fixture
def custom_on(monkeypatch):
    monkeypatch.setenv("FUSION_FUSED_QUANT_GEMV", "1")
    yield
    monkeypatch.delenv("FUSION_FUSED_QUANT_GEMV", raising=False)


@pytest.fixture
def custom_off(monkeypatch):
    monkeypatch.delenv("FUSION_FUSED_QUANT_GEMV", raising=False)


def test_default_off_delegates_to_native(custom_off):
    assert not is_fused_quant_gemv_enabled()
    mx.random.seed(42)
    lin = nn.QuantizedLinear(input_dims=64, output_dims=32, bits=4, group_size=32)
    x = mx.random.normal((1, 64))
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, 32, 4)
    mx.eval(y)
    y_ref = lin(x)
    diff = float(mx.max(mx.abs(y_ref[0] - y)))
    assert diff < 1e-5


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_custom_parity_small(custom_on):
    mx.random.seed(0)
    lin = nn.QuantizedLinear(input_dims=64, output_dims=32, bits=4, group_size=32)
    x = mx.random.normal((1, 64))
    y_ref = lin(x)
    y_kern = fused_dequant_gemv_int4(
        lin["weight"], lin["scales"], lin["biases"], x, 32, 4
    )
    mx.eval(y_kern)
    diff = float(mx.max(mx.abs(y_ref[0] - y_kern)))
    assert diff < 1e-4, f"parity diff {diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_custom_parity_qwen_shape(custom_on):
    mx.random.seed(1)
    K, M, gs = 4096, 11008, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((1, K))
    y_ref = lin(x)
    y_kern = fused_dequant_gemv_int4(
        lin["weight"], lin["scales"], lin["biases"], x, gs, 4
    )
    mx.eval(y_kern)
    diff = float(mx.max(mx.abs(y_ref[0] - y_kern)))
    assert diff < 1e-4, f"parity diff {diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_custom_batch_gt1_delegates(custom_on):
    mx.random.seed(2)
    lin = nn.QuantizedLinear(input_dims=64, output_dims=32, bits=4, group_size=32)
    x = mx.random.normal((4, 64))
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, 32, 4)
    mx.eval(y)
    y_ref = lin(x)
    diff = float(mx.max(mx.abs(y_ref - y)))
    assert diff < 1e-5


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_custom_k_gt_4096_delegates(custom_on):
    mx.random.seed(3)
    K, M, gs = 8192, 4096, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((1, K))
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, gs, 4)
    mx.eval(y)
    y_ref = lin(x)
    diff = float(mx.max(mx.abs(y_ref[0] - y)))
    assert diff < 1e-5


def test_bits_not_4_delegates(custom_off):
    mx.random.seed(4)
    lin = nn.QuantizedLinear(input_dims=64, output_dims=32, bits=8, group_size=32)
    x = mx.random.normal((1, 64))
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, 32, 8)
    mx.eval(y)
    y_ref = lin(x)
    diff = float(mx.max(mx.abs(y_ref[0] - y)))
    assert diff < 1e-5
