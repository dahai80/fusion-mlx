# SPDX-License-Identifier: Apache-2.0
"""Tests for fused INT4 dequant+GEMV -Ofast precompiled Metal kernel.

Parity verified vs mx.quantized_matmul:
  - K>=20480 precompiled path: relative diff < 4e-4 (fp16 output precision;
    abs diff ~0.1-0.2 at output magnitude ~450-560, < 1 fp16 ulp).
  - K<20480 native passthrough: exact 0.0.
  - batch>1 / bits!=4 / API unavailable: native passthrough, exact 0.0.

Precompiled path requires mx.fast.precompiled_metal_kernel (MLX fork, upstream
issue #4541). On stock PyPI MLX the API-detect check routes to native, so all
tests pass via the native fallback (CI-safe).
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.custom_kernels.fused_quant_gemv import (
    _precompiled_api_available,
    fused_dequant_gemv_int4,
    install_fused_quant_gemv_patch,
    is_fused_quant_gemv_enabled,
    uninstall_fused_quant_gemv_patch,
)

_HAS_METAL = mx.metal.is_available()
_HAS_PRECOMPILED = _precompiled_api_available()
_PRECOMPILED_AVAILABLE = _HAS_METAL and _HAS_PRECOMPILED


@pytest.fixture
def custom_on(monkeypatch):
    monkeypatch.setenv("FUSION_FUSED_QUANT_GEMV", "1")
    yield
    monkeypatch.delenv("FUSION_FUSED_QUANT_GEMV", raising=False)


@pytest.fixture
def custom_off(monkeypatch):
    monkeypatch.setenv("FUSION_FUSED_QUANT_GEMV", "0")
    yield
    monkeypatch.delenv("FUSION_FUSED_QUANT_GEMV", raising=False)


def test_default_on():
    assert is_fused_quant_gemv_enabled()


def test_disabled_off(custom_off):
    assert not is_fused_quant_gemv_enabled()


@pytest.mark.skipif(not _HAS_METAL, reason="Metal not available")
def test_small_k_native_passthrough(custom_on):
    mx.random.seed(0)
    K, M, gs = 8192, 2048, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((1, K)).astype(mx.float16)
    uninstall_fused_quant_gemv_patch()
    y_ref = lin(x)
    y_kern = fused_dequant_gemv_int4(
        lin["weight"], lin["scales"], lin["biases"], x, gs, 4
    )
    mx.eval(y_kern, y_ref)
    diff = float(mx.max(mx.abs(y_ref - y_kern)))
    assert diff < 1e-6, f"small-K native passthrough should be exact, diff {diff}"


@pytest.mark.skipif(not _PRECOMPILED_AVAILABLE, reason="precompiled API unavailable")
def test_precompiled_parity_large_k(custom_on):
    mx.random.seed(1)
    K, M, gs = 20480, 4096, 128
    w = mx.random.randint(0, 1 << 31, (M, K // 8)).astype(mx.uint32)
    ng = K // gs
    sc = mx.random.normal((M, ng)).astype(mx.float32) * 0.1
    bi = mx.random.normal((M, ng)).astype(mx.float32) * 0.01
    x = mx.random.normal((1, K)).astype(mx.float16)
    mx.eval(w, sc, bi, x)
    y_ref = mx.quantized_matmul(
        x,
        w,
        scales=sc,
        biases=bi,
        transpose=True,
        group_size=gs,
        bits=4,
        mode="affine",
    )
    y_kern = fused_dequant_gemv_int4(w, sc, bi, x, gs, 4)
    mx.eval(y_ref, y_kern)
    abs_diff = float(mx.max(mx.abs(y_ref - y_kern)))
    ref_max = float(mx.max(mx.abs(y_ref)))
    rel = abs_diff / ref_max if ref_max > 0 else abs_diff
    assert rel < 1e-3, f"precompiled parity rel {rel} (abs {abs_diff}, mag {ref_max})"


@pytest.mark.skipif(not _PRECOMPILED_AVAILABLE, reason="precompiled API unavailable")
def test_precompiled_parity_40960(custom_on):
    mx.random.seed(2)
    K, M, gs = 40960, 4096, 128
    w = mx.random.randint(0, 1 << 31, (M, K // 8)).astype(mx.uint32)
    ng = K // gs
    sc = mx.random.normal((M, ng)).astype(mx.float32) * 0.1
    bi = mx.random.normal((M, ng)).astype(mx.float32) * 0.01
    x = mx.random.normal((1, K)).astype(mx.float16)
    mx.eval(w, sc, bi, x)
    y_ref = mx.quantized_matmul(
        x,
        w,
        scales=sc,
        biases=bi,
        transpose=True,
        group_size=gs,
        bits=4,
        mode="affine",
    )
    y_kern = fused_dequant_gemv_int4(w, sc, bi, x, gs, 4)
    mx.eval(y_ref, y_kern)
    abs_diff = float(mx.max(mx.abs(y_ref - y_kern)))
    ref_max = float(mx.max(mx.abs(y_ref)))
    rel = abs_diff / ref_max if ref_max > 0 else abs_diff
    assert rel < 1e-3, f"precompiled parity rel {rel} (abs {abs_diff}, mag {ref_max})"


@pytest.mark.skipif(not _HAS_METAL, reason="Metal not available")
def test_batch_gt1_delegates(custom_on):
    mx.random.seed(3)
    K, M, gs = 20480, 2048, 128
    lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
    x = mx.random.normal((4, K)).astype(mx.float16)
    uninstall_fused_quant_gemv_patch()
    y = fused_dequant_gemv_int4(lin["weight"], lin["scales"], lin["biases"], x, gs, 4)
    mx.eval(y)
    y_ref = lin(x)
    mx.eval(y_ref)
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
    mx.eval(y_ref)
    diff = float(mx.max(mx.abs(y_ref - y)))
    assert diff < 1e-6


@pytest.mark.skipif(not _HAS_METAL, reason="Metal not available")
def test_patch_routes_large_k_decode(custom_on):
    if not _HAS_PRECOMPILED:
        pytest.skip("precompiled API unavailable — patch is no-op on stock MLX")
    install_fused_quant_gemv_patch()
    try:
        mx.random.seed(6)
        K, M, gs = 32768, 4096, 128
        lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
        x = mx.random.normal((1, K)).astype(mx.float16)
        y_patched = lin(x)
        uninstall_fused_quant_gemv_patch()
        y_native = lin(x)
        mx.eval([y_patched, y_native])
        abs_diff = float(mx.max(mx.abs(y_patched - y_native)))
        ref_max = float(mx.max(mx.abs(y_native)))
        rel = abs_diff / ref_max if ref_max > 0 else abs_diff
        assert rel < 1e-3, f"patched large-K parity rel {rel}"
    finally:
        uninstall_fused_quant_gemv_patch()


@pytest.mark.skipif(not _HAS_METAL, reason="Metal not available")
def test_patch_small_k_passthrough(custom_on):
    install_fused_quant_gemv_patch()
    try:
        mx.random.seed(7)
        K, M, gs = 4096, 2048, 128
        lin = nn.QuantizedLinear(input_dims=K, output_dims=M, bits=4, group_size=gs)
        x = mx.random.normal((1, K)).astype(mx.float16)
        y_patched = lin(x)
        uninstall_fused_quant_gemv_patch()
        y_native = lin(x)
        mx.eval([y_patched, y_native])
        diff = float(mx.max(mx.abs(y_patched - y_native)))
        assert diff < 1e-6, f"small-K should be exact passthrough, diff {diff}"
    finally:
        uninstall_fused_quant_gemv_patch()


@pytest.mark.skipif(not _HAS_METAL, reason="Metal not available")
def test_patch_noop_when_api_unavailable(custom_on, monkeypatch):
    if _HAS_PRECOMPILED:
        pytest.skip("precompiled API available — cannot test no-op path here")
    install_fused_quant_gemv_patch()
    try:
        from fusion_mlx.custom_kernels.fused_quant_gemv import _patch_installed

        assert not _patch_installed, "patch should be no-op when API unavailable"
    finally:
        uninstall_fused_quant_gemv_patch()
