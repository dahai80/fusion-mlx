# SPDX-License-Identifier: Apache-2.0
"""#911: SafeGroupNorm + graph_opt fusion tests.

MLX uses channels-last (NHWC) layout: conv output (N,H,W,C), GroupNorm normalizes
the last axis (C). Tests follow that convention.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.graph_opt import ConvGroupNormSiLU, fused_conv_groupnorm_silu
from fusion_mlx.nn_ext import SafeGroupNorm, safe_group_norm


def test_safe_group_norm_output_shape_4d():
    norm = SafeGroupNorm(32, 64, pytorch_compatible=True)
    x = mx.random.uniform(0, 1, (2, 16, 16, 64))  # NHWC
    out = norm(x)
    assert out.shape == (2, 16, 16, 64)


def test_safe_group_norm_preserves_dtype_fp16():
    norm = SafeGroupNorm(4, 8, pytorch_compatible=True)
    x = mx.random.uniform(0, 1, (1, 4, 4, 8)).astype(mx.float16)
    out = norm(x)
    assert out.dtype == mx.float16


def test_safe_group_norm_reduces_to_mean_zero_unit_var():
    norm = SafeGroupNorm(4, 8, eps=1e-6, pytorch_compatible=True)
    x = mx.random.uniform(5, 10, (1, 8, 8, 8))
    out = norm(x)
    mx.eval(out)
    m = out.reshape(1, 8, 8, 4, 2).mean(axis=(1, 2, 4), keepdims=True)
    mx.eval(m)
    assert mx.abs(m).max().item() < 0.1


def test_safe_group_norm_matches_stock_groupnorm_fp32():
    channels, groups = 32, 8
    stock = nn.GroupNorm(groups, channels, pytorch_compatible=True, eps=1e-6)
    safe = SafeGroupNorm(groups, channels, eps=1e-6, pytorch_compatible=True)
    safe.weight = stock.weight
    safe.bias = stock.bias
    x = mx.random.uniform(-2, 2, (1, 8, 8, channels))
    out_stock = stock(x)
    out_safe = safe(x)
    mx.eval(out_stock, out_safe)
    assert mx.allclose(out_stock, out_safe, atol=1e-4).item()


def test_safe_group_norm_rejects_bad_groups():
    with pytest.raises(ValueError, match="divisible"):
        SafeGroupNorm(3, 10)


def test_safe_group_norm_factory():
    norm = safe_group_norm(8, 32)
    assert isinstance(norm, SafeGroupNorm)
    assert norm.pytorch_compatible is True
    out = norm(mx.random.uniform(0, 1, (1, 4, 4, 32)))
    assert out.shape == (1, 4, 4, 32)


def test_conv_groupnorm_silu_fusion_output_shape():
    conv = nn.Conv2d(8, 16, 3, padding=1)
    gn = nn.GroupNorm(4, 16, pytorch_compatible=True)
    fused = ConvGroupNormSiLU(conv, gn)
    x = mx.random.uniform(0, 1, (1, 8, 8, 8))  # NHWC
    out = fused(x)
    assert out.shape == (1, 8, 8, 16)


def test_conv_groupnorm_silu_matches_unfused_fp32():
    conv = nn.Conv2d(8, 16, 3, padding=1)
    gn = nn.GroupNorm(4, 16, pytorch_compatible=True, eps=1e-6)
    fused = ConvGroupNormSiLU(conv, gn)
    x = mx.random.uniform(0, 1, (1, 8, 8, 8))
    ref = nn.silu(gn(conv(x)))
    out = fused(x)
    mx.eval(ref, out)
    assert mx.allclose(ref, out, atol=1e-4).item()


def test_conv_groupnorm_silu_upgrades_to_safe_groupnorm():
    conv = nn.Conv2d(8, 16, 3, padding=1)
    stock_gn = nn.GroupNorm(4, 16, pytorch_compatible=True)
    fused = ConvGroupNormSiLU(conv, stock_gn)
    assert isinstance(fused.groupnorm, SafeGroupNorm)
    assert mx.allclose(fused.groupnorm.weight, stock_gn.weight).item()
    assert mx.allclose(fused.groupnorm.bias, stock_gn.bias).item()


def test_fused_conv_groupnorm_silu_factory():
    conv = nn.Conv2d(4, 8, 1)
    gn = nn.GroupNorm(2, 8, pytorch_compatible=True)
    fused = fused_conv_groupnorm_silu(conv, gn)
    assert isinstance(fused, ConvGroupNormSiLU)
    out = fused(mx.random.uniform(0, 1, (1, 4, 4, 4)))
    assert out.shape == (1, 4, 4, 8)
