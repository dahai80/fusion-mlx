# SPDX-License-Identifier: Apache-2.0
"""#919: SmartConv2d shape dispatch + im2col parity + apply_smart_conv wrap."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.graph_opt.smart_conv import (
    SmartConv2d,
    apply_smart_conv,
    im2col_conv2d,
    set_im2col_rules,
    use_im2col_for_shape,
)


@pytest.fixture(autouse=True)
def _restore_rules():
    from fusion_mlx.graph_opt.smart_conv import _IM2COL_RULES

    yield
    set_im2col_rules(_IM2COL_RULES)


@pytest.fixture(autouse=True)
def _cpu():
    # GPU Metal GEMM accumulates differently (~1e-3 fp32 drift) — CPU gives
    # exact conv2d/im2col parity
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def test_im2col_parity_fp32():
    x = mx.random.normal((1, 16, 16, 8))
    w = mx.random.normal((8, 3, 3, 8)) * 0.1
    ref = mx.conv2d(x, w, stride=1, padding=1)
    got = im2col_conv2d(x, w, padding=1)
    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-4).item()


def test_im2col_parity_with_bias():
    x = mx.random.normal((2, 8, 8, 8))
    w = mx.random.normal((16, 3, 3, 8)) * 0.1
    b = mx.random.normal((16,)) * 0.1
    ref = mx.conv2d(x, w, stride=1, padding=1) + b
    got = im2col_conv2d(x, w, bias=b, padding=1)
    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-4).item()


def test_dispatch_rules_matching():
    set_im2col_rules([(512, 512, 4096)])
    assert use_im2col_for_shape(512, 512, 4096)
    assert use_im2col_for_shape(1024, 512, 4096)
    assert not use_im2col_for_shape(256, 512, 4096)
    assert not use_im2col_for_shape(512, 512, 1024)


def test_smart_conv2d_dispatches_by_shape():
    conv = nn.Conv2d(512, 512, 3, padding=1)
    smart = SmartConv2d(conv)
    x_small = mx.random.normal((1, 8, 8, 512)).astype(mx.float16)
    x_big = mx.random.normal((1, 64, 64, 512)).astype(mx.float16)
    ref_small = conv(x_small)
    ref_big = conv(x_big)
    out_small = smart(x_small)
    out_big = smart(x_big)
    mx.eval(ref_small, out_small, ref_big, out_big)
    assert mx.allclose(out_small, ref_small, atol=1e-2).item()
    assert mx.allclose(out_big, ref_big, atol=1e-2).item()


def test_apply_smart_conv_wraps_supported_convs():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv2d(8, 8, 3, padding=1)
            self.b = nn.Conv2d(8, 8, 3, stride=2, padding=1)
            self.c = nn.Conv2d(8, 8, 1)
            self.lst = [nn.Conv2d(8, 8, 3, padding=1), nn.Linear(4, 4)]

    m = M()
    n = apply_smart_conv(m)
    assert n == 2
    assert isinstance(m.a, SmartConv2d)
    assert not isinstance(m.b, SmartConv2d)
    assert not isinstance(m.c, SmartConv2d)
    assert isinstance(m.lst[0], SmartConv2d)
    assert isinstance(m.lst[1], nn.Linear)


def test_apply_smart_conv_weight_sharing():
    conv = nn.Conv2d(8, 8, 3, padding=1)
    holder = [conv]
    apply_smart_conv(holder)
    assert isinstance(holder[0], SmartConv2d)
    assert holder[0].conv is conv


def test_apply_smart_conv_idempotent():
    conv = nn.Conv2d(8, 8, 3, padding=1)
    holder = [conv]
    assert apply_smart_conv(holder) == 1
    assert apply_smart_conv(holder) == 0  # already wrapped — no double-wrap
    assert holder[0].conv is conv
