# SPDX-License-Identifier: Apache-2.0
# SmartConv2d - shape-dispatched conv backend (#919).
#
# Metal mx.conv2d has fp16 throughput cliffs at specific (C,H,W) shapes - up to
# 8x between adjacent configurations (e.g. 64x64 512->512 3x3 runs at 2.4
# TFLOPS Metal conv vs 7.1 TFLOPS for the same math as im2col GEMM, M5 Max /
# MLX 0.32.2). SmartConv2d routes each call to whichever backend wins.
#
# Dispatch is a pure shape predicate decided in code (Rule 5). Thresholds
# measured on M5 Max; regenerate with scripts/bench_smart_conv.py.

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# im2col wins (measured on M5 Max / MLX 0.32.2, fp16, B=1) at these
# (C_in, C_out, min_hw) bands with 3x3 s1 same-padding kernels:
#   64x64 512->512: native 0.04ms vs gemm 0.02ms (2x)
#   128x128 256->256: native 0.03ms vs gemm 0.02ms
# Native wins everywhere else measured (1280-ch, 32x32/16x16, small hw).
_IM2COL_RULES = [
    (512, 512, 4096),
    (256, 256, 8192),
]
_DEFAULT_RULES = None


def _rules():
    global _DEFAULT_RULES
    if _DEFAULT_RULES is None:
        _DEFAULT_RULES = list(_IM2COL_RULES)
    return _DEFAULT_RULES


def set_im2col_rules(rules):
    global _DEFAULT_RULES
    _DEFAULT_RULES = list(rules)
    logger.info("[graph_opt] im2col dispatch rules overridden: %s", rules)


def use_im2col_for_shape(c_in, c_out, hw):
    for r_ci, r_co, r_hw in _rules():
        if c_in >= r_ci and c_out >= r_co and hw >= r_hw:
            return True
    return False


def im2col_conv2d(x, weight, bias=None, padding=1):
    o, kh, kw, c = weight.shape
    if padding:
        x = mx.pad(x, [(0, 0), (padding, padding), (padding, padding), (0, 0)])
    h, w = x.shape[1] - 2, x.shape[2] - 2
    cols = mx.concatenate(
        [x[:, dy : dy + h, dx : dx + w, :] for dy in range(kh) for dx in range(kw)],
        axis=-1,
    )
    w2 = weight.reshape(o, kh * kw * c).transpose()
    out = cols @ w2
    if bias is not None:
        out = out + bias
    return out


class SmartConv2d(nn.Module):
    """Conv2d wrapper dispatching between mx.conv2d and im2col GEMM (#919)."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def __call__(self, x):
        conv = self.conv
        if _use_im2col(x, conv):
            # _use_im2col guarantees (1,1) same-padding — pass as int
            return im2col_conv2d(x, conv.weight, bias=conv.bias, padding=1)
        return conv(x)


def _use_im2col(x, conv) -> bool:
    if len(x.shape) != 4 or conv.groups != 1 or conv.stride != (1, 1):
        return False
    if conv.weight.shape[1:3] != (3, 3) or tuple(conv.padding) != (1, 1):
        return False
    b, h, w, c = x.shape
    o, kh, kw, ci = conv.weight.shape
    if c != ci:
        return False
    return use_im2col_for_shape(c, o, h * w)


def _wrappable(conv) -> bool:
    if not isinstance(conv, nn.Conv2d):
        return False
    return (
        conv.weight.shape[1:3] == (3, 3)
        and conv.stride == (1, 1)
        and conv.groups == 1
        and tuple(conv.padding) == (1, 1)
    )


def apply_smart_conv(root, _seen=None) -> int:
    """Wrap supported nn.Conv2d (k=3 s1 groups=1 padding=1) with SmartConv2d.

    Shares weights by reference (wrapper holds the conv). Call AFTER weight
    loading. Returns wrap count.
    """
    if _seen is None:
        _seen = set()
    if id(root) in _seen or isinstance(root, SmartConv2d):
        return 0
    _seen.add(id(root))
    wrapped = 0
    if isinstance(root, nn.Module):
        # MLX Module is a dict subclass; root[key] returns the LIVE stored
        # value (children() rebuilds container copies — mutations lost).
        for name in list(root.keys()):
            val = root[name]
            if isinstance(val, list):
                for i, c in enumerate(val):
                    if not isinstance(c, nn.Module):
                        continue
                    if _wrappable(c):
                        val[i] = SmartConv2d(c)
                        wrapped += 1
                    else:
                        wrapped += apply_smart_conv(c, _seen)
            elif isinstance(val, SmartConv2d):
                continue  # already wrapped — do not descend into .conv
            elif _wrappable(val):
                root[name] = SmartConv2d(val)
                wrapped += 1
            elif isinstance(val, nn.Module):
                wrapped += apply_smart_conv(val, _seen)
            elif isinstance(val, dict):
                for k, c in val.items():
                    if not isinstance(c, nn.Module):
                        continue
                    if _wrappable(c):
                        val[k] = SmartConv2d(c)
                        wrapped += 1
                    else:
                        wrapped += apply_smart_conv(c, _seen)
        return wrapped
    if isinstance(root, dict):
        for key, c in root.items():
            if isinstance(c, nn.Module):
                if _wrappable(c):
                    root[key] = SmartConv2d(c)
                    wrapped += 1
                else:
                    wrapped += apply_smart_conv(c, _seen)
        return wrapped
    if isinstance(root, list):
        for i, c in enumerate(root):
            if isinstance(c, nn.Module):
                if _wrappable(c):
                    root[i] = SmartConv2d(c)
                    wrapped += 1
                else:
                    wrapped += apply_smart_conv(c, _seen)
        return wrapped
    return wrapped
