# SPDX-License-Identifier: Apache-2.0
# SmartConv2d - shape-dispatched conv backend (#919).
#
# Metal mx.conv2d has fp16 throughput cliffs at specific (C,H,W) shapes - up to
# 8x between adjacent configurations (e.g. 64x64 512->512 3x3 runs at 2.4
# TFLOPS Metal conv vs 7.1 TFLOPS for the same math as im2col GEMM, M5 Max /
# MLX 0.32.2). SmartConv2d routes each call to whichever backend wins.
#
# Dispatch is a runtime autotune (#919 v2): at first call for each (c_in, c_out,
# hw) shape, mx.conv2d (MLX's tuned Metal conv kernel) is benchmarked against
# im2col_conv2d (an im2col gather + MLX matmul Metal kernel) and the winner is
# cached. Both candidates are Metal kernels — this is Metal kernel selection by
# measured Metal performance, not a hardcoded rule. Per-machine, per-MLX-version:
# thresholds regenerate with scripts/bench_smart_conv.py --autotune.
#
# Explicit overrides (determinism / CI / repro):
#   set_im2col_rules([(ci,co,hw),...])   — force im2col for listed shapes
#   set_backend_rules({(ci,co,hw):"native"|"im2col"})  — per-shape explicit
#   FUSION_SMART_CONV_RULES='{"ci,co,hw":"im2col"}'   — config-file override
#   FUSION_SMART_CONV_AUTOTUNE=0  — disable autotune, always native (zero overhead)

from __future__ import annotations

import json
import logging
import os
import time

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def _has_metal() -> bool:
    return hasattr(mx, "metal") and mx.metal.is_available()


def _autotune_enabled() -> bool:
    return _env_on("FUSION_SMART_CONV_AUTOTUNE", "1")


# --------------------------------------------------------------------------- #
# explicit rules (back-comat + determinism)
# --------------------------------------------------------------------------- #
_IM2COL_RULES = []  # legacy: list[(ci,co,hw)] forced to im2col
_BACKEND_RULES: dict | None = None  # explicit per-shape override


def _rules():
    return list(_IM2COL_RULES)


def set_im2col_rules(rules):
    global _IM2COL_RULES
    _IM2COL_RULES = list(rules)
    logger.info("[graph_opt] im2col rules overridden: %s", rules)


def set_backend_rules(mapping: dict):
    global _BACKEND_RULES
    _BACKEND_RULES = {
        tuple(k) if isinstance(k, (list, tuple)) else k: v for k, v in mapping.items()
    }
    logger.info("[graph_opt] backend rules overridden: %d entries", len(_BACKEND_RULES))


def clear_autotune_cache():
    _AUTOTUNE_CACHE.clear()


def _env_rules():
    raw = os.environ.get("FUSION_SMART_CONV_RULES")
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return {
            tuple(int(p) for p in k.split(",")): v
            for k, v in d.items()
            if v in ("native", "im2col")
        }
    except Exception as exc:
        logger.warning("[graph_opt] FUSION_SMART_CONV_RULES parse failed: %s", exc)
        return None


def use_im2col_for_shape(c_in, c_out, hw):
    for r_ci, r_co, r_hw in _rules():
        if c_in >= r_ci and c_out >= r_co and hw >= r_hw:
            return True
    return False


# --------------------------------------------------------------------------- #
# im2col GEMM conv (candidate backend)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# autotune: select best Metal backend per shape (#919 v2)
# --------------------------------------------------------------------------- #
_AUTOTUNE_CACHE: dict[tuple, str] = {}
_AUTOTUNE_WARM = 3
_AUTOTUNE_ITERS = 5


def _bench(fn) -> float:
    for _ in range(_AUTOTUNE_WARM):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(_AUTOTUNE_ITERS):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / _AUTOTUNE_ITERS * 1000


def _select_backend(c_in, c_out, hw, dtype) -> str:
    key = (c_in, c_out, hw)

    env_r = _env_rules()
    if env_r is not None:
        return env_r.get(key, "native")
    if _BACKEND_RULES is not None:
        return _BACKEND_RULES.get(key, "native")
    if use_im2col_for_shape(c_in, c_out, hw):
        return "im2col"
    # autotune bench path requires Metal + fp16
    if dtype != mx.float16 or not _has_metal() or not _autotune_enabled():
        return "native"
    if key in _AUTOTUNE_CACHE:
        return _AUTOTUNE_CACHE[key]

    side = int(round(hw**0.5))
    if side < 1:
        side = 1
    x = (mx.random.normal((1, side, side, c_in)) * 0.1).astype(mx.float16)
    w = (mx.random.normal((c_out, 3, 3, c_in)) * 0.02).astype(mx.float16)
    try:
        t_native = _bench(lambda: mx.conv2d(x, w, stride=1, padding=1))
        t_im2col = _bench(lambda: im2col_conv2d(x, w, padding=1))
    except Exception as exc:
        logger.warning("[graph_opt] autotune failed for %s: %s", key, exc)
        _AUTOTUNE_CACHE[key] = "native"
        return "native"
    winner = "im2col" if t_im2col < t_native else "native"
    _AUTOTUNE_CACHE[key] = winner
    logger.info(
        "[graph_opt] autotune %dx%d %d->%d: native %.3fms im2col %.3fms -> %s",
        hw,
        hw,
        c_in,
        c_out,
        t_native,
        t_im2col,
        winner,
    )
    return winner


# --------------------------------------------------------------------------- #
# SmartConv2d
# --------------------------------------------------------------------------- #
class SmartConv2d(nn.Module):
    """Conv2d wrapper dispatching between Metal conv backends (#919).

    Per-shape autotune selects mx.conv2d (tuned Metal conv) vs im2col GEMM
    (Metal matmul) at first call; winner cached. Pass-through to native when
    autotune is off or dtype is not fp16.
    """

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def __call__(self, x):
        conv = self.conv
        if _use_im2col(x, conv):
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
    return _select_backend(c, o, h * w, x.dtype) == "im2col"


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
                continue
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
