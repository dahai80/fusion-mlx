#!/usr/bin/env python3
import os
import sys
import traceback

os.environ.setdefault("FUSION_DISABLE_COMPILE", "1")

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, "/Users/dahai/claude-home/fusion-mlx")
from fusion_mlx.video.skyreels_v3.transformer_r2v import SkyReelsR2VDiT


def build_tiny():
    cfg = {
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 4,
        "num_layers": 4,
        "patch_size": (1, 2, 2),
        "in_dim": 16,
        "out_dim": 16,
        "text_dim": 64,
        "text_len": 32,
        "freq_dim": 32,
        "window_size": (-1, -1),
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }
    return SkyReelsR2VDiT(cfg)


def run_b(b, label):
    mx.random.seed(7)
    dit = build_tiny()
    H = W = 8
    T = 1
    C = 16
    L_patch = T * (H // 2) * (W // 2)
    L_ctx = 8
    x = mx.random.normal((b, C, T, H, W))
    t = mx.array([0.5 + 0.1 * i for i in range(b)])
    context = mx.random.normal((b, L_ctx, 64))
    seq_lens = [L_patch] * b
    grid_sizes = [(T, H // 2, W // 2)] * b
    out = dit(x, t, context, seq_lens, grid_sizes)
    mx.eval(out)
    return out, dit


def main():
    print("== b=2 forward (modulation broadcast fix) ==")
    try:
        out2, dit2 = run_b(2, "b2")
        print(f"B=2 forward OK, out shape {tuple(out2.shape)}")
    except Exception as e:
        print(f"B=2 forward FAILED: {e}")
        traceback.print_exc()
        return 1

    print("== b=1 forward (regression: must still work) ==")
    out1, _ = run_b(1, "b1")
    print(f"B=1 forward OK, out shape {tuple(out1.shape)}")

    print("== forward_partial(n_layers) == __call__ bit-identical @ b=2 ==")
    mx.random.seed(7)
    dit = build_tiny()
    b = 2
    H = W = 8
    T = 1
    C = 16
    L_patch = T * (H // 2) * (W // 2)
    L_ctx = 8
    x = mx.random.normal((b, C, T, H, W))
    t = mx.array([0.5 + 0.1 * i for range_i in range(1) for i in range(range_i, range_i + b)])
    t = mx.array([0.5 + 0.1 * i for i in range(b)])
    context = mx.random.normal((b, L_ctx, 64))
    seq_lens = [L_patch] * b
    grid_sizes = [(T, H // 2, W // 2)] * b
    full = dit(x, t, context, seq_lens, grid_sizes)
    part = dit.forward_partial(x, t, context, seq_lens, grid_sizes, n_blocks=dit.num_layers)
    mx.eval(full)
    mx.eval(part)
    diff = mx.max(mx.abs(full - part)).item()
    eq = mx.array_equal(full, part).item()
    print(f"max|full-part| = {diff}, bit-identical = {eq}")
    if not eq:
        print("FAIL: forward_partial != __call__")
        return 1

    print("== forward_partial(n_blocks=2) runs (draft path) @ b=2 ==")
    part2 = dit.forward_partial(x, t, context, seq_lens, grid_sizes, n_blocks=2)
    mx.eval(part2)
    print(f"draft (2/4 blocks) OK, out shape {tuple(part2.shape)}")

    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
