# SPDX-License-Identifier: Apache-2.0
"""#1201: op-level bench of manual attention vs mx.fast SDPA.

The 5 converted families (hunyuanvideo / svd / opensora / latentsync_mlx /
mage) have no cached weights and stub serving backends, so a full-model
step-time bench is not runnable today. Instead, bench the converted
attention op itself at each family's real denoise-loop shapes (taken from
the family configs), which is where the O(N^2) manual path hurts.

Usage: python benchmarks/sdpa_1201_bench.py [--iters 20]
"""

import argparse
import time

import mlx.core as mx
import numpy as np

# (family, site, B, H, S_q, S_kv, D) — real shapes from family configs.
SITES = [
    ("hunyuanvideo", "dit double-stream", 1, 24, 2816, 2816, 128),
    ("hunyuanvideo", "dit parallel", 1, 24, 2816, 2816, 128),
    ("hunyuanvideo", "text_encoder self", 1, 16, 512, 512, 128),
    ("hunyuanvideo", "text_encoder cross(GQA->repeated)", 1, 16, 512, 512, 128),
    ("hunyuanvideo", "vae mid (single head, C=512)", 1, 1, 4096, 4096, 512),
    ("svd", "unet spatial", 1, 8, 7200, 7200, 80),
    ("svd", "unet temporal", 1, 8, 25, 25, 80),
    ("svd", "unet cross", 1, 8, 7200, 77, 80),
    ("svd", "clip_vision", 1, 16, 257, 257, 64),
    ("opensora", "self+cross", 1, 16, 2260, 2260, 72),
    ("latentsync", "unet cross-attn", 1, 8, 1024, 96, 64),
    ("latentsync", "vae attn (single head)", 1, 1, 1024, 1024, 128),
    ("mage", "flow attention", 1, 16, 1024, 1024, 64),
]


def _manual(q, k, v, scale, mask=None):
    attn = (q * scale) @ k.transpose(0, 1, 3, 2)
    if mask is not None:
        attn = attn + mask
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    return attn @ v


def _bench(fn, q, k, v, scale, iters, mask=None):
    for _ in range(3):
        fn(q, k, v, scale, mask)
    mx.eval(mx.abs(q).sum())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k, v, scale, mask)
    mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"{'family':<15} {'site':<38} {'manual ms':>10} {'fast ms':>9} {'speedup':>8}")
    for fam, site, b, h, sq, skv, d in SITES:
        np.random.seed(0)
        q = mx.array(np.random.randn(b, h, sq, d)).astype(mx.bfloat16)
        k = mx.array(np.random.randn(b, h, skv, d)).astype(mx.bfloat16)
        v = mx.array(np.random.randn(b, h, skv, d)).astype(mx.bfloat16)
        scale = d**-0.5
        t_manual = _bench(_manual, q, k, v, scale, args.iters)
        t_fast = _bench(
            lambda q, k, v, s, m=None: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=s
            ),
            q,
            k,
            v,
            scale,
            args.iters,
        )
        print(
            f"{fam:<15} {site:<38} {t_manual * 1e3:>10.2f} {t_fast * 1e3:>9.2f}"
            f" {t_manual / t_fast:>7.2f}x"
        )


if __name__ == "__main__":
    main()
