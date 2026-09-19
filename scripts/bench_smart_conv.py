# SPDX-License-Identifier: Apache-2.0
# Microbenchmark: mx.conv2d vs im2col GEMM across conv shapes (#919).
#
# Regenerates the dispatch thresholds in fusion_mlx/graph_opt/smart_conv.py.
# Usage: python scripts/bench_smart_conv.py [--full]

import argparse
import json
import time

import mlx.core as mx

from fusion_mlx.graph_opt.smart_conv import im2col_conv2d


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    mx.eval(mx.zeros((1,)))
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1000


SHAPES = [
    (64, 64, 512, 512),
    (32, 32, 512, 512),
    (16, 16, 512, 512),
    (128, 128, 256, 256),
    (64, 64, 256, 256),
    (32, 32, 256, 256),
    (256, 256, 128, 256),
    (32, 32, 1280, 1280),
    (16, 16, 1280, 1280),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    results = []
    for (h, w, ci, co) in SHAPES:
        x = (mx.random.normal((1, h, w, ci)) * 0.1).astype(mx.float16)
        wgt = (mx.random.normal((co, 3, 3, ci)) * 0.02).astype(mx.float16)
        t_native = bench(lambda: mx.conv2d(x, wgt, stride=1, padding=1))
        t_gemm = bench(lambda: im2col_conv2d(x, wgt, padding=1))
        ref = mx.conv2d(x, wgt, stride=1, padding=1)
        got = im2col_conv2d(x, wgt, padding=1)
        mx.eval(ref)
        mx.eval(got)
        diff = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
        row = {
            "shape": f"{h}x{w} {ci}->{co}",
            "native_ms": round(t_native, 4),
            "im2col_ms": round(t_gemm, 4),
            "winner": "im2col" if t_gemm < t_native else "native",
            "maxdiff": round(diff, 5),
        }
        results.append(row)
        print(json.dumps(row))
    wins = [r for r in results if r["winner"] == "im2col"]
    print(f"\nim2col wins {len(wins)}/{len(results)} shapes")
    for r in wins:
        print(f"  {r['shape']}: {r['native_ms']} -> {r['im2col_ms']} ms")


if __name__ == "__main__":
    main()
