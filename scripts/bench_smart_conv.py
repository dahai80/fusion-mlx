# SPDX-License-Identifier: Apache-2.0
# Microbenchmark: mx.conv2d vs im2col GEMM across conv shapes (#919).
#
# Regenerates the per-shape dispatch decision in fusion_mlx/graph_opt/smart_conv.py.
# Usage:
#   python scripts/bench_smart_conv.py            # bench all shapes (eval-in-loop)
#   python scripts/bench_smart_conv.py --autotune  # dump per-shape winners as JSON
#   python scripts/bench_smart_conv.py --out rules.json
#
# NOTE: the timing loop MUST mx.eval each iteration — mx.conv2d is lazy and a
# bare fn() call only builds the graph (measures ~0.016ms, not compute).

import argparse
import json
import time

import mlx.core as mx

from fusion_mlx.graph_opt.smart_conv import im2col_conv2d


def bench(fn, n=30, warm=8):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
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
    (256, 256, 128, 128),
    (32, 32, 1280, 1280),
    (16, 16, 1280, 1280),
]


def run(autotune=False):
    results = []
    for h, w, ci, co in SHAPES:
        x = (mx.random.normal((1, h, w, ci)) * 0.1).astype(mx.float16)
        wgt = (mx.random.normal((co, 3, 3, ci)) * 0.02).astype(mx.float16)
        t_native = bench(lambda x=x, w=wgt: mx.conv2d(x, w, stride=1, padding=1))
        t_gemm = bench(lambda x=x, w=wgt: im2col_conv2d(x, w, padding=1))
        ref = mx.conv2d(x, wgt, stride=1, padding=1)
        got = im2col_conv2d(x, wgt, padding=1)
        mx.eval(ref)
        mx.eval(got)
        diff = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
        winner = "im2col" if t_gemm < t_native else "native"
        row = {
            "shape": f"{h}x{w} {ci}->{co}",
            "key": f"{ci},{co},{h*w}",
            "native_ms": round(t_native, 4),
            "im2col_ms": round(t_gemm, 4),
            "winner": winner,
            "maxdiff": round(diff, 5),
        }
        results.append(row)
        print(json.dumps(row))
    wins = [r for r in results if r["winner"] == "im2col"]
    print(f"\nim2col wins {len(wins)}/{len(results)} shapes")
    for r in wins:
        print(f"  {r['shape']}: {r['native_ms']} -> {r['im2col_ms']} ms")
    if autotune:
        rules = {r["key"]: r["winner"] for r in results}
        print("\n# FUSION_SMART_CONV_RULES (paste as env for deterministic override):")
        print(json.dumps(rules))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--autotune", action="store_true", help="dump per-shape winners as JSON"
    )
    ap.add_argument("--out", default=None, help="write results JSON to path")
    args = ap.parse_args()
    results = run(autotune=args.autotune)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
