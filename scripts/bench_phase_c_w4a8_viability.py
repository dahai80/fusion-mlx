#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase C W4A8 viability harness.

Measures the activation-quantization overhead that a native W4A8 (4-bit
weight + 8-bit activation) fused MatMul kernel must absorb or beat.

Background:
  - mx.quantized_matmul already fuses 4-bit WEIGHT dequant into MatMul, but
    only accepts fp16 ACTIVATIONS. W4A8 would also quantize activations to
    int8 and fuse that into MatMul.
  - MLX core has no int8-activation MatMul, so we cannot measure the upside
    directly. We measure the OVERHEAD (quantize A to int8 + dequant = the
    round-trip a fused kernel fuses away) relative to the baseline MatMul.
  - Verdict per regime: if overhead_frac is small, a native int8 MatMul could
    net-win (fusing removes the overhead AND int8 compute is cheaper). If
    overhead_frac >= 1, activation quant is prohibitively expensive at that
    batch size and W4A8 likely loses there.

Run:  python scripts/bench_phase_c_w4a8_viability.py
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx


def _bench(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000.0


def make_problem(
    hidden: int, intermediate: int, seq: int, batch: int, group_size: int, bits: int
):
    w_fp = mx.random.normal((hidden, intermediate))
    w_q, w_s, w_b = mx.quantize(w_fp, group_size=group_size, bits=bits)
    a_fp = mx.random.normal((batch, seq, hidden))
    return w_q, w_s, w_b, a_fp


def regime(name, hidden, intermediate, seq, batch, group_size=64, bits=4, iters=20):
    w_q, w_s, w_b, a_fp = make_problem(
        hidden, intermediate, seq, batch, group_size, bits
    )

    def baseline():
        return mx.quantized_matmul(
            a_fp, w_q, w_s, w_b, transpose=False, group_size=group_size, bits=bits
        )

    def a8_roundtrip():
        aq, as_, ab = mx.quantize(a_fp, group_size=group_size, bits=8)
        return mx.dequantize(aq, as_, ab, group_size=group_size, bits=8)

    def a8_naive():
        aq, as_, ab = mx.quantize(a_fp, group_size=group_size, bits=8)
        a_dq = mx.dequantize(aq, as_, ab, group_size=group_size, bits=8)
        return mx.quantized_matmul(
            a_dq, w_q, w_s, w_b, transpose=False, group_size=group_size, bits=bits
        )

    t_base = _bench(baseline, iters)
    t_rt = _bench(a8_roundtrip, iters)
    t_naive = _bench(a8_naive, iters)
    overhead_frac = t_rt / t_base if t_base > 0 else float("inf")
    print(
        f"  [{name:<14}] base={t_base:7.3f}ms  A8-roundtrip={t_rt:7.3f}ms  "
        f"A8-naive={t_naive:7.3f}ms  overhead={overhead_frac*100:6.1f}% of base"
    )
    return {
        "regime": name,
        "base_ms": t_base,
        "a8_roundtrip_ms": t_rt,
        "a8_naive_ms": t_naive,
        "overhead_frac": overhead_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print("Phase C W4A8 viability — activation-quant overhead vs W4 MatMul baseline")
    print(f"  mlx {mx.__version__}  device={mx.default_device()}")
    print("  baseline = mx.quantized_matmul (W4 weights, fp16 activations)")
    print("  A8-roundtrip = mx.quantize(A,8)+dequant (overhead a fused W4A8 absorbs)")
    print("  A8-naive = baseline on dequantized A (sanity: ~base + roundtrip)")
    print()
    print("  Verdict rule: overhead<20% → W4A8 promising (int8 upside may net-win);")
    print("                overhead>100% → W4A8 loses at that regime.")
    print()

    results = []
    results.append(regime("decode-b1", 4096, 11008, seq=1, batch=1, iters=args.iters))
    results.append(regime("decode-b4", 4096, 11008, seq=1, batch=4, iters=args.iters))
    results.append(
        regime("prefill-512", 4096, 11008, seq=512, batch=1, iters=args.iters)
    )
    results.append(
        regime("prefill-2048", 4096, 11008, seq=2048, batch=1, iters=args.iters)
    )

    print()
    print("Verdict:")
    for r in results:
        f = r["overhead_frac"]
        if f < 0.2:
            v = "PROMISING — int8 upside may net-win"
        elif f < 1.0:
            v = "MARGINAL — needs int8 MatMul >2x to net-win"
        else:
            v = "LOSES — activation quant overhead exceeds baseline"
        print(f"  {r['regime']:<14}: overhead={f*100:5.1f}% → {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
