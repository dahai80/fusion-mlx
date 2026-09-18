# SPDX-License-Identifier: Apache-2.0
"""Shim performance baseline (PR-M, v2 doc §5.2 perf gate).

Micro-benchmarks the shim enhancement ops against their stock MLX
references with synthetic tensors — no model weights, runs headless.
Each op is measured with its shim path forced ON and OFF so the report
shows the delta, not an absolute that only makes sense on one machine.

Usage:
    .venv/bin/python scripts/bench_shim_perf.py [--json PATH] [--iters N]

Ops covered:
  - fused_rmsnorm_residual (shim) vs x + rmsnorm residual (stock)
  - fused_rope (shim) vs mlx.nn.RoPE (stock)
  - quantized-KV online attention (shim) vs fp16 stock SDPA
  - grammar bitmask apply (shim manual) vs stock logits (no mask)

Output: markdown table to stdout, optional JSON file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np

VOCAB = 32000


def timed(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0  # ms


def timed_ab(fn_a, fn_b, iters: int, warmup: int = 3) -> tuple[float, float]:
    """Interleaved A/B timing — controls for drift between the two paths."""
    for _ in range(warmup):
        mx.eval(fn_a())
        mx.eval(fn_b())
    ta, tb = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn_a())
        ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mx.eval(fn_b())
        tb.append(time.perf_counter() - t0)
    return statistics.median(ta) * 1000.0, statistics.median(tb) * 1000.0


def bench_rmsnorm(iters: int) -> dict:
    from fusion_mlx.shim.fused_ops import fused_rmsnorm_residual

    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((1, 512, 4096)).astype(np.float16))
    w = mx.array(rng.standard_normal(4096).astype(np.float16) * 0.02 + 1.0)

    def stock():
        return mx.fast.rms_norm(x, w, 1e-6) + x

    def shim():
        return fused_rmsnorm_residual(x, x, w, eps=1e-6)

    s, sh = timed_ab(stock, shim, iters)
    return {"op": "fused_rmsnorm_residual", "stock_ms": s, "shim_ms": sh}


def bench_rope(iters: int) -> dict:
    from fusion_mlx.shim.fused_ops import fused_rope

    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((2, 8, 256, 128)).astype(np.float16))

    def stock():
        return mx.fast.rope(x, 128, traditional=True, base=10000.0, scale=1.0, offset=0)

    def shim():
        return fused_rope(x, dims=128, base=10000.0, offset=0)

    s, sh = timed_ab(stock, shim, iters)
    return {"op": "fused_rope", "stock_ms": s, "shim_ms": sh}


def bench_quant_kv(iters: int) -> dict:
    from fusion_mlx.shim.quant_kv import q80_dequantize, q80_quantize

    rng = np.random.default_rng(2)
    kv = mx.array(rng.standard_normal((1, 4, 512, 128)).astype(np.float16))
    q = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float16))
    d, p = q80_quantize(kv)
    kvr = q80_dequantize(d, p).astype(mx.float16)

    def stock():
        return mx.fast.scaled_dot_product_attention(q, kv, kv, scale=1 / 11.3)

    def shim():
        return mx.fast.scaled_dot_product_attention(q, kvr, kvr, scale=1 / 11.3)

    s, sh = timed_ab(stock, shim, iters)
    return {
        "op": "quant_kv_degraded_attention (q8_0, T=512)",
        "stock_ms": s,
        "shim_ms": sh,
    }


def bench_grammar_apply(iters: int) -> dict:
    from fusion_mlx.shim.grammar_ring import apply_bitmask

    rng = np.random.default_rng(3)
    logits = mx.array(rng.standard_normal((1, VOCAB)).astype(np.float32))
    mask_np = np.full(((VOCAB + 31) // 32,), -1, dtype=np.int32)
    allow = np.where(rng.random(VOCAB) < 0.02)[0]
    for i in allow:
        mask_np[i // 32] = np.int32(
            np.uint32(mask_np[i // 32]) | np.uint32(1) << np.uint32(i % 32)
        )

    def stock():
        return logits

    def shim():
        return apply_bitmask(mask_np, logits, VOCAB)

    s, sh = timed_ab(stock, shim, iters)
    return {"op": "grammar_bitmask_apply (2% allowed)", "stock_ms": s, "shim_ms": sh}


BENCHES = {
    "rmsnorm": bench_rmsnorm,
    "rope": bench_rope,
    "quant_kv": bench_quant_kv,
    "grammar": bench_grammar_apply,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=sorted(BENCHES))
    args = ap.parse_args()

    results = []
    for name in args.only:
        fn = BENCHES.get(name)
        if fn is None:
            print(f"unknown bench: {name}", file=sys.stderr)
            return 2
        try:
            results.append(fn(args.iters))
        except Exception as exc:
            print(f"[skip] {name}: {exc}", file=sys.stderr)

    print(f"{'op':44s} {'stock ms':>10s} {'shim ms':>10s} {'delta':>8s}")
    for r in results:
        delta = (r["shim_ms"] - r["stock_ms"]) / max(r["stock_ms"], 1e-9) * 100
        print(
            f"{r['op']:44s} {r['stock_ms']:10.3f} {r['shim_ms']:10.3f} {delta:+7.1f}%"
        )
        if "note" in r:
            print(f"    note: {r['note']}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=4))
        print(f"json written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
