# SPDX-License-Identifier: Apache-2.0
"""Shim performance baseline (PR-M, v2 doc §5.2 perf gate).

Micro-benchmarks the shim enhancement ops against their stock MLX
references with synthetic tensors — no model weights, runs headless.
Each op is measured with its shim path forced ON (via env flags) so the
report shows the real shim path, not the fallback.

The previous bench ran with flags OFF (default), measuring the fallback
path (identical to stock + Python overhead) — every op showed a phantom
regression. This version sets the flags before importing shim modules.

Usage:
    .venv/bin/python scripts/bench_shim_perf.py [--json PATH] [--iters N]

Ops covered:
  - fused_rmsnorm_residual (shim @mx.compile) vs stock rms_norm + add
  - fused_rope (shim @mx.compile) vs stock mx.fast.rope
  - quantized-KV: dequant+SDPA (shim) vs stock fp16 SDPA + memory savings
  - grammar bitmask apply (shim GPU expand) vs stock Python per-bit loop

Output: markdown table to stdout, optional JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Enable ALL shim flags BEFORE importing shim modules — measuring the real
# shim path, not the fallback (which is byte-identical to stock + overhead).
os.environ.setdefault("FUSION_SHIM_FUSED_RMSNORM", "1")
os.environ.setdefault("FUSION_SHIM_FUSED_ROPE", "1")
os.environ.setdefault("FUSION_SHIM_QUANT_KV", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np

VOCAB = 32000


def timed_ab(fn_a, fn_b, iters: int, warmup: int = 5) -> tuple[float, float]:
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


def _bench_rmsnorm_shape(shape, iters, label):
    from fusion_mlx.shim.fused_ops import fused_rmsnorm_residual

    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal(shape).astype(np.float16))
    w = mx.array(rng.standard_normal(shape[-1]).astype(np.float16) * 0.02 + 1.0)

    def stock():
        return mx.fast.rms_norm(x, w, 1e-6) + x

    def shim():
        return fused_rmsnorm_residual(x, x, w, eps=1e-6)

    s, sh = timed_ab(stock, shim, iters)
    return {"op": f"fused_rmsnorm_residual ({label})", "stock_ms": s, "shim_ms": sh}


def bench_rmsnorm(iters: int) -> dict:
    # Small shape (decode): Python dispatch overhead dominates — fusion
    # benefit negligible. Tests the @mx.compile fallback path.
    r_small = _bench_rmsnorm_shape((1, 512, 4096), iters, "decode seq=512")
    # Realistic prefill: 30%+ speedup expected (Metal kernel path).
    r_prefill = _bench_rmsnorm_shape((1, 4096, 4096), iters, "prefill seq=4096")
    # Batch decode: moderate win.
    r_batch = _bench_rmsnorm_shape((4, 512, 4096), iters, "batch4 seq=512")
    # Long context: biggest win (memory bandwidth bound).
    r_long = _bench_rmsnorm_shape((1, 4096, 8192), iters, "long seq=4096 dim=8192")
    return [r_small, r_prefill, r_batch, r_long]


def _bench_rope_shape(shape, dims, iters, label):
    from fusion_mlx.shim.fused_ops import fused_rope

    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal(shape).astype(np.float16))

    def stock():
        return mx.fast.rope(
            x, dims, traditional=True, base=10000.0, scale=1.0, offset=0
        )

    def shim():
        return fused_rope(x, dims=dims, base=10000.0, offset=0)

    s, sh = timed_ab(stock, shim, iters)
    return {"op": f"fused_rope ({label})", "stock_ms": s, "shim_ms": sh}


def bench_rope(iters: int) -> dict:
    # Standard path is direct passthrough to mx.fast.rope — zero overhead.
    # Value = YaRN/NTK capability (tested separately), not speed.
    r_decode = _bench_rope_shape((1, 32, 512, 128), 128, iters, "decode 32h seq=512")
    r_prefill = _bench_rope_shape(
        (1, 32, 4096, 128), 128, iters, "prefill 32h seq=4096"
    )
    r_long = _bench_rope_shape((1, 32, 8192, 128), 128, iters, "long 32h seq=8192")
    return [r_decode, r_prefill, r_long]


def _bench_quant_kv_shape(T, iters, label):
    from fusion_mlx.shim.quant_kv import q40_quantize, q80_quantize

    rng = np.random.default_rng(2)
    kv = mx.array(rng.standard_normal((1, 4, T, 128)).astype(np.float16))
    q = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float16))
    scale = 1 / 11.3

    d8, p8 = q80_quantize(kv)
    d4, p4 = q40_quantize(kv)

    fp16_bytes = kv.nbytes * 2
    q8_bytes = (d8.nbytes + p8.nbytes) * 2
    q4_bytes = (d4.nbytes + p4.nbytes) * 2
    q8_mem_pct = 100 - q8_bytes * 100 // fp16_bytes
    q4_mem_pct = 100 - q4_bytes * 100 // fp16_bytes

    def stock():
        return mx.fast.scaled_dot_product_attention(q, kv, kv, scale=scale)

    def shim_q8():
        from fusion_mlx.shim.quant_kv import q80_dequantize

        kvr = q80_dequantize(d8, p8).astype(mx.float16)
        return mx.fast.scaled_dot_product_attention(q, kvr, kvr, scale=scale)

    def shim_q4():
        from fusion_mlx.shim.quant_kv import q40_dequantize

        kvr = q40_dequantize(d4, p4).astype(mx.float16)
        return mx.fast.scaled_dot_product_attention(q, kvr, kvr, scale=scale)

    s8, sh8 = timed_ab(stock, shim_q8, iters)
    _, sh4 = timed_ab(stock, shim_q4, iters)
    return {
        "op": f"quant_kv_attention ({label}, dequant+SDPA)",
        "stock_ms": s8,
        "shim_ms": sh8,
        "shim_q4_ms": sh4,
        "note": f"memory: q8 -{q8_mem_pct}% q4 -{q4_mem_pct}% vs fp16; "
        f"q8={q8_bytes // 1024}KB q4={q4_bytes // 1024}KB fp16={fp16_bytes // 1024}KB",
    }


def bench_quant_kv(iters: int) -> dict:
    # Quant KV value proposition = memory savings (q8 -47%, q4 -72%),
    # not raw speed (dequant adds compute). Bench at decode + long context.
    r_decode = _bench_quant_kv_shape(512, iters, "T=512 decode")
    r_long = _bench_quant_kv_shape(4096, iters, "T=4096 long ctx")
    return [r_decode, r_long]


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
        neg_inf = float("-inf")
        allowed = np.zeros(VOCAB, dtype=bool)
        for i in range(0, min(mask_np.shape[0] * 32, VOCAB), 32):
            word = int(mask_np[i // 32])
            for bit in range(32):
                idx = i + bit
                if idx >= VOCAB:
                    break
                if word & (1 << bit):
                    allowed[idx] = True
        mask = np.where(allowed, 0.0, neg_inf).astype(np.float32)
        return logits + mx.array(mask).reshape(1, -1)

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
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=sorted(BENCHES))
    args = ap.parse_args()

    results = []
    for name in args.only:
        fn = BENCHES.get(name)
        if fn is None:
            print(f"unknown bench: {name}", file=sys.stderr)
            return 2
        try:
            r = fn(args.iters)
            if isinstance(r, list):
                results.extend(r)
            else:
                results.append(r)
        except Exception as exc:
            print(f"[skip] {name}: {exc}", file=sys.stderr)

    print(f"{'op':48s} {'stock ms':>10s} {'shim ms':>10s} {'delta':>8s}")
    for r in results:
        delta = (r["shim_ms"] - r["stock_ms"]) / max(r["stock_ms"], 1e-9) * 100
        print(
            f"{r['op']:48s} {r['stock_ms']:10.3f} {r['shim_ms']:10.3f} {delta:+7.1f}%"
        )
        if "note" in r:
            print(f"    {r['note']}")
        if "shim_q4_ms" in r:
            d4 = (r["shim_q4_ms"] - r["stock_ms"]) / max(r["stock_ms"], 1e-9) * 100
            print(f"    q4: {r['shim_q4_ms']:.3f}ms ({d4:+.1f}%)")

    if args.json:
        args.json.write_text(json.dumps(results, indent=4))
        print(f"json written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
