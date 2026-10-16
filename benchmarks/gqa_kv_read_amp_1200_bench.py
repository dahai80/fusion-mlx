#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1200: GQA KV-cache read-amplification gate bench (op level).

Path-mapping premise check (explored before writing this): LLM decode in
fusion-mlx runs mlx_lm's forward with mx.fast.scaled_dot_product_attention
(scheduler/sched_batch.py model() calls) — the custom_kernels/mfa and
flash_kda paths named in the issue have ZERO callers on the LLM decode
path (mfa_bridge is video-only; flash_kda has no prod callers per
custom_kernels/__init__.py). So the gate question is measured on the
kernel that actually runs: Apple's Metal SDPA over the [B, n_kv, S, D]
contiguous KV cache.

Method (mirrors benchmarks/sdpa_1201_bench.py conventions):
1. Device streaming reference: read-only reduction over a large contiguous
   array gives the achievable bandwidth ceiling on this machine.
2. Decode-shape SDPA: q [B, Hq, 1, D], k/v [B, Hkv, S, D] contiguous —
   the exact layout mlx_lm's KVCache grows into. Useful bytes = K + V
   read once (2 * Hkv * S * D * dtype_bytes); amplification = streaming
   reference BW / achieved useful BW.
3. Alignment probe: head_dim variants at constant total KV bytes.
   fp16 rows are D*2 bytes — 128B-aligned for D in {64, 128}, misaligned
   (192B) for D=96. If misaligned D is disproportionately slower relative
   to bytes, cache-line waste is real on the active path.

Gate (issue #1200): amplification > 1.3x anywhere -> implement aligned
layout + vectorized fetch; <= 1.3x everywhere -> close with archived
bench. Output JSON to benchmarks/reports/.

Usage: python benchmarks/gqa_kv_read_amp_1200_bench.py [--iters 50]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

GB = 1e9


def _sync():
    mx.eval(mx.zeros(1))
    mx.synchronize()


def _time(fn, iters, warmup=5):
    # mx.eval INSIDE the loop: MLX builds lazy graphs, so a post-loop eval
    # would only time the last iteration's graph construction.
    for _ in range(warmup):
        mx.eval(fn())
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    _sync()
    return (time.perf_counter() - t0) / iters


def streaming_reference_bw(iters, dtype=mx.float16):
    # Contiguous 2 GiB copy (scale by 2.0): reads + writes the whole
    # buffer per call — the standard achievable-bandwidth reference, far
    # beyond any cache level. Reductions (abs().sum()) are NOT a valid
    # reference here: they measured ~158 GB/s on this machine while SDPA
    # reached >200 GB/s, i.e. below the device's real streaming ceiling.
    x = mx.random.normal((1024, 1024, 1024), dtype=mx.float32).astype(dtype)
    bytes_moved = x.size * x.itemsize * 2  # read + write
    dt = min(_time(lambda: x * 2.0, iters) for _ in range(3))
    del x
    return bytes_moved / dt


def kv_only_bw(hkv, s_kv, d, dtype, iters):
    # Cross-check: how fast can a trivial reduction stream the SAME KV
    # buffers SDPA reads? If this hits ~copy peak, SDPA's shortfall is
    # real amplification, not a device quirk.
    k = mx.random.normal((1, hkv, s_kv, d)).astype(dtype)
    n_bytes = k.size * k.itemsize
    dt = min(_time(lambda: k.sum(), iters) for _ in range(3))
    del k
    return n_bytes / dt


def decode_sdpa(case, iters):
    b, hq, hkv, s_kv, d = case["b"], case["hq"], case["hkv"], case["s"], case["d"]
    dt = case["dtype"]
    scale = 1.0 / (d**0.5)
    q = mx.random.normal((b, hq, 1, d)).astype(dt)
    k = mx.random.normal((b, hkv, s_kv, d)).astype(dt)
    v = mx.random.normal((b, hkv, s_kv, d)).astype(dt)
    useful = 2 * b * hkv * s_kv * d * mx.array(0, dtype=dt).itemsize
    t = min(
        _time(
            lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
            iters,
        )
        for _ in range(3)
    )
    return {
        **case,
        "dtype": str(case["dtype"]).split(".")[-1].strip("><>"),
        "seconds": round(t, 8),
        "useful_bytes": useful,
        "useful_bw_gbs": round(useful / t / GB, 1),
        "amplification_vs_stream": round(
            case["stream_gbs"] / (useful / t / GB), 2
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="benchmarks/reports/gqa_kv_read_amp_1200.json")
    args = ap.parse_args()

    print("[1/3] streaming reference bandwidth")
    ref = streaming_reference_bw(args.iters)
    ref_gbs = ref / GB
    print(f"      read-only reduction: {ref_gbs:.0f} GB/s")

    print("[2/3] decode-shape SDPA (Qwen3.5 GQA: Hq=32, Hkv=4, D=128)")
    print(
        "      KV-only reduction cross-check: "
        f"{kv_only_bw(4, 65536, 128, mx.float16, args.iters) / GB:.0f} GB/s"
    )
    cases = []
    for s_kv in (2048, 8192, 32768, 65536, 131072):
        for dt in (mx.float16, mx.bfloat16):
            cases.append(
                {
                    "label": f"decode Hq32 Hkv4 D128 S{s_kv}",
                    "b": 1,
                    "hq": 32,
                    "hkv": 4,
                    "s": s_kv,
                    "d": 128,
                    "dtype": dt,
                    "stream_gbs": ref_gbs,
                }
            )
    rows = []
    for c in cases:
        r = decode_sdpa(c, args.iters)
        # Fixed kernel overhead dominates small-S decode (~0.2ms floor at
        # S2048-8192 regardless of KV size) — those rows measure launch
        # latency, not bandwidth. Only large-S rows are BW-bound and
        # speak to read amplification.
        r["bw_bound"] = c["s"] >= 32768
        rows.append(r)
        print(
            f"      {r['label']} [{r['dtype']}]: {r['seconds']*1e3:.2f}ms  "
            f"useful {r['useful_bw_gbs']} GB/s  amp {r['amplification_vs_stream']}x"
            f"{'  [BW-bound]' if r['bw_bound'] else '  [overhead-bound]'}"
        )

    print("[3/3] alignment probe (constant KV bytes, varying head_dim)")
    # Constant useful bytes: D * S must be constant -> scale S inversely.
    kv_elems = 4 * 32768 * 128
    align_rows = []
    for d in (64, 96, 128):
        s = kv_elems // (4 * d)
        c = {
            "label": f"align D{d} S{s}",
            "b": 1,
            "hq": 32,
            "hkv": 4,
            "s": s,
            "d": d,
            "dtype": mx.float16,
            "stream_gbs": ref_gbs,
        }
        r = decode_sdpa(c, args.iters)
        align_rows.append(r)
        print(
            f"      {r['label']}: {r['seconds']*1e3:.2f}ms  "
            f"useful {r['useful_bw_gbs']} GB/s  amp {r['amplification_vs_stream']}x"
        )

    max_amp = max(r["amplification_vs_stream"] for r in rows if r["bw_bound"])
    result = {
        "method": "op-level decode SDPA useful-BW vs 2GiB-copy streaming reference (min of 3 repeats)",
        "path_note": (
            "LLM decode runs mx.fast.scaled_dot_product_attention over the "
            "contiguous [B, n_kv, S, D] mlx_lm KV cache; custom_kernels/mfa "
            "and flash_kda have no LLM decode callers (explored 2026-10-16), "
            "so the active path is what is measured. Gate uses only BW-bound "
            "rows (S >= 32768); small-S decode is fixed-overhead-bound."
        ),
        "streaming_reference_gbs": round(ref_gbs, 1),
        "kv_only_reduction_gbs": round(
            kv_only_bw(4, 65536, 128, mx.float16, args.iters) / GB, 1
        ),
        "decode_rows": rows,
        "alignment_probe": align_rows,
        "max_amplification_bw_bound": max_amp,
        "gate": {
            "threshold": 1.3,
            "verdict": "IMPLEMENT (amp > 1.3x)" if max_amp > 1.3 else "NOT_WORTH_IT (amp <= 1.3x)",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    print(
        f"max amplification (BW-bound rows): {max_amp}x -> "
        f"{result['gate']['verdict']}"
    )


if __name__ == "__main__":
    main()
