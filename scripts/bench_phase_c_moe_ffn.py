#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase C MoE FFN fusion viability harness.

Measures whether fusing an expert FFN into a single gate_up projection +
SiLU-gate + down projection (the layout moe_ffn_fused.metal targets) beats
the unfused 3-matmul form, and how much headroom remains for the native
megakernel (which additionally keeps the intermediate activation in
threadgroup, eliminating the device-memory round-trip).

Background:
  - custom_kernels/phase_c/glm_moe_ffn.py moe_ffn_fused() falls back to the
    3-matmul path when _NATIVE_AVAILABLE is False (always today — the Metal
    source moe_ffn_fused.metal exists but no C++ extension is built).
  - The native megakernel fuses gate_up into one matmul AND keeps the
    inter_dim-wide intermediate in threadgroup. This harness measures the
    PROJECTION-fusion lower bound (1 gate_up + 1 down vs 3 matmuls) on real
    GLM-MoE / DeepSeek expert FFN shapes. A megakernel that also eliminates
    the intermediate round-trip can only do better.
  - Verdict per regime: fused speedup >= 1.10x -> fusion hits the PRD
    native-vs-separated Δ>=10% target on the projection floor alone, so a
    megakernel is clearly justified; ~1.0x -> break-even, the megakernel
    must win via the threadgroup-resident intermediate; <1.0x -> unfused fine.

Shapes (GLM-4 / DeepSeek-MoE expert FFN defaults):
  hidden=4096, intermediate=14336 (GLM MoE) ; we also probe 11008 (Llama FFN)
  for cross-architecture coverage.

Run:  python scripts/bench_phase_c_moe_ffn.py
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
import mlx.nn as nn


def _bench(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000.0


def make_weights(hidden: int, inter: int):
    w_gate = mx.random.normal((inter, hidden))
    w_up = mx.random.normal((inter, hidden))
    w_down = mx.random.normal((hidden, inter))
    w_gate_up = mx.concatenate([w_gate, w_up], axis=0)
    mx.eval(w_gate, w_up, w_down, w_gate_up)
    return w_gate, w_up, w_down, w_gate_up


def fused(x, w_gate_up, w_down):
    gate_up = mx.matmul(x, w_gate_up.T)
    gate = gate_up[..., : w_gate_up.shape[0] // 2]
    up = gate_up[..., w_gate_up.shape[0] // 2 :]
    hidden = nn.silu(gate) * up
    return mx.matmul(hidden, w_down.T)


def unfused(x, w_gate, w_up, w_down):
    gate = mx.matmul(x, w_gate.T)
    up = mx.matmul(x, w_up.T)
    hidden = nn.silu(gate) * up
    return mx.matmul(hidden, w_down.T)


def check_equivalence(hidden: int, inter: int):
    w_gate, w_up, w_down, w_gate_up = make_weights(hidden, inter)
    x = mx.random.normal((1, 16, hidden))
    f_out = fused(x, w_gate_up, w_down)
    u_out = unfused(x, w_gate, w_up, w_down)
    mx.eval(f_out, u_out)
    ok = mx.allclose(f_out, u_out, atol=1e-3)
    if not ok:
        print("  ERROR: fused != unfused", file=sys.stderr)
        return False
    return True


def regime(name, hidden, inter, seq, batch, iters):
    w_gate, w_up, w_down, w_gate_up = make_weights(hidden, inter)
    x = mx.random.normal((batch, seq, hidden))
    mx.eval(x)

    def f():
        return fused(x, w_gate_up, w_down)

    def u():
        return unfused(x, w_gate, w_up, w_down)

    t_fused = _bench(f, iters)
    t_unfused = _bench(u, iters)
    speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
    print(
        f"  [{name:<16}] fused={t_fused:7.3f}ms  unfused={t_unfused:7.3f}ms  "
        f"speedup={speedup:5.2f}x"
    )
    return {
        "regime": name,
        "hidden": hidden,
        "inter": inter,
        "fused_ms": t_fused,
        "unfused_ms": t_unfused,
        "speedup": speedup,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(
        "Phase C MoE FFN fusion viability — fused (gate_up+down, 2 matmul) vs unfused (3 matmul)"
    )
    print(f"  mlx {mx.__version__}  device={mx.default_device()}")
    print(
        "  fused   = 1 gate_up matmul (out=2*inter) + split + SiLU*up + 1 down matmul"
    )
    print("  unfused = 3 matmuls (gate, up, down) + SiLU*up")
    print("  native megakernel additionally keeps inter activation in threadgroup")
    print()
    print(
        "  Verdict rule: speedup>=1.10x -> hits native-vs-separated Δ>=10% target on floor;"
    )
    print(
        "                ~1.0x -> break-even, megakernel must win via threadgroup inter;"
    )
    print("                <1.0x -> unfused fine, megakernel the only path.")
    print()

    if not check_equivalence(4096, 14336):
        print("  correctness check FAILED — aborting", file=sys.stderr)
        return 1
    print("  correctness: fused-split == unfused  OK")
    print()

    results = []
    # GLM-MoE expert FFN: hidden=4096, inter=14336
    results.append(
        regime("glm-decode-b1", 4096, 14336, seq=1, batch=1, iters=args.iters)
    )
    results.append(
        regime("glm-decode-b4", 4096, 14336, seq=1, batch=4, iters=args.iters)
    )
    results.append(
        regime("glm-prefill-512", 4096, 14336, seq=512, batch=1, iters=args.iters)
    )
    results.append(
        regime("glm-prefill-2048", 4096, 14336, seq=2048, batch=1, iters=args.iters)
    )
    # Llama-style FFN: hidden=4096, inter=11008
    results.append(
        regime("llama-prefill-512", 4096, 11008, seq=512, batch=1, iters=args.iters)
    )
    results.append(
        regime("llama-prefill-2048", 4096, 11008, seq=2048, batch=1, iters=args.iters)
    )

    print()
    print("Verdict:")
    met_target = 0
    for r in results:
        s = r["speedup"]
        if s >= 1.10:
            v = "FUSED WINS -> native-vs-separated Δ>=10% met on projection floor"
            met_target += 1
        elif s > 0.9:
            v = "BREAK-EVEN -> megakernel must win via threadgroup inter"
        else:
            v = "UNFUSED FINE -> megakernel the only path"
        print(f"  {r['regime']:<16}: speedup={s:5.2f}x -> {v}")

    print()
    print(
        f"Summary: {met_target}/{len(results)} regimes meet Δ>=10% on projection floor."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
