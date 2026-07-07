#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase C Fused-GDN projection viability harness.

Measures whether fusing the GatedDeltaNet input projections (Qwen3.6-27B
linear-attention layers) into 2 matmuls beats the unfused 6-matmul form, and
how much headroom remains for a full fused megakernel.

Background:
  - The fork's qwen3_5.py GDN class already fuses q/k/v/z into one
    in_proj_qkvz matmul (out = key_dim*2 + value_dim*2 = 22528) and b/a into
    one in_proj_ba matmul (out = num_v_heads*2 = 128). Upstream unfused form
    would be 6 separate matmuls (q, k, v, z, b, a).
  - This harness measures the PROJECTION-fusion speedup (2 vs 6 matmuls) on
    real Qwen3.6-27B GDN shapes across prefill + decode regimes. It is the
    lower bound for a full fused megakernel (which would additionally fuse the
    conv1d + delta update + out_proj, eliminating the 22528-wide intermediate
    round-trip through global memory).
  - Verdict per regime: fused speedup >1.2x -> fused class clearly justified
    and a megakernel has a projection-fusion floor to build on; ~1.0x ->
    projection fusion is break-even, megakernel is the only remaining win;
    <1.0x -> unfused is fine.

Shapes (Qwen3.6-27B TextModelArgs defaults):
  hidden=4096, linear_num_key_heads=16, linear_num_value_heads=64,
  linear_key_head_dim=192, linear_value_head_dim=128 ->
    key_dim=3072, value_dim=8192, num_v_heads=64
  qkvz out = 3072*2 + 8192*2 = 22528 ; ba out = 64*2 = 128

Run:  python scripts/bench_phase_c_fused_gdn.py
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

HIDDEN = 4096
KEY_DIM = 3072
VALUE_DIM = 8192
NUM_V_HEADS = 64
QKVZ_OUT = KEY_DIM * 2 + VALUE_DIM * 2
BA_OUT = NUM_V_HEADS * 2


def _bench(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000.0


def make_weights():
    q_w = mx.random.normal((KEY_DIM, HIDDEN))
    k_w = mx.random.normal((KEY_DIM, HIDDEN))
    v_w = mx.random.normal((VALUE_DIM, HIDDEN))
    z_w = mx.random.normal((VALUE_DIM, HIDDEN))
    b_w = mx.random.normal((NUM_V_HEADS, HIDDEN))
    a_w = mx.random.normal((NUM_V_HEADS, HIDDEN))
    qkvz_w = mx.concatenate([q_w, k_w, v_w, z_w], axis=0)
    ba_w = mx.concatenate([b_w, a_w], axis=0)
    mx.eval(q_w, k_w, v_w, z_w, b_w, a_w, qkvz_w, ba_w)
    return q_w, k_w, v_w, z_w, b_w, a_w, qkvz_w, ba_w


def fused(x, qkvz_w, ba_w):
    qkvz = mx.matmul(x, qkvz_w.T)
    qkv_dim = KEY_DIM * 2 + VALUE_DIM
    qkv = qkvz[..., :qkv_dim]
    z = qkvz[..., qkv_dim:]
    ba = mx.matmul(x, ba_w.T)
    b = ba[..., :NUM_V_HEADS]
    a = ba[..., NUM_V_HEADS:]
    return qkv, z, b, a


def unfused(x, q_w, k_w, v_w, z_w, b_w, a_w):
    q = mx.matmul(x, q_w.T)
    k = mx.matmul(x, k_w.T)
    v = mx.matmul(x, v_w.T)
    z = mx.matmul(x, z_w.T)
    b = mx.matmul(x, b_w.T)
    a = mx.matmul(x, a_w.T)
    qkv = mx.concatenate([q, k, v], axis=-1)
    return qkv, z, b, a


def check_equivalence(qkvz_w, ba_w, q_w, k_w, v_w, z_w, b_w, a_w):
    x = mx.random.normal((1, 16, HIDDEN))
    fqkv, fz, fb, fa = fused(x, qkvz_w, ba_w)
    uqkv, uz, ub, ua = unfused(x, q_w, k_w, v_w, z_w, b_w, a_w)
    mx.eval(fqkv, fz, fb, fa, uqkv, uz, ub, ua)
    eq_qkv = mx.allclose(fqkv, uqkv, atol=1e-3)
    eq_z = mx.allclose(fz, uz, atol=1e-3)
    eq_b = mx.allclose(fb, ub, atol=1e-3)
    eq_a = mx.allclose(fa, ua, atol=1e-3)
    if not (eq_qkv and eq_z and eq_b and eq_a):
        print("  ERROR: fused != unfused (qkv={}, z={}, b={}, a={})".format(
            eq_qkv, eq_z, eq_b, eq_a), file=sys.stderr)
        return False
    return True


def regime(name, seq, batch, iters):
    q_w, k_w, v_w, z_w, b_w, a_w, qkvz_w, ba_w = make_weights()
    x = mx.random.normal((batch, seq, HIDDEN))
    mx.eval(x)

    def f():
        return fused(x, qkvz_w, ba_w)

    def u():
        return unfused(x, q_w, k_w, v_w, z_w, b_w, a_w)

    t_fused = _bench(f, iters)
    t_unfused = _bench(u, iters)
    speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
    print(
        f"  [{name:<14}] fused={t_fused:7.3f}ms  unfused={t_unfused:7.3f}ms  "
        f"speedup={speedup:5.2f}x"
    )
    return {
        "regime": name,
        "fused_ms": t_fused,
        "unfused_ms": t_unfused,
        "speedup": speedup,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print("Phase C Fused-GDN projection viability — fused (2 matmul) vs unfused (6 matmul)")
    print(f"  mlx {mx.__version__}  device={mx.default_device()}")
    print(f"  shapes: hidden={HIDDEN} key_dim={KEY_DIM} value_dim={VALUE_DIM} "
          f"num_v_heads={NUM_V_HEADS} qkvz_out={QKVZ_OUT} ba_out={BA_OUT}")
    print("  fused   = 1 qkvz matmul (out=22528) + split + 1 ba matmul (out=128) + split")
    print("  unfused = 6 matmuls (q,k,v,z,b,a) + qkv concat")
    print()
    print("  Verdict rule: speedup>1.2x -> fused justified, megakernel has floor;")
    print("                ~1.0x -> break-even, megakernel is the only remaining win;")
    print("                <1.0x -> unfused fine, megakernel the only path.")
    print()

    q_w, k_w, v_w, z_w, b_w, a_w, qkvz_w, ba_w = make_weights()
    if not check_equivalence(qkvz_w, ba_w, q_w, k_w, v_w, z_w, b_w, a_w):
        print("  correctness check FAILED — aborting", file=sys.stderr)
        return 1
    print("  correctness: fused-split == unfused  OK")
    print()

    results = []
    results.append(regime("decode-b1", seq=1, batch=1, iters=args.iters))
    results.append(regime("decode-b4", seq=1, batch=4, iters=args.iters))
    results.append(regime("prefill-512", seq=512, batch=1, iters=args.iters))
    results.append(regime("prefill-2048", seq=2048, batch=1, iters=args.iters))
    results.append(regime("prefill-8192", seq=8192, batch=1, iters=args.iters))

    print()
    print("Verdict:")
    for r in results:
        s = r["speedup"]
        if s > 1.2:
            v = "FUSED WINS -> fused class justified, megakernel has floor"
        elif s > 0.9:
            v = "BREAK-EVEN -> megakernel is the only remaining win"
        else:
            v = "UNFUSED FINE -> megakernel the only path"
        print(f"  {r['regime']:<14}: speedup={s:5.2f}x -> {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
