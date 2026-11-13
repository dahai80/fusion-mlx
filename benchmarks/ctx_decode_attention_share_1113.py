# SPDX-License-Identifier: Apache-2.0
"""R3 feasibility probe: is long-context b1 decode attention-bound?

Op-level bench of the per-step decode attention read at Qwen3-8B shapes
(36 layers, 32 q heads, 8 kv heads, head_dim 128, GQA). Chains 36
attention calls into ONE graph so a single mx.eval measures kernel time
without per-call CPU dispatch overhead, at growing S_kv. Reports
attention ms/step, theoretical KV-read bytes/time, and the 4-bit weights
read floor — the attention share decides whether Quest-style page
selection (R3) has headroom.

Run: python benchmarks/ctx_decode_attention_share_1113.py [--iters 30]
"""

import argparse
import time

import mlx.core as mx

N_LAYERS = 36
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
WEIGHTS_GB_4BIT = 4.5
BW_GBPS = 546.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    scale = 1.0 / (HEAD_DIM**0.5)
    w4_read_ms = WEIGHTS_GB_4BIT / BW_GBPS * 1000
    print(
        f"{'ctx':>6} {'attn_ms':>8} {'us/layer':>9} {'kv_gb':>7} "
        f"{'kv_read_ms':>10} {'w4_ms':>7} {'attn_share':>10}"
    )
    for ctx in (2048, 8192, 16384, 32768):
        q = mx.random.normal((1, N_Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, N_KV_HEADS, ctx, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, N_KV_HEADS, ctx, HEAD_DIM)).astype(mx.bfloat16)

        def full_step():
            # Chain all layers so one eval times pure kernel throughput.
            out = q
            for _ in range(N_LAYERS):
                a = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
                out = out + a.mean(axis=1, keepdims=True)
            return out

        mx.eval(full_step())
        t0 = time.perf_counter()
        for _ in range(args.iters):
            mx.eval(full_step())
        attn_ms = (time.perf_counter() - t0) / args.iters * 1000

        kv_gb = N_LAYERS * 2 * N_KV_HEADS * HEAD_DIM * ctx * 2 / 1e9
        kv_read_ms = kv_gb / BW_GBPS * 1000
        share = attn_ms / (attn_ms + w4_read_ms) * 100
        print(
            f"{ctx:>6} {attn_ms:>8.2f} {attn_ms / N_LAYERS * 1000:>9.1f} "
            f"{kv_gb:>7.2f} {kv_read_ms:>10.2f} {w4_read_ms:>7.2f} "
            f"{share:>9.1f}%"
        )


if __name__ == "__main__":
    main()
