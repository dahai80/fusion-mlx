# Scratch prototype v3: GQA decode attention split-KV Metal kernel.
# Fix for v2's 8x KV re-read: each threadgroup (256 threads, 8 warps, one
# (b, kvh, split)) cooperatively stages 8-row K/V tiles into threadgroup
# memory ONCE, then all 8 GQA heads share the staged tile:
#   score phase: warp w computes row w's dot against ALL 8 head q's
#                (simd_sum per (row, head)) from the staged K tile
#   accum phase: thread (head h, lane) owns dims lane*4..+3 of head h,
#                reads staged V tile, online-softmax m/l/acc in registers
# Global KV traffic per tile: 8 rows x (K+V) read ONCE (v2: 8x).
import math
import time

import mlx.core as mx

HQ = 32
HKV = 4
GQA = HQ // HKV
D = 128
TILE = 8


def sync():
    mx.eval(mx.zeros(1))
    mx.synchronize()


def make_split_kernel():
    src = r"""
        const uint gid = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = tid % 32;
        const uint warp = tid / 32;          // 0..7
        const uint b = gid / (N_KV_HEADS * N_SPLITS);
        const uint kvh = (gid / N_SPLITS) % N_KV_HEADS;
        const uint split = gid % N_SPLITS;
        const float scale = scale_in[0];

        const uint s_per_split = (S_TOTAL + N_SPLITS - 1) / N_SPLITS;
        const uint s_begin = split * s_per_split;
        const uint s_end = min(s_begin + s_per_split, (uint)S_TOTAL);

        const uint kv_base = (b * N_KV_HEADS + kvh) * S_TOTAL * HEAD_DIM;
        const uint d0 = lane * 4;

        threadgroup float k_tile[TILE * HEAD_DIM];
        threadgroup float v_tile[TILE * HEAD_DIM];
        threadgroup float scores[TILE * GQA];

        float acc = 0.0f;
        float m = -1e30f;
        float l = 0.0f;
        const uint h = warp;                 // head within GQA group
        const uint q_head = kvh * GQA + h;

        for (uint s0 = s_begin; s0 < s_end; s0 += TILE) {
          const uint rows = min(s0 + TILE, s_end) - s0;
          // cooperative stage of K/V rows into threadgroup memory
          for (uint i = tid; i < TILE * HEAD_DIM; i += 256) {
            const uint t = i / HEAD_DIM;
            const uint d = i % HEAD_DIM;
            if (t < rows) {
              k_tile[i] = keys[kv_base + (s0 + t) * HEAD_DIM + d];
              v_tile[i] = values[kv_base + (s0 + t) * HEAD_DIM + d];
            } else {
              k_tile[i] = 0.0f;
              v_tile[i] = 0.0f;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // score phase: warp w computes row w's dots for all 8 heads.
          // Each lane holds 4 dims -> float4 dot, simd_sum over 32 lanes.
          if (warp < rows) {
            const uint s = s0 + warp;
            for (uint hh = 0; hh < GQA; ++hh) {
              float4 qv = float4(q[(b * N_Q_HEADS + kvh * GQA + hh) * HEAD_DIM + d0 + 0],
                                 q[(b * N_Q_HEADS + kvh * GQA + hh) * HEAD_DIM + d0 + 1],
                                 q[(b * N_Q_HEADS + kvh * GQA + hh) * HEAD_DIM + d0 + 2],
                                 q[(b * N_Q_HEADS + kvh * GQA + hh) * HEAD_DIM + d0 + 3]);
              float4 kv = float4(k_tile[warp * HEAD_DIM + d0 + 0],
                                 k_tile[warp * HEAD_DIM + d0 + 1],
                                 k_tile[warp * HEAD_DIM + d0 + 2],
                                 k_tile[warp * HEAD_DIM + d0 + 3]);
              float partial = dot(qv, kv) * scale;
              if (lane == 0) {
                scores[warp * GQA + hh] = simd_sum(partial);
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // accum phase: online softmax per head h, rescale-then-accumulate
          float cmax = -1e30f;
          for (uint t = 0; t < rows; ++t) {
            cmax = max(cmax, scores[t * GQA + h]);
          }
          float m_new = max(m, cmax);
          float eold = exp(m - m_new);
          m = m_new;
          l *= eold;
          acc *= eold;
          for (uint t = 0; t < rows; ++t) {
            float p = exp(scores[t * GQA + h] - m);
            l += p;
            acc += p * v_tile[t * HEAD_DIM + d0];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // partials [B, HKV, N_SPLITS, GQA, 2 + HEAD_DIM]
        uint pbase = ((((b * N_KV_HEADS + kvh) * N_SPLITS) + split) * GQA + h) * (2 + HEAD_DIM);
        if (l > 0.0f) {
          partials[pbase + 0] = m;
          partials[pbase + 1] = l;
          partials[pbase + 2 + d0] = acc;
        } else {
          partials[pbase + 0] = -1e30f;
          partials[pbase + 1] = 0.0f;
        }
    """
    return mx.fast.metal_kernel(
        name="gqa_decode_split3",
        input_names=["q", "keys", "values", "scale_in"],
        output_names=["partials"],
        source=src,
    )


def make_combine_kernel():
    src = r"""
        const uint gid = thread_position_in_grid.x;
        const uint b = gid / (N_Q_HEADS * HEAD_DIM);
        uint rem = gid % (N_Q_HEADS * HEAD_DIM);
        uint h = rem / HEAD_DIM;
        uint d = rem % HEAD_DIM;
        uint kvh = h / (N_Q_HEADS / N_KV_HEADS);
        uint g = h % (N_Q_HEADS / N_KV_HEADS);

        float m = -1e30f;
        for (uint s = 0; s < N_SPLITS; ++s) {
          m = max(m, partials[((((b * N_KV_HEADS + kvh) * N_SPLITS) + s) * GQA + g) * (2 + HEAD_DIM)]);
        }
        float l = 0.0f;
        float acc = 0.0f;
        for (uint s = 0; s < N_SPLITS; ++s) {
          uint pbase = ((((b * N_KV_HEADS + kvh) * N_SPLITS) + s) * GQA + g) * (2 + HEAD_DIM);
          float ms_ = partials[pbase];
          float ls_ = partials[pbase + 1];
          float w = metal::exp(ms_ - m);
          l += w * ls_;
          acc += w * partials[pbase + 2 + d];
        }
        out[b * N_Q_HEADS * HEAD_DIM + h * HEAD_DIM + d] = acc / l;
    """
    return mx.fast.metal_kernel(
        name="gqa_decode_combine",
        input_names=["partials"],
        output_names=["out"],
        source=src,
    )


def gqa_decode(q, k, v, scale, n_splits):
    b = q.shape[0]
    s = k.shape[2]
    n_splits = max(1, min(n_splits, s))
    kerm = make_split_kernel()
    partials = kerm(
        inputs=[q.reshape(b, HQ * D), k, v, mx.array([scale], dtype=mx.float32)],
        template=[
            ("N_Q_HEADS", HQ),
            ("N_KV_HEADS", HKV),
            ("GQA", GQA),
            ("HEAD_DIM", D),
            ("S_TOTAL", s),
            ("N_SPLITS", n_splits),
            ("TILE", TILE),
        ],
        grid=(b * HKV * n_splits * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, HKV, n_splits, GQA, 2 + D)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    kerb = make_combine_kernel()
    out = kerb(
        inputs=[partials],
        template=[
            ("N_Q_HEADS", HQ),
            ("N_KV_HEADS", HKV),
            ("GQA", GQA),
            ("HEAD_DIM", D),
            ("N_SPLITS", n_splits),
        ],
        grid=(b * HQ * D, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(b, HQ, 1, D)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    return out.astype(q.dtype)


def bench(fn, iters=20):
    for _ in range(5):
        mx.eval(fn())
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    sync()
    return (time.perf_counter() - t0) / iters


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    scale = 1.0 / math.sqrt(D)
    for S in (32768, 65536, 131072):
        q = mx.random.normal((1, HQ, 1, D)).astype(mx.float16)
        k = mx.random.normal((1, HKV, S, D)).astype(mx.float16)
        v = mx.random.normal((1, HKV, S, D)).astype(mx.float16)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(ref)
        for ns in (32, 64, 128, 256):
            got = gqa_decode(q, k, v, scale, ns)
            mx.eval(got)
            err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
            # adjacent fair A/B: min-of-3, alternate order
            t_mine = min(bench(lambda: gqa_decode(q, k, v, scale, ns), 10) for _ in range(3))
            t_ref = min(bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale), 10) for _ in range(3))
            print(
                f"S={S} splits={ns}: maxerr={err:.4f}  mine={t_mine*1e3:.2f}ms  sdpa={t_ref*1e3:.2f}ms  speedup={t_ref/t_mine:.2f}x"
            )
