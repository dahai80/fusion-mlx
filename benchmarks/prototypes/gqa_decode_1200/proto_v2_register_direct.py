# Scratch prototype v2: GQA decode attention split-KV Metal kernel
# Design: threadgroup=256 (8 simdgroups). Each of 8 rows-in-flight handled
# by one simdgroup (32 lanes x 4 dims = D=128 per row dot, simd_sum reduce).
# V accumulation: thread (h, lane) owns dims lane*4..+3 of head h in
# registers, single m/l per head. KV read ONCE per kv head (vs 8x for
# per-q-head kernels). Split-KV across threadgroups + combine kernel.
import math
import time

import mlx.core as mx

HQ = 32
HKV = 4
GQA = HQ // HKV
D = 128


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

        // thread (warp, lane): warp iterates rows, lane owns dims.
        // V accumulator: this thread owns head w, dims lane*4..lane*4+3.
        const uint h = warp;                 // head index within GQA group
        float4 acc = float4(0.0f);
        float m = -1e30f;
        float l = 0.0f;

        const uint q_base = (b * N_Q_HEADS + kvh * GQA + h) * HEAD_DIM;
        const uint kv_base = (b * N_KV_HEADS + kvh) * S_TOTAL * HEAD_DIM;
        const uint d0 = lane * 4;

        // tile of 8 rows: warp w handles row (tile + w) score; all warps
        // then accumulate all 8 rows for their head. scores via tgmem.
        threadgroup float scores[8 * GQA];

        for (uint s0 = s_begin; s0 < s_end; s0 += 8) {
          const uint rows = min(s0 + 8, s_end) - s0;
          // score for my row (warp w -> row s0+w)
          if (warp < rows) {
            uint s = s0 + warp;
            float partial = 0.0f;
            // each lane covers 4 dims; loop 8 heads
            for (uint hh = 0; hh < GQA; ++hh) {
              float4 qv = float4(q[q_base - h * HEAD_DIM + hh * HEAD_DIM + d0 + 0],
                                 q[q_base - h * HEAD_DIM + hh * HEAD_DIM + d0 + 1],
                                 q[q_base - h * HEAD_DIM + hh * HEAD_DIM + d0 + 2],
                                 q[q_base - h * HEAD_DIM + hh * HEAD_DIM + d0 + 3]);
              float4 kv = float4(keys[kv_base + s * HEAD_DIM + d0 + 0],
                                 keys[kv_base + s * HEAD_DIM + d0 + 1],
                                 keys[kv_base + s * HEAD_DIM + d0 + 2],
                                 keys[kv_base + s * HEAD_DIM + d0 + 3]);
              partial = dot(qv, kv) * scale;
              partial = simd_sum(partial);
              if (lane == 0) {
                scores[warp * GQA + hh] = partial;
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // online-softmax merge: rescale to tile max FIRST, then
          // accumulate p = exp(score - m_new); p against the stale m
          // overflows to inf on the first tile (m = -1e30).
          float cmax2 = -1e30f;
          for (uint t = 0; t < rows; ++t) {
            cmax2 = max(cmax2, scores[t * GQA + h]);
          }
          float m_new = max(m, cmax2);
          float exp_m = exp(m - m_new);
          l *= exp_m;
          acc *= exp_m;
          m = m_new;
          for (uint t = 0; t < rows; ++t) {
            float p = exp(scores[t * GQA + h] - m);
            l += p;
            float4 vv = float4(values[kv_base + (s0 + t) * HEAD_DIM + d0 + 0],
                               values[kv_base + (s0 + t) * HEAD_DIM + d0 + 1],
                               values[kv_base + (s0 + t) * HEAD_DIM + d0 + 2],
                               values[kv_base + (s0 + t) * HEAD_DIM + d0 + 3]);
            acc += p * vv;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // partials [B, HKV, N_SPLITS, GQA, 2 + HEAD_DIM]
        uint pbase = ((((b * N_KV_HEADS + kvh) * N_SPLITS) + split) * GQA + h) * (2 + HEAD_DIM);
        if (l > 0.0f) {
          partials[pbase + 0] = m;
          partials[pbase + 1] = l;
          partials[pbase + 2 + d0 + 0] = acc.x;
          partials[pbase + 2 + d0 + 1] = acc.y;
          partials[pbase + 2 + d0 + 2] = acc.z;
          partials[pbase + 2 + d0 + 3] = acc.w;
        } else {
          partials[pbase + 0] = -1e30f;
          partials[pbase + 1] = 0.0f;
        }
    """
    return mx.fast.metal_kernel(
        name="gqa_decode_split2",
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
        for ns in (64, 256, 512):
            got = gqa_decode(q, k, v, scale, ns)
            mx.eval(got)
            err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
            t_mine = bench(lambda: gqa_decode(q, k, v, scale, ns))
            t_ref = bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale))
            print(
                f"S={S} splits={ns}: maxerr={err:.4f}  mine={t_mine*1e3:.2f}ms  sdpa={t_ref*1e3:.2f}ms  speedup={t_ref/t_mine:.2f}x"
            )
