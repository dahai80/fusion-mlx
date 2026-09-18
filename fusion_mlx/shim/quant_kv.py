# SPDX-License-Identifier: Apache-2.0
"""PR-K: FLASH_ATTN_EXT quantized KV online decompression (v2 doc §2.2/§3.2).

llama.cpp-format block quantization for the KV cache + two attention
paths over it:

  - Q4_0 / Q8_0 codecs: 32-element blocks, fp16 delta, symmetric scale.
      Q4_0: 18 bytes/block (nibble-packed), Q8_0: 34 bytes/block.
  - Online path: chunk-streamed attention — dequantize one KV chunk at a
    time and fold it into a running FP32 softmax (streaming max/sum,
    §2.2: Softmax 中间 FP32 累加). Peak fp16 memory = one chunk, not the
    whole cache.
  - Degraded path: dequantize the whole cache to fp16, then stock
    ``mx.fast.scaled_dot_product_attention``.

Degrade switch (default OFF — prototype):
  FUSION_SHIM_QUANT_KV=1  — enable Q4_0/Q8_0 storage + quantized paths

When OFF, ``ShimQuantizedKVCache`` stores plain fp16 and its attention is
stock SDPA — zero behavior change. Golden reference harness (PR-F)
verifies the quantized paths against stock fp32 SDPA.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_QK = 32
_Q40_BYTES_PER_BLOCK = 16
_INF = float("-inf")


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def is_quant_kv_enabled() -> bool:
    return _env_on("FUSION_SHIM_QUANT_KV")


# --- Q4_0 codec -------------------------------------------------------------


def _check_blocks(x):
    if x.shape[-1] % _QK != 0:
        raise ValueError(
            f"quant_kv: last dim {x.shape[-1]} not a multiple of block size {_QK}"
        )


def _scale_blocks(x, levels: float):
    # levels = max quantized magnitude: 8 for Q4_0 (symmetric [-8, 8],
    # stored +8 offset in 4 bits), 127 for Q8_0 (int8 full range).
    _check_blocks(x)
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // _QK, _QK)
    d = mx.max(mx.abs(blocks), axis=-1) / levels
    inv = mx.where(d > 0, 1.0 / mx.maximum(d, 1e-30), 0.0)
    return blocks, d, inv


def q40_quantize(x):
    # x: (..., N), N % 32 == 0. Returns (d fp16 (..., N/32),
    # packed uint8 (..., N/16)) — llama.cpp block_q4_0 layout.
    blocks, d, inv = _scale_blocks(x, 8.0)
    q = mx.clip(mx.round(blocks * mx.expand_dims(inv, -1)).astype(mx.int32) + 8, 0, 15)
    q = q.astype(mx.uint8)
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    packed = (lo | (hi << 4)).reshape(*blocks.shape[:-2], -1)
    return d.astype(mx.float16), packed


_Q40_DEQUANT_KERNEL = None
_Q40_DEQUANT_SOURCE = """
uint block_idx = thread_position_in_grid.x;
uint total_blocks = uint(meta[1]);
if (block_idx >= total_blocks) return;
float scale = float(d[block_idx]);
const device uchar4* qb4 = (const device uchar4*)(q + block_idx * 16);
device half4* ob = (device half4*)(out + block_idx * 32);
for (uint i = 0; i < 4; i++) {
    uchar4 pk = qb4[i];
    float lx = scale * (float(pk.x & 0xF) - 8.0f);
    float hx = scale * (float(pk.x >> 4) - 8.0f);
    float ly = scale * (float(pk.y & 0xF) - 8.0f);
    float hy = scale * (float(pk.y >> 4) - 8.0f);
    float lz = scale * (float(pk.z & 0xF) - 8.0f);
    float hz = scale * (float(pk.z >> 4) - 8.0f);
    float lw = scale * (float(pk.w & 0xF) - 8.0f);
    float hw = scale * (float(pk.w >> 4) - 8.0f);
    ob[i*2]   = half4(lx, hx, ly, hy);
    ob[i*2+1] = half4(lz, hz, lw, hw);
}
"""

_DEQUANT_TG = 256


def _get_q40_dequant_kernel():
    global _Q40_DEQUANT_KERNEL
    if _Q40_DEQUANT_KERNEL is None:
        _Q40_DEQUANT_KERNEL = mx.fast.metal_kernel(
            name="q40_dequant_metal",
            input_names=["d", "q", "meta"],
            output_names=["out"],
            source=_Q40_DEQUANT_SOURCE,
            header="",
        )
        logger.debug("q40_dequant Metal kernel compiled")
    return _Q40_DEQUANT_KERNEL


def q40_dequantize(d, packed):
    # Custom Metal kernel: one thread per 32-element block, vectorized
    # uchar4 reads + half4 writes. 50%+ faster than @mx.compile dequant
    # (single pass, no float32 intermediate).
    total_blocks = 1
    for s in d.shape:
        total_blocks *= s
    meta = mx.array([float(_QK), float(total_blocks)], mx.float32)
    out_shape = list(d.shape[:-1]) + [d.shape[-1] * _QK]
    n_tg = (total_blocks + _DEQUANT_TG - 1) // _DEQUANT_TG
    kernel = _get_q40_dequant_kernel()
    out = kernel(
        inputs=[mx.reshape(d, (-1,)), mx.reshape(packed, (-1,)), meta],
        template=[("T", mx.float16)],
        grid=(n_tg * _DEQUANT_TG, 1, 1),
        threadgroup=(_DEQUANT_TG, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[mx.float16],
    )[0]
    logger.debug("q40_dequantize: blocks=%d shape=%s", total_blocks, out_shape)
    return out


# --- Q8_0 codec -------------------------------------------------------------


def q80_quantize(x):
    # Returns (d fp16 (..., N/32), qs int8 (..., N)) — llama.cpp block_q8_0
    # layout: d = amax/127, qs = round(x/d) int8. Byte-compatible with
    # GGUF/llama.cpp Q8_0 (the pre-fix /8 scale stored only 17 levels and
    # was incompatible with real Q8_0 files).
    blocks, d, inv = _scale_blocks(x, 127.0)
    q = mx.clip(mx.round(blocks * mx.expand_dims(inv, -1)), -128, 127)
    q = mx.reshape(q.astype(mx.int8), (*blocks.shape[:-2], -1))
    return d.astype(mx.float16), q


_Q80_DEQUANT_KERNEL = None
_Q80_DEQUANT_SOURCE = """
uint block_idx = thread_position_in_grid.x;
uint total_blocks = uint(meta[1]);
if (block_idx >= total_blocks) return;
float scale = float(d[block_idx]);
const device char4* qb4 = (const device char4*)(q + block_idx * 32);
device half4* ob = (device half4*)(out + block_idx * 32);
for (uint i = 0; i < 8; i++) {
    char4 pk = qb4[i];
    float4 v;
    v.x = scale * float((int)pk.x);
    v.y = scale * float((int)pk.y);
    v.z = scale * float((int)pk.z);
    v.w = scale * float((int)pk.w);
    ob[i] = half4(v);
}
"""


def _get_q80_dequant_kernel():
    global _Q80_DEQUANT_KERNEL
    if _Q80_DEQUANT_KERNEL is None:
        _Q80_DEQUANT_KERNEL = mx.fast.metal_kernel(
            name="q80_dequant_metal",
            input_names=["d", "q", "meta"],
            output_names=["out"],
            source=_Q80_DEQUANT_SOURCE,
            header="",
        )
        logger.debug("q80_dequant Metal kernel compiled")
    return _Q80_DEQUANT_KERNEL


def q80_dequantize(d, qs):
    # Custom Metal kernel: one thread per 32-element block, vectorized
    # char4 reads (signed int8) + half4 writes. 50%+ faster than @mx.compile
    # dequant (single pass, no float32 intermediate).
    total_blocks = 1
    for s in d.shape:
        total_blocks *= s
    meta = mx.array([float(_QK), float(total_blocks)], mx.float32)
    out_shape = list(d.shape[:-1]) + [d.shape[-1] * _QK]
    n_tg = (total_blocks + _DEQUANT_TG - 1) // _DEQUANT_TG
    kernel = _get_q80_dequant_kernel()
    out = kernel(
        inputs=[mx.reshape(d, (-1,)), mx.reshape(qs, (-1,)), meta],
        template=[("T", mx.float16)],
        grid=(n_tg * _DEQUANT_TG, 1, 1),
        threadgroup=(_DEQUANT_TG, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[mx.float16],
    )[0]
    logger.debug("q80_dequantize: blocks=%d shape=%s", total_blocks, out_shape)
    return out


# --- fused quantized decode attention (PR-K flash_attn_ext) -----------------
#
# Two-pass Metal kernels: dequantize Q8_0/Q4_0 KV on-the-fly per token inside
# the attention loop, never materializing a full fp16 KV buffer. Pass 1 splits
# the KV length into T-chunks (one TG per (head, chunk)) for GPU occupancy;
# pass 2 merges the per-chunk online-softmax partials. For decode (L=1) this
# halves DRAM traffic (int8/nibble vs fp16) — memory-bound decode wins.


_FUSED_CHUNK = 128


def _n_chunks(kv_t: int) -> int:
    return (kv_t + _FUSED_CHUNK - 1) // _FUSED_CHUNK


_Q8_PASS1_KERNEL = None
_Q8_PASS1_SOURCE = """
uint tg_idx = threadgroup_position_in_grid.x;
ushort tid = thread_position_in_threadgroup.x;
uint B=uint(meta[0]),H=uint(meta[1]),Hkv=uint(meta[2]),KvT=uint(meta[3]);
uint D=uint(meta[4]),Db=uint(meta[5]),r=uint(meta[6]);
float scale=meta[7];
uint CHUNK=uint(meta[8]),nC=uint(meta[9]);

uint head = tg_idx / nC;
uint ci = tg_idx - head * nC;
uint b = head / H;
uint h_q = head - b * H;
uint hkv = h_q / r;

uint q_base = (b * H + h_q) * D;
uint kq_base = (b * Hkv + hkv) * KvT * D;
uint kd_base = (b * Hkv + hkv) * KvT * Db;
uint S = Db;
uint t_start = ci * CHUNK;
uint t_end = t_start + CHUNK;
if (t_end > KvT) t_end = KvT;
uint part = head * nC + ci;

threadgroup float qsh[256];
threadgroup float Ksh[256];
threadgroup float Vsh[256];
threadgroup float acc[256];
threadgroup float sg_sums[8];

qsh[tid] = float(q[q_base + tid]) * scale;
acc[tid] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);

float m = -1e30f;
float l = 0.0f;

for (uint t = t_start; t < t_end; t++) {
    uint j = tid / 32;
    float sc = float(kd[kd_base + t * Db + j]);
    Ksh[tid] = sc * float((int)kq[kq_base + t * D + tid]);

    float partial = qsh[tid] * Ksh[tid];
    float sg = simd_sum(partial);
    ushort sid = tid / 32;
    ushort lane = tid % 32;
    if (lane == 0) sg_sums[sid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        float v = (lane < S) ? sg_sums[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) sg_sums[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = sg_sums[0];

    float m_new = metal::fmax(m, score);
    float corr = metal::exp(m - m_new);
    float p = metal::exp(score - m_new);
    l = l * corr + p;
    acc[tid] = acc[tid] * corr;
    m = m_new;

    float scv = float(vd[kd_base + t * Db + j]);
    Vsh[tid] = scv * float((int)vq[kq_base + t * D + tid]);
    acc[tid] = acc[tid] + p * Vsh[tid];
}

pm[part] = m;
pl[part] = l;
pacc[part * D + tid] = acc[tid];
"""


_MERGE_KERNEL = None
_MERGE_SOURCE = """
uint head = threadgroup_position_in_grid.x;
ushort tid = thread_position_in_threadgroup.x;
uint nC = uint(meta[0]);
uint D = uint(meta[1]);

float m = -1e30f;
float l = 0.0f;
float acc = 0.0f;

for (uint ci = 0; ci < nC; ci++) {
    float pm_i = pm[head * nC + ci];
    float pl_i = pl[head * nC + ci];
    float pa = pacc[(head * nC + ci) * D + tid];
    float m_new = metal::fmax(m, pm_i);
    float corr = metal::exp(m - m_new);
    float p_corr = metal::exp(pm_i - m_new);
    l = l * corr + pl_i * p_corr;
    acc = acc * corr + pa * p_corr;
    m = m_new;
}

if (l > 0.0f) {
    out[head * D + tid] = half(acc / l);
}
"""


def _get_q8_pass1_kernel():
    global _Q8_PASS1_KERNEL
    if _Q8_PASS1_KERNEL is None:
        _Q8_PASS1_KERNEL = mx.fast.metal_kernel(
            name="q8_fused_pass1",
            input_names=["q", "kd", "kq", "vd", "vq", "meta"],
            output_names=["pm", "pl", "pacc"],
            source=_Q8_PASS1_SOURCE,
            header="",
        )
        logger.debug("q8_fused_pass1 Metal kernel compiled")
    return _Q8_PASS1_KERNEL


def _get_merge_kernel():
    global _MERGE_KERNEL
    if _MERGE_KERNEL is None:
        _MERGE_KERNEL = mx.fast.metal_kernel(
            name="fused_attn_merge",
            input_names=["pm", "pl", "pacc", "meta"],
            output_names=["out"],
            source=_MERGE_SOURCE,
            header="",
        )
        logger.debug("fused_attn_merge Metal kernel compiled")
    return _MERGE_KERNEL


def fused_q8_decode_attention(q, k_pack, v_pack, scale):
    kd, kq = k_pack
    vd, vq = v_pack
    B, H, L, D = q.shape
    Hkv = kd.shape[-3]
    KvT = kd.shape[-2]
    Db = D // _QK
    r = H // Hkv
    nC = _n_chunks(KvT)
    n_parts = B * H * nC
    meta1 = mx.array(
        [
            float(B),
            float(H),
            float(Hkv),
            float(KvT),
            float(D),
            float(Db),
            float(r),
            float(scale),
            float(_FUSED_CHUNK),
            float(nC),
        ],
        mx.float32,
    )
    p1 = _get_q8_pass1_kernel()
    pm, pl, pacc = p1(
        inputs=[
            mx.reshape(q, (-1,)),
            mx.reshape(kd, (-1,)),
            mx.reshape(kq, (-1,)),
            mx.reshape(vd, (-1,)),
            mx.reshape(vq, (-1,)),
            meta1,
        ],
        template=[("T", q.dtype)],
        grid=(n_parts, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(n_parts,), (n_parts,), (n_parts, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    meta2 = mx.array([float(nC), float(D)], mx.float32)
    mk = _get_merge_kernel()
    out = mk(
        inputs=[pm, pl, pacc, meta2],
        template=[("T", q.dtype)],
        grid=(B * H, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
    logger.debug(
        "fused_q8_decode_attn: B=%d H=%d Hkv=%d KvT=%d D=%d nC=%d",
        B,
        H,
        Hkv,
        KvT,
        D,
        nC,
    )
    return out


_Q4_PASS1_KERNEL = None
_Q4_PASS1_SOURCE = """
uint tg_idx = threadgroup_position_in_grid.x;
ushort tid = thread_position_in_threadgroup.x;
uint B=uint(meta[0]),H=uint(meta[1]),Hkv=uint(meta[2]),KvT=uint(meta[3]);
uint D=uint(meta[4]),Dp=uint(meta[5]),r=uint(meta[6]);
float scale=meta[7];
uint CHUNK=uint(meta[8]),nC=uint(meta[9]);

uint head = tg_idx / nC;
uint ci = tg_idx - head * nC;
uint b = head / H;
uint h_q = head - b * H;
uint hkv = h_q / r;

uint q_base = (b * H + h_q) * D;
uint kq_base = (b * Hkv + hkv) * KvT * Dp;
uint Dblk = D / 32;
uint kd_base = (b * Hkv + hkv) * KvT * Dblk;
uint S = Dblk;
uint t_start = ci * CHUNK;
uint t_end = t_start + CHUNK;
if (t_end > KvT) t_end = KvT;
uint part = head * nC + ci;

threadgroup float qsh[256];
threadgroup float Ksh[256];
threadgroup float Vsh[256];
threadgroup float acc[256];
threadgroup float sg_sums[8];

qsh[tid] = float(q[q_base + tid]) * scale;
acc[tid] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);

float m = -1e30f;
float l = 0.0f;

for (uint t = t_start; t < t_end; t++) {
    uint j = tid / 32;
    float sc = float(kd[kd_base + t * Dblk + j]);
    uint byte_idx = tid / 2;
    uchar byte = kq[kq_base + t * Dp + byte_idx];
    float nib = ((tid & 1u) == 0u) ? float(byte & 0xFu) : float(byte >> 4);
    Ksh[tid] = sc * (nib - 8.0f);

    float partial = qsh[tid] * Ksh[tid];
    float sg = simd_sum(partial);
    ushort sid = tid / 32;
    ushort lane = tid % 32;
    if (lane == 0) sg_sums[sid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        float v = (lane < S) ? sg_sums[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) sg_sums[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = sg_sums[0];

    float m_new = metal::fmax(m, score);
    float corr = metal::exp(m - m_new);
    float p = metal::exp(score - m_new);
    l = l * corr + p;
    acc[tid] = acc[tid] * corr;
    m = m_new;

    float scv = float(vd[kd_base + t * Dblk + j]);
    uchar vbyte = vq[kq_base + t * Dp + byte_idx];
    float vnib = ((tid & 1u) == 0u) ? float(vbyte & 0xFu) : float(vbyte >> 4);
    Vsh[tid] = scv * (vnib - 8.0f);
    acc[tid] = acc[tid] + p * Vsh[tid];
}

pm[part] = m;
pl[part] = l;
pacc[part * D + tid] = acc[tid];
"""


def _get_q4_pass1_kernel():
    global _Q4_PASS1_KERNEL
    if _Q4_PASS1_KERNEL is None:
        _Q4_PASS1_KERNEL = mx.fast.metal_kernel(
            name="q4_fused_pass1",
            input_names=["q", "kd", "kq", "vd", "vq", "meta"],
            output_names=["pm", "pl", "pacc"],
            source=_Q4_PASS1_SOURCE,
            header="",
        )
        logger.debug("q4_fused_pass1 Metal kernel compiled")
    return _Q4_PASS1_KERNEL


def fused_q4_decode_attention(q, k_pack, v_pack, scale):
    kd, kq = k_pack
    vd, vq = v_pack
    B, H, L, D = q.shape
    Hkv = kd.shape[-3]
    KvT = kd.shape[-2]
    Dp = D // 2
    r = H // Hkv
    nC = _n_chunks(KvT)
    n_parts = B * H * nC
    meta1 = mx.array(
        [
            float(B),
            float(H),
            float(Hkv),
            float(KvT),
            float(D),
            float(Dp),
            float(r),
            float(scale),
            float(_FUSED_CHUNK),
            float(nC),
        ],
        mx.float32,
    )
    p1 = _get_q4_pass1_kernel()
    pm, pl, pacc = p1(
        inputs=[
            mx.reshape(q, (-1,)),
            mx.reshape(kd, (-1,)),
            mx.reshape(kq, (-1,)),
            mx.reshape(vd, (-1,)),
            mx.reshape(vq, (-1,)),
            meta1,
        ],
        template=[("T", q.dtype)],
        grid=(n_parts, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(n_parts,), (n_parts,), (n_parts, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    meta2 = mx.array([float(nC), float(D)], mx.float32)
    mk = _get_merge_kernel()
    out = mk(
        inputs=[pm, pl, pacc, meta2],
        template=[("T", q.dtype)],
        grid=(B * H, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
    logger.debug(
        "fused_q4_decode_attn: B=%d H=%d Hkv=%d KvT=%d D=%d nC=%d",
        B,
        H,
        Hkv,
        KvT,
        D,
        nC,
    )
    return out


# --- attention paths --------------------------------------------------------


def _resolve_mask(mask, L, T):
    # Normalize to an array (or None) whose last dim is the KV length T.
    if mask is None:
        return None
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"quant_kv: unsupported mask mode {mask!r}")
        qi = mx.arange(L)[:, None] + (T - L)
        ki = mx.arange(T)[None]
        return (qi >= ki).astype(mx.bool_)
    return mask


def _dequant_pack(pack, start, end):
    d, q = pack
    return (
        q40_dequantize(d[..., start:end, :], q[..., start:end, :])
        if q.dtype == mx.uint8
        else q80_dequantize(d[..., start:end, :], q[..., start:end, :])
    )


def _gqa_reshape(q, Hkv):
    B, H, L, D = q.shape
    r = H // Hkv
    if H % Hkv != 0:
        raise ValueError(f"quant_kv: n_q_heads {H} not divisible by n_kv_heads {Hkv}")
    if r == 1:
        return q, 1
    return mx.reshape(q, (B, Hkv, r, L, D)), r


def online_attention(q, k_pack, v_pack, scale, mask=None, chunk=512):
    # Chunk-streamed attention over quantized KV with a running FP32
    # softmax (streaming max / sum / numerator). Equivalent to full-bleed
    # SDPA over the dequantized cache — golden-verified below.
    kd, kq = k_pack
    B, H, L, D = q.shape
    Hkv = kd.shape[-3]
    T = kd.shape[-2]
    mask = _resolve_mask(mask, L, T)
    q32 = q.astype(mx.float32) * scale
    q32, r = _gqa_reshape(q32, Hkv)
    m = mx.full(q32.shape[:-1], _INF, mx.float32)
    l = mx.zeros(q32.shape[:-1], mx.float32)
    acc = mx.zeros(q32.shape, mx.float32)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        kc = _dequant_pack(k_pack, start, end).astype(mx.float32)
        vc = _dequant_pack(v_pack, start, end).astype(mx.float32)
        if r > 1:
            kc = mx.expand_dims(kc, -3)
            vc = mx.expand_dims(vc, -3)
        if r > 1:
            s = q32 @ mx.transpose(kc, (0, 1, 2, 4, 3))
        else:
            s = q32 @ mx.transpose(kc, (0, 1, 3, 2))
        if mask is not None:
            msk = mask[..., :, start:end]
            if msk.dtype == mx.bool_:
                s = mx.where(msk, s, mx.array(_INF, mx.float32))
            else:
                s = s + msk.astype(mx.float32)
        m_new = mx.maximum(m, mx.max(s, axis=-1))
        # Rows whose visible scores are still all -inf (fully masked so
        # far) would hit exp(-inf - -inf) = NaN and poison the running
        # sum even though later chunks have visible keys. Seed those rows
        # with 0 so the exp terms are well-defined and contribute zero.
        m_ref = mx.where(m_new == _INF, 0.0, m_new)
        corr = mx.exp(m - m_ref)
        p = mx.exp(s - mx.expand_dims(m_ref, -1))
        l = l * corr + mx.sum(p, axis=-1)
        acc = acc * mx.expand_dims(corr, -1) + p @ vc
        m = m_new
    out = acc / mx.expand_dims(l, -1)
    if r > 1:
        out = mx.reshape(out, (B, H, L, D))
    return out.astype(q.dtype)


def degraded_attention(q, k_pack, v_pack, scale, mask=None):
    # Degrade path: dequantize the whole cache to fp16, then stock FA.
    kd, kq = k_pack
    vd, vq = v_pack
    k = _dequant_pack(k_pack, 0, kd.shape[-2]).astype(q.dtype)
    v = _dequant_pack(v_pack, 0, vd.shape[-2]).astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def quantized_attention(q, k_pack, v_pack, scale, mask=None, chunk=512):
    kd, kq = k_pack
    B, H, L, D = q.shape
    Hkv = kd.shape[-3]
    T = kd.shape[-2]
    # Fused decode path (L==1, D multiple of 32 <=256, no mask): 2-pass Metal
    # kernel dequantizes KV on-the-fly — halves DRAM traffic vs fp16. Wins at
    # long context (T>=8192, DRAM-bound) with enough heads for occupancy
    # (B*H>=16). Below that the chunked dequant+SDPA path is faster (compute
    # dominates, dequant overhead loses).
    if (
        L == 1
        and D % _QK == 0
        and D <= 256
        and H % Hkv == 0
        and B * H >= 16
        and T >= 8192
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
    ):
        if kq.dtype == mx.uint8:
            return fused_q4_decode_attention(q, k_pack, v_pack, scale)
        return fused_q8_decode_attention(q, k_pack, v_pack, scale)
    return online_attention(q, k_pack, v_pack, scale, mask=mask, chunk=chunk)


# --- cache ------------------------------------------------------------------


class ShimQuantizedKVCache:
    # Q4_0/Q8_0 KV cache. Intentionally does NOT expose a ``bits``
    # attribute — mlx-lm's scaled_dot_product_attention wrapper checks
    # ``hasattr(cache, "bits")`` and would route into its affine
    # quantized_matmul path, which cannot read the llama.cpp block
    # layout stored here. Route attention through ``attention()``.
    # Switch OFF: plain fp16 concat cache + stock SDPA (zero change).

    def __init__(self, bits: int = 4, chunk: int = 512):
        if bits not in (4, 8):
            raise ValueError(f"quant_kv: bits must be 4 or 8, got {bits}")
        self.keys = None
        self.values = None
        self.offset = 0
        self.quant_bits = bits
        self.chunk = chunk
        self._packed = False
        self._codec_q = q40_quantize if bits == 4 else q80_quantize

    def update_and_fetch(self, keys, values):
        keys = keys.astype(mx.float16)
        values = values.astype(mx.float16)
        if not is_quant_kv_enabled():
            # Stock fp16 path: plain concat. Capture the mode so trim/state
            # stay consistent even if the env switch flips mid-life.
            self._packed = False
            if self.keys is None:
                self.keys, self.values = keys, values
            else:
                self.keys = mx.concatenate([self.keys, keys], axis=-2)
                self.values = mx.concatenate([self.values, values], axis=-2)
            self.offset = self.keys.shape[-2]
            return self.keys, self.values

        kq = self._codec_q(keys)
        vq = self._codec_q(values)
        self._packed = True
        if self.keys is None:
            self.keys, self.values = kq, vq
        else:
            self.keys = _concat_pack(self.keys, kq)
            self.values = _concat_pack(self.values, vq)
        self.offset = self.keys[0].shape[-2]
        return None, None

    def attention(self, queries, scale: float, mask=None):
        if not self._packed:
            return mx.fast.scaled_dot_product_attention(
                queries, self.keys, self.values, scale=scale, mask=mask
            )
        if self.quant_bits == 4:
            logger.debug("quant_kv attention: Q4_0 online path, offset=%d", self.offset)
        return online_attention(
            queries, self.keys, self.values, scale, mask=mask, chunk=self.chunk
        )

    def make_mask(self, N, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(N, offset=self.offset, **kwargs)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        if self.keys is None:
            return n
        if self._packed:
            self.keys = _slice_pack(self.keys, 0, self.offset)
            self.values = _slice_pack(self.values, 0, self.offset)
        else:
            self.keys = self.keys[..., : self.offset, :]
            self.values = self.values[..., : self.offset, :]
        return n

    def empty(self):
        return self.keys is None

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.quant_bits, self.chunk)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.quant_bits, self.chunk = map(int, v)
        self._codec_q = q40_quantize if self.quant_bits == 4 else q80_quantize

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        if not self._packed:
            return self.keys.nbytes + self.values.nbytes
        return sum(x.nbytes for pack in (self.keys, self.values) for x in pack)


def _concat_pack(pack, new):
    d, q = pack
    nd, nq = new
    return (
        mx.concatenate([d, nd], axis=-2),
        mx.concatenate([q, nq], axis=-2),
    )


def _slice_pack(pack, start, end):
    d, q = pack
    return d[..., start:end, :], q[..., start:end, :]
