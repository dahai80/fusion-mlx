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


@mx.compile
def _q40_dequant_kernel(d, packed):
    p = packed.reshape(*packed.shape[:-1], -1, _Q40_BYTES_PER_BLOCK)
    lo = (p & 0xF).astype(mx.float32)
    hi = (p >> 4).astype(mx.float32)
    q = mx.stack([lo, hi], axis=-1).reshape(*p.shape[:-1], _QK) - 8.0
    out = q * mx.expand_dims(d.astype(mx.float32), -1)
    return mx.reshape(out, (*out.shape[:-2], -1))


def q40_dequantize(d, packed):
    return _q40_dequant_kernel(d, packed)


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


@mx.compile
def _q80_dequant_kernel(d, qs):
    q = qs.astype(mx.float32).reshape(*qs.shape[:-1], -1, _QK)
    out = q * mx.expand_dims(d.astype(mx.float32), -1)
    return mx.reshape(out, (*out.shape[:-2], -1))


def q80_dequantize(d, qs):
    return _q80_dequant_kernel(d, qs)


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
