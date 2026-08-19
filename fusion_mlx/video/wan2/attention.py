import logging
import os

import mlx.core as mx
import mlx.nn as nn

from .rope import rope_apply

logger = logging.getLogger(__name__)

try:
    from fusion_mlx.custom_kernels.xfuser_attention import (
        current_step as _fa_step,
    )
    from fusion_mlx.custom_kernels.xfuser_attention import (
        is_active as _fa_active,
    )
except Exception:  # pragma: no cover - xfuser strategy optional
    _fa_step = None
    _fa_active = None


def _q8_fp32_seq() -> int:
    # MLX QuantizedLinear with bfloat16 INPUT accumulates the per-group
    # dot product in bfloat16. At long patch-seq (Wan2.2-TI2V-5B
    # 1280x704x121f -> seq=27280) the self-attention o-projection (q8,
    # group_size=64, 3072 in-features) overflows the bf16 accumulator to
    # bf16 max (1.70141e+38) even though the SDPA output feeding it is
    # finite (am=5.65). The saturated value then propagates through the
    # residual and norms -> fp32 overflow -> all-NaN latents (#500/#503).
    # PyTorch fp16/bf16 matmuls accumulate in float32, so the official
    # Wan2 never hits this; it is MLX-q8-specific.
    # Cast the o-proj input to float32 above this seq threshold so the
    # q8 matmul accumulates in float32 (verified finite: am=19.5 vs
    # 1.7e38 saturated). 0 = always bf16 input (legacy, NaN at long seq).
    try:
        return max(0, int(os.getenv("FUSION_WAN2_Q8_FP32_SEQ", "16384")))
    except (TypeError, ValueError):
        return 16384


def _attn_chunk_size() -> int:
    # Metal hangs/freeze on very large attention matrices (seq>=~50k,
    # 40 heads, dim 5120). Chunking the Q sequence caps the per-op
    # attention matrix at (B, H, chunk, seq) instead of (B, H, seq, seq).
    # 0 = disabled (use single mx.fast SDPA). Env-only knob so existing
    # callers are untouched.
    try:
        return max(0, int(os.getenv("FUSION_WAN2_ATTN_CHUNK", "0")))
    except (TypeError, ValueError):
        return 0


def _sdpa(q, k, v, scale, mask=None, *, fast_attn=None, step=0, batch_size=None):
    if fast_attn is not None:
        return fast_attn(q, k, v, step, scale=scale, mask=mask, batch_size=batch_size)
    # NOTE: skip mfa_bridge for wan2 attention. Wan2 uses num_heads=40 which
    # exceeds the _normalize_qkv_layout heuristic range (1-32), causing it to
    # mis-transpose q/k/v when seq_len <= 32. Since wan2 already emits (B,H,N,D)
    # layout, call mx.fast.scaled_dot_product_attention directly.
    if mask is not None:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def _sdpa_chunked(q, k, v, scale, mask=None):
    # Exact chunked SDPA: split Q along seq (dim 2), each chunk attends to
    # the full K/V. softmax(Q_chunk @ K^T) @ V is identical to the unchunked
    # result because each Q row is independent. Cuts peak attention memory
    # from seq*seq to chunk*seq. Used when seq exceeds Metal freeze threshold.
    b, h, s, d = q.shape
    chunk = _attn_chunk_size()
    if chunk <= 0 or s <= chunk:
        return _sdpa(q, k, v, scale, mask)
    logger.info(
        "wan2 chunked sdpa: seq=%d heads=%d chunk=%d (FUSION_WAN2_ATTN_CHUNK)",
        s,
        h,
        chunk,
    )
    outs = []
    for start in range(0, s, chunk):
        end = min(start + chunk, s)
        q_chunk = q[:, :, start:end, :]
        out_chunk = _sdpa(q_chunk, k, v, scale, mask)
        outs.append(out_chunk)
        mx.eval(out_chunk)
    return mx.concatenate(outs, axis=2)


def _linear_dtype(layer) -> mx.Dtype:
    # Unwrap LoRA wrapper to get the underlying linear layer
    inner = getattr(layer, "linear", layer)
    if isinstance(inner, nn.QuantizedLinear):
        return inner.scales.dtype
    # FP8Linear 无 .weight (存 fp8_weight), 用 compute_dtype (fp8_matmul 实际 dtype) (#142).
    # 不能用 fp8_weight.dtype: FP8 硬件下为 float8, x.astype(float8) 与 bf16 权重 matmul 错配.
    compute_dtype = getattr(inner, "compute_dtype", None)
    if compute_dtype is not None:
        return compute_dtype
    return inner.weight.dtype


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class WanLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.elementwise_affine:
            return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)
        else:
            return mx.fast.layer_norm(x, None, None, self.eps)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim

        # Cast to compute dtype for efficient matmul (bfloat16 matching official autocast).
        # q8 q/k/v share the bf16-accumulator overflow of the o-proj (#500/#503):
        # at later blocks the residual grows (b1 entry am~73) and the q8
        # q-proj (3072 in-features) overflows its bf16 accumulator on bf16
        # input. Cast to float32 above the seq threshold so the q8 matmuls
        # accumulate in float32. Gated by seq so short-seq runs keep bf16.
        w_dtype = _linear_dtype(self.q)
        if s > _q8_fp32_seq():
            x_w = x.astype(mx.float32)
        else:
            x_w = x.astype(w_dtype)

        q = self.q(x_w)
        k = self.k(x_w)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        q = q.reshape(b, s, n, d)
        k = k.reshape(b, s, n, d)
        v = self.v(x_w).reshape(b, s, n, d)

        # RoPE in float32 for precision (official uses float64)
        q = rope_apply(
            q.astype(mx.float32), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin
        )
        k = rope_apply(
            k.astype(mx.float32), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin
        )

        # Cast back to weight dtype for efficient attention (matching official q.to(v.dtype))
        q = q.astype(w_dtype).transpose(0, 2, 1, 3)
        k = k.astype(w_dtype).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Use precomputed mask or build from seq_lens
        mask = attn_mask
        if mask is None and any(sl < s for sl in seq_lens):
            mask = mx.zeros((b, 1, 1, s), dtype=q.dtype)
            for i, sl in enumerate(seq_lens):
                mask[i, :, :, sl:] = -1e9

        # Use memory-efficient scaled dot-product attention
        # mx.fast.scaled_dot_product_attention expects [B, N, L, D]
        fa = getattr(self, "_fast_attn", None)
        if fa is not None and _fa_active is not None and _fa_active():
            out = _sdpa(
                q,
                k,
                v,
                self.scale,
                mask,
                fast_attn=fa,
                step=_fa_step(),
                batch_size=b,
            )
        else:
            # Fail-visible: large seq freezes Metal on a single seq*seq
            # attention matrix. Warn once per freeze-sized call so the
            # user knows to set FUSION_WAN2_ATTN_CHUNK.
            chunk = _attn_chunk_size()
            if chunk <= 0 and s > 16384:
                logger.warning(
                    "wan2 self-attn seq=%d exceeds safe threshold; "
                    "set FUSION_WAN2_ATTN_CHUNK (e.g. 8192) to avoid "
                    "Metal freeze",
                    s,
                )
            out = _sdpa_chunked(q, k, v, self.scale, mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        # q8 o-proj: cast input to float32 at long seq to avoid bf16
        # accumulator overflow in the QuantizedLinear matmul (#500/#503).
        # SDPA emits bf16; feeding bf16 to the q8 o-proj saturates to
        # 1.7e38 at seq>=~16k. fp32 input accumulates in fp32 -> finite.
        if s > _q8_fp32_seq():
            out = out.astype(mx.float32)
        return self.o(out)


class WanCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

    def prepare_kv(self, context: mx.array) -> tuple:
        b = context.shape[0]
        n, d = self.num_heads, self.head_dim
        # Cast to compute dtype for efficient matmul
        w_dtype = _linear_dtype(self.k)
        ctx = context.astype(w_dtype)
        k = self.k(ctx)
        if self.norm_k is not None:
            k = self.norm_k(k)
        k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v = self.v(ctx).reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        return k, v

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        context_lens: list | None = None,
        kv_cache: tuple | None = None,
    ) -> mx.array:
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim

        # Cast to compute dtype for efficient matmul (bfloat16 matching official autocast).
        # q8 q/k/v share the bf16-accumulator overflow of the o-proj (#500/#503):
        # the q-proj input is the residual stream, which grows at later blocks
        # (b1 entry am~73) and overflows the q8 bf16 accumulator. Cast to
        # float32 above the seq threshold (denoiser token count x.shape[1]).
        w_dtype = _linear_dtype(self.q)
        s = x.shape[1]
        q_in = x.astype(mx.float32) if s > _q8_fp32_seq() else x.astype(w_dtype)
        q = self.q(q_in)
        if self.norm_q is not None:
            q = self.norm_q(q)
        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            k, v = kv_cache
        else:
            ctx_in = (
                context.astype(mx.float32)
                if s > _q8_fp32_seq()
                else context.astype(w_dtype)
            )
            k = self.k(ctx_in)
            if self.norm_k is not None:
                k = self.norm_k(k)
            k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
            v = self.v(ctx_in).reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # Optional context masking
        mask = None
        if context_lens is not None:
            ctx_len = k.shape[2]
            mask = mx.zeros((b, 1, 1, ctx_len), dtype=q.dtype)
            for i, cl in enumerate(context_lens):
                mask[i, :, :, cl:] = -1e9

        fa = getattr(self, "_fast_attn", None)
        if fa is not None and _fa_active is not None and _fa_active():
            out = _sdpa(
                q,
                k,
                v,
                self.scale,
                mask,
                fast_attn=fa,
                step=_fa_step(),
                batch_size=b,
            )
        else:
            # Cross-attn Q seq is short (text tokens) so freeze risk is
            # low; still route through chunked helper for consistency.
            out = _sdpa_chunked(q, k, v, self.scale, mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * d)
        # q8 o-proj: same bf16-accumulator overflow as self-attn (#500/#503).
        # The o-proj output feeds back into the long-seq residual, so gate on
        # the denoiser token count (x.shape[1], e.g. 27280 for 1280x704x121f),
        # not the short text-context length. Verified: bf16 input (am=1.27
        # finite) -> cross.o OUT nan=106170; fp32 input -> finite.
        if x.shape[1] > _q8_fp32_seq():
            out = out.astype(mx.float32)
        return self.o(out)
