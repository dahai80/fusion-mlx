import mlx.core as mx
import mlx.nn as nn

from .rope import rope_apply

try:
    from fusion_mlx.custom_kernels.mfa_bridge import (
        flash_attention as _mfa_flash_attention,
    )
    from fusion_mlx.custom_kernels.mfa_bridge import (
        is_available as _mfa_available,
    )
except Exception:  # pragma: no cover - optional Metal Flash Attention extension
    _mfa_flash_attention = None
    _mfa_available = None

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


def _sdpa(q, k, v, scale, mask=None, *, fast_attn=None, step=0, batch_size=None):
    if fast_attn is not None:
        return fast_attn(q, k, v, step, scale=scale, mask=mask, batch_size=batch_size)
    if _mfa_available is not None and _mfa_available():
        return _mfa_flash_attention(q, k, v, scale=scale, mask=mask)
    if mask is not None:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


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

        # Cast to compute dtype for efficient matmul (bfloat16 matching official autocast)
        w_dtype = _linear_dtype(self.q)
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
            out = _sdpa(q, k, v, self.scale, mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
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

        # Cast to compute dtype for efficient matmul (bfloat16 matching official autocast)
        w_dtype = _linear_dtype(self.q)
        q = self.q(x.astype(w_dtype))
        if self.norm_q is not None:
            q = self.norm_q(q)
        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            k, v = kv_cache
        else:
            ctx = context.astype(w_dtype)
            k = self.k(ctx)
            if self.norm_k is not None:
                k = self.norm_k(k)
            k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
            v = self.v(ctx).reshape(b, -1, n, d).transpose(0, 2, 1, 3)

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
            out = _sdpa(q, k, v, self.scale, mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * d)
        return self.o(out)
