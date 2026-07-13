# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 attention (vendored from mlx-video).
# Phase 4 LTX-2 direct-MLX port: model-layer foundation.
import math

import mlx.core as mx
import mlx.nn as nn

from .config import LTXRopeType
from .rope import apply_rotary_emb

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


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    mask: mx.array | None = None,
    *,
    fast_attn=None,
    step: int = 0,
    batch_size: int | None = None,
) -> mx.array:

    b, q_seq_len, dim = q.shape
    _, kv_seq_len, _ = k.shape
    dim_head = dim // heads

    q = mx.reshape(q, (b, q_seq_len, heads, dim_head))
    k = mx.reshape(k, (b, kv_seq_len, heads, dim_head))
    v = mx.reshape(v, (b, kv_seq_len, heads, dim_head))

    q = mx.swapaxes(q, 1, 2)
    k = mx.swapaxes(k, 1, 2)
    v = mx.swapaxes(v, 1, 2)

    if mask is not None:
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=0)
        if mask.ndim == 3:
            mask = mx.expand_dims(mask, axis=1)

    scale = 1.0 / math.sqrt(dim_head)

    if fast_attn is not None:
        out = fast_attn(q, k, v, step, scale=scale, mask=mask, batch_size=batch_size)
    elif _mfa_available is not None and _mfa_available():
        out = _mfa_flash_attention(q, k, v, scale=scale, mask=mask)
    else:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    out = mx.swapaxes(out, 1, 2)
    out = mx.reshape(out, (b, q_seq_len, heads * dim_head))

    return out


class Attention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        has_gate_logits: bool = False,
    ):
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

        if has_gate_logits:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        pe: tuple[mx.array, mx.array] | None = None,
        k_pe: tuple[mx.array, mx.array] | None = None,
        skip_attention: bool = False,
    ) -> mx.array:
        gate = None
        if hasattr(self, "to_gate_logits"):
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))

        context = x if context is None else context
        v = self.to_v(context)

        if skip_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k_pe_to_use = pe if k_pe is None else k_pe
                k = apply_rotary_emb(k, k_pe_to_use, self.rope_type)

            fa = getattr(self, "_fast_attn", None)
            if fa is not None and _fa_active is not None and _fa_active():
                out = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    self.heads,
                    mask,
                    fast_attn=fa,
                    step=_fa_step(),
                )
            else:
                out = scaled_dot_product_attention(q, k, v, self.heads, mask)

        if gate is not None:
            b, seq_len, _ = out.shape
            out = mx.reshape(out, (b, seq_len, self.heads, self.dim_head))
            out = out * gate[..., None]
            out = mx.reshape(out, (b, seq_len, -1))

        return self.to_out(out)
