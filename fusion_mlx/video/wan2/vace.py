# SPDX-License-Identifier: Apache-2.0
# VACE (Video-Conditioned Auxiliary Control Encoding) blocks for Wan2.
# These run a parallel control branch alongside the main DiT blocks.
# At specified vace_layers, the control output is injected into the main
# DiT hidden states as a residual: x = x + control_hint * scale.

import mlx.core as mx
import mlx.nn as nn

from .attention import WanCrossAttention, WanLayerNorm, WanSelfAttention
from .transformer import WanFFN


class VACEBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        has_before_proj: bool = False,
    ):
        super().__init__()

        self.has_before_proj = has_before_proj
        if has_before_proj:
            self.before_proj = nn.Linear(dim, dim)

        # Self-attention
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # Cross-attention
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else None
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm, eps)

        # Feed-forward
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)

        # Output projection (always present in VACE blocks)
        self.after_proj = nn.Linear(dim, dim)

        # Learned modulation: 6 vectors for scale/shift/gate
        self.modulation = (mx.random.normal((1, 6, dim)) * (dim**-0.5)).astype(
            mx.float32
        )

    def __call__(
        self,
        x: mx.array,
        ctrl: mx.array,
        e: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
        context: mx.array,
        context_lens: list | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
    ) -> tuple[mx.array | None, mx.array]:
        # before_proj: project control state and add main hidden state
        if self.has_before_proj:
            ctrl = self.before_proj(ctrl)
            ctrl = ctrl + x

        # Modulation
        mod = self.modulation + e
        e0, e1, e2, e3, e4, e5 = (
            mod[:, :, 0, :],
            mod[:, :, 1, :],
            mod[:, :, 2, :],
            mod[:, :, 3, :],
            mod[:, :, 4, :],
            mod[:, :, 5, :],
        )

        # Self-attention
        ctrl_mod = self.norm1(ctrl) * (1 + e1) + e0
        y = self.self_attn(
            ctrl_mod,
            seq_lens,
            grid_sizes,
            freqs,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )
        ctrl = ctrl + y * e2

        # Cross-attention
        ctrl_cross = self.norm3(ctrl) if self.norm3 is not None else ctrl
        ctrl = ctrl + self.cross_attn(ctrl_cross, context, context_lens)

        # FFN
        ctrl_mod = self.norm2(ctrl) * (1 + e4) + e3
        y = self.ffn(ctrl_mod)
        ctrl = ctrl + y * e5

        # Output projection -> conditioning hint for main DiT
        conditioning = self.after_proj(ctrl)

        return conditioning, ctrl
