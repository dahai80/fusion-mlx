import mlx.core as mx
import mlx.nn as nn

from .attention import WanCrossAttention, WanLayerNorm, WanSelfAttention, _linear_dtype


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)

        # Cross-attention (with optional norm on context)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else None
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm, eps)

        # Feed-forward
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)

        # Learned modulation: 6 vectors for scale/shift/gate (kept in float32 for precision)
        self.modulation = (mx.random.normal((1, 6, dim)) * (dim**-0.5)).astype(
            mx.float32
        )

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
        context: mx.array,
        context_lens: list | None = None,
        cross_kv_cache: tuple | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        # Modulation: compute in float32 for precision, matching the reference
        # which keeps residual x in float32 via torch.amp.autocast(dtype=float32).
        # By keeping modulation in float32, type promotion ensures the residual
        # stream stays float32 throughout all 30 layers (gate * output + x → float32).
        mod = self.modulation + e  # float32
        e0, e1, e2, e3, e4, e5 = (
            mod[:, :, 0, :],  # shift for self-attn
            mod[:, :, 1, :],  # scale for self-attn
            mod[:, :, 2, :],  # gate for self-attn
            mod[:, :, 3, :],  # shift for ffn
            mod[:, :, 4, :],  # scale for ffn
            mod[:, :, 5, :],  # gate for ffn
        )

        # Self-attention with modulation (hidden state stays in w_dtype)
        x_mod = self.norm1(x) * (1 + e1) + e0
        y = self.self_attn(
            x_mod,
            seq_lens,
            grid_sizes,
            freqs,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )
        x = x + y * e2

        # Cross-attention (no modulation, just norm)
        x_cross = self.norm3(x) if self.norm3 is not None else x
        x = x + self.cross_attn(x_cross, context, context_lens, kv_cache=cross_kv_cache)

        # FFN with modulation
        x_mod = self.norm2(x) * (1 + e4) + e3
        y = self.ffn(x_mod)
        x = x + y * e5

        return x


class WanFFN(nn.Module):

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.act = nn.GELU(approx="tanh")
        self.fc2 = nn.Linear(ffn_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Cast to compute dtype for efficient matmul (bfloat16 matching official autocast)
        x_w = x.astype(_linear_dtype(self.fc1))
        return self.fc2(self.act(self.fc1(x_w)))
