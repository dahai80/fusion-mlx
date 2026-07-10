import math

import mlx.core as mx
import mlx.nn as nn


class T5LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class T5RelativeEmbedding(nn.Module):

    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        bidirectional: bool = True,
        max_dist: int = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, rel_pos: mx.array) -> mx.array:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(mx.int32) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = mx.zeros_like(rel_pos, dtype=mx.int32)
            rel_pos = mx.maximum(-rel_pos, mx.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact

        rel_pos_f = rel_pos.astype(mx.float32)
        rel_pos_large = max_exact + (
            mx.log(rel_pos_f / max_exact)
            / math.log(self.max_dist / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        rel_pos_large = mx.minimum(
            rel_pos_large,
            mx.full(rel_pos_large.shape, num_buckets - 1, dtype=mx.int32),
        )

        rel_buckets = rel_buckets + mx.where(
            is_small, rel_pos.astype(mx.int32), rel_pos_large
        )
        return rel_buckets

    def __call__(self, lq: int, lk: int) -> mx.array:
        positions_k = mx.arange(lk)[None, :]  # [1, lk]
        positions_q = mx.arange(lq)[:, None]  # [lq, 1]
        rel_pos = positions_k - positions_q  # [lq, lk]

        buckets = self._relative_position_bucket(rel_pos)
        embeds = self.embedding(buckets)  # [lq, lk, num_heads]
        embeds = embeds.transpose(2, 0, 1)[None, :, :, :]  # [1, N, lq, lk]
        return embeds


class T5Attention(nn.Module):

    def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        pos_bias: mx.array | None = None,
    ) -> mx.array:
        context = x if context is None else context
        b, n, c = x.shape[0], self.num_heads, self.head_dim

        q = self.q(x).reshape(b, -1, n, c)  # [B, Lq, N, C]
        k = self.k(context).reshape(b, -1, n, c)  # [B, Lk, N, C]
        v = self.v(context).reshape(b, -1, n, c)

        # T5 uses no scaling — compute attention manually with float32 softmax
        # to match official: F.softmax(attn.float(), dim=-1).type_as(attn)
        # Using SDPA with bfloat16 inputs causes precision loss in softmax
        # since unscaled logits can be very large (no 1/sqrt(d) division).
        q = q.transpose(0, 2, 1, 3)  # [B, N, Lq, C]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # QK^T (no scaling) — compute in float32 for precision
        attn = q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)

        # Add position bias
        if pos_bias is not None:
            attn = attn + pos_bias.astype(mx.float32)

        # Apply attention mask (use dtype min like official, not -1e9)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]  # [B, 1, 1, Lk]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]  # [B, 1, Lq, Lk]
            additive_mask = mx.where(mask == 0, -3.389e38, 0.0).astype(mx.float32)
            attn = attn + additive_mask

        # Softmax in float32 (matches official), then cast back
        attn = mx.softmax(attn, axis=-1).astype(q.dtype)

        # Attention @ V
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, -1, n * c)
        return self.o(out)


class T5FeedForward(nn.Module):

    def __init__(self, dim: int, dim_ffn: int):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.gate_proj = nn.Linear(dim, dim_ffn, bias=False)
        self.gate_act = nn.GELU(approx="tanh")
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.fc1(x) * self.gate_act(self.gate_proj(x)))


class T5SelfAttentionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
    ):
        super().__init__()
        self.shared_pos = shared_pos
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        pos_bias: mx.array | None = None,
    ) -> mx.array:
        e = pos_bias if self.shared_pos else self.pos_embedding(x.shape[1], x.shape[1])
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


class T5Encoder(nn.Module):

    def __init__(
        self,
        vocab_size: int = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        shared_pos: bool = False,
    ):
        super().__init__()
        self.dim = dim

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
            if shared_pos
            else None
        )
        self.blocks = [
            T5SelfAttentionBlock(
                dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos
            )
            for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    def __call__(self, ids: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.token_embedding(ids)

        e = self.pos_embedding(x.shape[1], x.shape[1]) if self.pos_embedding else None
        for block in self.blocks:
            x = block(x, mask=mask, pos_bias=e)

        x = self.norm(x)
        return x
