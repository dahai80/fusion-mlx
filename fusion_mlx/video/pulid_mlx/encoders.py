"""PuLID IDFormer encoder — Perceiver-resampler for identity embedding.

Fuses ArcFace (1280-d) + EVA-CLIP (5 x 1024-d hidden layers) into a
2048-d output that gets injected into Flux DiT via PerceiverAttentionCA.

Pure MLX port of pulid/encoders_transformer.py (IDFormer only).
"""
import math
import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = [
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.Linear(inner_dim, dim, bias=False),
        ]

    def __call__(self, x):
        x = self.net[0](x)
        x = nn.gelu(self.net[1](x))
        return self.net[2](x)


class PerceiverAttention(nn.Module):
    """Perceiver cross-attention: latents attend to context (x + latents)."""

    def __init__(self, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        kv_dim = kv_dim or dim

        self.norm1 = nn.LayerNorm(kv_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv_input = mx.concatenate([x, latents], axis=1)
        k, v = mx.split(self.to_kv(kv_input), 2, axis=-1)

        q = q.reshape(b, seq_len, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, k.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, v.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(0, 1, 3, 2)
        weight = mx.softmax(weight.astype(mx.float32), axis=-1).astype(q.dtype)
        out = weight @ v

        out = out.transpose(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.to_out(out)


class MappingLayer(nn.Module):
    """MLP mapping from EVA-CLIP hidden (1024) to IDFormer dim."""

    def __init__(self, in_dim=1024, out_dim=1024):
        super().__init__()
        self.net = [
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        ]

    def __call__(self, x):
        x = nn.leaky_relu(self.net[0](x))
        x = nn.leaky_relu(self.net[2](x))
        return self.net[4](x)


class IDEmbeddingMapping(nn.Module):
    """Maps concatenated ArcFace + EVA-CLIP CLS -> num_id_token * dim."""

    def __init__(self, in_dim=1280, hidden_dim=1024, out_dim=1024, num_id_token=5):
        super().__init__()
        self.num_id_token = num_id_token
        self.net = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim * num_id_token),
        ]

    def __call__(self, x):
        x = nn.leaky_relu(self.net[0](x))
        x = nn.leaky_relu(self.net[2](x))
        x = self.net[4](x)
        return x.reshape(x.shape[0], self.num_id_token, -1)


class IDFormer(nn.Module):
    """Perceiver-resampler that fuses identity embeddings for PuLID.

    Architecture:
    - 5 mapping layers for 5 EVA-CLIP hidden states (1024 -> dim)
    - ID embedding mapping (1280 -> num_id_token * dim)
    - 10 Perceiver layers (2 per scale) with self+cross attention
    - Output projection (dim -> output_dim=2048)
    """

    def __init__(
        self,
        dim=1024,
        depth=10,
        dim_head=64,
        heads=16,
        num_id_token=5,
        num_queries=32,
        output_dim=2048,
        ff_mult=4,
    ):
        super().__init__()
        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth_per_scale = depth // 5

        scale = dim ** -0.5
        self.latents = mx.random.normal((1, num_queries, dim)) * scale
        self.proj_out = mx.random.normal((dim, output_dim)) * scale

        self.layers = []
        for _ in range(depth):
            self.layers.append([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ])

        self.mappings = []
        for i in range(5):
            self.mappings.append(MappingLayer(1024, dim))

        self.id_embedding_mapping = IDEmbeddingMapping(
            in_dim=1280, hidden_dim=1024, out_dim=dim, num_id_token=num_id_token
        )

    def __call__(self, id_cond, vit_hidden):
        """Forward pass.

        Args:
            id_cond: (B, 1280) concatenated ArcFace + EVA-CLIP [CLS]
            vit_hidden: list of 5 tensors, each (B, seq, 1024)
        Returns:
            (B, num_queries, output_dim) ID embedding for Flux injection
        """
        b = id_cond.shape[0]
        latents = mx.broadcast_to(self.latents, (b, self.num_queries, self.dim))

        id_tokens = self.id_embedding_mapping(id_cond)
        latents = mx.concatenate([latents, id_tokens], axis=1)

        for scale_idx in range(5):
            vit_feature = self.mappings[scale_idx](vit_hidden[scale_idx])
            ctx_feature = mx.concatenate([id_tokens, vit_feature], axis=1)

            start = scale_idx * self.depth_per_scale
            end = start + self.depth_per_scale
            for attn, ff in self.layers[start:end]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents
