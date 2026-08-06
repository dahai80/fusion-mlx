import logging

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def _act(name: str):
    if name == "quick_gelu":
        return lambda x: x * mx.sigmoid(1.702 * x)
    if name == "gelu":
        return nn.gelu
    return nn.gelu


class CLIPAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        b, s, _ = hidden.shape
        q = (
            self.q_proj(hidden)
            .reshape(b, s, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(hidden)
            .reshape(b, s, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(hidden)
            .reshape(b, s, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        out = scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.num_heads * self.head_dim)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, dims: int, intermediate: int, act: str):
        super().__init__()
        self.fc1 = nn.Linear(dims, intermediate)
        self.fc2 = nn.Linear(intermediate, dims)
        self.act = _act(act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, dims: int, num_heads: int, intermediate: int, act: str):
        super().__init__()
        self.self_attn = CLIPAttention(dims, num_heads)
        self.layer_norm1 = nn.LayerNorm(dims)
        self.mlp = CLIPMLP(dims, intermediate, act)
        self.layer_norm2 = nn.LayerNorm(dims)

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        h = hidden + self.self_attn(self.layer_norm1(hidden), mask)
        h = h + self.mlp(self.layer_norm2(h))
        return h


class CLIPEncoder(nn.Module):
    def __init__(
        self, dims: int, num_layers: int, num_heads: int, intermediate: int, act: str
    ):
        super().__init__()
        self.layers = [
            CLIPEncoderLayer(dims, num_heads, intermediate, act)
            for _ in range(num_layers)
        ]

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return hidden


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, dims: int, vocab: int, max_pos: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dims)
        self.position_embedding = nn.Embedding(max_pos, dims)

    def __call__(self, tokens: mx.array) -> mx.array:
        s = tokens.shape[-1]
        pos = mx.arange(s).reshape(1, s)
        return self.token_embedding(tokens) + self.position_embedding(pos)


class CLIPTextModel(nn.Module):
    def __init__(
        self,
        dims: int = 1280,
        num_layers: int = 32,
        num_heads: int = 16,
        intermediate: int = 5120,
        act: str = "gelu",
        vocab: int = 49408,
        max_pos: int = 77,
    ):
        super().__init__()
        self.dims = dims
        self.text_model = nn.Module()
        self.text_model.embeddings = CLIPTextEmbeddings(dims, vocab, max_pos)
        self.text_model.encoder = CLIPEncoder(
            dims, num_layers, num_heads, intermediate, act
        )
        self.text_model.final_layer_norm = nn.LayerNorm(dims)

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = self.text_model.embeddings(tokens)
        mask = _causal_mask(hidden.shape).astype(hidden.dtype)
        hidden = self.text_model.encoder(hidden, mask)
        hidden = self.text_model.final_layer_norm(hidden)
        eos = mx.argmax(tokens, axis=-1)
        pooled = hidden[mx.arange(hidden.shape[0]), eos]
        return pooled


def _causal_mask(shape: tuple) -> mx.array:
    b, s, _ = shape
    mask = mx.tril(mx.ones((s, s)))
    mask = (1 - mask) * -3.4e38
    mask = mask.reshape(1, 1, s, s)
    return mx.broadcast_to(mask, (b, 1, s, s))
