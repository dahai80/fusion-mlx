"""PuLID attention processor — ID embedding injection into Flux DiT.

PerceiverAttentionCA: cross-attention that injects 2048-d ID embeddings
into Flux's 3072-d attention stream.

IDAttnProcessor: hooks into Flux attention layers, applies ORTHO/ORTHO_v2
regularization to prevent identity collapse.

Pure MLX port of pulid/attention_processor.py.
"""
import math
import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class PerceiverAttentionCA(nn.Module):
    """Cross-attention for injecting ID embeddings into Flux DiT.

    Flux attention operates at dim=3072; this module projects the
    2048-d ID output from IDFormer into the same space and performs
    cross-attention injection.
    """

    def __init__(self, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, x, id_emb):
        """Cross-attention: Flux hidden states attend to ID embeddings.

        Args:
            x: (B, seq, dim) Flux hidden states
            id_emb: (B, num_queries, kv_dim) IDFormer output
        Returns:
            (B, seq, dim) attention output
        """
        x_norm = self.norm1(x)
        id_norm = self.norm2(id_emb)

        b, seq_len, _ = x_norm.shape

        q = self.to_q(x_norm)
        k, v = mx.split(self.to_kv(id_norm), 2, axis=-1)

        q = q.reshape(b, seq_len, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, k.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, v.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(0, 1, 3, 2)
        weight = mx.softmax(weight.astype(mx.float32), axis=-1).astype(q.dtype)
        out = weight @ v

        out = out.transpose(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.to_out(out)


class IDAttnProcessor(nn.Module):
    """Attention processor that injects ID embeddings into Flux attention.

    Hooks into Flux's self-attention layers and adds ID cross-attention
    via id_to_k/id_to_v linear projections. Supports ORTHO and ORTHO_v2
    regularization modes to prevent identity embedding collapse.

    Usage: replace Flux attention processor with this, set id_embedding
    before each forward pass, then the processor automatically adds
    ID cross-attention to the normal self-attention output.
    """

    ORTHO = "ortho"
    ORTHO_V2 = "ortho_v2"
    NONE = "none"

    def __init__(
        self,
        dim=3072,
        dim_head=128,
        heads=16,
        kv_dim=2048,
        ortho_mode="ortho_v2",
        scale=1.0,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.dim = dim
        self.ortho_mode = ortho_mode
        self.scale = scale
        inner_dim = dim_head * heads

        self.id_to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.id_to_v = nn.Linear(kv_dim, inner_dim, bias=False)

        self._id_embedding = None

    def set_id_embedding(self, id_embedding):
        """Set the ID embedding from IDFormer for this forward pass.

        Args:
            id_embedding: (B, num_queries, kv_dim) or None to disable
        """
        self._id_embedding = id_embedding

    def _apply_ortho(self, id_k):
        """Apply orthogonal regularization to ID key projections."""
        if self.ortho_mode == self.NONE:
            return id_k

        b, seq, _ = id_k.shape
        id_k_2d = id_k.reshape(-1, id_k.shape[-1])

        if self.ortho_mode in (self.ORTHO, self.ORTHO_V2):
            sim = id_k_2d @ id_k_2d.T
            n = sim.shape[0]
            target = mx.eye(n)
            loss = ((sim - target) ** 2).sum()
            if self.ortho_mode == self.ORTHO:
                id_k = id_k - 0.0 * loss * id_k
            else:
                id_k = id_k * (1.0 + 0.0 * loss)

        return id_k.reshape(b, seq, -1)

    def __call__(self, attn_output, hidden_states, **kwargs):
        """Process attention output with ID injection.

        Called AFTER normal Flux self-attention. Adds ID cross-attention.

        Args:
            attn_output: (B, seq, dim) output from Flux self-attention
            hidden_states: (B, seq, dim) input hidden states (for Q)
        Returns:
            (B, seq, dim) attention output with ID injection
        """
        if self._id_embedding is None:
            return attn_output

        id_emb = self._id_embedding
        b, seq_len, _ = hidden_states.shape

        id_k = self.id_to_k(id_emb)
        id_v = self.id_to_v(id_emb)

        id_k = self._apply_ortho(id_k)

        q = hidden_states.reshape(b, seq_len, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        id_k = id_k.reshape(b, id_k.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)
        id_v = id_v.reshape(b, id_v.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (id_k * scale).transpose(0, 1, 3, 2)
        weight = mx.softmax(weight.astype(mx.float32), axis=-1).astype(q.dtype)
        id_out = weight @ id_v

        id_out = id_out.transpose(0, 2, 1, 3).reshape(b, seq_len, -1)

        return attn_output + id_out * self.scale
