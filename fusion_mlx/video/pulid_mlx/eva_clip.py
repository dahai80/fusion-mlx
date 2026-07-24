"""EVA-CLIP vision encoder — pure MLX port.

EVA02-CLIP-L-14-336 configuration:
  - image_size=336, patch_size=14, in_chans=3
  - embed_dim=1024, depth=24, num_heads=16 (head_width=64)
  - mlp_ratio=2.6667, naiveswiglu=True, subln=True
  - rope=True (VisionRotaryEmbeddingFast), intp_freq=True, pt_hw_seq_len=16

Outputs 5 hidden states (layers 4,8,12,16,20) for IDFormer plus CLS token.
Each hidden state shape: (B, seq_len, 1024).

Pure MLX port of eva_clip/eva_vit_model.py + eva_clip/rope.py.
Zero PyTorch dependency.
"""
import math
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def _rotate_half(x):
    """Split last dim in half, rotate: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


class VisionRotaryEmbeddingFast:
    """Precomputed 2D rotary positional embedding for vision tokens.

    Builds a (seq_h * seq_w, dim) frequency table and applies RoPE
    via elementwise cos/sin multiplication. No learnable parameters.
    """

    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None):
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        theta = 10000.0
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

        t = mx.arange(ft_seq_len).astype(mx.float32) / ft_seq_len * pt_seq_len
        freqs_t = mx.expand_dims(t, -1) * mx.expand_dims(freqs, 0)
        freqs_t = mx.repeat(freqs_t, 2, axis=-1)

        freqs_2d = mx.concatenate([
            mx.expand_dims(freqs_t, 1) * mx.ones((1, freqs_t.shape[0], 1)),
            mx.ones((freqs_t.shape[0], 1, 1)) * mx.expand_dims(freqs_t, 0),
        ], axis=-1)

        freqs_flat = freqs_2d.reshape(-1, freqs_2d.shape[-1])
        self.freqs_cos = mx.cos(freqs_flat)
        self.freqs_sin = mx.sin(freqs_flat)
        logger.info(f"VisionRoPE freq shape: {self.freqs_cos.shape}")

    def __call__(self, t):
        """Apply RoPE: t * cos + rotate_half(t) * sin."""
        return t * self.freqs_cos + _rotate_half(t) * self.freqs_sin


class PatchEmbed(nn.Module):
    """Image -> patch token embedding via Conv2d."""

    def __init__(self, img_size=336, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def __call__(self, x):
        b, c, h, w = x.shape
        x = self.proj(x)
        x = x.reshape(b, x.shape[1], -1).transpose(0, 2, 1)
        return x


class SwiGLU(nn.Module):
    """SwiGLU MLP: w1(silu) * w2 -> ffn_ln -> w3."""

    def __init__(self, dim, hidden_dim, subln=True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.ffn_ln = nn.LayerNorm(hidden_dim) if subln else None
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        if self.ffn_ln is not None:
            hidden = self.ffn_ln(hidden)
        return self.w3(hidden)


class Attention(nn.Module):
    """Multi-head attention with optional RoPE and subln (separate Q/K/V)."""

    def __init__(self, dim, num_heads=16, subln=True, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.subln = subln
        self.rope = rope

        if subln:
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.inner_attn_ln = nn.LayerNorm(dim) if subln else None
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        b, n, _ = x.shape

        if self.subln:
            q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        else:
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            q_cls = q[:, :, :1, :]
            q_tok = self.rope(q[:, :, 1:, :])
            q = mx.concatenate([q_cls, q_tok], axis=2)

            k_cls = k[:, :, :1, :]
            k_tok = self.rope(k[:, :, 1:, :])
            k = mx.concatenate([k_cls, k_tok], axis=2)

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, -1)

        if self.inner_attn_ln is not None:
            out = self.inner_attn_ln(out)
        return self.proj(out)


class Block(nn.Module):
    """Transformer block: norm -> attn -> norm -> SwiGLU, with layer-scale."""

    def __init__(self, dim, num_heads, mlp_ratio=2.6667, subln=True,
                 rope=None, init_values=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, subln=subln, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden, subln=subln)

        if init_values is not None and init_values > 0:
            self.gamma_1 = mx.ones((dim,)) * init_values
            self.gamma_2 = mx.ones((dim,)) * init_values
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def __call__(self, x):
        if self.gamma_1 is not None:
            x = x + self.gamma_1 * self.attn(self.norm1(x))
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class EVAVisionTransformer(nn.Module):
    """EVA02-CLIP vision transformer (vision-only, no text encoder).

    Outputs 5 intermediate hidden states for IDFormer consumption.

    Config for EVA02-CLIP-L-14-336:
        image_size=336, patch_size=14, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=2.6667, naiveswiglu=True, subln=True,
        rope=True, pt_hw_seq_len=16, intp_freq=True
    """

    def __init__(
        self,
        img_size=336,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=2.6667,
        subln=True,
        naiveswiglu=True,
        use_abs_pos_emb=True,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        init_values=None,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mx.zeros((1, 1, embed_dim))

        if use_abs_pos_emb:
            self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None

        rope_module = None
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            ft_seq = hw_seq_len if intp_freq else None
            rope_module = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=ft_seq,
            )

        dpr = [min(drop_path_rate, drop_path_rate * i / max(depth - 1, 1)) for i in range(depth)]

        self.blocks = []
        for i in range(depth):
            self.blocks.append(Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                subln=subln,
                rope=rope_module,
                init_values=init_values,
            ))

        self.norm = nn.LayerNorm(embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim)

        self.hidden_indices = [4, 8, 12, 16, 20]
        logger.info(f"EVA ViT: depth={depth}, embed_dim={embed_dim}, "
                     f"num_heads={num_heads}, rope={rope}")

    def __call__(self, x, return_hidden=True):
        """Forward pass.

        Args:
            x: (B, 3, 336, 336) float16 image tensor
            return_hidden: if True, return list of 5 hidden states for IDFormer
        Returns:
            If return_hidden: list of 5 (B, seq, 1024) hidden states
            Else: (B, 1024) CLS pooled features
        """
        x = self.patch_embed(x)
        b, seq_len, _ = x.shape

        cls_tokens = mx.broadcast_to(self.cls_token, (b, 1, self.embed_dim))
        x = mx.concatenate([cls_tokens, x], axis=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        hidden_states = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if return_hidden and (idx + 1) in self.hidden_indices:
                hidden_states.append(x.astype(mx.float16))

        x = self.norm(x)

        if return_hidden:
            if len(hidden_states) < 5:
                while len(hidden_states) < 5:
                    hidden_states.append(mx.zeros_like(x))
            return hidden_states

        pooled = self.fc_norm(x.mean(axis=1))
        return pooled


class EVACLIPEncoder:
    """High-level EVA-CLIP encoder wrapper for PuLID pipeline.

    Loads EVAVisionTransformer weights and provides a simple encode interface.
    """

    def __init__(self, model, dtype=mx.float16):
        self.model = model
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, model_dir, dtype=mx.float16):
        """Load from a directory containing EVA-CLIP safetensors.

        Args:
            model_dir: path with weights.safetensors or model.safetensors
            dtype: compute dtype
        """
        model_dir = Path(model_dir)
        logger.info(f"Loading EVA-CLIP from {model_dir}")

        model = EVAVisionTransformer()

        weight_file = model_dir / "weights.safetensors"
        if not weight_file.exists():
            weight_file = model_dir / "model.safetensors"
        if not weight_file.exists():
            candidates = list(model_dir.glob("*.safetensors"))
            if candidates:
                weight_file = candidates[0]

        if weight_file.exists():
            weights = mx.load(str(weight_file))
            filtered = {}
            for k, v in weights.items():
                new_k = k
                if k.startswith("visual."):
                    new_k = k[7:]
                if any(skip in new_k for skip in ["text.", "logit_scale", "mask_token"]):
                    continue
                filtered[new_k] = v
            model.load_weights(list(filtered.items()))
            logger.info(f"Loaded {len(filtered)} weight tensors from {weight_file.name}")
        else:
            logger.warning(f"No weight file found in {model_dir}, using random init")

        return cls(model, dtype)

    def __call__(self, x):
        """Encode image.

        Args:
            x: (B, 3, 336, 336) preprocessed image tensor
        Returns:
            list of 5 (B, seq, 1024) hidden states from layers 4,8,12,16,20
        """
        x = x.astype(self.dtype)
        return self.model(x, return_hidden=True)

    def encode_image(self, x):
        """Encode image to CLS pooled features.

        Args:
            x: (B, 3, 336, 336) preprocessed image tensor
        Returns:
            (B, 1024) CLS pooled features
        """
        x = x.astype(self.dtype)
        return self.model(x, return_hidden=False)
