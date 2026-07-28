# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo dual text encoder: CLIP-L + CLIP-G.
# Returns concatenated embeddings for DiT conditioning.

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _silu(x):
    return x * mx.sigmoid(x)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=768, image_size=224, patch_size=32, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.class_embedding = mx.zeros((hidden_size,), dtype=mx.float32)
        self.patch_embedding = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.position_embedding = mx.zeros(
            (num_patches + 1, hidden_size), dtype=mx.float32
        )

    def __call__(self, x):
        B = x.shape[0]
        patch_embs = self.patch_embedding(x)
        # MLX Conv2d outputs NHWC: (B, H', W', C_out) -> (B, num_patches, C_out)
        patch_embs = patch_embs.reshape(B, -1, patch_embs.shape[-1])
        cls_emb = self.class_embedding.reshape(1, 1, -1).repeat(B, axis=0)
        embs = mx.concatenate([cls_emb, patch_embs], axis=1)
        return embs + self.position_embedding


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def __call__(self, x):
        h = self.layer_norm1(x)
        x = x + self.self_attn(h, h, h)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12, is_large=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_large = is_large
        self.token_embedding = nn.Embedding(49408, hidden_size)
        self.position_embedding = mx.zeros((77, hidden_size), dtype=mx.float32)
        self.layers = [
            CLIPEncoderLayer(hidden_size, num_heads) for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(hidden_size)

    def __call__(self, input_ids):
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        pooled = x[:, 0]
        return x, pooled


class HunyuanDualTextEncoder(nn.Module):
    # HunyuanVideo uses CLIP-L (text) + CLIP-G (text) for dual conditioning.
    # Returns concatenated sequence embeddings + pooled embeddings.

    def __init__(
        self,
        clip_l_hidden=768,
        clip_l_heads=12,
        clip_l_layers=12,
        clip_g_hidden=1280,
        clip_g_heads=20,
        clip_g_layers=32,
    ):
        super().__init__()
        self.clip_l = CLIPTextEncoder(
            hidden_size=clip_l_hidden,
            num_heads=clip_l_heads,
            num_layers=clip_l_layers,
            is_large=False,
        )
        self.clip_g = CLIPTextEncoder(
            hidden_size=clip_g_hidden,
            num_heads=clip_g_heads,
            num_layers=clip_g_layers,
            is_large=True,
        )

    def __call__(self, input_ids_l, input_ids_g=None):
        seq_l, pooled_l = self.clip_l(input_ids_l)
        if input_ids_g is None:
            input_ids_g = input_ids_l
        seq_g, pooled_g = self.clip_g(input_ids_g)
        # Concatenate sequence embeddings along feature dim
        combined_seq = mx.concatenate([seq_l, seq_g], axis=-1)
        combined_pooled = mx.concatenate([pooled_l, pooled_g], axis=-1)
        return combined_seq, combined_pooled

    @property
    def output_dim(self):
        return self.clip_l.hidden_size + self.clip_g.hidden_size

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        import glob
        import os

        enc = cls(**kwargs)
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("no safetensors at %s, random init", model_path)
            return enc
        from mlx.utils import tree_flatten, tree_unflatten

        all_params = {}
        for sf in safetensor_files:
            weights = mx.load(sf)
            all_params.update(weights)
        flat = tree_flatten(enc.parameters())
        loaded = {}
        for k, v in flat:
            w = all_params.get(k, v)
            loaded[k] = w.astype(mx.float16) if hasattr(w, "dtype") and w.dtype != mx.float16 else w
        enc.update(tree_unflatten(loaded))
        return enc
