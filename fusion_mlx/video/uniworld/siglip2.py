# SPDX-License-Identifier: Apache-2.0
# Pure-MLX SigLIP2-SO400M vision encoder (google/siglip2-so400m-patch16-512).
# Architecture: ViT-SO400M, 27 transformer layers, 1152 hidden, 16 heads,
# patch_size=16, image_size=512. Outputs last_hidden_state (B, N, 1152).
# Reference: fusion_mlx/video/pulid_mlx/eva_clip.py (EVAVisionTransformer pattern).

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class SigLIP2PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 512, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 1152):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[1] == 3 and x.ndim == 4:
            x = x.transpose(0, 2, 3, 1)
        x = self.proj(x)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        return x


class SigLIP2Attention(nn.Module):
    def __init__(self, dim: int = 1152, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, -1)
        x = self.proj(x)
        return x


class SigLIP2MLP(nn.Module):
    def __init__(self, dim: int = 1152, hidden_dim: int = 4304):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu_approx(x)
        x = self.fc2(x)
        return x


class SigLIP2EncoderLayer(nn.Module):
    def __init__(self, dim: int = 1152, num_heads: int = 16,
                 mlp_hidden: int = 4304):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SigLIP2Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SigLIP2MLP(dim, mlp_hidden)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SigLIP2VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1152,
        num_heads: int = 16,
        num_layers: int = 27,
        mlp_hidden: int = 4304,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.patch_embed = SigLIP2PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = mx.zeros((1, num_patches, embed_dim))
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(SigLIP2EncoderLayer(dim=embed_dim, num_heads=num_heads,
                                                    mlp_hidden=mlp_hidden))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        logger.info(
            "SigLIP2 ViT: layers=%d, embed_dim=%d, num_heads=%d, "
            "image_size=%d, patch_size=%d, num_patches=%d",
            num_layers, embed_dim, num_heads, image_size, patch_size, num_patches,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        b = x.shape[0]
        if self.pos_embed.shape[1] != x.shape[1]:
            logger.warning(
                "SigLIP2 pos_embed mismatch: %d vs %d patches, interpolating",
                self.pos_embed.shape[1], x.shape[1],
            )
            x = x + _interpolate_pos_embed(self.pos_embed, x.shape[1])
        else:
            x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class SigLIP2VisionEncoder:
    def __init__(self, model: SigLIP2VisionTransformer, dtype=mx.float16):
        self.model = model
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, dtype=mx.float16) -> "SigLIP2VisionEncoder":
        model_dir = Path(model_dir)
        logger.info("Loading SigLIP2 from %s", model_dir)
        model = SigLIP2VisionTransformer()
        weight_file = model_dir / "model.safetensors"
        if not weight_file.exists():
            candidates = list(model_dir.glob("*.safetensors"))
            if candidates:
                weight_file = candidates[0]
        if weight_file.exists():
            weights = mx.load(str(weight_file))
            filtered = _remap_siglip_weights(weights)
            model.load_weights(list(filtered.items()))
            mx.eval(model.parameters())
            logger.info("SigLIP2 loaded %d weights from %s", len(filtered), weight_file.name)
        else:
            logger.warning("No weight file found for SigLIP2 in %s", model_dir)
        return cls(model, dtype)

    def encode_image(self, pixel_values: mx.array) -> mx.array:
        if pixel_values.dtype != self.dtype:
            pixel_values = pixel_values.astype(self.dtype)
        out = self.model(pixel_values)
        mx.eval(out)
        return out

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.encode_image(pixel_values)


def _interpolate_pos_embed(pos_embed: mx.array, target_len: int) -> mx.array:
    import numpy as np
    src_len = pos_embed.shape[1]
    if src_len == target_len:
        return pos_embed
    src_h = int(src_len ** 0.5)
    tgt_h = int(target_len ** 0.5)
    dim = pos_embed.shape[-1]
    pe_np = np.array(pos_embed).reshape(1, src_h, src_h, dim)
    pe_np = np.resize(pe_np, (1, tgt_h, tgt_h, dim))
    return mx.array(pe_np.reshape(1, target_len, dim))


def _remap_siglip_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    remapped = {}
    for k, v in weights.items():
        new_k = k
        if k.startswith("vision_model."):
            new_k = k[len("vision_model."):]
        if k.startswith("model.vision_model."):
            new_k = k[len("model.vision_model."):]
        for prefix in ["embeddings.", "encoder."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        new_k = new_k.replace("patch_embedding.", "patch_embed.proj.")
        new_k = new_k.replace("position_embedding.", "pos_embed.")
        new_k = new_k.replace("layers.", "layers.")
        new_k = new_k.replace(".layer_norm1.", ".norm1.")
        new_k = new_k.replace(".layer_norm2.", ".norm2.")
        new_k = new_k.replace(".self_attn.", ".attn.")
        new_k = new_k.replace(".mlp.fc1.", ".mlp.fc1.")
        new_k = new_k.replace(".mlp.fc2.", ".mlp.fc2.")
        new_k = new_k.replace("post_layernorm.", "norm.")
        if any(skip in new_k for skip in ["text_model", "logit_scale"]):
            continue
        remapped[new_k] = v
    return remapped
