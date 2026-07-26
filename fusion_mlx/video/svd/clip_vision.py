# SPDX-License-Identifier: Apache-2.0
# CLIP Vision encoder adapter for SVD.
# Extracts image embeddings using a CLIP ViT-L/14 vision transformer.

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.class_embedding = mx.zeros((hidden_size,), dtype=mx.float32)
        scale = 1.0 / (hidden_size**0.5)
        self.patch_embedding_weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(hidden_size, num_channels, patch_size, patch_size),
            dtype=mx.float32,
        )
        self.patch_embedding_bias = mx.zeros((hidden_size,), dtype=mx.float32)
        num_positions = num_patches + 1
        self.position_embedding = mx.zeros(
            (num_positions, hidden_size), dtype=mx.float32
        )

    def __call__(self, x):
        N, C, H, W = x.shape
        p = self.patch_size
        # Conv2d patch embedding
        patches = _conv2d(
            x,
            self.patch_embedding_weight,
            self.patch_embedding_bias,
            (p, p),
            (p, p),
            (0, 0),
        )
        PH, PW = patches.shape[2], patches.shape[3]
        patches = patches.reshape(N, self.hidden_size, PH * PW).transpose(0, 2, 1)
        cls_emb = self.class_embedding.reshape(1, 1, -1).broadcast_to((N, 1, -1))
        embeddings = mx.concatenate([cls_emb, patches], axis=1)
        embeddings = embeddings + self.position_embedding
        return embeddings


def _conv2d(x, weight, bias, stride, padding, groups=1):
    if padding[0] > 0 or padding[1] > 0:
        pad_widths = [(0, 0), (0, 0)] + [
            (padding[0], padding[0]),
            (padding[1], padding[1]),
        ]
        x = mx.pad(x, pad_widths)
    N, C, H, W = x.shape
    OC, IC, KH, KW = weight.shape
    OH = (H - KH) // stride[0] + 1
    OW = (W - KW) // stride[1] + 1
    cols = []
    for kh in range(KH):
        for kw in range(KW):
            cols.append(
                x[
                    :,
                    :,
                    kh : kh + OH * stride[0] : stride[0],
                    kw : kw + OW * stride[1] : stride[1],
                ]
            )
    cols = mx.stack(cols, axis=-1).reshape(N, IC * KH * KW, OH * OW)
    w = weight.reshape(OC, IC * KH * KW)
    out = mx.matmul(w, cols).reshape(N, OC, OH, OW)
    out = out + bias.reshape(1, -1, 1, 1)
    return out


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x):
        B, L, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, intermediate_size=4096):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        num_layers=24,
        intermediate_size=4096,
        image_size=224,
        patch_size=14,
    ):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(hidden_size, image_size, patch_size)
        self.pre_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.encoder = [
            CLIPEncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)

    def __call__(self, x):
        x = self.embeddings(x)
        x = self.pre_layernorm(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.post_layernorm(x)
        pooled = x[:, 0, :]
        return pooled, x


class SVDCLIPVisionEncoder:
    def __init__(self, model_path=None, dtype=mx.float16):
        self._model = None
        self._model_path = model_path
        self._dtype = dtype
        self._preprocess = _CLIPPreprocess()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        logger.info("SVD CLIP vision: loading from %s", self._model_path)
        self._model = CLIPVisionTransformer()
        if self._model_path:
            import os

            weight_dir = (
                os.path.join(self._model_path, "image_encoder")
                if os.path.isdir(self._model_path)
                else self._model_path
            )
            weight_files = []
            if os.path.isdir(weight_dir):
                import glob

                weight_files = sorted(
                    glob.glob(os.path.join(weight_dir, "*.safetensors"))
                )
            if weight_files:
                from mlx.utils import load_weights

                weights = load_weights(weight_dir)
                self._model.update(weights)
        self._model = self._model.astype(self._dtype)
        logger.info("SVD CLIP vision: loaded dtype=%s", self._dtype)

    def encode_image(self, image_path_or_array):
        self._ensure_loaded()
        if isinstance(image_path_or_array, str):
            from PIL import Image as PILImage

            pil_img = PILImage.open(image_path_or_array).convert("RGB")
            pixel_values = self._preprocess(pil_img)
        else:
            pixel_values = image_path_or_array
        pooled, sequence = self._model(pixel_values)
        mx.eval(pooled, sequence)
        return pooled, sequence


class _CLIPPreprocess:
    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, pil_img):
        img = pil_img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None, :, :, :]
        return mx.array(arr, dtype=mx.float16)
