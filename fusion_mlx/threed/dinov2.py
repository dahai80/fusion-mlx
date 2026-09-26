# SPDX-License-Identifier: Apache-2.0
# DINOv2-Large ViT conditioner for Hunyuan3D-2.1 (issue #989 Session 1).
# Image -> 1370 tokens x 1024 (1 cls + 37*37 patch tokens). 8bit mlx-quantized
# (uint32 packed, group 64). Separate q/k/v/out projections (not fused QKV).
# Architecture ported from tencent/Hunyuan3D-2.1 diffusers reference + DINOv2
# ViT-L/14 spec; weights loaded from ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit
# conditioner.safetensors (MLX-native, no conversion needed).
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from .config import ShapeConfig

logger = logging.getLogger(__name__)


class _QuantizedLinear(nn.quantized.QuantizedLinear):
    # Subclass only to bypass the random-init in __init__ — weights are set
    # explicitly from the loaded safetensors tensors via load_weights. Keeps
    # the dequant forward (mx.quantized_matmul) from the base class.
    pass


class DINOv2Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, group_size: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = _QuantizedLinear(dim, dim, bias=bias, group_size=group_size, bits=8)
        self.k = _QuantizedLinear(dim, dim, bias=bias, group_size=group_size, bits=8)
        self.v = _QuantizedLinear(dim, dim, bias=bias, group_size=group_size, bits=8)
        self.out = _QuantizedLinear(dim, dim, bias=bias, group_size=group_size, bits=8)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out(attn)


class DINOv2MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, group_size: int):
        super().__init__()
        self.fc1 = _QuantizedLinear(
            dim, hidden, bias=True, group_size=group_size, bits=8
        )
        self.fc2 = _QuantizedLinear(
            hidden, dim, bias=True, group_size=group_size, bits=8
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class DINOv2Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, group_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv2Attention(dim, num_heads, group_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOv2MLP(dim, dim * mlp_ratio, group_size)
        self.ls1 = mx.ones((dim,))
        self.ls2 = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DINOv2Conditioner(nn.Module):
    # DINOv2-Large ViT: 24 layers, hidden 1024, 16 heads, patch 14, image 518.
    # Outputs (B, 1370, 1024) — 1 cls + 1369 patch tokens. The cls token is
    # dropped by Hunyuan3D's shape DiT (uses patch tokens as cross-attn context),
    # but we return the full sequence and let the caller slice.

    def __init__(self, cfg: ShapeConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dino_hidden
        self.num_heads = cfg.dino_heads
        self.patch_size = cfg.dino_patch
        self.image_size = cfg.dino_image_size
        self.num_patches = (cfg.dino_image_size // cfg.dino_patch) ** 2
        gs = cfg.dino_group_size
        self.cls_token = mx.zeros((1, 1, dim))
        self.pos_embed = mx.zeros((1, cfg.dino_num_tokens, dim))
        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=cfg.dino_patch, stride=cfg.dino_patch, bias=True
        )
        self.blocks = [
            DINOv2Block(dim, cfg.dino_heads, 4, gs) for _ in range(cfg.dino_layers)
        ]
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, 3, 518, 518) float32 in [0,1] (ImageNet-normalized by caller).
        # mlx Conv2d expects NHWC.
        B = x.shape[0]
        x_nchw = x
        x_nhwc = x_nchw.transpose(0, 2, 3, 1)
        patches = self.patch_embed(x_nhwc)  # (B, 37, 37, dim)
        h = patches.shape[1]
        patches = patches.reshape(B, h * h, -1)  # (B, 1369, dim)
        cls = mx.broadcast_to(self.cls_token, (B, 1, patches.shape[-1]))
        x = mx.concatenate([cls, patches], axis=1)  # (B, 1370, dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def load_dinov2_conditioner(weights_path: str, cfg: ShapeConfig) -> DINOv2Conditioner:
    # Load conditioner.safetensors into DINOv2Conditioner. The safetensors keys
    # map 1:1 to the module tree: cls_token, pos_embed, patch_embed.{weight,bias},
    # layers.N.{attn.{q,k,v,out}.{weight,scales,biases,bias}, mlp.{fc1,fc2}.*,
    # norm{1,2}.*, ls1, ls2}, norm.{weight,bias}. QuantizedLinear expects
    # {weight, scales, biases, bias} — same names, direct set.
    model = DINOv2Conditioner(cfg)
    from safetensors import safe_open

    weights: dict[str, mx.array] = {}
    with safe_open(weights_path, framework="np") as f:
        for k in f.keys():  # noqa: SIM118
            weights[k] = mx.array(f.get_tensor(k))
    # Remap "layers.N" -> "blocks.N" (module attr name). Conv2d weight is
    # PyTorch layout (C_out, C_in, H, W) in the checkpoint; mlx conv2d expects
    # (H, W, C_in, C_out) — transpose patch_embed.weight.
    remapped: list[tuple[str, mx.array]] = []
    for k, v in weights.items():
        if k.startswith("layers."):
            nk = "blocks." + k[len("layers.") :]
        else:
            nk = k
        if nk == "patch_embed.weight":
            # PyTorch (C_out, C_in, H, W) -> mlx.nn.Conv2d (C_out, H, W, C_in).
            v = v.transpose(0, 2, 3, 1)
        remapped.append((nk, v))
    # load_weights returns self; strict=False tolerates order/extra. Diff
    # manually via tree_flatten (prefix="") to surface mismatches (Rule 12).
    flat_before: dict = {}
    nn.utils.tree_flatten(model, destination=flat_before)
    model.load_weights(remapped, strict=False)
    remapped_keys = {k for k, _ in remapped}
    module_keys = set(flat_before.keys())
    missing = [k for k in module_keys if k not in remapped_keys]
    skipped = [k for k in remapped_keys if k not in module_keys]
    if missing:
        logger.warning("DINOv2 missing weights (%d): %s", len(missing), missing[:6])
    if skipped:
        logger.warning("DINOv2 unexpected weights (%d): %s", len(skipped), skipped[:6])
    logger.info(
        "DINOv2 conditioner loaded: %d tensors, %d layers, %d tokens (mapped=%d)",
        len(weights),
        cfg.dino_layers,
        cfg.dino_num_tokens,
        len(remapped) - len(skipped),
    )
    return model
