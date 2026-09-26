# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 paint VAE decoder (issue #989 Session 4).
# Standard diffusers AutoencoderKL decoder: latent (B,4,H,W) -> /scaling ->
# post_quant_conv -> conv_in -> mid_block(resnet,attn,resnet) -> 4 up_blocks
# (3 resnets each; up0/1/2 have 2x nearest-upsampler) -> conv_norm_out+silu
# -> conv_out -> RGB (B,3,8H,8W). NHWC throughout (MLX Conv2d is NHWC,
# weight layout [out,kh,kw,in]; checkpoint conv weights already MLX layout).
# Weights: paint/vae.safetensors (pure fp16, no quant). scaling_factor 0.18215.
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from fusion_mlx.threed.config import PaintVAEConfig

logger = logging.getLogger(__name__)


def _conv(in_c: int, out_c: int, k: int = 3, pad: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, k, stride=1, padding=pad, bias=True)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class _ResnetBlock(nn.Module):
    # norm1 + silu + conv1 + norm2 + silu + conv2 + conv_shortcut (if in!=out).
    def __init__(self, in_c: int, out_c: int, cfg: PaintVAEConfig):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            cfg.norm_num_groups, in_c, pytorch_compatible=True, eps=cfg.norm_eps
        )
        self.conv1 = _conv(in_c, out_c)
        self.norm2 = nn.GroupNorm(
            cfg.norm_num_groups, out_c, pytorch_compatible=True, eps=cfg.norm_eps
        )
        self.conv2 = _conv(out_c, out_c)
        self.in_c = in_c
        self.out_c = out_c
        if in_c != out_c:
            self.conv_shortcut = nn.Conv2d(in_c, out_c, 1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(_silu(self.norm1(x)))
        h = self.conv2(_silu(self.norm2(h)))
        if self.in_c != self.out_c:
            x = self.conv_shortcut(x)
        return x + h


class _Attention(nn.Module):
    # Single-head spatial self-attention over H*W tokens, dim C.
    # group_norm -> reshape (B,H*W,C) -> q/k/v/proj_attn. Standard VAE attn.
    def __init__(self, dim: int, cfg: PaintVAEConfig):
        super().__init__()
        self.group_norm = nn.GroupNorm(
            cfg.norm_num_groups, dim, pytorch_compatible=True, eps=cfg.norm_eps
        )
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj_attn = nn.Linear(dim, dim)
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (NHWC).
        B, H, W, C = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, H * W, C)
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        scale = 1.0 / mx.sqrt(mx.array(float(C)))
        attn = mx.softmax((q @ k.transpose(0, 2, 1)) * scale, axis=-1)
        out = attn @ v
        out = self.proj_attn(out)
        out = out.reshape(B, H, W, C)
        return x + out


class _MidBlock(nn.Module):
    # resnet -> attention -> resnet.
    def __init__(self, dim: int, cfg: PaintVAEConfig):
        super().__init__()
        self.resnets = [
            _ResnetBlock(dim, dim, cfg),
            _ResnetBlock(dim, dim, cfg),
        ]
        self.attentions = [_Attention(dim, cfg)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class _Upsampler(nn.Module):
    # nearest 2x then 3x3 conv.
    def __init__(self, dim: int):
        super().__init__()
        self.conv = _conv(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = x[:, :, None, :, None, :]
        x = mx.broadcast_to(x, (B, H, 2, W, 2, C))
        x = x.reshape(B, H * 2, W * 2, C)
        return self.conv(x)


class _UpBlock(nn.Module):
    # 3 resnets (layers_per_block+1) + optional upsampler.
    def __init__(self, in_c: int, out_c: int, cfg: PaintVAEConfig, upsample: bool):
        super().__init__()
        self.resnets = [
            _ResnetBlock(in_c if i == 0 else out_c, out_c, cfg)
            for i in range(cfg.layers_per_block + 1)
        ]
        self.upsamplers = [_Upsampler(out_c)] if upsample else []
        self.out_c = out_c

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        for u in self.upsamplers:
            x = u(x)
        return x


class PaintVAEDecoder(nn.Module):
    # latent (B, latent_channels, H, W) NCHW -> RGB (B, 3, 8H, 8W) NCHW.
    def __init__(self, cfg: PaintVAEConfig):
        super().__init__()
        self.cfg = cfg
        rch = list(reversed(cfg.block_out_channels))  # [512,512,256,128]
        self.post_quant_conv = nn.Conv2d(
            cfg.latent_channels, cfg.latent_channels, 1, bias=True
        )
        self.conv_in = _conv(cfg.latent_channels, rch[0])
        self.mid_block = _MidBlock(rch[0], cfg)
        # up0/1/2 upsample, up3 no upsample. in_c = rch[i-1] (or rch[0] for up0).
        self.up_blocks = [
            _UpBlock(rch[0], rch[0], cfg, upsample=True),
            _UpBlock(rch[0], rch[1], cfg, upsample=True),
            _UpBlock(rch[1], rch[2], cfg, upsample=True),
            _UpBlock(rch[2], rch[3], cfg, upsample=False),
        ]
        self.conv_norm_out = nn.GroupNorm(
            cfg.norm_num_groups, rch[3], pytorch_compatible=True, eps=cfg.norm_eps
        )
        self.conv_out = _conv(rch[3], cfg.out_channels)

    def __call__(self, latent: mx.array) -> mx.array:
        # latent: NCHW -> /scaling -> NHWC.
        x = latent / self.cfg.scaling_factor
        x = x.transpose(0, 2, 3, 1)
        x = self.post_quant_conv(x)
        x = self.conv_in(x)
        x = self.mid_block(x)
        for ub in self.up_blocks:
            x = ub(x)
        x = _silu(self.conv_norm_out(x))
        x = self.conv_out(x)
        return x.transpose(0, 3, 1, 2)  # NHWC -> NCHW


def load_paint_vae(weights_path: str, cfg: PaintVAEConfig) -> PaintVAEDecoder:
    from safetensors import safe_open

    net = PaintVAEDecoder(cfg)
    raw: dict = {}
    with safe_open(weights_path, framework="numpy") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open handle, not dict)
            if k.startswith("decoder.") or k.startswith("post_quant_conv."):
                t = f.get_tensor(k)
                arr = np.asarray(t)
                # strip decoder. prefix so keys match module tree (conv_in.*,
                # mid_block.*, up_blocks.*, conv_norm_out.*, conv_out.*).
                mk = k[len("decoder.") :] if k.startswith("decoder.") else k
                raw[mk] = mx.array(arr)
    # Module is nested: decoder.{conv_in,mid_block,up_blocks,conv_norm_out,
    # conv_out} + post_quant_conv (top-level on net).
    params = net.parameters()
    flat: dict = {}
    nn.utils.tree_flatten(params, destination=flat)
    loaded = {}
    missing = []
    for key in flat:
        if key in raw:
            loaded[key] = raw[key]
        else:
            missing.append(key)
    unexpected = [k for k in raw if k not in flat]
    net.load_weights(list(loaded.items()))
    if missing:
        logger.warning("PaintVAE missing %d weights: %s", len(missing), missing[:8])
    if unexpected:
        logger.warning(
            "PaintVAE unexpected %d weights: %s", len(unexpected), unexpected[:8]
        )
    logger.info(
        "PaintVAE loaded %d/%d weights (missing=%d unexpected=%d)",
        len(loaded),
        len(flat),
        len(missing),
        len(unexpected),
    )
    return net
