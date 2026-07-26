# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of the SVD Temporal UNet.
# Based on stabilityai/svd UNet with temporal attention + spatial attention.
# I2V: receives CLIP vision embeddings as cross-attention + VAE-encoded image concat.

import logging
import math

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=1280):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=True)
        self.linear2 = nn.Linear(out_channels, out_channels, bias=True)

    def __call__(self, x):
        x = nn.silu(self.linear1(x))
        return self.linear2(x)


def _timestep_proj(t, dim):
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TemporalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def __call__(self, x):
        return self.conv(x)


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(32, dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, T * H * W).transpose(0, 2, 1)  # (B, THW, C)
        q = (
            self.q(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T * H * W, C)
        out = self.out(out)
        out = out.transpose(0, 2, 1).reshape(B, C, T, H, W)
        return x + out


class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(32, dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        # Reshape: (B*H*W, T, C) for temporal attention
        h = h.transpose(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        q = (
            self.q(h)
            .reshape(B * H * W, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(h)
            .reshape(B * H * W, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(h)
            .reshape(B * H * W, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B * H * W, T, C)
        out = self.out(out)
        out = out.reshape(B, H, W, T, C).transpose(0, 4, 3, 1, 2)
        return x + out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=1024, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(32, query_dim)
        self.q = nn.Linear(query_dim, query_dim, bias=True)
        self.k = nn.Linear(context_dim, query_dim, bias=True)
        self.v = nn.Linear(context_dim, query_dim, bias=True)
        self.out = nn.Linear(query_dim, query_dim, bias=True)

    def __call__(self, x, context):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, T * H * W).transpose(0, 2, 1)  # (B, THW, C)
        q = (
            self.q(h)
            .reshape(B, T * H * W, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(context)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(context)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T * H * W, C)
        out = self.out(out)
        out = out.transpose(0, 2, 1).reshape(B, C, T, H, W)
        return x + out


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_ch = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = None
        if in_channels != out_ch:
            self.skip = nn.Conv3d(in_channels, out_ch, kernel_size=1, padding=0)

    def __call__(self, x, temb=None):
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + temb
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        return x + h


class SVDUNetBlock(nn.Module):
    def __init__(self, dim, context_dim=1024, num_heads=8, use_temporal=True):
        super().__init__()
        self.resnet1 = ResnetBlock(dim)
        self.attn_spatial = SpatialAttention(dim, num_heads)
        self.attn_temporal = TemporalAttention(dim, num_heads) if use_temporal else None
        self.attn_cross = CrossAttention(dim, context_dim, num_heads)
        self.resnet2 = ResnetBlock(dim)

    def __call__(self, x, context=None, temb=None):
        x = self.resnet1(x, temb)
        x = self.attn_spatial(x)
        if self.attn_temporal is not None:
            x = self.attn_temporal(x)
        if context is not None:
            x = self.attn_cross(x, context)
        x = self.resnet2(x, temb)
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        context_dim=1024,
        num_layers=2,
        downsample=True,
        num_heads=8,
    ):
        super().__init__()
        self.blocks = [
            SVDUNetBlock(in_dim if i == 0 else out_dim, context_dim, num_heads)
            for i in range(num_layers)
        ]
        self.downsample = downsample
        if downsample:
            self.conv_down = nn.Conv3d(
                out_dim, out_dim, kernel_size=3, stride=2, padding=1
            )

    def __call__(self, x, context=None, temb=None):
        for block in self.blocks:
            x = block(x, context, temb)
        if self.downsample:
            x = self.conv_down(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        context_dim=1024,
        num_layers=2,
        upsample=True,
        num_heads=8,
    ):
        super().__init__()
        self.blocks = [
            SVDUNetBlock(in_dim if i == 0 else out_dim, context_dim, num_heads)
            for i in range(num_layers)
        ]
        self.upsample = upsample
        if upsample:
            self.conv_up = nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1)

    def __call__(self, x, context=None, temb=None):
        for block in self.blocks:
            x = block(x, context, temb)
        if self.upsample:
            B, C, T, H, W = x.shape
            x_up = mx.zeros((B, C, T * 2, H * 2, W * 2), dtype=x.dtype)
            x = self.conv_up(x_up)
        return x


class SVDTemporalUNet(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=4,
        context_dim=1024,
        dims=(320, 640, 1280, 1280),
        num_heads=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.cfg = type("Cfg", (), {"in_channels": in_channels})()
        # Time embedding
        self.time_proj = TimestepEmbedding(320, 1280)
        self.time_embed = nn.Sequential(
            nn.Linear(1280, 1280, bias=True),
            nn.SiLU(),
            nn.Linear(1280, 1280, bias=True),
        )
        # Input conv
        self.conv_in = nn.Conv3d(in_channels, dims[0], kernel_size=3, padding=1)
        # Downsampling
        self.down_blocks = [
            DownBlock(
                dims[0],
                dims[0],
                context_dim,
                num_layers=2,
                downsample=True,
                num_heads=num_heads,
            ),
            DownBlock(
                dims[0],
                dims[1],
                context_dim,
                num_layers=2,
                downsample=True,
                num_heads=num_heads,
            ),
            DownBlock(
                dims[1],
                dims[2],
                context_dim,
                num_layers=2,
                downsample=True,
                num_heads=num_heads,
            ),
            DownBlock(
                dims[2],
                dims[3],
                context_dim,
                num_layers=2,
                downsample=False,
                num_heads=num_heads,
            ),
        ]
        # Mid
        self.mid_block1 = SVDUNetBlock(dims[3], context_dim, num_heads)
        self.mid_block2 = SVDUNetBlock(dims[3], context_dim, num_heads)
        # Upsampling
        self.up_blocks = [
            UpBlock(
                dims[3],
                dims[2],
                context_dim,
                num_layers=2,
                upsample=True,
                num_heads=num_heads,
            ),
            UpBlock(
                dims[2],
                dims[1],
                context_dim,
                num_layers=2,
                upsample=True,
                num_heads=num_heads,
            ),
            UpBlock(
                dims[1],
                dims[0],
                context_dim,
                num_layers=2,
                upsample=True,
                num_heads=num_heads,
            ),
            UpBlock(
                dims[0],
                dims[0],
                context_dim,
                num_layers=2,
                upsample=False,
                num_heads=num_heads,
            ),
        ]
        # Output
        self.conv_norm_out = nn.GroupNorm(32, dims[0])
        self.conv_out = nn.Conv3d(dims[0], out_channels, kernel_size=3, padding=1)

    def __call__(self, x, timestep=None, context=None):
        # Time embedding
        if timestep is not None:
            t_emb = _timestep_proj(timestep, 320)
            t_emb = self.time_proj(t_emb)
            t_emb = self.time_embed(t_emb)
            # Reshape for broadcast: (B, C, 1, 1, 1)
            temb = t_emb[:, :, None, None, None]
        else:
            temb = None
        # Forward
        h = self.conv_in(x)
        # Down
        skips = [h]
        for block in self.down_blocks:
            h = block(h, context, temb)
            skips.append(h)
        # Mid
        h = self.mid_block1(h, context, temb)
        h = self.mid_block2(h, context, temb)
        # Up with skip connections
        for i, block in enumerate(self.up_blocks):
            if i < len(skips):
                skip = skips[-(i + 1)]
                if h.shape == skip.shape:
                    h = h + skip
            h = block(h, context, temb)
        # Output
        h = self.conv_norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h

    @classmethod
    def from_pretrained(cls, path, dtype=mx.float32):
        model = cls()
        import glob
        import os

        weight_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if weight_files:
            from mlx.utils import load_weights

            weights = load_weights(path)
            model.update(weights)
        model = model.astype(dtype)
        logger.info("SVD UNet loaded dtype=%s", dtype)
        return model
