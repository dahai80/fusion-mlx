# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of the SVD Video VAE (temporal-autoencoder).
# Based on stabilityai/svd temporal VAE with 3D convolutions.
# Latent dim=4, spatial compression 8x, temporal compression 4x.

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )
        scale = (
            1.0
            / (
                in_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
        ) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, in_channels, *self.kernel_size),
            dtype=mx.float32,
        )
        self.bias = mx.zeros((out_channels,), dtype=mx.float32)

    def __call__(self, x):
        return _conv3d(x, self.weight, self.bias, self.stride, self.padding)


def _conv3d(x, weight, bias, stride, padding):
    if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
        pad_widths = [(0, 0)] * (x.ndim - 3) + [
            (padding[0], padding[0]),
            (padding[1], padding[1]),
            (padding[2], padding[2]),
        ]
        x = mx.pad(x, pad_widths)
    out = _conv3d_core(x, weight, stride)
    out = out + bias.reshape((1, -1) + (1,) * (out.ndim - 2))
    return out


def _conv3d_core(x, weight, stride):
    N, C, T, H, W = x.shape
    OC, IC, KT, KH, KW = weight.shape
    OT = (T - KT) // stride[0] + 1
    OH = (H - KH) // stride[1] + 1
    OW = (W - KW) // stride[2] + 1
    # Unfold input into columns
    cols = []
    for kt in range(KT):
        for kh in range(KH):
            for kw in range(KW):
                t_start = kt
                t_end = t_start + OT * stride[0]
                h_start = kh
                h_end = h_start + OH * stride[1]
                w_start = kw
                w_end = w_start + OW * stride[2]
                cols.append(
                    x[
                        :,
                        :,
                        t_start : t_end : stride[0],
                        h_start : h_end : stride[1],
                        w_start : w_end : stride[2],
                    ]
                )
    cols = mx.stack(cols, axis=-1)  # (N, C, OT, OH, OW, KT*KH*KW)
    cols = cols.reshape(N, C, OT * OH * OW, KT * KH * KW)
    w = weight.reshape(OC, IC * KT * KH * KW)
    # Batched matmul: (N, OC, IC*KKK) @ (N, IC*KKK, OT*OH*OW)
    out = mx.matmul(w, cols)  # (N, OC, OT*OH*OW)
    out = out.reshape(N, OC, OT, OH, OW)
    return out


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_ch = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = CausalConv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = None
        if in_channels != out_ch:
            self.skip = CausalConv3d(in_channels, out_ch, kernel_size=1, padding=0)

    def __call__(self, x):
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        return x + h


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, downsample=True):
        super().__init__()
        self.resnets = [
            ResnetBlock3D(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ]
        self.downsample = downsample
        if downsample:
            self.conv_down = CausalConv3d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsample:
            x = self.conv_down(x)
        return x


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, upsample=True):
        super().__init__()
        self.resnets = [
            ResnetBlock3D(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ]
        self.upsample = upsample
        if upsample:
            self.conv_up = CausalConv3d(
                out_channels, out_channels, kernel_size=3, padding=1
            )

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsample:
            B, C, t, h, w = x.shape
            x = mx.broadcast_to(
                x.reshape(B, C, t, 1, h, 1, w, 1),
                (B, C, t, 2, h, 2, w, 2),
            )
            x = x.reshape(B, C, t * 2, h * 2, w * 2)
            x = self.conv_up(x)
        return x


class SVDVideoVAE(nn.Module):
    def __init__(self, latent_channels=4, out_channels=3):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        # Encoder: progressive downsampling 8x spatial, 4x temporal
        self.encoder_conv_in = CausalConv3d(out_channels, 64, kernel_size=3, padding=1)
        self.encoder_down = [
            DownBlock3D(64, 64, num_layers=1, downsample=True),
            DownBlock3D(64, 128, num_layers=2, downsample=True),
            DownBlock3D(128, 256, num_layers=2, downsample=True),
            DownBlock3D(256, 512, num_layers=2, downsample=False),
        ]
        self.encoder_mid_block1 = ResnetBlock3D(512, 512)
        self.encoder_mid_block2 = ResnetBlock3D(512, 512)
        self.encoder_norm_out = nn.GroupNorm(32, 512)
        self.encoder_conv_out = CausalConv3d(
            512, 2 * latent_channels, kernel_size=3, padding=1
        )
        # Decoder
        self.decoder_conv_in = CausalConv3d(
            latent_channels, 512, kernel_size=3, padding=1
        )
        self.decoder_mid_block1 = ResnetBlock3D(512, 512)
        self.decoder_mid_block2 = ResnetBlock3D(512, 512)
        self.decoder_up = [
            UpBlock3D(512, 256, num_layers=2, upsample=True),
            UpBlock3D(256, 128, num_layers=2, upsample=True),
            UpBlock3D(128, 64, num_layers=2, upsample=True),
            UpBlock3D(64, 64, num_layers=1, upsample=False),
        ]
        self.decoder_norm_out = nn.GroupNorm(32, 64)
        self.decoder_conv_out = CausalConv3d(64, out_channels, kernel_size=3, padding=1)

    def encode(self, x):
        h = self.encoder_conv_in(x)
        for block in self.encoder_down:
            h = block(h)
        h = self.encoder_mid_block1(h)
        h = self.encoder_mid_block2(h)
        h = self.encoder_norm_out(h)
        h = nn.silu(h)
        h = self.encoder_conv_out(h)
        mean, logvar = mx.split(h, 2, axis=1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(shape=mean.shape, dtype=mean.dtype)
        z = mean + std * eps
        return z

    def decode(self, z):
        h = self.decoder_conv_in(z)
        h = self.decoder_mid_block1(h)
        h = self.decoder_mid_block2(h)
        for block in self.decoder_up:
            h = block(h)
        h = self.decoder_norm_out(h)
        h = nn.silu(h)
        h = self.decoder_conv_out(h)
        return h

    @classmethod
    def from_pretrained(cls, path, dtype=mx.float32):
        model = cls()
        import glob
        import os

        weight_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not weight_files:
            weight_files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if weight_files:
            from mlx.utils import load_weights

            weights = load_weights(path)
            model.update(weights)
        model = model.astype(dtype)
        logger.info("SVD VAE loaded dtype=%s", dtype)
        return model
