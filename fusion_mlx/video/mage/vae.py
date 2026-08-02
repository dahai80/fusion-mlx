# SPDX-License-Identifier: Apache-2.0
import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class MageVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        hidden_channels: int = 128,
        depth: int = 4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = _VAEEncoder(in_channels, hidden_channels, latent_channels, depth)
        self.decoder = _VAEDecoder(
            latent_channels, hidden_channels, out_channels, depth
        )
        self.quant_conv = nn.Linear(latent_channels * 2, latent_channels * 2)
        self.post_quant_conv = nn.Linear(latent_channels, latent_channels)

    def encode(self, x: mx.array) -> mx.array:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = mx.split(moments, 2, axis=-1)
        return mean + mx.random.normal(shape=mean.shape) * mx.exp(0.5 * logvar)

    def decode(self, z: mx.array) -> mx.array:
        z = self.post_quant_conv(z)
        return self.decoder(z)


class _VAEEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, latent_ch: int, depth: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.down_blocks = [
            nn.Conv2d(
                hidden_ch * (2**i),
                hidden_ch * (2 ** (i + 1)),
                3,
                stride=2,
                padding=1,
            )
            for i in range(depth)
        ]
        final_ch = hidden_ch * (2**depth)
        self.conv_out = nn.Conv2d(final_ch, latent_ch * 2, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = nn.silu(block(x))
        x = self.conv_out(nn.silu(x))
        return x


class _VAEDecoder(nn.Module):
    def __init__(self, latent_ch: int, hidden_ch: int, out_ch: int, depth: int):
        super().__init__()
        final_ch = hidden_ch * (2**depth)
        self.conv_in = nn.Conv2d(latent_ch, final_ch, 3, padding=1)
        self.up_blocks = []
        for i in range(depth):
            in_c = hidden_ch * (2 ** (depth - i))
            out_c = hidden_ch * (2 ** (depth - i - 1)) if i < depth - 1 else hidden_ch
            self.up_blocks.append(nn.Conv2d(in_c, out_c, 3, padding=1))
        self.conv_out = nn.Conv2d(hidden_ch, out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.up_blocks:
            x = nn.silu(block(x))
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        x = self.conv_out(nn.silu(x))
        return x
