# SPDX-License-Identifier: Apache-2.0
# Vendored from mflux 0.20.0 (https://github.com/filipstrand/mflux) PR#736
# MIT-licensed pure-MLX Qwen-Image-2.1 reference (txt2img + 4-ch RGBA VAE).
# Pinned to fusion-mlx's mlx stack. Upstream-tracked; do not modify forward
# numerics without parity-testing against mflux qwen21.

import mlx.core as mx
from mlx import nn

from .qwen21_avg_down import Qwen21AvgDown
from .qwen21_causal_conv import Qwen21CausalConv
from .qwen21_mid_block import Qwen21MidBlock
from .qwen21_res_block import Qwen21ResBlock
from .qwen21_resample import Qwen21Resample
from .qwen21_rms_norm import Qwen21RMSNorm


class Qwen21ResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temperal_downsample: bool,
        down_flag: bool = True,
    ):
        super().__init__()
        self.resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks):
            self.resnets.append(Qwen21ResBlock(current_dim, out_dim))
            current_dim = out_dim
        self.downsampler = Qwen21Resample(out_dim, out_dim, "downsample") if down_flag else None
        # the shortcut always exists; at the bottleneck (no downsample, equal dims) it is a
        # factor-1 identity pooling and the residual add is still applied
        self.avg_shortcut = Qwen21AvgDown(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(x_copy)


class Qwen21Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 96,
        z_dim: int = 128,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True, True),
    ):
        super().__init__()
        # [1] + dim_mult doubles the first multiplier: dims = [96, 96, 192, 384, 768, 768],
        # five residual down blocks where the first four downsample and the last is a plain
        # residual block group at the bottleneck width.
        dims = [dim * mult for mult in [1] + list(dim_mult)]

        self.conv_in = Qwen21CausalConv(in_channels, dims[0], 3, 1)
        self.down_blocks = [
            Qwen21ResidualDownBlock(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                num_res_blocks=num_res_blocks,
                temperal_downsample=temperal_downsample[i] if i < len(temperal_downsample) else False,
                down_flag=i < len(dims) - 2,
            )
            for i in range(len(dims) - 1)
        ]
        self.mid_block = Qwen21MidBlock(dims[-1])
        self.norm_out = Qwen21RMSNorm(dims[-1])
        self.conv_out = Qwen21CausalConv(dims[-1], z_dim, 3, 1)

    def __call__(self, x):
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        x = nn.silu(self.norm_out(x))
        return self.conv_out(x)
