# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE upsampling (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
import mlx.core as mx
import mlx.nn as nn

from ..config import CausalityAxis
from .attention import AttentionType, make_attn
from .causal_conv_2d import make_conv2d
from .normalization import NormType
from .resnet import ResnetBlock


def nearest_neighbor_upsample(x: mx.array, scale_factor: int = 2) -> mx.array:
    x = mx.repeat(x, scale_factor, axis=1)
    x = mx.repeat(x, scale_factor, axis=2)
    return x


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )

    def __call__(self, x: mx.array) -> mx.array:
        x = nearest_neighbor_upsample(x, scale_factor=2)
        if self.with_conv:
            x = self.conv(x)
            if self.causality_axis == CausalityAxis.NONE:
                pass
            elif self.causality_axis == CausalityAxis.HEIGHT:
                x = x[:, 1:, :, :]
            elif self.causality_axis == CausalityAxis.WIDTH:
                x = x[:, :, 1:, :]
            elif self.causality_axis == CausalityAxis.WIDTH_COMPATIBILITY:
                pass
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")
        return x


def build_upsampling_path(
    *,
    ch: int,
    ch_mult: tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: set[int],
    resamp_with_conv: bool,
    initial_block_channels: int,
) -> tuple[dict, int]:
    up_modules = {}
    block_in = initial_block_channels
    curr_res = resolution // (2 ** (num_resolutions - 1))
    for level in reversed(range(num_resolutions)):
        stage = {}
        stage["block"] = {}
        stage["attn"] = {}
        block_out = ch * ch_mult[level]
        for i_block in range(num_res_blocks + 1):
            stage["block"][i_block] = ResnetBlock(
                in_channels=block_in,
                out_channels=block_out,
                temb_channels=temb_channels,
                dropout=dropout,
                norm_type=norm_type,
                causality_axis=causality_axis,
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                stage["attn"][i_block] = make_attn(
                    block_in, attn_type=attn_type, norm_type=norm_type
                )
        if level != 0:
            stage["upsample"] = Upsample(
                block_in, resamp_with_conv, causality_axis=causality_axis
            )
            curr_res *= 2
        up_modules[level] = stage
    return up_modules, block_in
