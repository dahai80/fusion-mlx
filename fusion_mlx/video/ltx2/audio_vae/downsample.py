# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE downsampling (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
import mlx.core as mx
import mlx.nn as nn

from ..config import CausalityAxis
from .attention import AttentionType, make_attn
from .normalization import NormType
from .resnet import ResnetBlock


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("causality is only supported when `with_conv=True`.")
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def __call__(self, x: mx.array) -> mx.array:
        if self.with_conv:
            if self.causality_axis == CausalityAxis.NONE:
                pad = [(0, 0), (0, 1), (0, 1), (0, 0)]
            elif self.causality_axis == CausalityAxis.WIDTH:
                pad = [(0, 0), (0, 1), (2, 0), (0, 0)]
            elif self.causality_axis == CausalityAxis.HEIGHT:
                pad = [(0, 0), (2, 0), (0, 1), (0, 0)]
            elif self.causality_axis == CausalityAxis.WIDTH_COMPATIBILITY:
                pad = [(0, 0), (0, 1), (1, 0), (0, 0)]
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")
            x = mx.pad(x, pad, constant_values=0)
            x = self.conv(x)
        else:
            n, h, w, c = x.shape
            x = x.reshape(n, h // 2, 2, w // 2, 2, c)
            x = mx.mean(x, axis=(2, 4))
        return x


def build_downsampling_path(
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
) -> tuple[dict, int]:
    down_modules = {}
    curr_res = resolution
    in_ch_mult = (1, *tuple(ch_mult))
    block_in = ch
    for i_level in range(num_resolutions):
        stage = {}
        stage["block"] = {}
        stage["attn"] = {}
        block_in = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult[i_level]
        for i_block in range(num_res_blocks):
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
        if i_level != num_resolutions - 1:
            stage["downsample"] = Downsample(
                block_in, resamp_with_conv, causality_axis=causality_axis
            )
            curr_res = curr_res // 2
        down_modules[i_level] = stage
    return down_modules, block_in
