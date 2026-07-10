# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE / vocoder resnet blocks (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
import mlx.core as mx
import mlx.nn as nn

from ..config import CausalityAxis
from .causal_conv_2d import make_conv2d
from .normalization import NormType, build_normalization_layer

LRELU_SLOPE = 0.1


def leaky_relu(x: mx.array, negative_slope: float = LRELU_SLOPE) -> mx.array:
    return mx.maximum(x, x * negative_slope)


class ResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=(kernel_size - 1) * d // 2,
            )
            for i, d in enumerate(dilation)
        }
        self.convs2 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            )
            for i in range(len(dilation))
        }

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(len(self.convs1)):
            xt = leaky_relu(x, LRELU_SLOPE)
            xt = self.convs1[i](xt)
            xt = leaky_relu(xt, LRELU_SLOPE)
            xt = self.convs2[i](xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int] = (1, 3),
    ):
        super().__init__()
        self.convs = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=(kernel_size - 1) * d // 2,
            )
            for i, d in enumerate(dilation)
        }

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(len(self.convs)):
            xt = leaky_relu(x, LRELU_SLOPE)
            xt = self.convs[i](xt)
            x = xt + x
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        if self.causality_axis != CausalityAxis.NONE and norm_type == NormType.GROUP:
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.temb_channels = temb_channels
        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.conv1 = make_conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout_rate = dropout
        self.conv2 = make_conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    causality_axis=causality_axis,
                )

    def __call__(
        self,
        x: mx.array,
        temb: mx.array | None = None,
    ) -> mx.array:
        h = x
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)
        if temb is not None and self.temb_channels > 0:
            h = h + mx.expand_dims(
                mx.expand_dims(nn.silu(self.temb_proj(temb)), axis=1), axis=1
            )
        h = self.norm2(h)
        h = nn.silu(h)
        if self.dropout_rate > 0:
            h = nn.Dropout(self.dropout_rate)(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h
