# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE causal convolutions (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
import mlx.core as mx
import mlx.nn as nn

from ..config import CausalityAxis


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        kernel_size = _pair(kernel_size)
        dilation = _pair(dilation)
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]
        if self.causality_axis == CausalityAxis.NONE:
            self.padding = (
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
            )
        elif self.causality_axis in (
            CausalityAxis.WIDTH,
            CausalityAxis.WIDTH_COMPATIBILITY,
        ):
            self.padding = (pad_h // 2, pad_h - pad_h // 2, pad_w, 0)
        elif self.causality_axis == CausalityAxis.HEIGHT:
            self.padding = (pad_h, 0, pad_w // 2, pad_w - pad_w // 2)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = self.padding
        if any(p > 0 for p in self.padding):
            x = mx.pad(
                x,
                [(0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)],
            )
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int = 1,
    padding: int | tuple[int, int] | None = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
) -> nn.Module:
    if causality_axis is not None:
        return CausalConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            causality_axis,
        )
    if padding is None:
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )
