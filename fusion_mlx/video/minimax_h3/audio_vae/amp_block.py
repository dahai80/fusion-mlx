# SPDX-License-Identifier: Apache-2.0
# AMPBlock1：SnakeBeta 抗混叠残差块（BigVGAN multi-periodicity）。
# 上游 dac_bigvgan.py AMPBlock1。
# 通道 last：(B, T, C)，Conv1d weight (out, k, in)。
import mlx.nn as nn

from .activations import SnakeBeta
from .alias_free import Activation1d


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    # channels: 通道数。kernel_size: conv 核。dilation: (1,3,5) per layer。
    # snake_logscale: SnakeBeta alpha_logscale。
    def __init__(
        self, channels, kernel_size=3, dilation=(1, 3, 5), snake_logscale=True
    ):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for d in dilation
        ]
        self.convs2 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for _ in range(len(dilation))
        ]
        self.num_layers = len(self.convs1) + len(self.convs2)
        # num_layers 个抗混叠激活，按层分配。
        self.activations = [
            Activation1d(SnakeBeta(channels, alpha_logscale=snake_logscale))
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        # x (B, T, C)。acts1 = activations[::2]，acts2 = activations[1::2]。
        acts1 = self.activations[::2]
        acts2 = self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


__all__ = ["AMPBlock1", "get_padding"]
