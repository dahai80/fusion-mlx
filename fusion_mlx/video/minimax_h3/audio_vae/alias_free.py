# SPDX-License-Identifier: Apache-2.0
# Alias-free 上/下采样 + 抗混叠激活（MLX 移植）。
# 上游 dac_alias_free_filter.py / dac_alias_free_resample.py / dac_alias_free_act.py。
# 关键 MLX 适配：
#   - 通道 last：(B, T, C)，权重 (out, k, in/groups)
#   - mx.pad 仅支持 constant/edge（无 replicate）→ edge = 边缘复制，等价
#   - kaiser-sinc filter 在 init 时用 numpy 一次性算（CPU 固定 buffer，不参与训练）
import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    # 返回 (1, 1, kernel_size) 低通滤波器，sum=1 归一化。
    # 上游：2*cutoff*window*sinc(2*cutoff*time)，sinc = sin(pi x)/(pi x)。
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * np.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = np.kaiser(kernel_size, beta)
    if even:
        time = np.arange(-half_size, half_size) + 0.5
    else:
        time = np.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = np.zeros_like(time, dtype=np.float32)
    else:
        sinc = np.sinc(2 * cutoff * time)  # np.sinc 已 pi 归一化，等价 torch.sinc
        filter_ = 2 * cutoff * window * sinc
        filter_ = filter_ / filter_.sum()
    filter_ = filter_.astype(np.float32).reshape(1, 1, kernel_size)
    return mx.array(filter_)


class LowPassFilter1d(nn.Module):
    # depthwise 低通卷积，padding=edge（等价上游 replicate）。
    def __init__(
        self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, padding=True
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("cutoff must be >= 0")
        if cutoff > 0.5:
            raise ValueError("cutoff > 0.5 invalid")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)  # (1,1,k)
        self.filter = filt  # (1,1,k)，forward 时广播到 (C,k,1)

    def __call__(self, x):
        # x (B, T, C)
        _, _, C = x.shape
        if self.padding:
            pads = [(0, 0), (self.pad_left, self.pad_right), (0, 0)]
            x = mx.pad(x, pads, mode="edge")
        # filter (1,1,k) -> (k,1) -> (C,k,1)
        w = mx.broadcast_to(
            self.filter.reshape(self.kernel_size, 1), (C, self.kernel_size, 1)
        )
        return mx.conv1d(x, w, stride=self.stride, padding=0, groups=C)


class UpSample1d(nn.Module):
    # depthwise 反卷积上采样 + 低通。
    # 上游：F.pad(replicate) → conv_transpose1d(groups=C) → 裁剪。
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filt = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.filter = filt  # (1,1,k)

    def __call__(self, x):
        # x (B, T, C)
        _, _, C = x.shape
        x = mx.pad(x, [(0, 0), (self.pad, self.pad), (0, 0)], mode="edge")
        # filter (1,1,k) -> (k,1) -> (C,k,1)
        w = mx.broadcast_to(
            self.filter.reshape(self.kernel_size, 1), (C, self.kernel_size, 1)
        )
        x = self.ratio * mx.conv_transpose1d(
            x, w, stride=self.stride, padding=0, groups=C
        )
        # 裁剪两侧
        if self.pad_right == 0:
            x = x[:, self.pad_left :, :]
        else:
            x = x[:, self.pad_left : -self.pad_right, :]
        return x


class DownSample1d(nn.Module):
    # 低通 + stride 下采样。
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    # 上采样 → 激活 → 下采样（抗混叠激活）。
    # 上游 act=SnakeBeta，up/down=UpSample1d/DownSample1d ratio=2。
    def __init__(
        self,
        activation,
        up_ratio=2,
        down_ratio=2,
        up_kernel_size=12,
        down_kernel_size=12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(ratio=up_ratio, kernel_size=up_kernel_size)
        self.downsample = DownSample1d(ratio=down_ratio, kernel_size=down_kernel_size)

    def __call__(self, x):
        # x (B, T, C)
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


__all__ = [
    "kaiser_sinc_filter1d",
    "LowPassFilter1d",
    "UpSample1d",
    "DownSample1d",
    "Activation1d",
]
