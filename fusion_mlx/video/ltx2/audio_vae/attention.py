# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE attention (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
from enum import Enum

import mlx.core as mx
import mlx.nn as nn

from .normalization import NormType, build_normalization_layer


class AttentionType(Enum):
    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def __call__(self, x: mx.array) -> mx.array:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, h, w, c = q.shape
        q = q.reshape(b, h * w, c)
        k = k.reshape(b, h * w, c)
        v = v.reshape(b, h * w, c)
        scale = float(c) ** (-0.5)
        w_ = mx.matmul(q, k.transpose(0, 2, 1)) * scale
        w_ = mx.softmax(w_, axis=-1)
        h_ = mx.matmul(w_, v)
        h_ = h_.reshape(b, h, w, c)
        h_ = self.proj_out(h_)
        return x + h_


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


def make_attn(
    in_channels: int,
    attn_type: AttentionType = AttentionType.VANILLA,
    norm_type: NormType = NormType.GROUP,
) -> nn.Module:
    if attn_type == AttentionType.VANILLA:
        return AttnBlock(in_channels, norm_type=norm_type)
    if attn_type == AttentionType.NONE:
        return Identity()
    if attn_type == AttentionType.LINEAR:
        raise NotImplementedError(
            f"Attention type {attn_type.value} is not supported yet."
        )
    raise ValueError(f"Unknown attention type: {attn_type}")
