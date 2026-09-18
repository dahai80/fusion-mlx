# SPDX-License-Identifier: Apache-2.0
# SafeGroupNorm — FP32-protected GroupNorm for FP16 vision models (#911).
#
# Stock mlx.nn.GroupNorm computes mean/var in the input dtype. Under FP16,
# GroupNorm statistics drift (variance underflow near small activations) causing
# PRD V2 tier-2 cosine >= 0.98 violations on VAE/UNet norm layers. SafeGroupNorm
# upcasts to FP32 for the reduction stats, applies the affine in FP32, then casts
# the normalized output back to the input dtype — numerically stable, no runtime
# torch/mmpose dependency.
#
# Matches MLX's channels-last (NHWC) layout convention: the last axis of x is the
# channel dim C. pytorch_compatible mirrors nn.GroupNorm's weight/bias broadcast.

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SafeGroupNorm(nn.Module):
    """GroupNorm with FP32-protected mean/variance (#911).

    Identical interface to ``mlx.nn.GroupNorm`` (groups, dims, eps, pytorch_compatible)
    but the reduction (mean/var) and affine transform run in FP32 internally, then
    the result is cast back to the caller's dtype. This avoids FP16 variance
    underflow on VAE/UNet GroupNorm layers (PRD V2 cosine >= 0.98 gate).

    Layout: channels-last (NHWC) — the last axis of x is C, matching MLX conv/groupnorm.
    """

    def __init__(
        self,
        groups: int,
        dims: int,
        eps: float = 1e-6,
        pytorch_compatible: bool = False,
    ):
        super().__init__()
        if dims % groups != 0:
            raise ValueError(
                f"SafeGroupNorm: dims ({dims}) must be divisible by groups ({groups})"
            )
        self.num_groups = groups
        self.dims = dims
        self.eps = eps
        self.pytorch_compatible = pytorch_compatible
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))

    def _pytorch_compatible_group_norm_fp32(self, x32):
        num_groups = self.num_groups
        batch, *rest, dims = x32.shape
        group_size = dims // num_groups
        x = x32.reshape(batch, -1, num_groups, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, num_groups, -1)
        # layer_norm normalizes over last axis (the per-group flattened features).
        x = mx.fast.layer_norm(x, eps=self.eps, weight=None, bias=None)
        x = x.reshape(batch, num_groups, -1, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
        return x

    def _group_norm_fp32(self, x32):
        num_groups = self.num_groups
        batch, *rest, dims = x32.shape
        x = x32.reshape(batch, -1, num_groups)
        means = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        x = x.reshape(batch, *rest, dims)
        return x

    def __call__(self, x):
        input_dtype = x.dtype
        x32 = x.astype(mx.float32)
        group_norm = (
            self._pytorch_compatible_group_norm_fp32
            if self.pytorch_compatible
            else self._group_norm_fp32
        )
        x = group_norm(x32)
        # Affine in FP32, weight/bias broadcast over channels-last dim.
        w = self.weight.astype(mx.float32)
        b = self.bias.astype(mx.float32)
        out = w * x + b
        return out.astype(input_dtype)


def safe_group_norm(groups: int, channels: int, eps: float = 1e-6):
    """Factory matching the ``nn.GroupNorm(num_groups, dims, eps)`` call shape (#911)."""
    return SafeGroupNorm(groups, channels, eps=eps, pytorch_compatible=True)
