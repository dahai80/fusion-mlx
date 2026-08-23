# SPDX-License-Identifier: Apache-2.0
# SnakeBeta 激活函数（BigVGAN 周期性激活）。
# 上游 dac_activations.py：snakebeta(x, alpha, beta) = x + 1/b * sin^2(a*x)。
# alpha 控制频率，beta 控制幅度。alpha_logscale=True 时存储 log 空间，forward exp 还原。
import mlx.core as mx
import mlx.nn as nn


class SnakeBeta(nn.Module):
    # in_features: 通道数。alpha/beta 形状 (channels,)，forward 时广播到 (B,C,T)。
    # alpha_logscale: True 时 alpha/beta 初始化为 0（log 空间），forward 用 exp。
    def __init__(self, in_features, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            # log 空间初始化为 0 → exp(0)=1（对应 linear 空间 alpha=1）。
            self.alpha = mx.zeros((in_features,), dtype=mx.float32)
            self.beta = mx.zeros((in_features,), dtype=mx.float32)
        else:
            self.alpha = mx.ones((in_features,), dtype=mx.float32)
            self.beta = mx.ones((in_features,), dtype=mx.float32)

    def __call__(self, x):
        # x: (B, T, C) channel-last。alpha/beta 广播 (1, 1, C)。
        alpha = self.alpha.reshape(1, 1, -1)
        beta = self.beta.reshape(1, 1, -1)
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        # snakebeta: x + (1/b) * sin^2(a*x)
        return x + mx.reciprocal(beta + 1e-9) * mx.square(mx.sin(alpha * x))


__all__ = ["SnakeBeta"]
