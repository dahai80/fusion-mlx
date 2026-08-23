# SPDX-License-Identifier: Apache-2.0
# BigVGAN 解码器（MLX 移植，推理用）。
# 上游 dac_bigvgan.py BigVGAN。
# 通道 last：(B, T, C)。ConvTranspose1d weight (out, k, in)（上游 PyTorch (in,out,k) → 加载时转置）。
import logging

import mlx.core as mx
import mlx.nn as nn

from .activations import SnakeBeta
from .alias_free import Activation1d
from .amp_block import AMPBlock1

logger = logging.getLogger(__name__)


class BigVGAN(nn.Module):
    # num_mels: latent_dim（dec_in_proj 输出通道）= 2048。
    # upsample_rates/kernels: [5,5,2,2,2,2,2] / [9,9,4,4,4,4,4]。
    # initial_channel: decoder_dim = 1024，每层减半。
    # resblock_kernel_sizes [3,7,11]，dilation_sizes [[1,3,5]]*3。
    def __init__(
        self,
        num_mels,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        snake_logscale=True,
        use_tanh_at_final=False,
        use_bias_at_final=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final

        # 前置 conv：num_mels -> initial_channel，k7 pad3。
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)

        # 上采样（ConvTranspose1d，无抗混叠）。
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(in_ch, out_ch, k, stride=u, padding=(k - u) // 2)
            )

        # 残差块：每个 upsample 阶段挂 num_kernels 个 AMPBlock1（输出通道 = 该阶段 out_ch）。
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    AMPBlock1(ch, k, d, snake_logscale=snake_logscale)
                )

        # 后置激活 + conv_post（out_ch -> 1, k7 pad3, bias=use_bias_at_final）。
        final_ch = upsample_initial_channel // (2 ** len(upsample_rates))
        self.activation_post = Activation1d(
            SnakeBeta(final_ch, alpha_logscale=snake_logscale)
        )
        self.conv_post = nn.Conv1d(final_ch, 1, 7, 1, padding=3, bias=use_bias_at_final)

    def __call__(self, x):
        # x (B, T, num_mels)。
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            x = mx.tanh(x)
        else:
            x = mx.clip(x, -1.0, 1.0)
        return x  # (B, T_out, 1)


__all__ = ["BigVGAN"]
