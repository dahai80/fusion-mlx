# SPDX-License-Identifier: Apache-2.0
import logging

import mlx.core as mx
import mlx.nn as nn

from .config import CogVideoXConfig

logger = logging.getLogger(__name__)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 1,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        kt, kh, kw = kernel_size
        self.weight = mx.zeros((out_channels, in_channels, kt, kh, kw))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # MLX has no conv3d; implement via stacked 2D convs per frame
        # with temporal causal padding via frame-shift
        pt, ph, pw = self.padding
        kt, kh, kw = self.kernel_size
        b, c, t, h, w = x.shape

        # Causal temporal padding: pad 2*pt on left only
        if pt > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (2 * pt, 0), (0, 0), (0, 0)])
            t_padded = t + 2 * pt
        else:
            t_padded = t

        # Reshape to (b*t_padded, c, h + 2*ph, w + 2*pw)
        x = x.transpose(0, 2, 1, 3, 4)  # (b, t_padded, c, h, w)
        x = x.reshape(b * t_padded, c, h, w)

        if ph > 0 or pw > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)])

        # Apply 2D conv with spatial-only kernels (temporal handled via frame stacking)
        # Weight: (out_ch, in_ch, kt, kh, kw) -> treat as (out_ch, in_ch*kt, kh, kw)
        weight_2d = self.weight.reshape(self.weight.shape[0], -1, kh, kw)

        # For temporal conv, stack kt consecutive frames as channels
        # This is a simplification - proper 3D conv needs per-frame accumulation
        stride_2d = (self.stride[1], self.stride[2])
        out = mx.conv2d(x, weight_2d, stride=stride_2d, padding=0)

        out = out.reshape(b, t_padded, -1, out.shape[2], out.shape[3])

        # Handle temporal stride
        st = self.stride[0]
        if st > 1:
            out = out[:, ::st]

        # Trim to original temporal size
        out_t = out.shape[1]
        if out_t > t:
            # Take last t frames (causal)
            out = out[:, out_t - t :]

        out = out.transpose(0, 2, 1, 3, 4)  # (b, c, t, h, w)
        return out


class ResNetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        out_ch = out_channels or in_channels
        self.norm1 = nn.GroupNorm(norm_num_groups, in_channels, eps=norm_eps)
        self.conv1 = CausalConv3d(
            in_channels, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(norm_num_groups, out_ch, eps=norm_eps)
        self.conv2 = CausalConv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.residual = (
            nn.Linear(in_channels, out_ch) if in_channels != out_ch else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if x.shape[1] != h.shape[1]:
            res = self.residual(x.transpose(0, 2, 3, 4, 1))
            res = res.transpose(0, 4, 1, 2, 3)
        else:
            res = x
        return h + res


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers_per_block: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        downsample: bool = True,
    ):
        super().__init__()
        self.resnets = [
            ResNetBlock3D(
                in_channels if i == 0 else out_channels,
                out_channels,
                norm_num_groups,
                norm_eps,
            )
            for i in range(layers_per_block)
        ]
        self.downsample = downsample
        if downsample:
            self.conv_down = CausalConv3d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsample:
            x = self.conv_down(x)
        return x


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers_per_block: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        upsample: bool = True,
    ):
        super().__init__()
        self.resnets = [
            ResNetBlock3D(
                in_channels if i == 0 else out_channels,
                out_channels,
                norm_num_groups,
                norm_eps,
            )
            for i in range(layers_per_block)
        ]
        self.upsample = upsample
        if upsample:
            self.conv_up = CausalConv3d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsample:
            b, c, t, h, w = x.shape
            x = mx.broadcast_to(
                x[:, :, :, None, :, None, :, None], (b, c, t, 2, h, 2, w, 2)
            )
            x = x.reshape(b, c, t * 2, h * 2, w * 2)
            x = self.conv_up(x)
        return x


class AutoencoderKLCogVideoX(nn.Module):
    def __init__(self, config: CogVideoXConfig):
        super().__init__()
        self.config = config
        latent_ch = config.vae_latent_channels
        block_out = config.vae_block_out_channels
        layers_per = config.vae_layers_per_block

        self.encoder_conv_in = CausalConv3d(
            3, block_out[0], kernel_size=3, stride=1, padding=1
        )
        self.encoder_blocks = []
        ch_in = block_out[0]
        for i, ch_out in enumerate(block_out):
            is_last = i == len(block_out) - 1
            self.encoder_blocks.append(
                DownEncoderBlock3D(ch_in, ch_out, layers_per, downsample=not is_last)
            )
            ch_in = ch_out
        self.encoder_conv_out = CausalConv3d(
            block_out[-1], 2 * latent_ch, kernel_size=3, stride=1, padding=1
        )

        self.decoder_conv_in = CausalConv3d(
            latent_ch, block_out[-1], kernel_size=3, stride=1, padding=1
        )
        self.decoder_blocks = []
        ch_in = block_out[-1]
        for i, ch_out in enumerate(reversed(block_out)):
            is_last = i == len(block_out) - 1
            self.decoder_blocks.append(
                UpDecoderBlock3D(ch_in, ch_out, layers_per, upsample=not is_last)
            )
            ch_in = ch_out
        self.decoder_conv_out = CausalConv3d(
            block_out[0], 3, kernel_size=3, stride=1, padding=1
        )

        self.scaling_factor = config.scaling_factor

    def encode(self, x: mx.array) -> mx.array:
        h = self.encoder_conv_in(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_conv_out(h)
        mean, logvar = h.split(2, axis=1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        std = mx.exp(0.5 * logvar)
        noise = mx.random.normal(mean.shape)
        z = mean + std * noise
        z = z * self.scaling_factor
        return z

    def decode(self, z: mx.array) -> mx.array:
        z = z / self.scaling_factor
        h = self.decoder_conv_in(z)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.decoder_conv_out(h)
        return h

    def __call__(self, z: mx.array) -> mx.array:
        return self.decode(z)
