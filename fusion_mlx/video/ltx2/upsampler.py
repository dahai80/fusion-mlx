# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 upsampler (vendored from mlx-video).
# Phase 4 LTX-2 direct-MLX port: latent spatial upsampler (x2 / x1.5).
import logging
from fractions import Fraction

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        scale = (
            1.0
            / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]) ** 0.5
        )
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(
                out_channels,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
                in_channels,
            ),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class GroupNorm3d(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        n, d, h, w, c = x.shape
        input_dtype = x.dtype

        x = x.astype(mx.float32)

        x = mx.reshape(x, (n, d * h * w, self.num_groups, c // self.num_groups))

        mean = mx.mean(x, axis=(1, 3), keepdims=True)
        var = mx.var(x, axis=(1, 3), keepdims=True)

        x = (x - mean) / mx.sqrt(var + self.eps)

        x = mx.reshape(x, (n, d, h, w, c))

        weight = self.weight.astype(mx.float32)
        bias = self.bias.astype(mx.float32)
        x = x * weight + bias

        x = x.astype(input_dtype)

        return x


class PixelShuffle2D(nn.Module):
    def __init__(self, upscale_factor_h: int = 2, upscale_factor_w: int = 2):
        super().__init__()
        self.rh = upscale_factor_h
        self.rw = upscale_factor_w

    def __call__(self, x: mx.array) -> mx.array:
        n, h, w, c = x.shape
        rh, rw = self.rh, self.rw
        out_c = c // (rh * rw)

        x = mx.reshape(x, (n, h, w, out_c, rh, rw))

        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))

        x = mx.reshape(x, (n, h * rh, w * rw, out_c))

        return x


class BlurDownsample(nn.Module):
    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride
        k = mx.array([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel_2d = mx.outer(k, k)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.kernel = kernel_2d.reshape(1, 5, 5, 1)

    def __call__(self, x: mx.array) -> mx.array:
        n, h, w, c = x.shape

        x = mx.pad(x, [(0, 0), (2, 2), (2, 2), (0, 0)], mode="edge")

        x = mx.transpose(x, (0, 3, 1, 2))
        x = mx.reshape(x, (n * c, h + 4, w + 4, 1))

        x = mx.conv2d(x, self.kernel, stride=(self.stride, self.stride))

        _, h_out, w_out, _ = x.shape
        x = mx.reshape(x, (n, c, h_out, w_out))
        x = mx.transpose(x, (0, 2, 3, 1))

        return x


class SpatialUpsampler2x(nn.Module):
    def __init__(self, mid_channels: int = 1024):
        super().__init__()
        self.scale = 2.0
        self.conv = nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle2D(2, 2)

    def __call__(self, x: mx.array) -> mx.array:
        n, d, h, w, c = x.shape
        x = mx.reshape(x, (n * d, h, w, c))
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = mx.reshape(x, (n, d, h * 2, w * 2, c))
        return x


class SpatialRationalResampler(nn.Module):
    def __init__(self, mid_channels: int = 1024, scale: float = 1.5):
        super().__init__()
        self.scale = scale

        num, den = _rational_for_scale(scale)
        self.num = num
        self.den = den

        self.conv = nn.Conv2d(
            mid_channels, num * num * mid_channels, kernel_size=3, padding=1
        )
        self.pixel_shuffle = PixelShuffle2D(num, num)
        self.blur_down = BlurDownsample(stride=den)

    def __call__(self, x: mx.array) -> mx.array:
        n, d, h, w, c = x.shape
        x = mx.reshape(x, (n * d, h, w, c))

        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)

        _, h_out, w_out, _ = x.shape
        x = mx.reshape(x, (n, d, h_out, w_out, c))
        return x


def _rational_for_scale(scale: float) -> tuple[int, int]:
    frac = Fraction(scale).limit_denominator(10)
    return frac.numerator, frac.denominator


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm3d(32, channels)
        self.conv2 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm3d(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = nn.silu(x + residual)

        return x


class LatentUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.spatial_scale = spatial_scale

        self.initial_conv = Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = GroupNorm3d(32, mid_channels)

        self.res_blocks = {
            i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)
        }

        if rational_resampler:
            self.upsampler = SpatialRationalResampler(
                mid_channels=mid_channels, scale=spatial_scale
            )
        else:
            self.upsampler = SpatialUpsampler2x(mid_channels=mid_channels)

        self.post_upsample_res_blocks = {
            i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)
        }

        self.final_conv = Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, latent: mx.array, debug: bool = False) -> mx.array:
        def debug_stats(name, t):
            if debug:
                mx.eval(t)
                logger.debug(
                    "    %s: shape=%s min=%.4f max=%.4f mean=%.4f",
                    name,
                    t.shape,
                    t.min().item(),
                    t.max().item(),
                    t.mean().item(),
                )

        if debug:
            logger.debug("[DEBUG] LatentUpsampler forward pass:")
            debug_stats("Input (channels first)", latent)

        x = mx.transpose(latent, (0, 2, 3, 4, 1))

        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)

        for i in sorted(self.res_blocks.keys()):
            x = self.res_blocks[i](x)

        x = self.upsampler(x)
        if debug:
            debug_stats(f"After upsampler (spatial {self.spatial_scale}x)", x)

        for i in sorted(self.post_upsample_res_blocks.keys()):
            x = self.post_upsample_res_blocks[i](x)

        x = self.final_conv(x)

        x = mx.transpose(x, (0, 4, 1, 2, 3))
        if debug:
            debug_stats("Output (channels first)", x)

        return x


def upsample_latents(
    latent: mx.array,
    upsampler: LatentUpsampler,
    latent_mean: mx.array,
    latent_std: mx.array,
    debug: bool = False,
) -> mx.array:
    latent_mean = latent_mean.reshape(1, -1, 1, 1, 1)
    latent_std = latent_std.reshape(1, -1, 1, 1, 1)
    latent = latent * latent_std + latent_mean

    latent = upsampler(latent, debug=debug)

    latent = (latent - latent_mean) / latent_std

    return latent


def load_upsampler(weights_path: str) -> tuple[LatentUpsampler, float]:
    logger.info("Loading spatial upsampler from %s", weights_path)
    raw_weights = mx.load(weights_path)

    sample_key = "res_blocks.0.conv1.weight"
    if sample_key in raw_weights:
        mid_channels = raw_weights[sample_key].shape[0]
    else:
        mid_channels = 1024

    conv_key = (
        "upsampler.conv.weight"
        if "upsampler.conv.weight" in raw_weights
        else "upsampler.0.weight"
    )
    if conv_key in raw_weights:
        out_channels = raw_weights[conv_key].shape[0]
        ratio = out_channels // mid_channels
        rational_resampler = ratio == 9
        spatial_scale = 1.5 if rational_resampler else 2.0
    else:
        rational_resampler = False
        spatial_scale = 2.0

    logger.info(
        "Detected upsampler: mid_channels=%d scale=%sx rational=%s",
        mid_channels,
        spatial_scale,
        rational_resampler,
    )

    upsampler = LatentUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
        spatial_scale=spatial_scale,
        rational_resampler=rational_resampler,
    )

    sanitized = {}
    for key, value in raw_weights.items():
        new_key = key

        if key.startswith("upsampler.0."):
            new_key = key.replace("upsampler.0.", "upsampler.conv.")

        if "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        if ("weight" in new_key or "kernel" in new_key) and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    upsampler.load_weights(list(sanitized.items()), strict=False)

    logger.info("Loaded %d upsampler weights", len(sanitized))

    return upsampler, spatial_scale
