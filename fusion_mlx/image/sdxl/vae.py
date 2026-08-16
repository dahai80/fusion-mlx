import logging

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


class GroupNorm(nn.Module):
    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        b, h, w, c = x.shape
        x = x.reshape(b, h, w, self.num_groups, c // self.num_groups)
        mean = mx.mean(x, axis=(1, 2, 4), keepdims=True)
        var = mx.var(x, axis=(1, 2, 4), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = x.reshape(b, h, w, c)
        x = x * self.weight + self.bias
        return x


class Conv2d(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1
    ):
        super().__init__()
        scale = 1.0 / (in_ch * k * k) ** 0.5
        self.weight = mx.random.normal((out_ch, k, k, in_ch)) * scale
        self.bias = mx.zeros((out_ch,))
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            x = mx.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )
        y = mx.conv2d(x, self.weight, stride=self.stride)
        y = y + self.bias
        return y


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = GroupNorm(in_ch)
        self.conv1 = Conv2d(in_ch, out_ch)
        self.norm2 = GroupNorm(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch)
        self.conv_shortcut = (
            Conv2d(in_ch, out_ch, k=1, padding=0) if in_ch != out_ch else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x.astype(mx.float32) + h


class Attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = GroupNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        b, hh, w, c = x.shape
        normed = self.group_norm(x)
        q = self.to_q(normed).reshape(b, hh * w, 1, c).transpose(0, 2, 1, 3)
        k = self.to_k(normed).reshape(b, hh * w, 1, c).transpose(0, 2, 1, 3)
        v = self.to_v(normed).reshape(b, hh * w, 1, c).transpose(0, 2, 1, 3)
        scale = 1.0 / mx.sqrt(mx.array(c, dtype=mx.float32))
        attn = scaled_dot_product_attention(q, k, v, scale=scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, hh, w, c)
        attn = self.to_out[0](attn)
        return x + attn


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        b, hh, w, c = x.shape
        h = mx.broadcast_to(x[:, :, None, :, None, :], (b, hh, 2, w, 2, c))
        h = h.reshape(b, hh * 2, w * 2, c)
        h = self.conv(h)
        return h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels, k=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        h = self.conv(h)
        return h


class MidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.resnets = [
            ResnetBlock(channels, channels),
            ResnetBlock(channels, channels),
        ]
        self.attentions = [Attention(channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class DownBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, num_res_blocks: int, add_downsample: bool
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_ch if i == 0 else out_ch, out_ch)
            for i in range(num_res_blocks)
        ]
        self.downsamplers = [Downsample(out_ch)] if add_downsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for res in self.resnets:
            x = res(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, num_res_blocks: int, add_upsample: bool
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_ch if i == 0 else out_ch, out_ch)
            for i in range(num_res_blocks)
        ]
        self.upsamplers = [Upsample(out_ch)] if add_upsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for res in self.resnets:
            x = res(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class VAEEncoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.conv_in = Conv2d(3, ch)
        self.down_blocks = []
        prev = ch
        for level, mult in enumerate(ch_mult):
            cur = ch * mult
            self.down_blocks.append(
                DownBlock(
                    prev, cur, num_res_blocks, add_downsample=(level < len(ch_mult) - 1)
                )
            )
            prev = cur
        self.mid_block = MidBlock(prev)
        self.conv_norm_out = GroupNorm(prev)
        self.conv_out = Conv2d(prev, 2 * latent_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = _nchw_to_nhwc(x)
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        h = self.conv_norm_out(x)
        h = nn.silu(h)
        h = self.conv_out(h)
        return _nhwc_to_nchw(h)


class VAEDecoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks: int = 3,
        latent_channels: int = 4,
    ):
        super().__init__()
        reversed_mult = list(reversed(ch_mult))
        self.conv_in = Conv2d(latent_channels, ch * reversed_mult[0])
        self.mid_block = MidBlock(ch * reversed_mult[0])
        self.up_blocks = []
        prev = ch * reversed_mult[0]
        for level, mult in enumerate(reversed_mult):
            cur = ch * mult
            self.up_blocks.append(
                UpBlock(
                    prev, cur, num_res_blocks, add_upsample=(level < len(ch_mult) - 1)
                )
            )
            prev = cur
        self.conv_norm_out = GroupNorm(ch * ch_mult[0])
        self.conv_out = Conv2d(ch * ch_mult[0], 3)

    def __call__(self, x: mx.array) -> mx.array:
        x = _nchw_to_nhwc(x)
        x = self.conv_in(x)
        x = self.mid_block(x)
        for blk in self.up_blocks:
            x = blk(x)
        h = self.conv_norm_out(x)
        h = nn.silu(h)
        h = self.conv_out(h)
        return _nhwc_to_nchw(h)


class SDXLVAE(nn.Module):
    scaling_factor = 0.13025
    spatial_scale = 8
    latent_channels = 4

    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()
        # 1x1 convs sitting between encoder/decoder and the latent space
        # (diffusers AutoencoderKL: quant_conv after encoder, post_quant_conv
        # before decoder). Without these the latents are wrong (#482).
        self.quant_conv = Conv2d(
            2 * self.latent_channels, 2 * self.latent_channels, k=1, padding=0
        )
        self.post_quant_conv = Conv2d(
            self.latent_channels, self.latent_channels, k=1, padding=0
        )

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        scaled = latents.astype(mx.float32) / self.scaling_factor
        # post_quant_conv is a Conv2d (expects NHWC); latents are NCHW here.
        scaled = _nchw_to_nhwc(scaled)
        scaled = self.post_quant_conv(scaled)
        scaled = _nhwc_to_nchw(scaled)
        decoded = self.decoder(scaled)
        return decoded

    def encode(self, image: mx.array) -> mx.array:
        latents = self.encoder(image)
        # quant_conv is a Conv2d (expects NHWC); encoder returns NCHW.
        latents = _nchw_to_nhwc(latents)
        latents = self.quant_conv(latents)
        latents = _nhwc_to_nchw(latents)
        mean, _ = mx.split(latents, 2, axis=1)
        latent = mean * self.scaling_factor
        return latent
