import logging

import mlx.core as mx
from mlx.core.fast import scaled_dot_product_attention

from mlx import nn

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
    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = False):
        super().__init__()
        self.norm1 = GroupNorm(in_ch)
        self.conv1 = Conv2d(in_ch, out_ch)
        self.norm2 = GroupNorm(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch)
        self.nin_shortcut = Conv2d(in_ch, out_ch, k=1, padding=0) if shortcut else None

    def __call__(self, x: mx.array) -> mx.array:
        xh = _nchw_to_nhwc(x)
        h = self.norm1(xh)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            xh = self.nin_shortcut(xh)
        return _nhwc_to_nchw(xh.astype(mx.float32) + h)


class Attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = _nchw_to_nhwc(x)
        b, hh, w, c = h.shape
        normed = self.norm(h)
        q = self.q(normed).reshape(b, hh * w, 1, c).transpose(0, 2, 1, 3)
        k = self.k(normed).reshape(b, hh * w, 1, c).transpose(0, 2, 1, 3)
        v = self.v(normed).reshape(b, hh * w, 1, c).transpose(0, 2, 1, 3)
        scale = 1.0 / mx.sqrt(mx.array(c, dtype=mx.float32))
        attn = scaled_dot_product_attention(q, k, v, scale=scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, hh, w, c)
        attn = self.proj_out(attn)
        return _nhwc_to_nchw(h + attn)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = _nchw_to_nhwc(x)
        b, hh, w, c = h.shape
        h = mx.broadcast_to(h[:, :, None, :, None, :], (b, hh, 2, w, 2, c))
        h = h.reshape(b, hh * 2, w * 2, c)
        h = self.conv(h)
        return _nhwc_to_nchw(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels, k=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
        h = _nchw_to_nhwc(h)
        h = self.conv(h)
        return _nhwc_to_nchw(h)


class MidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = Attention(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.block_1(x)
        x = self.attn_1(x)
        x = self.block_2(x)
        return x


class _DownLevel(nn.Module):
    def __init__(self, blocks, downsample):
        super().__init__()
        self.block = blocks
        self.downsample = downsample

    def __call__(self, x: mx.array) -> mx.array:
        for blk in self.block:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class _UpLevel(nn.Module):
    def __init__(self, blocks, upsample):
        super().__init__()
        self.block = blocks
        self.upsample = upsample

    def __call__(self, x: mx.array) -> mx.array:
        for blk in self.block:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, ch: int = 128, ch_mult=(1, 2, 4, 4), num_res_blocks: int = 2):
        super().__init__()
        self.conv_in = Conv2d(3, ch)
        self.down = []
        prev = ch
        for level, mult in enumerate(ch_mult):
            cur = ch * mult
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(prev, cur, shortcut=(prev != cur)))
                prev = cur
            self.down.append(
                _DownLevel(
                    blocks, Downsample(prev) if level < len(ch_mult) - 1 else None
                )
            )
        self.mid = MidBlock(prev)
        self.norm_out = GroupNorm(prev)
        self.conv_out = Conv2d(prev, 32)

    def __call__(self, x: mx.array) -> mx.array:
        x = _nchw_to_nhwc(x)
        x = self.conv_in(x)
        x = _nhwc_to_nchw(x)
        for lvl in self.down:
            x = lvl(x)
        x = self.mid(x)
        h = _nchw_to_nhwc(x)
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return _nhwc_to_nchw(h)


class Decoder(nn.Module):
    def __init__(self, ch: int = 128, ch_mult=(1, 2, 4, 4), num_res_blocks: int = 3):
        super().__init__()
        self.conv_in = Conv2d(16, ch * ch_mult[-1])
        reversed_mult = list(reversed(ch_mult))
        prev = ch * reversed_mult[0]
        built = []
        for level, mult in enumerate(reversed_mult):
            cur = ch * mult
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(prev, cur, shortcut=(prev != cur)))
                prev = cur
            has_up = level < len(ch_mult) - 1
            built.append(_UpLevel(blocks, Upsample(cur) if has_up else None))
        # ckpt indexes up.0 = shallowest (128ch), up.3 = deepest (512ch);
        # we built deepest-first, so reverse to align param names with ckpt.
        self.up = list(reversed(built))
        self.mid = MidBlock(ch * reversed_mult[0])
        self.norm_out = GroupNorm(ch * ch_mult[0])
        self.conv_out = Conv2d(ch * ch_mult[0], 3)

    def __call__(self, x: mx.array) -> mx.array:
        x = _nchw_to_nhwc(x)
        x = self.conv_in(x)
        x = _nhwc_to_nchw(x)
        x = self.mid(x)
        for lvl in reversed(self.up):
            x = lvl(x)
        h = _nchw_to_nhwc(x)
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return _nhwc_to_nchw(h)


class SD3VAE(nn.Module):
    scaling_factor = 1.5305
    shift_factor = 0.0609
    spatial_scale = 8
    latent_channels = 16

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        scaled = (latents / self.scaling_factor) + self.shift_factor
        decoded = self.decoder(scaled)
        return decoded

    def encode(self, image: mx.array) -> mx.array:
        latents = self.encoder(image)
        mean, _ = mx.split(latents, 2, axis=1)
        latent = (mean - self.shift_factor) * self.scaling_factor
        return latent
