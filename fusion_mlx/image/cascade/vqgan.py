import logging

import mlx.core as mx
from mlx import nn

logger = logging.getLogger(__name__)


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


class Conv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        scale = 1.0 / (in_ch * k * k) ** 0.5
        self.weight = mx.random.normal((out_ch, k, k, in_ch)) * scale
        self.bias = mx.zeros((out_ch,)) if bias else None
        self.stride = stride
        self.padding = padding
        self._has_bias = bias

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
        if self._has_bias and self.bias is not None:
            y = y + self.bias
        return y


class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, k: int = 4, stride: int = 2, padding: int = 1
    ):
        super().__init__()
        scale = 1.0 / (in_ch * k * k) ** 0.5
        self.weight = mx.random.normal((out_ch, k, k, in_ch)) * scale
        self.bias = mx.zeros((out_ch,))
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose2d(
            x, self.weight, stride=self.stride, padding=self.padding
        )
        return y + self.bias


class BatchNorm2d(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((ch,))
        self.bias = mx.zeros((ch,))
        self.running_mean = mx.zeros((ch,))
        self.running_var = mx.ones((ch,))
        self.num_batches_tracked = mx.zeros((1,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        ch = x.shape[-1]
        mean = self.running_mean.reshape(1, 1, 1, ch)
        var = self.running_var.reshape(1, 1, 1, ch)
        w = self.weight.reshape(1, 1, 1, ch)
        b = self.bias.reshape(1, 1, 1, ch)
        return (x - mean) * mx.rsqrt(var + self.eps) * w + b


def _pixel_unshuffle_nhwc(x: mx.array, factor: int) -> mx.array:
    # NHWC (b,h,w,c) -> (b, h/factor, w/factor, c*factor^2). Downsamples
    # spatial, expands channels — matches torch.nn.PixelUnshuffle.
    b, h, w, c = x.shape
    x = x.reshape(b, h // factor, factor, w // factor, factor, c)
    x = x.transpose(0, 1, 3, 4, 5, 2)
    x = x.reshape(b, h // factor, w // factor, c * factor * factor)
    return x


def _pixel_shuffle_nhwc(x: mx.array, factor: int) -> mx.array:
    # NHWC (b,h,w,c) -> (b, h*factor, w*factor, c/factor^2). Upsamples
    # spatial, contracts channels — matches torch.nn.PixelShuffle.
    b, h, w, c = x.shape
    x = x.reshape(b, h, w, c // (factor * factor), factor, factor)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, h * factor, w * factor, c // (factor * factor))
    return x


class _ReplicationPad2d(nn.Module):
    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def __call__(self, x: mx.array) -> mx.array:
        p = self.pad
        return mx.pad(
            x,
            (
                (0, 0),
                (p, p),
                (p, p),
                (0, 0),
            ),
            mode="edge",
        )


class DepthwiseConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        padding: int = 0,
        replication: bool = False,
    ):
        super().__init__()
        scale = 1.0 / (in_ch * k * k) ** 0.5
        self.weight = mx.random.normal((out_ch, k, k, 1)) * scale
        self.bias = mx.zeros((out_ch,))
        self.padding = padding
        self.replication = replication

    def __call__(self, x: mx.array) -> mx.array:
        if self.replication and self.padding > 0:
            p = self.padding
            x = mx.pad(
                x,
                (
                    (0, 0),
                    (p, p),
                    (p, p),
                    (0, 0),
                ),
                mode="edge",
            )
        elif self.padding > 0:
            x = mx.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )
        # Depthwise: weight (out_ch, k, k, 1) with groups=in_ch. mlx conv2d
        # expects (out_ch, kh, kw, in_ch/groups) = (in_ch, k, k, 1) for
        # depthwise (out_ch == in_ch == groups).
        y = mx.conv2d(x, self.weight, stride=1, groups=x.shape[-1])
        return y + self.bias


class _AffineFreeLayerNorm(nn.Module):
    # nn.LayerNorm(elementwise_affine=False, eps=1e-6) operating on the
    # last axis (channel) of an NHWC tensor.

    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.ch = ch
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) / mx.sqrt(var + self.eps)


class MixingResidualBlock(nn.Module):
    # depthwise = Sequential(ReplicationPad2d(1), Conv2d) -> key depthwise.1
    # channelwise = Sequential(Linear, GELU, Linear) -> keys channelwise.0/2
    # norm1/norm2 affine-free -> NO weight/bias keys
    # gammas shape (6,)

    def __init__(self, in_ch: int, embed_dim: int):
        super().__init__()
        self.norm1 = _AffineFreeLayerNorm(in_ch)
        self.depthwise = [
            _ReplicationPad2d(1),
            DepthwiseConv2d(in_ch, in_ch, k=3, replication=True),
        ]
        self.norm2 = _AffineFreeLayerNorm(in_ch)
        self.channelwise = [
            nn.Linear(in_ch, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_ch),
        ]
        self.gammas = mx.zeros((6,))

    def __call__(self, x: mx.array) -> mx.array:
        # Operates in NHWC: norm (last axis), depthwise conv, channelwise
        # Linear all consume NHWC directly. gammas are scalar modulations.
        mods = self.gammas
        x_temp = self.norm1(x) * (1 + mods[0]) + mods[1]
        dw = self.depthwise[0](x_temp)
        dw = self.depthwise[1](dw)
        x = x + dw * mods[2]
        x_temp = self.norm2(x) * (1 + mods[3]) + mods[4]
        cw = self.channelwise[0](x_temp)
        cw = self.channelwise[1](cw)
        cw = self.channelwise[2](cw)
        x = x + cw * mods[5]
        return x


class _NormBlock(nn.Module):
    # down_blocks.{last} = Sequential(Conv2d(bias=False), BatchNorm2d) -> keys .0 / .1.
    # Store as a plain 2-element list so the params flatten directly under
    # down_blocks.{last}.0 / .1 (a wrapper module would add an extra prefix).

    def __init__(self, in_ch: int, latent_ch: int):
        super().__init__()
        self.layers = [
            Conv2d(in_ch, latent_ch, k=1, bias=False),
            BatchNorm2d(latent_ch),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class PaellaVQModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_down_scale_factor: int = 2,
        levels: int = 2,
        bottleneck_blocks: int = 12,
        embed_dim: int = 384,
        latent_channels: int = 4,
        scale_factor: float = 0.3764,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.up_down_scale_factor = up_down_scale_factor
        c_levels = [embed_dim // (2**i) for i in reversed(range(levels))]
        # in_block = [PixelUnshuffle, Conv2d] -> key in_block.1
        self.in_block = [
            up_down_scale_factor,
            Conv2d(in_channels * up_down_scale_factor**2, c_levels[0], k=1),
        ]
        # down_blocks = flat list: [ResBlock, (Conv), ResBlock, ..., NormBlock]
        self.down_blocks = []
        for i in range(levels):
            if i > 0:
                self.down_blocks.append(
                    Conv2d(c_levels[i - 1], c_levels[i], k=4, stride=2, padding=1)
                )
            self.down_blocks.append(MixingResidualBlock(c_levels[i], c_levels[i] * 4))
        self.down_blocks.append(
            [
                Conv2d(c_levels[-1], latent_ch := latent_channels, k=1, bias=False),
                BatchNorm2d(latent_ch),
            ]
        )
        # up_blocks = flat list; each element matches diffusers nn.Sequential
        # indexing. up_blocks.0 = Sequential(Conv2d) -> key up_blocks.0.0.
        self.up_blocks = [[Conv2d(latent_channels, c_levels[-1], k=1)]]
        for i in range(levels):
            n_blocks = bottleneck_blocks if i == 0 else 1
            for _ in range(n_blocks):
                ch = c_levels[levels - 1 - i]
                self.up_blocks.append(MixingResidualBlock(ch, ch * 4))
            if i < levels - 1:
                self.up_blocks.append(
                    ConvTranspose2d(
                        c_levels[levels - 1 - i],
                        c_levels[levels - 2 - i],
                        k=4,
                        stride=2,
                        padding=1,
                    )
                )
        # out_block = [Conv2d, PixelShuffle] -> key out_block.0
        self.out_block = [
            Conv2d(c_levels[0], out_channels * up_down_scale_factor**2, k=1),
            up_down_scale_factor,
        ]
        logger.debug(
            "PaellaVQModel built: levels=%d embed_dim=%d down=%d up=%d",
            levels,
            embed_dim,
            len(self.down_blocks),
            len(self.up_blocks),
        )

    def decode(self, h: mx.array, force_not_quantize: bool = True) -> mx.array:
        x = _nchw_to_nhwc(h)
        for block in self.up_blocks:
            if isinstance(block, list):
                for layer in block:
                    x = layer(x)
            else:
                x = block(x)
        x = self.out_block[0](x)
        x = _pixel_shuffle_nhwc(x, self.out_block[1])
        x = _nhwc_to_nchw(x)
        return x

    def __call__(self, h: mx.array) -> mx.array:
        return self.decode(h)
