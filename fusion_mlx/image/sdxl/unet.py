import logging
import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


def timestep_embedding(
    timesteps: mx.array, dim: int, max_period: int = 10000
) -> mx.array:
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32)
    exponent = exponent / (half - 1)
    freqs = mx.exp(exponent)
    emb = timesteps.reshape(-1, 1).astype(mx.float32) * freqs.reshape(1, -1)
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2 == 1:
        emb = mx.pad(emb, ((0, 0), (0, 1)))
    return emb


class GroupNorm(nn.Module):
    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-5):
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
    def __init__(self, in_ch: int, out_ch: int, temb_channels: int):
        super().__init__()
        self.norm1 = GroupNorm(in_ch)
        self.conv1 = Conv2d(in_ch, out_ch)
        self.time_emb_proj = nn.Linear(temb_channels, out_ch)
        self.norm2 = GroupNorm(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch)
        self.conv_shortcut = (
            Conv2d(in_ch, out_ch, k=1, padding=0) if in_ch != out_ch else None
        )

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        h = _nchw_to_nhwc(x)
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)
        t = nn.silu(temb)
        t = self.time_emb_proj(t)[:, None, None, :]
        h = h + t
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            xh = self.conv_shortcut(_nchw_to_nhwc(x))
            return _nhwc_to_nchw(xh.astype(mx.float32) + h)
        return _nhwc_to_nchw(_nchw_to_nhwc(x).astype(mx.float32) + h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels, k=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
        h = _nchw_to_nhwc(h)
        h = self.conv(h)
        return _nhwc_to_nchw(h)


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


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, cross_dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(cross_dim, inner, bias=False)
        self.to_v = nn.Linear(cross_dim, inner, bias=False)
        self.to_out = [nn.Linear(inner, query_dim), nn.Identity()]

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = self.to_q(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (
            self.to_k(context)
            .reshape(b, -1, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.to_v(context)
            .reshape(b, -1, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        out = scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.heads * self.head_dim)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    # diffusers FeedForward with geglu: net.0 = GEGLU(proj), net.2 = Linear

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.net_0_proj = nn.Linear(dim, inner * 2)
        self.net_2 = nn.Linear(inner, dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.net_0_proj(x)
        proj, gate = mx.split(h, 2, axis=-1)
        h = proj * nn.gelu(gate)
        return self.net_2(h)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, cross_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn1 = CrossAttention(dim, dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.attn2 = CrossAttention(dim, cross_dim, heads, head_dim)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim, dim * 4)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = x + self.attn1(self.norm1(x), self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class Transformer2D(nn.Module):
    def __init__(
        self, channels: int, heads: int, head_dim: int, cross_dim: int, num_layers: int
    ):
        super().__init__()
        self.norm = GroupNorm(channels, eps=1e-6)
        inner = heads * head_dim
        self.proj_in = nn.Linear(channels, inner)
        self.transformer_blocks = [
            TransformerBlock(inner, heads, head_dim, cross_dim)
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner, channels)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        b, c, h, w = x.shape
        residual = x
        x = _nchw_to_nhwc(x)
        x = self.norm(x)
        x = x.reshape(b, h * w, c)
        x = self.proj_in(x)
        for blk in self.transformer_blocks:
            x = blk(x, context)
        x = self.proj_out(x)
        x = x.reshape(b, h, w, c).transpose(0, 3, 1, 2)
        return x + residual


class CrossAttnDownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_ch: int,
        heads: int,
        head_dim: int,
        cross_dim: int,
        num_layers: int,
        num_tf_layers: int,
        add_downsample: bool,
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_ch if i == 0 else out_ch, out_ch, temb_ch)
            for i in range(num_layers)
        ]
        self.attentions = [
            Transformer2D(out_ch, heads, head_dim, cross_dim, num_tf_layers)
            for _ in range(num_layers)
        ]
        self.downsamplers = [Downsample(out_ch)] if add_downsample else None

    def __call__(self, x: mx.array, temb: mx.array, context: mx.array):
        outputs = []
        for res, att in zip(self.resnets, self.attentions):
            x = res(x, temb)
            x = att(x, context)
            outputs.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            outputs.append(x)
        return x, outputs


class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_ch: int,
        num_layers: int,
        add_downsample: bool,
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_ch if i == 0 else out_ch, out_ch, temb_ch)
            for i in range(num_layers)
        ]
        self.downsamplers = [Downsample(out_ch)] if add_downsample else None

    def __call__(self, x: mx.array, temb: mx.array, context: mx.array):
        outputs = []
        for res in self.resnets:
            x = res(x, temb)
            outputs.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            outputs.append(x)
        return x, outputs


class MidCrossAttnBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        temb_ch: int,
        heads: int,
        head_dim: int,
        cross_dim: int,
        num_tf_layers: int,
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock(channels, channels, temb_ch),
            ResnetBlock(channels, channels, temb_ch),
        ]
        self.attentions = [
            Transformer2D(channels, heads, head_dim, cross_dim, num_tf_layers)
        ]

    def __call__(self, x: mx.array, temb: mx.array, context: mx.array) -> mx.array:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        x = self.resnets[1](x, temb)
        return x


class CrossAttnUpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        prev_ch: int,
        skip_ch: int,
        temb_ch: int,
        heads: int,
        head_dim: int,
        cross_dim: int,
        num_layers: int,
        num_tf_layers: int,
        add_upsample: bool,
    ):
        super().__init__()
        self.resnets = []
        self.attentions = []
        # diffusers: res_skip_channels = in_ch if i==num_layers-1 else out_ch;
        # resnet_in = prev_ch if i==0 else out_ch; in = resnet_in + res_skip.
        for i in range(num_layers):
            res_skip = in_ch if i == num_layers - 1 else out_ch
            resnet_in = prev_ch if i == 0 else out_ch
            self.resnets.append(ResnetBlock(resnet_in + res_skip, out_ch, temb_ch))
            self.attentions.append(
                Transformer2D(out_ch, heads, head_dim, cross_dim, num_tf_layers)
            )
        self.upsamplers = [Upsample(out_ch)] if add_upsample else None

    def __call__(self, x: mx.array, skips: list, temb: mx.array, context: mx.array):
        for res, att in zip(self.resnets, self.attentions):
            x = mx.concatenate([x, skips.pop()], axis=1)
            x = res(x, temb)
            x = att(x, context)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        prev_ch: int,
        skip_ch: int,
        temb_ch: int,
        num_layers: int,
        add_upsample: bool,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            res_skip = in_ch if i == num_layers - 1 else out_ch
            resnet_in = prev_ch if i == 0 else out_ch
            self.resnets.append(ResnetBlock(resnet_in + res_skip, out_ch, temb_ch))
        self.upsamplers = [Upsample(out_ch)] if add_upsample else None

    def __call__(self, x: mx.array, skips: list, temb: mx.array, context: mx.array):
        for res in self.resnets:
            x = mx.concatenate([x, skips.pop()], axis=1)
            x = res(x, temb)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class SDXLUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        uc = config.block_out_channels
        ad = config.attention_head_dim
        tl = config.transformer_layers_per_block
        cross = config.cross_attention_dim
        temb_ch = uc[0] * 4
        self.conv_in = Conv2d(config.in_channels, uc[0])
        time_dim = uc[0] * 4
        self.time_embedding = [
            nn.Linear(uc[0], time_dim),
            nn.Linear(time_dim, time_dim),
        ]
        add_dim = config.projection_class_embeddings_input_dim
        self.add_embedding = [
            nn.Linear(add_dim, time_dim),
            nn.Linear(time_dim, time_dim),
        ]
        self.down_blocks = []
        self.down_blocks.append(
            DownBlock(uc[0], uc[0], temb_ch, num_layers=2, add_downsample=True)
        )
        self.down_blocks.append(
            CrossAttnDownBlock(
                uc[0],
                uc[1],
                temb_ch,
                ad[1],
                uc[1] // ad[1],
                cross,
                num_layers=2,
                num_tf_layers=tl[1],
                add_downsample=True,
            )
        )
        self.down_blocks.append(
            CrossAttnDownBlock(
                uc[1],
                uc[2],
                temb_ch,
                ad[2],
                uc[2] // ad[2],
                cross,
                num_layers=2,
                num_tf_layers=tl[2],
                add_downsample=False,
            )
        )
        self.mid_block = MidCrossAttnBlock(
            uc[2], temb_ch, ad[2], uc[2] // ad[2], cross, num_tf_layers=tl[2]
        )
        self.up_blocks = []
        # diffusers reverses block_out_channels -> [1280, 640, 320].
        # up_blocks get num_layers = layers_per_block + 1 = 3 (consume 3 skips).
        # i=0: in_ch=640, out_ch=1280, prev_ch=1280; heads=20, tf=10.
        self.up_blocks.append(
            CrossAttnUpBlock(
                uc[1],
                uc[2],
                uc[2],
                uc[1],
                temb_ch,
                ad[2],
                uc[2] // ad[2],
                cross,
                num_layers=3,
                num_tf_layers=tl[2],
                add_upsample=True,
            )
        )
        # i=1: in_ch=320(reversed[2]), out_ch=640, prev_ch=1280; heads=10, tf=2.
        self.up_blocks.append(
            CrossAttnUpBlock(
                uc[0],
                uc[1],
                uc[2],
                uc[0],
                temb_ch,
                ad[1],
                uc[1] // ad[1],
                cross,
                num_layers=3,
                num_tf_layers=tl[1],
                add_upsample=True,
            )
        )
        # i=2 (final): in_ch=320, out_ch=320, prev_ch=640; UpBlock, num_layers=3.
        self.up_blocks.append(
            UpBlock(
                uc[0], uc[0], uc[1], uc[0], temb_ch, num_layers=3, add_upsample=False
            )
        )
        self.conv_norm_out = GroupNorm(uc[0])
        self.conv_out = Conv2d(uc[0], config.out_channels)

    def _time_embed(self, t: mx.array) -> mx.array:
        # sinusoidal embedding dim = block_out_channels[0] (320), then
        # TimestepEmbedding linear_1: 320 -> time_dim(1280).
        t_emb = timestep_embedding(t, self.config.block_out_channels[0])
        h = self.time_embedding[0](t_emb)
        h = nn.silu(h)
        h = self.time_embedding[1](h)
        return h

    def _add_embed(self, text_embeds: mx.array, time_ids: mx.array) -> mx.array:
        time_emb = timestep_embedding(
            time_ids.flatten(), self.config.addition_time_embed_dim
        )
        time_emb = time_emb.reshape((text_embeds.shape[0], -1))
        add = mx.concatenate([text_embeds, time_emb], axis=-1)
        h = self.add_embedding[0](add)
        h = nn.silu(h)
        h = self.add_embedding[1](h)
        return h

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        text_embeds: mx.array,
        time_ids: mx.array,
    ) -> mx.array:
        t = self._time_embed(timestep)
        aug = self._add_embed(text_embeds, time_ids)
        temb = t + aug
        x = _nchw_to_nhwc(sample)
        x = self.conv_in(x)
        x = _nhwc_to_nchw(x)
        # diffusers seeds res_samples with the conv_in output as the first
        # skip so up_blocks (num_layers=layers_per_block+1) can pop it.
        skips = [x]
        for blk in self.down_blocks:
            x, outs = blk(x, temb, encoder_hidden_states)
            skips.extend(outs)
        x = self.mid_block(x, temb, encoder_hidden_states)
        for blk in self.up_blocks:
            x = blk(x, skips, temb, encoder_hidden_states)
        h = _nchw_to_nhwc(x)
        h = self.conv_norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return _nhwc_to_nchw(h)
