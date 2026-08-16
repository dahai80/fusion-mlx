import logging

import mlx.core as mx
from mlx import nn

# SD1.5 differs from SDXL in two ways beyond block counts:
#   1. use_linear_projection=False -> Transformer2D proj_in/proj_out are
#      Conv2d 1x1 (NCHW), not Linear (sequence). SDXL uses Linear.
#   2. NO add_embedding (no text_embeds/time_ids augmentation).
# We reuse the SDXL leaf primitives (GroupNorm/Conv2d/ResnetBlock/Downsample/
# Upsample/CrossAttention/FeedForward/TransformerBlock) but provide an SD1.5
# Transformer2D + block classes that wire the Conv2d 1x1 projections, since
# the SDXL Transformer2D hardcodes nn.Linear proj_in/proj_out.
from fusion_mlx.image.sdxl.unet import (
    Conv2d,
    Downsample,
    GroupNorm,
    ResnetBlock,
    TransformerBlock,
    Upsample,
    _nchw_to_nhwc,
    _nhwc_to_nchw,
    timestep_embedding,
)

logger = logging.getLogger(__name__)


class Transformer2DConv(nn.Module):
    # SD1.5 Transformer2D with use_linear_projection=False: proj_in/proj_out
    # are Conv2d 1x1 (weights OIHW -> OHWI). Flow (diffusers
    # _operate_on_continuous_inputs / _get_output_for_continuous_inputs):
    #   norm(NCHW) -> proj_in Conv1x1 (NCHW->inner) -> permute NHWC -> seq
    #   blocks (seq) -> reshape NHWC -> permute NCHW -> proj_out Conv1x1 -> +res

    def __init__(
        self, channels: int, heads: int, head_dim: int, cross_dim: int, num_layers: int
    ):
        super().__init__()
        self.norm = GroupNorm(channels, eps=1e-6)
        inner = heads * head_dim
        self.proj_in = Conv2d(channels, inner, k=1, padding=0)
        self.transformer_blocks = [
            TransformerBlock(inner, heads, head_dim, cross_dim)
            for _ in range(num_layers)
        ]
        self.proj_out = Conv2d(inner, channels, k=1, padding=0)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        b, c, h, w = x.shape
        residual = x
        # GroupNorm operates NHWC; proj_in is Conv2d (OHWI weight, NHWC input).
        x = _nchw_to_nhwc(x)
        x = self.norm(x)
        x = self.proj_in(x)  # Conv1x1 NHWC -> NHWC (inner channels)
        x = x.reshape(b, h * w, -1)  # -> seq (b, h*w, inner)
        for blk in self.transformer_blocks:
            x = blk(x, context)
        x = x.reshape(b, h, w, -1)  # seq -> NHWC
        x = self.proj_out(x)  # Conv1x1 NHWC -> NHWC (out channels)
        x = _nhwc_to_nchw(x)
        return x + residual


class SD15CrossAttnDownBlock(nn.Module):
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
            Transformer2DConv(out_ch, heads, head_dim, cross_dim, num_tf_layers)
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


class SD15DownBlock(nn.Module):
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


class SD15MidCrossAttnBlock(nn.Module):
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
            Transformer2DConv(channels, heads, head_dim, cross_dim, num_tf_layers)
        ]

    def __call__(self, x: mx.array, temb: mx.array, context: mx.array) -> mx.array:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        x = self.resnets[1](x, temb)
        return x


class SD15CrossAttnUpBlock(nn.Module):
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
        for i in range(num_layers):
            res_skip = in_ch if i == num_layers - 1 else out_ch
            resnet_in = prev_ch if i == 0 else out_ch
            self.resnets.append(ResnetBlock(resnet_in + res_skip, out_ch, temb_ch))
            self.attentions.append(
                Transformer2DConv(out_ch, heads, head_dim, cross_dim, num_tf_layers)
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


class SD15UpBlock(nn.Module):
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


class SD15UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        uc = config.block_out_channels  # (320, 640, 1280, 1280)
        head_dim = config.attention_head_dim  # 8 -> heads = ch // 8
        tl = config.transformer_layers_per_block  # 1
        cross = config.cross_attention_dim  # 768
        temb_ch = uc[0] * 4  # 1280

        self.conv_in = Conv2d(config.in_channels, uc[0])
        time_dim = uc[0] * 4
        self.time_embedding = [
            nn.Linear(uc[0], time_dim),
            nn.Linear(time_dim, time_dim),
        ]

        # down_blocks: [CrossAttn x3, Down x1] (diffusers down_block_types).
        self.down_blocks = []
        self.down_blocks.append(
            SD15CrossAttnDownBlock(
                uc[0],
                uc[0],
                temb_ch,
                uc[0] // head_dim,
                head_dim,
                cross,
                num_layers=config.layers_per_block,
                num_tf_layers=tl,
                add_downsample=True,
            )
        )
        self.down_blocks.append(
            SD15CrossAttnDownBlock(
                uc[0],
                uc[1],
                temb_ch,
                uc[1] // head_dim,
                head_dim,
                cross,
                num_layers=config.layers_per_block,
                num_tf_layers=tl,
                add_downsample=True,
            )
        )
        self.down_blocks.append(
            SD15CrossAttnDownBlock(
                uc[1],
                uc[2],
                temb_ch,
                uc[2] // head_dim,
                head_dim,
                cross,
                num_layers=config.layers_per_block,
                num_tf_layers=tl,
                add_downsample=True,
            )
        )
        self.down_blocks.append(
            SD15DownBlock(
                uc[2],
                uc[3],
                temb_ch,
                num_layers=config.layers_per_block,
                add_downsample=False,
            )
        )

        self.mid_block = SD15MidCrossAttnBlock(
            uc[3], temb_ch, uc[3] // head_dim, head_dim, cross, num_tf_layers=tl
        )

        # up_blocks: reversed(uc) = (1280, 1280, 640, 320).
        r = list(reversed(uc))
        self.up_blocks = []
        self.up_blocks.append(
            SD15UpBlock(
                r[1],
                r[0],
                r[0],
                r[1],
                temb_ch,
                num_layers=config.layers_per_block + 1,
                add_upsample=True,
            )
        )
        self.up_blocks.append(
            SD15CrossAttnUpBlock(
                r[2],
                r[1],
                r[0],
                r[2],
                temb_ch,
                r[1] // head_dim,
                head_dim,
                cross,
                num_layers=config.layers_per_block + 1,
                num_tf_layers=tl,
                add_upsample=True,
            )
        )
        self.up_blocks.append(
            SD15CrossAttnUpBlock(
                r[3],
                r[2],
                r[1],
                r[3],
                temb_ch,
                r[2] // head_dim,
                head_dim,
                cross,
                num_layers=config.layers_per_block + 1,
                num_tf_layers=tl,
                add_upsample=True,
            )
        )
        self.up_blocks.append(
            SD15CrossAttnUpBlock(
                r[3],
                r[3],
                r[2],
                r[3],
                temb_ch,
                r[3] // head_dim,
                head_dim,
                cross,
                num_layers=config.layers_per_block + 1,
                num_tf_layers=tl,
                add_upsample=False,
            )
        )
        self.conv_norm_out = GroupNorm(uc[0])
        self.conv_out = Conv2d(uc[0], config.out_channels)

    def _time_embed(self, t: mx.array) -> mx.array:
        t_emb = timestep_embedding(t, self.config.block_out_channels[0])
        h = self.time_embedding[0](t_emb)
        h = nn.silu(h)
        h = self.time_embedding[1](h)
        return h

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        # SD1.5 has NO add_embedding (no text_embeds/time_ids). temb = time embed
        # only, unlike SDXL which adds aug = _add_embed(text_embeds, time_ids).
        temb = self._time_embed(timestep)
        x = _nchw_to_nhwc(sample)
        x = self.conv_in(x)
        x = _nhwc_to_nchw(x)
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
