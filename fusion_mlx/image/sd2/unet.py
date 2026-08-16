import logging

import mlx.core as mx
from mlx import nn

# SD2.1 UNet is a hybrid of SD1.5 structure and SDXL attention:
#   - use_linear_projection=True -> reuse the SDXL Transformer2D (Linear
#     proj_in/proj_out), NOT SD1.5's Transformer2DConv (Conv 1x1).
#   - cross_attention_dim=1024 (ViT-H), vs SD1.5 768.
#   - attention_head_dim is a per-block LIST [5,10,20,20] = head_DIM;
#     heads = ch // head_dim. SD1.5 used a scalar head-count (8).
#   - NO add_embedding (like SD1.5, unlike SDXL) -> temb = time embed only.
#   - time_embedding.linear_1/linear_2 (diffusers default), mapped to
#     time_embedding.0/1 by the shared SDXL weight remap.
# Block topology (down: CrossAttn,CrossAttn,CrossAttn,Down; up: Up,CrossAttn,
# CrossAttn,CrossAttn) matches SD1.5 exactly; only the transformer primitive
# and head config differ, so we reuse SD1.5's block classes' wiring but swap
# Transformer2DConv -> SDXL Transformer2D.
from fusion_mlx.image.sdxl.unet import (
    Conv2d,
    Downsample,
    GroupNorm,
    ResnetBlock,
    Transformer2D,
    Upsample,
    _nchw_to_nhwc,
    _nhwc_to_nchw,
    timestep_embedding,
)

logger = logging.getLogger(__name__)


class SD2CrossAttnDownBlock(nn.Module):
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


class SD2DownBlock(nn.Module):
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


class SD2MidCrossAttnBlock(nn.Module):
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


class SD2CrossAttnUpBlock(nn.Module):
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


class SD2UpBlock(nn.Module):
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


class SD2UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        uc = config.block_out_channels  # (320, 640, 1280, 1280)
        ad = config.attention_head_dim  # [5, 10, 20, 20] = head_DIM per block
        tl = config.transformer_layers_per_block  # 1
        cross = config.cross_attention_dim  # 1024
        temb_ch = uc[0] * 4  # 1280

        self.conv_in = Conv2d(config.in_channels, uc[0])
        time_dim = uc[0] * 4
        self.time_embedding = [
            nn.Linear(uc[0], time_dim),
            nn.Linear(time_dim, time_dim),
        ]

        # down_blocks: [CrossAttn x3, Down x1]. attention_head_dim ad[i] is
        # the NUMBER OF HEADS (diffusers convention, verified vs SD2.1 unet:
        # block0 heads=5 head_dim=64). head_dim = uc[i] // ad[i] = 64.
        self.down_blocks = []
        self.down_blocks.append(
            SD2CrossAttnDownBlock(
                uc[0],
                uc[0],
                temb_ch,
                ad[0],
                uc[0] // ad[0],
                cross,
                num_layers=config.layers_per_block,
                num_tf_layers=tl,
                add_downsample=True,
            )
        )
        self.down_blocks.append(
            SD2CrossAttnDownBlock(
                uc[0],
                uc[1],
                temb_ch,
                ad[1],
                uc[1] // ad[1],
                cross,
                num_layers=config.layers_per_block,
                num_tf_layers=tl,
                add_downsample=True,
            )
        )
        self.down_blocks.append(
            SD2CrossAttnDownBlock(
                uc[1],
                uc[2],
                temb_ch,
                ad[2],
                uc[2] // ad[2],
                cross,
                num_layers=config.layers_per_block,
                num_tf_layers=tl,
                add_downsample=True,
            )
        )
        self.down_blocks.append(
            SD2DownBlock(
                uc[2],
                uc[3],
                temb_ch,
                num_layers=config.layers_per_block,
                add_downsample=False,
            )
        )

        self.mid_block = SD2MidCrossAttnBlock(
            uc[3], temb_ch, ad[3], uc[3] // ad[3], cross, num_tf_layers=tl
        )

        # up_blocks mirror SD1.5: reversed(uc)=(1280,1280,640,320). diffusers
        # builds up blocks with num_attention_heads=reversed(ad) (NOT forward
        # ad[i]) while attention_head_dim fallback stays forward. Since SD2.1
        # attention_head_dim=[5,10,20,20] IS the head count (head_dim=64
        # uniformly), the up-block heads = reversed(ad)=[20,20,10,5].
        # head_dim = out_ch // heads = 64 everywhere. A wrong head_dim yields a
        # wrong attention scale (1/sqrt(head_dim)) -> std mismatch.
        r = list(reversed(uc))  # (1280, 1280, 640, 320)
        rad = list(reversed(ad))  # (20, 20, 10, 5) = up-block head counts
        self.up_blocks = []
        # up0: UpBlock (no attn), out=r[0]=1280, prev=r[0], num_layers=3.
        self.up_blocks.append(
            SD2UpBlock(
                r[1],
                r[0],
                r[0],
                r[1],
                temb_ch,
                num_layers=config.layers_per_block + 1,
                add_upsample=True,
            )
        )
        # up1: CrossAttn, out=r[1]=1280, heads=rad[1]=20 (head_dim=64).
        self.up_blocks.append(
            SD2CrossAttnUpBlock(
                r[2],
                r[1],
                r[0],
                r[2],
                temb_ch,
                rad[1],
                r[1] // rad[1],
                cross,
                num_layers=config.layers_per_block + 1,
                num_tf_layers=tl,
                add_upsample=True,
            )
        )
        # up2: CrossAttn, out=r[2]=640, heads=rad[2]=10 (head_dim=64).
        self.up_blocks.append(
            SD2CrossAttnUpBlock(
                r[3],
                r[2],
                r[1],
                r[3],
                temb_ch,
                rad[2],
                r[2] // rad[2],
                cross,
                num_layers=config.layers_per_block + 1,
                num_tf_layers=tl,
                add_upsample=True,
            )
        )
        # up3: CrossAttn, out=r[3]=320, heads=rad[3]=5 (head_dim=64).
        self.up_blocks.append(
            SD2CrossAttnUpBlock(
                r[3],
                r[3],
                r[2],
                r[3],
                temb_ch,
                rad[3],
                r[3] // rad[3],
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
        # SD2.1 has NO add_embedding (no text_embeds/time_ids). temb = time
        # embed only, identical to SD1.5 forward path.
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
