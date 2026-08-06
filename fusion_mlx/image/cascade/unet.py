import logging

import mlx.core as mx
from mlx import nn

from fusion_mlx.image.cascade.common import (
    AttnBlock,
    ResBlock,
    TimestepBlock,
    WuerstchenLayerNorm,
    gen_r_embedding,
)

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


def _pixel_unshuffle_nhwc(x: mx.array, patch: int) -> mx.array:
    # NHWC (b,h,w,c) -> (b, h/patch, w/patch, c*patch^2). Downsamples
    # spatial, expands channels — matches torch.nn.PixelUnshuffle.
    b, h, w, c = x.shape
    x = x.reshape(b, h // patch, patch, w // patch, patch, c)
    x = x.transpose(0, 1, 3, 4, 5, 2)
    x = x.reshape(b, h // patch, w // patch, c * patch * patch)
    return x


def _pixel_shuffle_nhwc(x: mx.array, patch: int) -> mx.array:
    # NHWC (b,h,w,c) -> (b, h*patch, w*patch, c/patch^2). Upsamples
    # spatial, contracts channels — matches torch.nn.PixelShuffle.
    b, h, w, c = x.shape
    x = x.reshape(b, h, w, c // (patch * patch), patch, patch)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, h * patch, w * patch, c // (patch * patch))
    return x


def _resize_axis(x: mx.array, target: int, axis: int) -> mx.array:
    # Bilinear resize (align_corners=True) along a single spatial axis.
    src = x.shape[axis]
    if target == src:
        return x
    # Source coordinate for each output index: align_corners=True mapping.
    if target > 1:
        coord = mx.arange(target, dtype=mx.float32) * (src - 1) / (target - 1)
    else:
        coord = mx.array([0.0], dtype=mx.float32)
    lo = mx.floor(coord).astype(mx.int32)
    hi = mx.minimum(lo + 1, src - 1)
    frac = (coord - lo.astype(mx.float32))[:, None]  # (target, 1)
    lo_g = mx.take(x, lo, axis=axis)
    hi_g = mx.take(x, hi, axis=axis)
    # Broadcast frac against the gathered spatial axis (now at axis `axis`).
    shape = [1] * lo_g.ndim
    shape[axis] = target
    frac = mx.broadcast_to(mx.reshape(frac, shape), lo_g.shape)
    out = lo_g * (1.0 - frac) + hi_g * frac
    return out


def _interp_bilinear(x: mx.array, h: int, w: int) -> mx.array:
    # NHWC bilinear resize of spatial axes (1=h, 2=w) to (h, w).
    x = x.astype(mx.float32)
    x = _resize_axis(x, h, axis=1)
    x = _resize_axis(x, w, axis=2)
    return x


def _upsample_nearest_2x(x: mx.array) -> mx.array:
    b, h, w, c = x.shape
    x = x[:, :, None, :, None, :]
    x = mx.broadcast_to(x, (b, h, 2, w, 2, c))
    x = x.reshape(b, h * 2, w * 2, c)
    return x


class _BilinearScale(nn.Module):
    # Bilinear up (x2) or down (x0.5) on NHWC spatial axes. Used by
    # UpDownBlock2d when switch_level is enabled (matches diffusers
    # nn.Upsample(mode='bilinear', align_corners=True)).

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def __call__(self, x: mx.array) -> mx.array:
        h = max(1, int(round(x.shape[1] * self.factor)))
        w = max(1, int(round(x.shape[2] * self.factor)))
        return _interp_bilinear(x, h, w)


class UpDownBlock2d(nn.Module):
    # diffusers UpDownBlock2d: blocks = [mapping, interp] for down,
    # [interp, mapping] for up. mapping = Conv2d 1x1; interp = bilinear
    # scale (x2 up / x0.5 down) when enabled else Identity. Keys:
    # blocks.0 / blocks.1 (Identity has no params, Conv2d at the
    # non-Identity index). Operates in NHWC.

    def __init__(self, in_ch: int, out_ch: int, mode: str, enabled: bool = True):
        super().__init__()
        mapping = Conv2d(in_ch, out_ch, k=1)
        if enabled:
            interp = _BilinearScale(2.0 if mode == "up" else 0.5)
        else:
            interp = _Identity()
        if mode == "up":
            self.blocks = [interp, mapping]
        else:
            self.blocks = [mapping, interp]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return x


class _PixelUnshuffle(nn.Module):
    def __init__(self, patch: int):
        super().__init__()
        self.patch = patch

    def __call__(self, x: mx.array) -> mx.array:
        x = _nchw_to_nhwc(x)
        return _pixel_unshuffle_nhwc(x, self.patch)


class _PixelShuffle(nn.Module):
    def __init__(self, patch: int):
        super().__init__()
        self.patch = patch

    def __call__(self, x: mx.array) -> mx.array:
        out = _pixel_shuffle_nhwc(x, self.patch)
        return _nhwc_to_nchw(out)


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class ConvTranspose2d(nn.Module):
    # up_upscalers.{i}.1 = ConvTranspose2d(k=2, stride=2, padding=0).
    # weight (out, k, k, in) = (out, 2, 2, in) — MLX OHWI matches PyTorch
    # ConvTranspose2d (out, in, k, k) after transpose (0, 2, 3, 1).

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 2, stride: int = 2, padding: int = 0
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


def _make_block(
    block_type, c, nhead, cond_dim, c_r, conds, c_skip, kernel_size, dropout, self_attn
):
    if block_type == "SDCascadeResBlock":
        return ResBlock(c, c_skip, kernel_size, dropout)
    if block_type == "SDCascadeAttnBlock":
        return AttnBlock(c, cond_dim, nhead, self_attn, dropout)
    if block_type == "SDCascadeTimestepBlock":
        return TimestepBlock(c, c_r, conds)
    raise ValueError(f"Block type {block_type} not supported")


def _block_kind(block: nn.Module) -> str:
    if isinstance(block, ResBlock):
        return "res"
    if isinstance(block, AttnBlock):
        return "attn"
    if isinstance(block, TimestepBlock):
        return "ts"
    raise ValueError(f"Unknown block type: {type(block)}")


class StableCascadeUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        boc = list(cfg.block_out_channels)
        if isinstance(cfg.dropout, float):
            dropout = (cfg.dropout,) * len(boc)
        else:
            dropout = tuple(cfg.dropout)
        if isinstance(cfg.self_attn, bool):
            self_attn = (cfg.self_attn,) * len(boc)
        else:
            self_attn = tuple(cfg.self_attn)
        conds = tuple(cfg.timestep_conditioning_type)
        c_r = cfg.timestep_ratio_embedding_dim
        cond_dim = cfg.conditioning_dim
        patch = cfg.patch_size
        self.patch_size = patch

        if cfg.effnet_in_channels is not None:
            # [Conv, GELU, Conv, LayerNorm(affine=False)] -> keys .0/.2
            self.effnet_mapper = [
                Conv2d(cfg.effnet_in_channels, boc[0] * 4, k=1),
                nn.GELU(),
                Conv2d(boc[0] * 4, boc[0], k=1),
                WuerstchenLayerNorm(boc[0]),
            ]
        if cfg.pixel_mapper_in_channels is not None:
            self.pixels_mapper = [
                Conv2d(cfg.pixel_mapper_in_channels, boc[0] * 4, k=1),
                nn.GELU(),
                Conv2d(boc[0] * 4, boc[0], k=1),
                WuerstchenLayerNorm(boc[0]),
            ]
        self.clip_txt_pooled_mapper = nn.Linear(
            cfg.clip_text_pooled_in_channels, cond_dim * cfg.clip_seq
        )
        self.clip_txt_mapper = None
        if cfg.clip_text_in_channels is not None:
            self.clip_txt_mapper = nn.Linear(cfg.clip_text_in_channels, cond_dim)
        self.clip_img_mapper = None
        if cfg.clip_image_in_channels is not None:
            self.clip_img_mapper = nn.Linear(
                cfg.clip_image_in_channels, cond_dim * cfg.clip_seq
            )
        self.clip_norm = nn.LayerNorm(cond_dim, eps=1e-6)
        self.clip_norm.weight = None
        self.clip_norm.bias = None

        # embedding = [PixelUnshuffle, Conv, LayerNorm(affine=False)] -> key .1
        self.embedding = [
            _PixelUnshuffle(patch),
            Conv2d(cfg.in_channels * (patch * patch), boc[0], k=1),
            WuerstchenLayerNorm(boc[0]),
        ]

        self.down_blocks = []
        self.down_downscalers = []
        self.down_repeat_mappers = []
        for i in range(len(boc)):
            if i > 0:
                # down_downscalers.{i} = [LN(affine-free), scaler].
                # When switch_level is None (decoder): scaler = Conv2d
                # k=2 s=2 -> key down_downscalers.{i}.1.weight.
                # When switch_level is not None (prior): scaler =
                # UpDownBlock2d(mode='down', enabled=switch_level[i-1])
                # -> key down_downscalers.{i}.1.blocks.{0|1}.weight.
                if cfg.switch_level is not None:
                    scaler = UpDownBlock2d(
                        boc[i - 1], boc[i], mode="down", enabled=cfg.switch_level[i - 1]
                    )
                else:
                    scaler = Conv2d(boc[i - 1], boc[i], k=2, stride=2)
                self.down_downscalers.append([WuerstchenLayerNorm(boc[i - 1]), scaler])
            else:
                self.down_downscalers.append(_Identity())
            down_block = []
            for _ in range(cfg.down_num_layers_per_block[i]):
                for bt in cfg.block_types_per_layer[i]:
                    down_block.append(
                        _make_block(
                            bt,
                            boc[i],
                            cfg.num_attention_heads[i],
                            cond_dim,
                            c_r,
                            conds,
                            0,
                            cfg.kernel_size,
                            dropout[i],
                            self_attn[i],
                        )
                    )
            self.down_blocks.append(down_block)
            reps = []
            if cfg.down_blocks_repeat_mappers is not None:
                for _ in range(cfg.down_blocks_repeat_mappers[i] - 1):
                    reps.append(Conv2d(boc[i], boc[i], k=1))
            self.down_repeat_mappers.append(reps)

        self.up_blocks = []
        self.up_upscalers = []
        self.up_repeat_mappers = []
        up_layers = list(cfg.up_num_layers_per_block)
        up_reps = (
            list(cfg.up_blocks_repeat_mappers)
            if cfg.up_blocks_repeat_mappers is not None
            else [1] * len(boc)
        )
        for idx, i in enumerate(reversed(range(len(boc)))):
            if i > 0:
                # up_upscalers.{idx} = [LN(affine-free), scaler].
                # switch_level None (decoder): ConvTranspose2d k=2 s=2 ->
                # key up_upscalers.{idx}.1.weight.
                # switch_level not None (prior): UpDownBlock2d(mode='up',
                # enabled=switch_level[i-1]) -> key
                # up_upscalers.{idx}.1.blocks.1.weight (mapping at idx 1).
                if cfg.switch_level is not None:
                    scaler = UpDownBlock2d(
                        boc[i], boc[i - 1], mode="up", enabled=cfg.switch_level[i - 1]
                    )
                else:
                    scaler = ConvTranspose2d(
                        boc[i], boc[i - 1], k=2, stride=2, padding=0
                    )
                self.up_upscalers.append([WuerstchenLayerNorm(boc[i]), scaler])
            else:
                self.up_upscalers.append(_Identity())
            up_block = []
            for j in range(up_layers[idx]):
                for k_idx, bt in enumerate(cfg.block_types_per_layer[i]):
                    c_skip = boc[i] if i < len(boc) - 1 and j == k_idx == 0 else 0
                    up_block.append(
                        _make_block(
                            bt,
                            boc[i],
                            cfg.num_attention_heads[i],
                            cond_dim,
                            c_r,
                            conds,
                            c_skip,
                            cfg.kernel_size,
                            dropout[i],
                            self_attn[i],
                        )
                    )
            self.up_blocks.append(up_block)
            reps = []
            for _ in range(up_reps[idx] - 1):
                reps.append(Conv2d(boc[i], boc[i], k=1))
            self.up_repeat_mappers.append(reps)

        # clf = [LayerNorm(affine=False), Conv, PixelShuffle] -> key .1
        self.clf = [
            WuerstchenLayerNorm(boc[0]),
            Conv2d(boc[0], cfg.out_channels * (patch * patch), k=1),
            _PixelShuffle(patch),
        ]

    def get_timestep_ratio_embedding(self, timestep_ratio: mx.array) -> mx.array:
        return gen_r_embedding(timestep_ratio, self.cfg.timestep_ratio_embedding_dim)

    def get_clip_embeddings(self, clip_txt_pooled, clip_txt=None, clip_img=None):
        b = clip_txt_pooled.shape[0]
        clip_seq = self.cfg.clip_seq
        cond_dim = self.cfg.conditioning_dim
        if clip_txt_pooled.ndim == 2:
            clip_txt_pooled = clip_txt_pooled[:, None, :]
        pooled = self.clip_txt_pooled_mapper(clip_txt_pooled)
        pooled = pooled.reshape(b, clip_txt_pooled.shape[1] * clip_seq, cond_dim)
        if (
            clip_txt is not None
            and self.clip_txt_mapper is not None
            and clip_img is not None
            and self.clip_img_mapper is not None
        ):
            clip_txt = self.clip_txt_mapper(clip_txt)
            if clip_img.ndim == 2:
                clip_img = clip_img[:, None, :]
            clip_img = self.clip_img_mapper(clip_img)
            clip_img = clip_img.reshape(b, clip_img.shape[1] * clip_seq, cond_dim)
            clip = mx.concatenate([clip_txt, pooled, clip_img], axis=1)
        else:
            clip = pooled
        return self.clip_norm(clip)

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        for i in range(len(self.down_blocks)):
            scaler = self.down_downscalers[i]
            if isinstance(scaler, list):
                for layer in scaler:
                    x = layer(x)
            else:
                x = scaler(x)
            reps = self.down_repeat_mappers[i]
            for r in range(len(reps) + 1):
                for block in self.down_blocks[i]:
                    kind = _block_kind(block)
                    if kind == "res":
                        x = block(x)
                    elif kind == "attn":
                        x = block(x, clip)
                    elif kind == "ts":
                        x = block(x, r_embed)
                if r < len(reps):
                    x = reps[r](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        for i in range(len(self.up_blocks)):
            reps = self.up_repeat_mappers[i]
            for r in range(len(reps) + 1):
                for k_idx, block in enumerate(self.up_blocks[i]):
                    kind = _block_kind(block)
                    if kind == "res":
                        skip = level_outputs[i] if k_idx == 0 and i > 0 else None
                        if skip is not None and (
                            x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]
                        ):
                            x = _interp_bilinear(x, skip.shape[1], skip.shape[2])
                        x = block(x, skip)
                    elif kind == "attn":
                        x = block(x, clip)
                    elif kind == "ts":
                        x = block(x, r_embed)
                if r < len(reps):
                    x = reps[r](x)
            scaler = self.up_upscalers[i]
            if isinstance(scaler, list):
                for layer in scaler:
                    x = layer(x)
            else:
                x = scaler(x)
        return x

    def __call__(
        self,
        sample,
        timestep_ratio,
        clip_text_pooled,
        clip_text=None,
        clip_img=None,
        effnet=None,
        pixels=None,
        sca=None,
        crp=None,
    ):
        if pixels is None and hasattr(self, "pixels_mapper"):
            pixels = mx.zeros((sample.shape[0], 3, 8, 8), dtype=mx.float32)
        r_embed = self.get_timestep_ratio_embedding(timestep_ratio)
        for c in self.cfg.timestep_conditioning_type:
            if c == "sca":
                cond = sca
            elif c == "crp":
                cond = crp
            else:
                cond = None
            t_cond = cond if cond is not None else mx.zeros_like(timestep_ratio)
            r_embed = mx.concatenate(
                [r_embed, self.get_timestep_ratio_embedding(t_cond)], axis=1
            )
        clip = self.get_clip_embeddings(clip_text_pooled, clip_text, clip_img)
        x = self.embedding[0](sample)
        x = self.embedding[1](x)
        x = self.embedding[2](x)
        if hasattr(self, "effnet_mapper") and effnet is not None:
            # effnet conditioning arrives as NCHW (prior output convention);
            # convert to NHWC for the conv mapper.
            mapped = _nchw_to_nhwc(effnet)
            for layer in self.effnet_mapper:
                mapped = layer(mapped)
            mapped = _interp_bilinear(mapped, x.shape[1], x.shape[2])
            x = x + mapped
        if hasattr(self, "pixels_mapper"):
            # pixels conditioning arrives as NCHW; convert to NHWC.
            mapped = _nchw_to_nhwc(pixels)
            for layer in self.pixels_mapper:
                mapped = layer(mapped)
            mapped = _interp_bilinear(mapped, x.shape[1], x.shape[2])
            x = x + mapped
        level_outputs = self._down_encode(x, r_embed, clip)
        x = self._up_decode(level_outputs, r_embed, clip)
        out = self.clf[0](x)
        out = self.clf[1](out)
        out = self.clf[2](out)
        return out
