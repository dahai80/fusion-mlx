# SPDX-License-Identifier: Apache-2.0
# Cosmos continuous VAE (NVIDIA "Factorized" tokenizer) for latent video
# encode/decode. Mirrors the diffusers AutoencoderKLCosmos checkpoint tree
# exactly (conv_s/conv_t factored convs, to_q/to_k/to_v/to_out.0 attention,
# norm.norm.* groupnorms) so from_pretrained loads every key with only a
# prefix swap. See fusion-mlx issue #461.
#
# Output is in [-1, 1] (NOT [0,1]) — the diffusers VAE does not clip; the
# caller normalizes. Latent dim 16, 8x spatial + 8x temporal compression
# (4x from patch_size=4 Haar patcher + 2x from one hybrid up/downsample).

import glob
import logging
import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

COSMOS_VAE_CONFIG = {
    "in_channels": 3,
    "latent_channels": 16,
    "decode_block_out_channels": [256, 512, 512, 512],
    "encoder_block_out_channels": [128, 256, 512, 512],
    "num_layers": 2,
    "patch_size": 4,
    "patch_type": "haar",
    "resolution": 1024,
    "spatial_compression_ratio": 8,
    "temporal_compression_ratio": 8,
    "attention_resolutions": [32],
}


def _vae_checkpoint_enabled():
    return os.getenv("FUSION_COSMOS_VAE_CHECKPOINT", "1").strip() not in (
        "0",
        "",
        "false",
        "False",
    )


def _silu(x):
    return x * mx.sigmoid(x)


def _replication_pad_time(x, time_pad=2):
    x_prev = mx.repeat(x[:, :, :1, ...], time_pad, axis=2)
    return mx.concatenate([x_prev, x], axis=2)


def _repeat_interleave(x, repeats, axis):
    x_e = mx.expand_dims(x, axis + 1)
    tiles = [1] * x_e.ndim
    tiles[axis + 1] = repeats
    x_e = mx.tile(x_e, tiles)
    new_shape = list(x.shape)
    new_shape[axis] = new_shape[axis] * repeats
    return x_e.reshape(new_shape)


class _CausalNorm(nn.Module):
    def __init__(self, channels, num_groups=1):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups, channels, pytorch_compatible=True)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(B * T, H, W, C)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, C).transpose(0, 4, 1, 2, 3)
        return x


class _SpatialConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight = mx.zeros((out_ch, in_ch, 1, 3, 3), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def __call__(self, x):
        w = self.weight.squeeze(2).transpose(0, 2, 3, 1)
        B, C, T, H, W = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(B * T, H, W, C)
        y = mx.conv2d(x, w, stride=1, padding=1) + self.bias
        y = y.reshape(B, T, H, W, -1).transpose(0, 4, 1, 2, 3)
        return y


class _TemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_pad=2):
        super().__init__()
        self.time_pad = time_pad
        self.weight = mx.zeros((out_ch, in_ch, 3, 1, 1), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def __call__(self, x):
        x = _replication_pad_time(x, self.time_pad)
        B, C, T1, H, W = x.shape
        kt = self.weight.shape[2]
        ot = T1 - kt + 1
        windows = []
        for ti in range(ot):
            windows.append(x[:, :, ti : ti + kt])
        w3 = mx.stack(windows, axis=2)
        w3 = w3.transpose(0, 1, 3, 2, 4, 5)
        w3 = w3.reshape(B, C * kt, ot * H * W)
        w_t = self.weight.reshape(self.weight.shape[0], C * kt)
        y = w_t @ w3
        y = y.reshape(B, self.weight.shape[0], ot, H, W) + self.bias.reshape(
            1, -1, 1, 1, 1
        )
        return y


class _PointwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight = mx.zeros((out_ch, in_ch, 1, 1, 1), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def __call__(self, x):
        w = self.weight.reshape(self.weight.shape[0], self.weight.shape[1])
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T * H * W)
        y = w @ x
        y = y.reshape(B, w.shape[0], T, H, W) + self.bias.reshape(1, -1, 1, 1, 1)
        return y


class FactoredConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_s = _SpatialConv(in_ch, out_ch)
        self.conv_t = _TemporalConv(out_ch, out_ch, time_pad=2)

    def __call__(self, x):
        return self.conv_t(self.conv_s(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=1):
        super().__init__()
        self.norm1 = _CausalNorm(in_ch, num_groups)
        self.conv1 = FactoredConv(in_ch, out_ch)
        self.norm2 = _CausalNorm(out_ch, num_groups)
        self.conv2 = FactoredConv(out_ch, out_ch)
        if in_ch != out_ch:
            self.conv_shortcut = _PointwiseConv(in_ch, out_ch)
        else:
            self.conv_shortcut = None

    def __call__(self, x):
        residual = x
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(x)
        h = self.norm1(x)
        h = _silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _silu(h)
        h = self.conv2(h)
        return h + residual


def _spatial_attn(q, k, v):
    B, C, T, H, W = q.shape
    q = q.transpose(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
    k = k.transpose(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
    v = v.transpose(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
    scale = C ** (-0.5)
    w_ = (q @ k.transpose(0, 2, 1)) * scale
    w_ = mx.softmax(w_, axis=-1)
    out = w_ @ v
    out = out.reshape(B, T, H, W, C).transpose(0, 4, 1, 2, 3)
    return out


def _temporal_attn(q, k, v):
    B, C, T, H, W = q.shape
    q = q.transpose(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
    k = k.transpose(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
    v = v.transpose(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
    scale = C ** (-0.5)
    w_ = (q @ k.transpose(0, 2, 1)) * scale
    mask = mx.tril(mx.ones((T, T), dtype=w_.dtype))
    w_ = mx.where(mask == 0, mx.full(w_.shape, -1e9), w_)
    w_ = mx.softmax(w_, axis=-1)
    out = w_ @ v
    out = out.reshape(B, H, W, T, C).transpose(0, 4, 3, 1, 2)
    return out


class AttnBlock(nn.Module):
    def __init__(self, channels, num_groups=1):
        super().__init__()
        self.norm = _CausalNorm(channels, num_groups)
        self.to_q = _PointwiseConv(channels, channels)
        self.to_k = _PointwiseConv(channels, channels)
        self.to_v = _PointwiseConv(channels, channels)
        self.to_out = [_PointwiseConv(channels, channels)]

    def __call__(self, x):
        residual = x
        h = self.norm(x)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        h = _spatial_attn(q, k, v)
        h = self.to_out[0](h)
        return h + residual


class TemporalAttnBlock(nn.Module):
    def __init__(self, channels, num_groups=1):
        super().__init__()
        self.norm = _CausalNorm(channels, num_groups)
        self.to_q = _PointwiseConv(channels, channels)
        self.to_k = _PointwiseConv(channels, channels)
        self.to_v = _PointwiseConv(channels, channels)
        self.to_out = [_PointwiseConv(channels, channels)]

    def __call__(self, x):
        residual = x
        h = self.norm(x)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        h = _temporal_attn(q, k, v)
        h = self.to_out[0](h)
        return h + residual


class _SpatialConvStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight = mx.zeros((out_ch, in_ch, 1, 3, 3), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def __call__(self, x):
        w = self.weight.squeeze(2).transpose(0, 2, 3, 1)
        B, C, T, H, W = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(B * T, H, W, C)
        y = mx.conv2d(x, w, stride=(2, 2), padding=0) + self.bias
        y = y.reshape(B, T, y.shape[1], y.shape[2], -1).transpose(0, 4, 1, 2, 3)
        return y


class _TemporalConvStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.time_pad = 1
        self.weight = mx.zeros((out_ch, in_ch, 3, 1, 1), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def __call__(self, x):
        x = _replication_pad_time(x, self.time_pad)
        B, C, T1, H, W = x.shape
        kt = self.weight.shape[2]
        st = 2
        ot = (T1 - kt) // st + 1
        windows = []
        for ti in range(ot):
            windows.append(x[:, :, ti * st : ti * st + kt])
        w3 = mx.stack(windows, axis=2)
        w3 = w3.transpose(0, 1, 3, 2, 4, 5)
        w3 = w3.reshape(B, C * kt, ot * H * W)
        w_t = self.weight.reshape(self.weight.shape[0], C * kt)
        y = w_t @ w3
        y = y.reshape(B, self.weight.shape[0], ot, H, W) + self.bias.reshape(
            1, -1, 1, 1, 1
        )
        return y


def _avg_pool3d(x, kernel, stride):
    kt, kh, kw = kernel
    st, sh, sw = stride
    B, C, T, H, W = x.shape
    ot = (T - kt) // st + 1
    oh = (H - kh) // sh + 1
    ow = (W - kw) // sw + 1
    out = mx.zeros((B, C, ot, oh, ow), dtype=x.dtype)
    for ti in range(ot):
        for hi in range(oh):
            for wi in range(ow):
                slab = x[
                    :,
                    :,
                    ti * st : ti * st + kt,
                    hi * sh : hi * sh + kh,
                    wi * sw : wi * sw + kw,
                ]
                out[:, :, ti, hi, wi] = slab.mean(axis=(2, 3, 4))
    return out


class HybridUpsample(nn.Module):
    def __init__(self, channels, spatial_up=True, temporal_up=True):
        super().__init__()
        self.spatial_up = spatial_up
        self.temporal_up = temporal_up
        if not spatial_up and not temporal_up:
            return
        if temporal_up:
            self.conv1 = _TemporalConv(channels, channels, time_pad=2)
        if spatial_up:
            self.conv2 = _SpatialConv(channels, channels)
        self.conv3 = _PointwiseConv(channels, channels)

    def __call__(self, x):
        if not self.spatial_up and not self.temporal_up:
            return x
        if self.temporal_up:
            B, C, T, H, W = x.shape
            time_factor = 2 if T > 1 else 1
            if time_factor > 1:
                x = _repeat_interleave(x, 2, axis=2)
                x = x[:, :, 1:, ...]
            x = self.conv1(x) + x
        if self.spatial_up:
            x = _repeat_interleave(x, 2, axis=3)
            x = _repeat_interleave(x, 2, axis=4)
            x = self.conv2(x) + x
        x = self.conv3(x)
        return x


class HybridDownsample(nn.Module):
    def __init__(self, channels, spatial_down=True, temporal_down=True):
        super().__init__()
        self.spatial_down = spatial_down
        self.temporal_down = temporal_down
        if not spatial_down and not temporal_down:
            return
        if spatial_down:
            self.conv1 = _SpatialConvStride2(channels, channels)
        if temporal_down:
            self.conv2 = _TemporalConvStride2(channels, channels)
        self.conv3 = _PointwiseConv(channels, channels)

    def __call__(self, x):
        if not self.spatial_down and not self.temporal_down:
            return x
        if self.spatial_down:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1), (0, 1)])
            x1 = self.conv1(x)
            x2 = _avg_pool3d(x, kernel=(1, 2, 2), stride=(1, 2, 2))
            x = x1 + x2
        if self.temporal_down:
            x = _replication_pad_time(x, time_pad=1)
            x1 = self.conv2(x)
            x2 = _avg_pool3d(x, kernel=(2, 1, 1), stride=(2, 1, 1))
            x = x1 + x2
        x = self.conv3(x)
        return x


_HAAR = 0.7071067811865476


def _dwt3d_step(x, rescale):
    def split_axis(a, axis):
        sl0 = [slice(None)] * a.ndim
        sl1 = [slice(None)] * a.ndim
        sl0[axis] = slice(0, None, 2)
        sl1[axis] = slice(1, None, 2)
        low = (a[tuple(sl0)] + a[tuple(sl1)]) * _HAAR
        high = (a[tuple(sl0)] - a[tuple(sl1)]) * _HAAR
        return low, high

    def split_w(a):
        sl0 = [slice(None)] * a.ndim
        sl1 = [slice(None)] * a.ndim
        sl0[4] = slice(0, None, 2)
        sl1[4] = slice(1, None, 2)
        low = (a[tuple(sl0)] + a[tuple(sl1)]) * _HAAR
        high = (a[tuple(sl0)] - a[tuple(sl1)]) * _HAAR
        return low, high

    t_low, t_high = split_axis(x, 2)
    out_bands = []
    for tb in [t_low, t_high]:
        h_low, h_high = split_axis(tb, 3)
        for hb in [h_low, h_high]:
            w_low, w_high = split_w(hb)
            out_bands.append(w_low)
            out_bands.append(w_high)
    out = mx.concatenate(out_bands, axis=1)
    if rescale:
        out = out / (8.0**0.5)
    return out


def haar_patcher3d(x, patch_size=4):
    levels = int(math.log2(patch_size))
    B, C, T, H, W = x.shape
    xi = x[:, :, :1, ...]
    xv = x[:, :, 1:, ...]
    x = mx.concatenate([_repeat_interleave(xi, patch_size, axis=2), xv], axis=2)
    for _ in range(levels):
        x = _dwt3d_step(x, rescale=True)
    return x


def _idwt3d_step(x, rescale):
    B, C8, T, H, W = x.shape
    g = C8 // 8
    bands = mx.split(x, 8, axis=1)
    xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = bands

    def inv_axis(low, high, axis):
        inv = 1.0 / _HAAR
        a = (low + high) * inv * 0.5
        b = (low - high) * inv * 0.5
        out_shape = list(low.shape)
        out_shape[axis] = low.shape[axis] * 2
        out = mx.zeros(out_shape, dtype=low.dtype)
        sl_a = [slice(None)] * low.ndim
        sl_b = [slice(None)] * low.ndim
        sl_a[axis] = slice(0, None, 2)
        sl_b[axis] = slice(1, None, 2)
        out[tuple(sl_a)] = a
        out[tuple(sl_b)] = b
        return out

    xll = inv_axis(xlll, xllh, 4)
    xlh = inv_axis(xlhl, xlhh, 4)
    xhl = inv_axis(xhll, xhlh, 4)
    xhh = inv_axis(xhhl, xhhh, 4)
    xl = inv_axis(xll, xlh, 3)
    xh = inv_axis(xhl, xhh, 3)
    out = inv_axis(xl, xh, 2)
    if rescale:
        out = out * (8.0**0.5)
    return out


def haar_unpatcher3d(x, patch_size=4):
    levels = int(math.log2(patch_size))
    for _ in range(levels):
        x = _idwt3d_step(x, rescale=True)
    x = x[:, :, patch_size - 1 :, ...]
    return x


class _MidBlock(nn.Module):
    def __init__(self, channels, num_layers=1, num_groups=1):
        super().__init__()
        resnets = [ResnetBlock(channels, channels, num_groups)]
        attentions = []
        temp_attentions = []
        for _ in range(num_layers):
            attentions.append(AttnBlock(channels, num_groups))
            temp_attentions.append(TemporalAttnBlock(channels, num_groups))
            resnets.append(ResnetBlock(channels, channels, num_groups))
        self.resnets = resnets
        self.attentions = attentions
        self.temp_attentions = temp_attentions

    def __call__(self, x):
        h = self.resnets[0](x)
        for attn, temp_attn, resnet in zip(
            self.attentions, self.temp_attentions, self.resnets[1:]
        ):
            h = attn(h)
            h = temp_attn(h)
            h = resnet(h)
        return h


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        num_res,
        use_attention,
        use_upsample,
        spatial_up,
        temporal_up,
        num_groups=1,
    ):
        super().__init__()
        resnets = []
        block_in = in_ch
        for _ in range(num_res):
            resnets.append(ResnetBlock(block_in, out_ch, num_groups))
            block_in = out_ch
        self.resnets = resnets
        self.attentions = []
        self.temp_attentions = []
        if use_attention:
            for _ in range(num_res):
                self.attentions.append(AttnBlock(out_ch, num_groups))
                self.temp_attentions.append(TemporalAttnBlock(out_ch, num_groups))
        self.upsamplers = []
        if use_upsample:
            self.upsamplers.append(
                HybridUpsample(out_ch, spatial_up=spatial_up, temporal_up=temporal_up)
            )

    def __call__(self, x):
        for i, resnet in enumerate(self.resnets):
            x = resnet(x)
            if i < len(self.attentions) and self.attentions:
                x = self.attentions[i](x)
                x = self.temp_attentions[i](x)
        if self.upsamplers:
            x = self.upsamplers[0](x)
        return x


class _DownBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        num_res,
        use_attention,
        use_downsample,
        spatial_down,
        temporal_down,
        num_groups=1,
    ):
        super().__init__()
        resnets = []
        block_in = in_ch
        for _ in range(num_res):
            resnets.append(ResnetBlock(block_in, out_ch, num_groups))
            block_in = out_ch
        self.resnets = resnets
        self.attentions = []
        self.temp_attentions = []
        if use_attention:
            for _ in range(num_res):
                self.attentions.append(AttnBlock(out_ch, num_groups))
                self.temp_attentions.append(TemporalAttnBlock(out_ch, num_groups))
        self.downsamplers = []
        if use_downsample:
            self.downsamplers.append(
                HybridDownsample(
                    out_ch, spatial_down=spatial_down, temporal_down=temporal_down
                )
            )

    def __call__(self, x):
        for i, resnet in enumerate(self.resnets):
            x = resnet(x)
            if i < len(self.attentions) and self.attentions:
                x = self.attentions[i](x)
                x = self.temp_attentions[i](x)
        if self.downsamplers:
            x = self.downsamplers[0](x)
        return x


class CosmosVideoVAE(nn.Module):
    def __init__(self, latent_channels=16, in_channels=3, config=None):
        super().__init__()
        cfg = config or COSMOS_VAE_CONFIG
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.patch_size = cfg["patch_size"]
        dec_chs = cfg["decode_block_out_channels"]
        enc_chs = cfg["encoder_block_out_channels"]
        num_res = cfg["num_layers"]
        patch = self.patch_size
        num_spatial = int(math.log2(cfg["spatial_compression_ratio"])) - int(
            math.log2(patch)
        )
        num_temporal = int(math.log2(cfg["temporal_compression_ratio"])) - int(
            math.log2(patch)
        )
        attn_res = cfg.get("attention_resolutions", [32])
        resolution = cfg.get("resolution", 1024)

        rev_dec = list(reversed(dec_chs))
        block_in = rev_dec[0]
        out_ch_inner = in_channels * patch**3
        self.dec_conv_in = FactoredConv(latent_channels, block_in)
        self.dec_mid_block = _MidBlock(block_in, num_layers=1, num_groups=1)
        self.dec_up_blocks = []
        curr_res = (resolution // patch) // 2 ** (len(dec_chs) - 2)
        for i in range(len(dec_chs) - 1):
            in_ch = rev_dec[i]
            out_ch = rev_dec[i + 1]
            use_attention = curr_res in attn_res
            spatial_up = temporal_up = False
            if i < len(dec_chs) - 2:
                use_upsample = True
                temporal_up = 0 < i < num_temporal + 1
                spatial_up = temporal_up or (
                    i < num_spatial and num_spatial > num_temporal
                )
                curr_res = curr_res * 2
            else:
                use_upsample = False
            self.dec_up_blocks.append(
                _UpBlock(
                    in_ch,
                    out_ch,
                    num_res + 1,
                    use_attention,
                    use_upsample,
                    spatial_up,
                    temporal_up,
                )
            )
        self.dec_norm_out = _CausalNorm(rev_dec[-1], 1)
        self.dec_conv_out = FactoredConv(rev_dec[-1], out_ch_inner)

        inner_dim_enc = in_channels * patch**3
        enc_block_in = enc_chs[0]
        self.enc_conv_in = FactoredConv(inner_dim_enc, enc_block_in)
        self.enc_down_blocks = []
        curr_res = resolution // patch
        for i in range(len(enc_chs) - 1):
            in_ch = enc_chs[i]
            out_ch = enc_chs[i + 1]
            use_attention = curr_res in attn_res
            spatial_down = temporal_down = False
            if i < len(enc_chs) - 2:
                use_downsample = True
                spatial_down = i < num_spatial
                temporal_down = i < num_temporal
                curr_res = curr_res // 2
            else:
                use_downsample = False
            self.enc_down_blocks.append(
                _DownBlock(
                    in_ch,
                    out_ch,
                    num_res,
                    use_attention,
                    use_downsample,
                    spatial_down,
                    temporal_down,
                )
            )
        self.enc_mid_block = _MidBlock(enc_chs[-1], num_layers=1, num_groups=1)
        self.enc_norm_out = _CausalNorm(enc_chs[-1], 1)
        self.enc_conv_out = FactoredConv(enc_chs[-1], latent_channels)

        self.post_quant_conv = _PointwiseConv(latent_channels, latent_channels)
        self.quant_conv = _PointwiseConv(latent_channels, latent_channels)

    def decode(self, z, *, checkpoint=None):
        logger.info("cosmos vae decode: latent shape=%s", z.shape)
        if checkpoint is None:
            checkpoint = _vae_checkpoint_enabled()
        h = self.post_quant_conv(z)
        h = self.dec_conv_in(h)
        h = self.dec_mid_block(h)
        if checkpoint:
            mx.eval(h)
            mx.clear_cache()
        for ub in self.dec_up_blocks:
            h = ub(h)
            if checkpoint:
                mx.eval(h)
                mx.clear_cache()
        h = self.dec_norm_out(h)
        h = _silu(h)
        h = self.dec_conv_out(h)
        h = haar_unpatcher3d(h, self.patch_size)
        logger.info("cosmos vae decode: output shape=%s", h.shape)
        return h

    def decode_tiled(
        self,
        z,
        tile_t=8,
        tile_h=32,
        tile_w=32,
        overlap_t=2,
        overlap_h=4,
        overlap_w=4,
    ):
        logger.info(
            "cosmos vae decode_tiled: latent shape=%s tile=(%d,%d,%d)",
            z.shape,
            tile_t,
            tile_h,
            tile_w,
        )
        B, C, T, H, W = z.shape
        need_t = tile_t < T
        need_h = tile_h < H
        need_w = tile_w < W
        if not need_t and not need_h and not need_w:
            return self.decode(z)
        out_t, out_h, out_w = self._compute_output_shape(T, H, W)
        C_out = self.in_channels
        output = np.zeros((B, C_out, out_t, out_h, out_w), dtype=np.float32)
        weight_sum = np.zeros((B, C_out, out_t, out_h, out_w), dtype=np.float32)
        t_positions = self._tile_positions(T, tile_t, overlap_t) if need_t else [(0, T)]
        h_positions = self._tile_positions(H, tile_h, overlap_h) if need_h else [(0, H)]
        w_positions = self._tile_positions(W, tile_w, overlap_w) if need_w else [(0, W)]
        total = len(t_positions) * len(h_positions) * len(w_positions)
        idx = 0
        for t_start, t_end in t_positions:
            for h_start, h_end in h_positions:
                for w_start, w_end in w_positions:
                    idx += 1
                    tile_z = z[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                    logger.debug(
                        "cosmos vae decode_tiled: tile %d/%d shape=%s",
                        idx,
                        total,
                        tile_z.shape,
                    )
                    tile_out = self.decode(tile_z)
                    mx.eval(tile_out)
                    tile_np = np.array(tile_out, dtype=np.float32)
                    del tile_out
                    mx.clear_cache()
                    scale = self.patch_size * 2
                    ots = t_start * scale
                    ohs = h_start * scale
                    ows = w_start * scale
                    ote = min(ots + tile_np.shape[2], out_t)
                    ohe = min(ohs + tile_np.shape[3], out_h)
                    owe = min(ows + tile_np.shape[4], out_w)
                    tile_np = tile_np[:, :, : ote - ots, : ohe - ohs, : owe - ows]
                    wt = self._blend_weights_1d(
                        tile_np.shape[2], t_start, t_end, T, overlap_t * scale
                    )
                    wh = self._blend_weights_1d(
                        tile_np.shape[3], h_start, h_end, H, overlap_h * scale
                    )
                    ww = self._blend_weights_1d(
                        tile_np.shape[4], w_start, w_end, W, overlap_w * scale
                    )
                    w3d = (
                        wt.reshape(1, 1, -1, 1, 1)
                        * wh.reshape(1, 1, 1, -1, 1)
                        * ww.reshape(1, 1, 1, 1, -1)
                    )
                    w3d = np.broadcast_to(w3d, (1, C_out) + w3d.shape[2:]).copy()
                    output[:, :, ots:ote, ohs:ohe, ows:owe] += tile_np * w3d
                    weight_sum[:, :, ots:ote, ohs:ohe, ows:owe] += w3d
                    del tile_np
        mask = weight_sum > 0
        output[mask] /= weight_sum[mask]
        logger.info("cosmos vae decode_tiled: output shape=%s", output.shape)
        return mx.array(output)

    def _compute_output_shape(self, T, H, W):
        scale = self.patch_size * 2
        return T * scale, H * scale, W * scale

    @staticmethod
    def _tile_positions(dim_size, tile_size, overlap):
        positions = []
        start = 0
        while start < dim_size:
            end = min(start + tile_size, dim_size)
            positions.append((start, end))
            if end >= dim_size:
                break
            start = end - overlap
            if start >= dim_size - 1:
                break
        return positions

    @staticmethod
    def _blend_weights_1d(tile_out_size, tile_start, tile_end, dim_size, overlap):
        weights = np.ones(tile_out_size, dtype=np.float32)
        if overlap <= 0 or tile_out_size <= 0:
            return weights
        fade_len = min(overlap, tile_out_size // 2)
        if fade_len <= 0:
            return weights
        if tile_start > 0:
            weights[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)
        if tile_end < dim_size:
            weights[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)
        return weights

    def encode(self, x):
        logger.info("cosmos vae encode: input shape=%s", x.shape)
        h = haar_patcher3d(x, self.patch_size)
        h = self.enc_conv_in(h)
        for db in self.enc_down_blocks:
            h = db(h)
        h = self.enc_mid_block(h)
        h = self.enc_norm_out(h)
        h = _silu(h)
        h = self.enc_conv_out(h)
        h = self.quant_conv(h)
        logger.info("cosmos vae encode: latent shape=%s", h.shape)
        return h

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        vae = cls(**kwargs)
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("no safetensors found at %s, using random init", model_path)
            return vae
        from mlx.utils import tree_flatten, tree_unflatten

        all_params = {}
        for sf in safetensor_files:
            all_params.update(mx.load(sf))
        mapped = _remap_vae_weights(all_params)
        flat = tree_flatten(vae.parameters())
        flat_keys = {k for k, _ in flat}
        loaded = {}
        matched = 0
        for k, v in flat:
            if k in mapped:
                mv = mapped[k]
                if mv.dtype != mx.float16:
                    mv = mv.astype(mx.float16)
                if mv.shape != v.shape:
                    logger.warning(
                        "vae: shape mismatch %s: ckpt %s vs module %s",
                        k,
                        tuple(mv.shape),
                        tuple(v.shape),
                    )
                    loaded[k] = v
                else:
                    loaded[k] = mv
                    matched += 1
            else:
                loaded[k] = v
        unmatched_ckpt = [ck for ck in mapped if ck not in flat_keys]
        vae.update(tree_unflatten(loaded))
        logger.info(
            "cosmos vae from_pretrained: matched %d/%d params, %d ckpt keys unmatched",
            matched,
            len(flat),
            len(unmatched_ckpt),
        )
        if unmatched_ckpt:
            logger.warning("cosmos vae: unmatched ckpt keys: %s", unmatched_ckpt[:20])
        return vae


def _remap_vae_weights(params):
    out = {}
    for k, v in params.items():
        nk = k
        nk = nk.replace("decoder.", "dec_")
        nk = nk.replace("encoder.", "enc_")
        out[nk] = v
    return out
