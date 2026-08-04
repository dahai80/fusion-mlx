# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo causal 3D VAE — matches official weight format.
# Latent dim 16, 8x spatial, 4x temporal compression.
# Called by: generate.py via HunyuanVideoVAE.from_pretrained(), .encode(), .decode(), .decode_tiled()
# API unchanged: constructor(latent_channels=16, in_channels=3), all method signatures preserved.

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

BASE_CH = 128
CH_MULT = [1, 2, 4, 4]
NUM_RES_BLOCKS_DEC = [3, 3, 3, 3]
NUM_RES_BLOCKS_ENC = [2, 2, 2, 2]
LATENT_CHANNELS = 16
IN_CHANNELS = 3

DECODER_TEMPORAL_UP = [False, True, True, True]
DECODER_SPATIAL_UP = [False, True, True, True]
ENCODER_TEMPORAL_DOWN = [False, True, True, False]
ENCODER_SPATIAL_DOWN = [True, True, True, False]


def _silu(x):
    return x * mx.sigmoid(x)


def _group_norm_5d(norm, x):
    B, C, T, H, W = x.shape
    x_cl = x.reshape(B * T, C, H, W).transpose(0, 2, 3, 1)
    y_cl = norm(x_cl)
    y = y_cl.transpose(0, 3, 1, 2).reshape(B, C, T, H, W)
    return y


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, wrap_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._wrap_conv = wrap_conv
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)
        w = mx.zeros((out_channels, in_channels) + self.kernel_size, dtype=mx.float32)
        b = mx.zeros((out_channels,), dtype=mx.float32)
        if wrap_conv:
            # e.g. "decoder.conv_in.conv.weight" (ResBlock/Upsample/Downsample/conv_in/conv_out)
            self.conv = nn.Module()
            self.conv.weight = w
            self.conv.bias = b
        else:
            # e.g. "decoder.mid.attn_1.q.weight" (MidAttention/quant_conv/post_quant_conv)
            self.weight = w
            self.bias = b

    def _get_wb(self):
        if self._wrap_conv:
            return self.conv.weight, self.conv.bias
        return self.weight, self.bias

    def __call__(self, x):
        B, C, T, H, W = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        _, ph, pw = self.padding
        pt = kt - 1
        logger.debug(
            "CausalConv3d: input=(%s) k=(%d,%d,%d) s=(%d,%d,%d) pad=(%d,%d,%d)",
            x.shape, kt, kh, kw, st, sh, sw, pt, ph, pw,
        )
        cw, cb = self._get_wb()
        x_cl = x.transpose(0, 2, 3, 4, 1)
        w_cl = cw.transpose(0, 2, 3, 4, 1)
        padding = ([pt, ph, pw], [0, ph, pw])
        stride = (st, sh, sw)
        out = mx.conv_general(x_cl, w_cl, stride=stride, padding=padding)
        out = out.transpose(0, 4, 1, 2, 3)
        out = out + cb.reshape(1, -1, 1, 1, 1)
        logger.debug("CausalConv3d: output=(%s)", out.shape)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, 1, 1)
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, 1, 1, 0)

    def __call__(self, x):
        h = _group_norm_5d(self.norm1, x)
        h = _silu(h)
        h = self.conv1(h)
        h = _group_norm_5d(self.norm2, h)
        h = _silu(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class MidAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = CausalConv3d(channels, channels, 1, 1, 0, wrap_conv=False)
        self.k = CausalConv3d(channels, channels, 1, 1, 0, wrap_conv=False)
        self.v = CausalConv3d(channels, channels, 1, 1, 0, wrap_conv=False)
        self.proj_out = CausalConv3d(channels, channels, 1, 1, 0, wrap_conv=False)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        h = _group_norm_5d(self.norm, x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        q = q.reshape(B * T, C, H * W).transpose(0, 2, 1)
        k = k.reshape(B * T, C, H * W).transpose(0, 2, 1)
        v = v.reshape(B * T, C, H * W).transpose(0, 2, 1)
        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(0, 2, 1)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1).reshape(B, C, T, H, W)
        out = self.proj_out(out)
        return x + out


class Upsample(nn.Module):
    def __init__(self, channels, temporal_up=True):
        super().__init__()
        self.temporal_up = temporal_up
        self.conv = CausalConv3d(channels, channels, 3, 1, 1)

    def __call__(self, x):
        B, C, T, H, W = x.shape
        # Spatial 2x upsampling via repeat
        h = x.reshape(B, C, T, H, 1, W, 1)
        h = mx.broadcast_to(h, (B, C, T, H, 2, W, 2))
        h = h.reshape(B, C, T, H * 2, W * 2)
        # Temporal upsampling: each frame after the first gets duplicated
        if self.temporal_up and T > 1:
            first = h[:, :, 0:1, :, :]  # (B, C, 1, H*2, W*2)
            rest = h[:, :, 1:, :, :]    # (B, C, T-1, H*2, W*2)
            rest = rest.reshape(B, C, T - 1, 1, H * 2, W * 2)
            rest = mx.broadcast_to(rest, (B, C, T - 1, 2, H * 2, W * 2))
            rest = rest.reshape(B, C, (T - 1) * 2, H * 2, W * 2)
            h = mx.concatenate([first, rest], axis=2)
        h = self.conv(h)
        return h


class Downsample(nn.Module):
    def __init__(self, channels, temporal_down=True):
        super().__init__()
        stride = (2, 2, 2) if temporal_down else (1, 2, 2)
        self.conv = CausalConv3d(channels, channels, 3, stride=stride, padding=1)

    def __call__(self, x):
        return self.conv(x)


class DecoderUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks, has_upsample, temporal_up):
        super().__init__()
        # Use .block container with numeric sub-attrs to match weight keys:
        # "decoder.up.0.block.0.conv1.conv.weight"
        self.block = nn.Module()
        for i in range(num_blocks):
            if i == 0:
                setattr(self.block, str(i), ResBlock(in_ch, out_ch))
            else:
                setattr(self.block, str(i), ResBlock(out_ch, out_ch))
        self._num_blocks = num_blocks
        self.upsample = None
        if has_upsample:
            self.upsample = Upsample(out_ch, temporal_up=temporal_up)

    def __call__(self, x):
        for i in range(self._num_blocks):
            x = getattr(self.block, str(i))(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class EncoderDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks, has_downsample, temporal_down):
        super().__init__()
        self.block = nn.Module()
        for i in range(num_blocks):
            if i == 0:
                setattr(self.block, str(i), ResBlock(in_ch, out_ch))
            else:
                setattr(self.block, str(i), ResBlock(out_ch, out_ch))
        self._num_blocks = num_blocks
        self.downsample = None
        if has_downsample:
            self.downsample = Downsample(out_ch, temporal_down=temporal_down)

    def __call__(self, x):
        for i in range(self._num_blocks):
            x = getattr(self.block, str(i))(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class HunyuanVideoVAE(nn.Module):
    def __init__(self, latent_channels=LATENT_CHANNELS, in_channels=IN_CHANNELS):
        super().__init__()
        self.latent_channels = latent_channels
        self.in_channels = in_channels

        dec_ch = [BASE_CH * m for m in reversed(CH_MULT)]
        # dec_ch = [512, 512, 256, 128]

        self.decoder = nn.Module()
        self.decoder.conv_in = CausalConv3d(latent_channels, dec_ch[0], 3, 1, 1)
        self.decoder.mid = nn.Module()
        self.decoder.mid.block_1 = ResBlock(dec_ch[0], dec_ch[0])
        self.decoder.mid.attn_1 = MidAttention(dec_ch[0])
        self.decoder.mid.block_2 = ResBlock(dec_ch[0], dec_ch[0])
        self.decoder.norm_out = nn.GroupNorm(32, dec_ch[-1])
        self.decoder.conv_out = CausalConv3d(dec_ch[-1], in_channels, 3, 1, 1)

        self.decoder.up = nn.Module()
        for i in range(len(CH_MULT)):
            out_ch = dec_ch[i]
            in_ch = dec_ch[i]
            # up.0: in=256 (from up.1 output after upsample) → out=128
            # Wait, the blocks chain: up.3 output → up.2 input, etc.
            # Need to check: each up block's in_ch = previous block's out_ch
            # But the first block of each group needs in_ch from the previous group
            if i == 0:
                # up.0: dec_ch[0]=512 → but weights show block.0.conv1 in=256
                # This is because up blocks are ordered differently
                pass
            has_up = DECODER_SPATIAL_UP[i]
            temporal_up = DECODER_TEMPORAL_UP[i]

        # Re-derive decoder up block channel sizes from weight shapes:
        # decoder.up.0.block.0.conv1.weight: (128, 256, 3, 3, 3) → in=256, out=128
        # decoder.up.1.block.0.conv1.weight: (256, 512, 3, 3, 3) → in=512, out=256
        # decoder.up.2.block.0.conv1.weight: (512, 512, 3, 3, 3) → in=512, out=512
        # decoder.up.3.block.0.conv1.weight: (512, 512, 3, 3, 3) → in=512, out=512
        dec_up_in = [256, 512, 512, 512]
        dec_up_out = [128, 256, 512, 512]

        self.decoder.up = nn.Module()
        for i in range(4):
            setattr(
                self.decoder.up, str(i),
                DecoderUpBlock(
                    dec_up_in[i], dec_up_out[i],
                    NUM_RES_BLOCKS_DEC[i], DECODER_SPATIAL_UP[i], DECODER_TEMPORAL_UP[i]
                )
            )

        # Encoder
        enc_ch = [BASE_CH * m for m in CH_MULT]
        # enc_ch = [128, 256, 512, 512]

        self.encoder = nn.Module()
        self.encoder.conv_in = CausalConv3d(in_channels, enc_ch[0], 3, 1, 1)
        self.encoder.mid = nn.Module()
        self.encoder.mid.block_1 = ResBlock(enc_ch[-1], enc_ch[-1])
        self.encoder.mid.attn_1 = MidAttention(enc_ch[-1])
        self.encoder.mid.block_2 = ResBlock(enc_ch[-1], enc_ch[-1])
        self.encoder.norm_out = nn.GroupNorm(32, enc_ch[-1])
        self.encoder.conv_out = CausalConv3d(enc_ch[-1], latent_channels * 2, 3, 1, 1)

        # Encoder down block channel sizes from weight shapes:
        # encoder.down.0.block.0.conv1.weight: (128, 128, 3, 3, 3) → in=128, out=128
        # encoder.down.1.block.0.conv1.weight: (256, 128, 3, 3, 3) → in=128, out=256
        # encoder.down.2.block.0.conv1.weight: (512, 256, 3, 3, 3) → in=256, out=512
        # encoder.down.3.block.0.conv1.weight: (512, 512, 3, 3, 3) → in=512, out=512
        enc_down_in = [128, 128, 256, 512]
        enc_down_out = [128, 256, 512, 512]

        self.encoder.down = nn.Module()
        for i in range(4):
            setattr(
                self.encoder.down, str(i),
                EncoderDownBlock(
                    enc_down_in[i], enc_down_out[i],
                    NUM_RES_BLOCKS_ENC[i], ENCODER_SPATIAL_DOWN[i], ENCODER_TEMPORAL_DOWN[i]
                )
            )

        self.quant_conv = CausalConv3d(latent_channels * 2, latent_channels * 2, 1, 1, 0, wrap_conv=False)
        self.post_quant_conv = CausalConv3d(latent_channels, latent_channels, 1, 1, 0, wrap_conv=False)

    def encode(self, x):
        logger.info("hunyuan vae encode: input shape=%s", x.shape)
        h = self.encoder.conv_in(x)
        for i in range(4):
            h = getattr(self.encoder.down, str(i))(h)
        h = self.encoder.mid.block_1(h)
        h = self.encoder.mid.attn_1(h)
        h = self.encoder.mid.block_2(h)
        h = _group_norm_5d(self.encoder.norm_out, h)
        h = _silu(h)
        h = self.encoder.conv_out(h)
        h = self.quant_conv(h)
        mean, logvar = mx.split(h, 2, axis=1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mean.shape, dtype=mean.dtype)
        z = mean + std * eps
        logger.info("hunyuan vae encode: latent shape=%s", z.shape)
        return z

    def decode(self, z):
        logger.info("hunyuan vae decode: latent shape=%s", z.shape)
        h = self.post_quant_conv(z)
        h = self.decoder.conv_in(h)
        h = self.decoder.mid.block_1(h)
        h = self.decoder.mid.attn_1(h)
        h = self.decoder.mid.block_2(h)
        for i in range(3, -1, -1):
            h = getattr(self.decoder.up, str(i))(h)
        h = _group_norm_5d(self.decoder.norm_out, h)
        h = _silu(h)
        h = self.decoder.conv_out(h)
        h = mx.clip(h, 0.0, 1.0)
        logger.info("hunyuan vae decode: output shape=%s", h.shape)
        return h

    def decode_tiled(
        self, z, tile_t=8, tile_h=32, tile_w=32, overlap_t=2, overlap_h=4, overlap_w=4
    ):
        logger.info(
            "hunyuan vae decode_tiled: latent shape=%s tile=(%d,%d,%d) overlap=(%d,%d,%d)",
            z.shape, tile_t, tile_h, tile_w, overlap_t, overlap_h, overlap_w,
        )
        B, C, T, H, W = z.shape
        need_t = tile_t < T
        need_h = tile_h < H
        need_w = tile_w < W

        if not need_t and not need_h and not need_w:
            logger.info("hunyuan vae decode_tiled: no tiling needed, using decode()")
            return self.decode(z)

        out_t, out_h, out_w = self._compute_output_shape(T, H, W)
        out_c = 3
        output = np.zeros((B, out_c, out_t, out_h, out_w), dtype=np.float32)
        weight_sum = np.zeros((B, out_c, out_t, out_h, out_w), dtype=np.float32)

        t_positions = self._tile_positions(T, tile_t, overlap_t) if need_t else [(0, T)]
        h_positions = self._tile_positions(H, tile_h, overlap_h) if need_h else [(0, H)]
        w_positions = self._tile_positions(W, tile_w, overlap_w) if need_w else [(0, W)]

        total_tiles = len(t_positions) * len(h_positions) * len(w_positions)
        tile_idx = 0
        for t_start, t_end in t_positions:
            for h_start, h_end in h_positions:
                for w_start, w_end in w_positions:
                    tile_idx += 1
                    tile_z = z[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                    logger.debug(
                        "hunyuan vae decode_tiled: tile %d/%d slice=[%d:%d,%d:%d,%d:%d] shape=%s",
                        tile_idx, total_tiles, t_start, t_end, h_start, h_end, w_start, w_end,
                        tile_z.shape,
                    )
                    tile_out = self.decode(tile_z)
                    mx.eval(tile_out)
                    tile_np = np.array(tile_out, dtype=np.float32)
                    del tile_out
                    mx.clear_cache()

                    ot_start = self._latent_to_output_pos(t_start, is_temporal=True)
                    ot_end = ot_start + tile_np.shape[2]
                    oh_start = h_start * 8
                    oh_end = oh_start + tile_np.shape[3]
                    ow_start = w_start * 8
                    ow_end = ow_start + tile_np.shape[4]

                    ot_end = min(ot_end, out_t)
                    oh_end = min(oh_end, out_h)
                    ow_end = min(ow_end, out_w)

                    tile_np = tile_np[
                        :, :,
                        : ot_end - ot_start,
                        : oh_end - oh_start,
                        : ow_end - ow_start,
                    ]

                    w_t = self._blend_weights_1d(
                        tile_np.shape[2], t_start, t_end, T, overlap_t
                    )
                    w_h = self._blend_weights_1d(
                        tile_np.shape[3], h_start, h_end, H, overlap_h
                    )
                    w_w = self._blend_weights_1d(
                        tile_np.shape[4], w_start, w_end, W, overlap_w
                    )

                    w_3d = (
                        w_t.reshape(1, 1, -1, 1, 1)
                        * w_h.reshape(1, 1, 1, -1, 1)
                        * w_w.reshape(1, 1, 1, 1, -1)
                    )
                    w_3d = np.broadcast_to(w_3d, (1, out_c) + w_3d.shape[2:]).copy()

                    output[:, :, ot_start:ot_end, oh_start:oh_end, ow_start:ow_end] += (
                        tile_np * w_3d
                    )
                    weight_sum[
                        :, :, ot_start:ot_end, oh_start:oh_end, ow_start:ow_end
                    ] += w_3d

                    del tile_np

        mask = weight_sum > 0
        output[mask] /= weight_sum[mask]

        logger.info(
            "hunyuan vae decode_tiled: output shape=%s total_tiles=%d",
            output.shape, total_tiles,
        )
        return mx.array(output)

    def _compute_output_shape(self, T, H, W):
        t, h, w = T, H, W
        for i in range(4):
            if DECODER_SPATIAL_UP[i]:
                h = h * 2
                w = w * 2
            if DECODER_TEMPORAL_UP[i] and t > 1:
                t = 1 + (t - 1) * 2
        return t, h, w

    def _latent_to_output_pos(self, latent_pos, is_temporal=False):
        if is_temporal:
            if latent_pos == 0:
                return 0
            return 1 + (latent_pos - 1) * 2
        else:
            return latent_pos * 8

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
        if tile_start > 0:
            ramp_len = min(overlap, tile_out_size)
            for i in range(ramp_len):
                weights[i] = (i + 1) / (ramp_len + 1)
        if tile_end < dim_size:
            ramp_len = min(overlap, tile_out_size)
            for i in range(ramp_len):
                weights[tile_out_size - 1 - i] = min(
                    weights[tile_out_size - 1 - i], (i + 1) / (ramp_len + 1)
                )
        return weights

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        import glob
        import os

        vae = cls(**kwargs)
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("no safetensors found at %s, using random init", model_path)
            return vae
        from mlx.utils import tree_flatten

        all_params = {}
        for sf in safetensor_files:
            weights = mx.load(sf)
            all_params.update(weights)

        flat = tree_flatten(vae.parameters())
        loaded = {}
        matched = 0
        unmatched = []
        for k, v in flat:
            if k in all_params:
                loaded[k] = (
                    all_params[k].astype(mx.float16)
                    if all_params[k].dtype != mx.float16
                    else all_params[k]
                )
                matched += 1
            else:
                loaded[k] = v
                unmatched.append(k)
        logger.info(
            "hunyuan vae: loaded %d/%d params from %s",
            matched, len(flat), model_path,
        )
        if unmatched:
            logger.debug("hunyuan vae: unmatched params (%d): %s", len(unmatched), unmatched[:20])

        # Build nested dict preserving numeric string keys as dict keys
        # (tree_unflatten converts numeric keys to list indices, breaking nn.Module.update)
        nested = {}
        for key, val in loaded.items():
            parts = key.split(".")
            d = nested
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = val
        vae.update(nested)
        return vae
