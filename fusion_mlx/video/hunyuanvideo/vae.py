# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo causal 3D VAE.
# Latent dim 16, 8x spatial, 4x temporal compression.

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def _silu(x):
    return x * mx.sigmoid(x)


def _group_norm_5d(norm, x):
    B, C, T, H, W = x.shape
    x_cl = x.reshape(B * T, C, H, W).transpose(0, 2, 3, 1)
    y_cl = norm(x_cl)
    y = y_cl.transpose(0, 3, 1, 2).reshape(B, C, T, H, W)
    return y


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        self.weight = mx.zeros(
            (out_channels, in_channels) + self.kernel_size, dtype=mx.float32
        )
        self.bias = mx.zeros((out_channels,), dtype=mx.float32)

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
        x_cl = x.transpose(0, 2, 3, 4, 1)
        w_cl = self.weight.transpose(0, 2, 3, 4, 1)
        padding = ([pt, ph, pw], [0, ph, pw])
        stride = (st, sh, sw)
        out = mx.conv_general(x_cl, w_cl, stride=stride, padding=padding)
        out = out.transpose(0, 4, 1, 2, 3)
        out = out + self.bias.reshape(1, -1, 1, 1, 1)
        logger.debug("CausalConv3d: output=(%s)", out.shape)
        return out


class HVResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = CausalConv3d(channels, channels, 3, 1, 1)
        self.conv2 = CausalConv3d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)

    def __call__(self, x):
        h = _group_norm_5d(self.norm1, x)
        h = _silu(h)
        h = self.conv1(h)
        h = _group_norm_5d(self.norm2, h)
        h = _silu(h)
        h = self.conv2(h)
        return x + h


class HVDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_conv_down=True, temporal_downsample=True):
        super().__init__()
        self.res1 = HVResBlock(in_ch)
        self.res2 = HVResBlock(in_ch)
        self.use_conv_down = use_conv_down
        self.temporal_downsample = temporal_downsample
        if use_conv_down:
            st = (2, 2, 2) if temporal_downsample else (1, 2, 2)
            self.conv_down = CausalConv3d(in_ch, out_ch, 3, stride=st, padding=1)
        else:
            self.conv_resample = CausalConv3d(in_ch, out_ch, 3, 1, 1)

    def __call__(self, x):
        h = self.res1(x)
        h = self.res2(h)
        if self.use_conv_down:
            return self.conv_down(h)
        return self.conv_resample(h)


class HVUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_conv_up=True, temporal_upsample=True):
        super().__init__()
        self.res1 = HVResBlock(in_ch)
        self.res2 = HVResBlock(in_ch)
        self.use_conv_up = use_conv_up
        self.temporal_upsample = temporal_upsample
        if use_conv_up:
            self.conv_up = CausalConv3d(in_ch, out_ch, 3, 1, 1)
        else:
            self.conv_resample = CausalConv3d(in_ch, out_ch, 3, 1, 1)

    def __call__(self, x):
        h = self.res1(x)
        h = self.res2(h)
        if self.use_conv_up:
            B, C, T, H, W = h.shape
            if self.temporal_upsample and T > 1:
                # Official UpsampleCausal3D behavior:
                # first frame stays as 1 frame (spatial-only upsample)
                # other frames double temporally -> 2*(T-1) frames
                # total: 1 + 2*(T-1) = 2T-1 frames
                first = h[:, :, 0:1, :, :]  # (B, C, 1, H, W)
                other = h[:, :, 1:, :, :]  # (B, C, T-1, H, W)
                # Spatial upsample for first frame
                first = first.reshape(B, C, 1, H, 1, W, 1)
                first = mx.broadcast_to(first, (B, C, 1, H, 2, W, 2))
                first = first.reshape(B, C, 1, H * 2, W * 2)
                # Temporal + spatial upsample for other frames
                other = other.reshape(B, C, T - 1, 1, H, 1, W, 1)
                other = mx.broadcast_to(other, (B, C, T - 1, 2, H, 2, W, 2))
                other = other.reshape(B, C, (T - 1) * 2, H * 2, W * 2)
                h = mx.concatenate([first, other], axis=2)
            else:
                # Spatial-only upsample (no temporal)
                h = h.reshape(B, C, T, H, 1, W, 1)
                h = mx.broadcast_to(h, (B, C, T, H, 2, W, 2))
                h = h.reshape(B, C, T, H * 2, W * 2)
            return self.conv_up(h)
        return self.conv_resample(h)


class HunyuanVideoVAE(nn.Module):
    # Official decoder upsample pattern:
    #   i=0: spatial_up=True,  temporal_up=False  -> (1,2,2) spatial only
    #   i=1: spatial_up=True,  temporal_up=True   -> (2,2,2) temporal+spatial
    #   i=2: spatial_up=True,  temporal_up=True   -> (2,2,2) temporal+spatial
    #   i=3: spatial_up=False, temporal_up=False  -> no upsample
    # Official encoder downsample pattern:
    #   i=0: spatial_down=True,  temporal_down=False -> stride (1,2,2)
    #   i=1: spatial_down=True,  temporal_down=True  -> stride (2,2,2)
    #   i=2: spatial_down=True,  temporal_down=True  -> stride (2,2,2)
    #   i=3: spatial_down=False, temporal_down=False -> no downsample

    TEMPORAL_UP_BLOCKS = [False, True, True, False]
    SPATIAL_UP_BLOCKS = [True, True, True, False]
    TEMPORAL_DOWN_BLOCKS = [False, True, True, False]
    SPATIAL_DOWN_BLOCKS = [True, True, True, False]

    def __init__(self, latent_channels=16, in_channels=3):
        super().__init__()
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        ch_mult = [1, 2, 4, 4]
        base_ch = 128
        # Encoder
        self.enc_conv_in = CausalConv3d(in_channels, base_ch, 3, 1, 1)
        enc_blocks = []
        prev_ch = base_ch
        for i, mult in enumerate(ch_mult):
            cur_ch = base_ch * mult
            use_down = self.SPATIAL_DOWN_BLOCKS[i] or self.TEMPORAL_DOWN_BLOCKS[i]
            temporal_down = self.TEMPORAL_DOWN_BLOCKS[i]
            enc_blocks.append(HVDownBlock(prev_ch, cur_ch, use_conv_down=use_down, temporal_downsample=temporal_down))
            prev_ch = cur_ch
        self.enc_blocks = enc_blocks
        self.enc_mid1 = HVResBlock(prev_ch)
        self.enc_mid2 = HVResBlock(prev_ch)
        self.enc_conv_out = CausalConv3d(prev_ch, latent_channels * 2, 3, 1, 1)
        # Decoder
        self.dec_conv_in = CausalConv3d(latent_channels, prev_ch, 3, 1, 1)
        self.dec_mid1 = HVResBlock(prev_ch)
        self.dec_mid2 = HVResBlock(prev_ch)
        dec_blocks = []
        for i, mult in reversed(list(enumerate(ch_mult))):
            cur_ch = base_ch * mult
            use_up = self.SPATIAL_UP_BLOCKS[i] or self.TEMPORAL_UP_BLOCKS[i]
            temporal_up = self.TEMPORAL_UP_BLOCKS[i]
            dec_blocks.append(HVUpBlock(prev_ch, cur_ch, use_conv_up=use_up, temporal_upsample=temporal_up))
            prev_ch = cur_ch
        self.dec_blocks = dec_blocks
        self.dec_conv_out = CausalConv3d(prev_ch, in_channels, 3, 1, 1)

    def encode(self, x):
        logger.info("hunyuan vae encode: input shape=%s", x.shape)
        h = self.enc_conv_in(x)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_mid1(h)
        h = self.enc_mid2(h)
        h = self.enc_conv_out(h)
        mean, logvar = mx.split(h, 2, axis=1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mean.shape, dtype=mean.dtype)
        z = mean + std * eps
        logger.info("hunyuan vae encode: latent shape=%s", z.shape)
        return z

    def decode(self, z):
        logger.info("hunyuan vae decode: latent shape=%s", z.shape)
        h = self.dec_conv_in(z)
        h = self.dec_mid1(h)
        h = self.dec_mid2(h)
        for block in self.dec_blocks:
            h = block(h)
        h = self.dec_conv_out(h)
        h = mx.clip(h, 0.0, 1.0)
        logger.info("hunyuan vae decode: output shape=%s", h.shape)
        return h

    def decode_tiled(self, z, tile_t=8, tile_h=32, tile_w=32,
                     overlap_t=2, overlap_h=4, overlap_w=4):
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
        output = np.zeros((B, 3, out_t, out_h, out_w), dtype=np.float32)
        weight_sum = np.zeros((B, 1, out_t, out_h, out_w), dtype=np.float32)

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
                        tile_idx, total_tiles, t_start, t_end, h_start, h_end, w_start, w_end, tile_z.shape,
                    )
                    tile_out = self.decode(tile_z)
                    mx.eval(tile_out)
                    tile_np = np.array(tile_out, dtype=np.float32)
                    del tile_out
                    mx.clear_cache()

                    # Compute output positions
                    ot_start = self._latent_to_output_pos(t_start, is_temporal=True)
                    ot_end = ot_start + tile_np.shape[2]
                    oh_start = h_start * 8
                    oh_end = oh_start + tile_np.shape[3]
                    ow_start = w_start * 8
                    ow_end = ow_start + tile_np.shape[4]

                    # Clamp to output bounds
                    ot_end = min(ot_end, out_t)
                    oh_end = min(oh_end, out_h)
                    ow_end = min(ow_end, out_w)

                    tile_np = tile_np[:, :, :ot_end - ot_start, :oh_end - oh_start, :ow_end - ow_start]

                    # Compute blending weights (feathered in overlap regions)
                    w_t = self._blend_weights_1d(tile_np.shape[2], t_start, t_end, T, overlap_t)
                    w_h = self._blend_weights_1d(tile_np.shape[3], h_start, h_end, H, overlap_h)
                    w_w = self._blend_weights_1d(tile_np.shape[4], w_start, w_end, W, overlap_w)

                    # Outer product -> (1, 1, t, h, w)
                    w_3d = w_t.reshape(1, 1, -1, 1, 1) * w_h.reshape(1, 1, 1, -1, 1) * w_w.reshape(1, 1, 1, 1, -1)

                    output[:, :, ot_start:ot_end, oh_start:oh_end, ow_start:ow_end] += tile_np * w_3d
                    weight_sum[:, :, ot_start:ot_end, oh_start:oh_end, ow_start:ow_end] += w_3d

                    del tile_np
                    tile_idx_logged = tile_idx

        # Normalize by weight sum
        mask = weight_sum > 0
        output[mask] /= weight_sum[mask]

        logger.info("hunyuan vae decode_tiled: output shape=%s total_tiles=%d", output.shape, total_tiles)
        return mx.array(output)

    def _compute_output_shape(self, T, H, W):
        t, h, w = T, H, W
        for i in range(len(self.dec_blocks)):
            if self.SPATIAL_UP_BLOCKS[i]:
                h = h * 2
                w = w * 2
            if self.TEMPORAL_UP_BLOCKS[i] and t > 1:
                # Official UpsampleCausal3D: first frame stays, others double
                t = 1 + (t - 1) * 2
        return t, h, w

    def _latent_to_output_pos(self, latent_pos, is_temporal=False):
        if is_temporal:
            # First latent frame -> output frame 0
            # Each subsequent latent frame -> 2 output frames (2T-1 pattern)
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
        # Ramp up at start if not first tile
        if tile_start > 0:
            ramp_len = min(overlap, tile_out_size)
            for i in range(ramp_len):
                weights[i] = (i + 1) / (ramp_len + 1)
        # Ramp down at end if not last tile
        if tile_end < dim_size:
            ramp_len = min(overlap, tile_out_size)
            for i in range(ramp_len):
                weights[tile_out_size - 1 - i] = min(weights[tile_out_size - 1 - i], (i + 1) / (ramp_len + 1))
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
        from mlx.utils import tree_flatten, tree_unflatten

        all_params = {}
        for sf in safetensor_files:
            weights = mx.load(sf)
            all_params.update(weights)
        mapped = _remap_vae_weights(all_params)
        flat = tree_flatten(vae.parameters())
        loaded = {}
        for k, v in flat:
            if k in mapped:
                loaded[k] = (
                    mapped[k].astype(mx.float16)
                    if mapped[k].dtype != mx.float16
                    else mapped[k]
                )
            else:
                loaded[k] = v
                logger.debug("vae: unmatched param %s", k)
        vae.update(tree_unflatten(loaded))
        return vae


def _remap_vae_weights(params):
    out = {}
    for k, v in params.items():
        nk = k
        nk = nk.replace("decoder.", "dec_")
        nk = nk.replace("encoder.", "enc_")
        nk = nk.replace("mid_block.resnets.0.", "mid1.")
        nk = nk.replace("mid_block.resnets.1.", "mid2.")
        out[nk] = v
    return out
