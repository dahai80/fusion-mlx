# SPDX-License-Identifier: Apache-2.0
# Cosmos continuous VAE for latent video encoding/decoding.
# Latent dim 16, 8x spatial compression, 4x temporal compression.

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def _silu(x):
    return x * mx.sigmoid(x)


def _group_norm_5d(norm, x):
    # MLX GroupNorm.__call__ does (self.weight * x + self.bias) but
    # weight shape (C,) cannot broadcast to (N, C, H, W) when N > 1.
    # We bypass by calling the internal normalize step and applying
    # weight/bias manually with correct reshape.
    orig_5d = x.ndim == 5
    if orig_5d:
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

    group_fn = (
        norm._pytorch_compatible_group_norm
        if norm.pytorch_compatible
        else norm._group_norm
    )
    x = group_fn(x)

    if "weight" in norm:
        w = norm.weight.reshape(1, -1, *([1] * (x.ndim - 2)))
        x = w * x
    if "bias" in norm:
        b = norm.bias.reshape(1, -1, *([1] * (x.ndim - 2)))
        x = x + b

    if orig_5d:
        x = x.reshape(B, C, T, H, W)
    return x


class CosmosVAEConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )
        self.weight = mx.zeros(
            (out_channels, in_channels) + self.kernel_size, dtype=mx.float32
        )
        self.bias = mx.zeros((out_channels,), dtype=mx.float32)

    def __call__(self, x):
        # x: (B, C, T, H, W) -> manual conv3d via chunked slice matmul
        B, C, T, H, W = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding
        if any(p > 0 for p in (pt, ph, pw)):
            x = mx.pad(x, [(0, 0), (0, 0), (pt, pt), (ph, ph), (pw, pw)])
        Tp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        ot = (Tp - kt) // st + 1
        oh = (Hp - kh) // sh + 1
        ow = (Wp - kw) // sw + 1
        total_patches = ot * oh * ow
        if total_patches == 0:
            return mx.zeros((B, self.out_channels, ot, oh, ow), dtype=x.dtype)
        w = self.weight.reshape(self.out_channels, -1)  # (out, C*kt*kh*kw)
        # Chunk output positions to limit peak memory
        CHUNK_SIZE = 512
        out_chunks = []
        positions = [
            (ti, hi, wi) for ti in range(ot) for hi in range(oh) for wi in range(ow)
        ]
        for chunk_start in range(0, total_patches, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_patches)
            patches = []
            for ti, hi, wi in positions[chunk_start:chunk_end]:
                patch = x[
                    :,
                    :,
                    ti * st : ti * st + kt,
                    hi * sh : hi * sh + kh,
                    wi * sw : wi * sw + kw,
                ]
                patches.append(patch.reshape(B, -1))
            chunk = mx.stack(patches, axis=0)  # (chunk_len, B, C*kt*kh*kw)
            chunk = chunk.reshape(-1, B, C * kt * kh * kw)
            out_chunk = chunk @ w.T  # (chunk_len, B, out)
            out_chunks.append(out_chunk)
        out = mx.concatenate(out_chunks, axis=0)  # (total, B, out)
        out = out.reshape(ot, oh, ow, B, self.out_channels)
        out = out.transpose(3, 4, 0, 1, 2)  # (B, out, ot, oh, ow)
        out = out + self.bias.reshape(1, -1, 1, 1, 1)
        return out


class CosmosVAEResBlock(nn.Module):
    def __init__(self, channels, temb_channels=None):
        super().__init__()
        self.conv1 = CosmosVAEConv3d(channels, channels, 3, 1, 1)
        self.conv2 = CosmosVAEConv3d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        if temb_channels is not None:
            self.temb_proj = nn.Linear(temb_channels, channels)
        else:
            self.temb_proj = None

    def __call__(self, x, temb=None):
        h = _group_norm_5d(self.norm1, x)
        h = _silu(h)
        h = self.conv1(h)
        if temb is not None and self.temb_proj is not None:
            t = self.temb_proj(_silu(temb))
            if t.ndim == 1:
                t = t[None, :]
            h = h + t[:, :, None, None, None]
        h = _group_norm_5d(self.norm2, h)
        h = _silu(h)
        h = self.conv2(h)
        return x + h


class CosmosVAEDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.res1 = CosmosVAEResBlock(in_ch)
        self.res2 = CosmosVAEResBlock(in_ch)
        self.downsample = downsample
        if downsample:
            self.conv_down = CosmosVAEConv3d(
                in_ch, out_ch, 3, stride=(2, 2, 2), padding=1
            )
        else:
            self.conv_down = CosmosVAEConv3d(in_ch, out_ch, 3, 1, 1)

    def __call__(self, x):
        h = self.res1(x)
        h = self.res2(h)
        return self.conv_down(h)


class CosmosVAEUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.res1 = CosmosVAEResBlock(in_ch)
        self.res2 = CosmosVAEResBlock(in_ch)
        self.upsample = upsample
        if upsample:
            self.conv_up = CosmosVAEConv3d(in_ch, out_ch, 3, 1, 1)
        else:
            self.conv_up = CosmosVAEConv3d(in_ch, out_ch, 3, 1, 1)

    def __call__(self, x):
        h = self.res1(x)
        h = self.res2(h)
        if self.upsample:
            B, C, T, H, W = h.shape
            h = mx.broadcast_to(
                h.reshape(B, C, T, 1, H, 1, W, 1),
                (B, C, T, 2, H, 2, W, 2),
            )
            h = h.reshape(B, C, T * 2, H * 2, W * 2)
        return self.conv_up(h)


class CosmosVideoVAE(nn.Module):
    UP_BLOCKS = [False, True, True, True]

    def __init__(self, latent_channels=16, in_channels=3):
        super().__init__()
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        ch_mult = [1, 2, 4, 4]
        base_ch = 128
        # Encoder
        self.enc_conv_in = CosmosVAEConv3d(in_channels, base_ch, 3, 1, 1)
        enc_blocks = []
        prev_ch = base_ch
        for i, mult in enumerate(ch_mult):
            cur_ch = base_ch * mult
            down = i < len(ch_mult) - 1
            enc_blocks.append(CosmosVAEDownBlock(prev_ch, cur_ch, downsample=down))
            prev_ch = cur_ch
        self.enc_blocks = enc_blocks
        self.enc_mid1 = CosmosVAEResBlock(prev_ch)
        self.enc_mid2 = CosmosVAEResBlock(prev_ch)
        self.enc_conv_out = CosmosVAEConv3d(prev_ch, latent_channels * 2, 3, 1, 1)
        # Decoder
        self.dec_conv_in = CosmosVAEConv3d(latent_channels, prev_ch, 3, 1, 1)
        self.dec_mid1 = CosmosVAEResBlock(prev_ch)
        self.dec_mid2 = CosmosVAEResBlock(prev_ch)
        dec_blocks = []
        for i, mult in reversed(list(enumerate(ch_mult))):
            cur_ch = base_ch * mult
            up = i > 0
            dec_blocks.append(CosmosVAEUpBlock(prev_ch, cur_ch, upsample=up))
            prev_ch = cur_ch
        self.dec_blocks = dec_blocks
        self.dec_conv_out = CosmosVAEConv3d(prev_ch, in_channels, 3, 1, 1)

    def encode(self, x):
        logger.info("cosmos vae encode: input shape=%s", x.shape)
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
        logger.info("cosmos vae encode: latent shape=%s", z.shape)
        return z

    def decode(self, z):
        logger.info("cosmos vae decode: latent shape=%s", z.shape)
        h = self.dec_conv_in(z)
        h = self.dec_mid1(h)
        h = self.dec_mid2(h)
        for block in self.dec_blocks:
            h = block(h)
        h = self.dec_conv_out(h)
        h = mx.clip(h, 0.0, 1.0)
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
            "cosmos vae decode_tiled: latent shape=%s tile=(%d,%d,%d) overlap=(%d,%d,%d)",
            z.shape,
            tile_t,
            tile_h,
            tile_w,
            overlap_t,
            overlap_h,
            overlap_w,
        )
        B, C, T, H, W = z.shape
        need_t = tile_t < T
        need_h = tile_h < H
        need_w = tile_w < W

        if not need_t and not need_h and not need_w:
            logger.info("cosmos vae decode_tiled: no tiling needed, using decode()")
            return self.decode(z)

        out_t, out_h, out_w = self._compute_output_shape(T, H, W)
        C_out = self.in_channels
        output = np.zeros((B, C_out, out_t, out_h, out_w), dtype=np.float32)
        weight_sum = np.zeros((B, C_out, out_t, out_h, out_w), dtype=np.float32)

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
                        "cosmos vae decode_tiled: tile %d/%d slice=[%d:%d,%d:%d,%d:%d] shape=%s",
                        tile_idx,
                        total_tiles,
                        t_start,
                        t_end,
                        h_start,
                        h_end,
                        w_start,
                        w_end,
                        tile_z.shape,
                    )
                    tile_out = self.decode(tile_z)
                    mx.eval(tile_out)
                    tile_np = np.array(tile_out, dtype=np.float32)
                    del tile_out
                    mx.clear_cache()

                    ot_start = t_start * 8
                    ot_end = ot_start + tile_np.shape[2]
                    oh_start = h_start * 8
                    oh_end = oh_start + tile_np.shape[3]
                    ow_start = w_start * 8
                    ow_end = ow_start + tile_np.shape[4]

                    ot_end = min(ot_end, out_t)
                    oh_end = min(oh_end, out_h)
                    ow_end = min(ow_end, out_w)

                    tile_np = tile_np[
                        :,
                        :,
                        : ot_end - ot_start,
                        : oh_end - oh_start,
                        : ow_end - ow_start,
                    ]

                    w_t = self._blend_weights_1d(
                        tile_np.shape[2], t_start, t_end, T, overlap_t * 8
                    )
                    w_h = self._blend_weights_1d(
                        tile_np.shape[3], h_start, h_end, H, overlap_h * 8
                    )
                    w_w = self._blend_weights_1d(
                        tile_np.shape[4], w_start, w_end, W, overlap_w * 8
                    )

                    w_3d = (
                        w_t.reshape(1, 1, -1, 1, 1)
                        * w_h.reshape(1, 1, 1, -1, 1)
                        * w_w.reshape(1, 1, 1, 1, -1)
                    )
                    w_3d = np.broadcast_to(w_3d, (1, C_out) + w_3d.shape[2:]).copy()

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
            "cosmos vae decode_tiled: output shape=%s total_tiles=%d",
            output.shape,
            total_tiles,
        )
        return mx.array(output)

    def _compute_output_shape(self, T, H, W):
        t, h, w = T, H, W
        for up in self.UP_BLOCKS:
            if up:
                h = h * 2
                w = w * 2
                t = t * 2
        return t, h, w

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
            ramp = np.linspace(0, 1, fade_len, dtype=np.float32)
            weights[:fade_len] *= ramp
        if tile_end < dim_size:
            ramp = np.linspace(1, 0, fade_len, dtype=np.float32)
            weights[-fade_len:] *= ramp
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
        nk = nk.replace("time_emb_proj.", "temb_proj.")
        out[nk] = v
    return out
