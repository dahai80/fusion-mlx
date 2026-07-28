# SPDX-License-Identifier: Apache-2.0
# Cosmos continuous VAE for latent video encoding/decoding.
# Latent dim 16, 8x spatial compression, 4x temporal compression.

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _silu(x):
    return x * mx.sigmoid(x)


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
        # x: (B, C, T, H, W) -> manual conv3d via slice matmul
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
        # Unfold patches
        patches = []
        for ti in range(ot):
            for hi in range(oh):
                for wi in range(ow):
                    patch = x[
                        :,
                        :,
                        ti * st : ti * st + kt,
                        hi * sh : hi * sh + kh,
                        wi * sw : wi * sw + kw,
                    ]
                    patches.append(patch.reshape(B, -1))
        if len(patches) == 0:
            return mx.zeros((B, self.out_channels, ot, oh, ow), dtype=x.dtype)
        patches = mx.stack(patches, axis=0)  # (ot*oh*ow, B, C*kt*kh*kw)
        patches = patches.reshape(-1, B, C * kt * kh * kw)  # (L, B, D)
        w = self.weight.reshape(self.out_channels, -1)  # (out, C*kt*kh*kw)
        out = patches @ w.T  # (L, B, out)
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
        h = self.norm1(x)
        h = _silu(h)
        h = self.conv1(h)
        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(_silu(temb))[:, :, None, None, None]
        h = self.norm2(h)
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
                loaded[k] = mapped[k].astype(mx.float16) if mapped[k].dtype != mx.float16 else mapped[k]
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
