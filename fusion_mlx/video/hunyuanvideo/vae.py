# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo causal 3D VAE.
# Latent dim 16, 8x spatial, 4x temporal compression.

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _silu(x):
    return x * mx.sigmoid(x)


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
        # Causal: only pad spatial dims, temporal uses causal padding
        pt, ph, pw = self.padding
        # Causal temporal padding: only pad left (before)
        if pt > 0 or ph > 0 or pw > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pt, 0), (ph, ph), (pw, pw)])
        Tp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        ot = (Tp - kt) // st + 1
        oh = (Hp - kh) // sh + 1
        ow = (Wp - kw) // sw + 1
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
        patches = mx.stack(patches, axis=0)
        patches = patches.reshape(-1, B, C * kt * kh * kw)
        w = self.weight.reshape(self.out_channels, -1)
        out = patches @ w.T
        out = out.reshape(ot, oh, ow, B, self.out_channels)
        out = out.transpose(3, 4, 0, 1, 2)
        out = out + self.bias.reshape(1, -1, 1, 1, 1)
        return out


class HVResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = CausalConv3d(channels, channels, 3, 1, 1)
        self.conv2 = CausalConv3d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)

    def __call__(self, x):
        h = self.norm1(x)
        h = _silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _silu(h)
        h = self.conv2(h)
        return x + h


class HVDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_conv_down=True):
        super().__init__()
        self.res1 = HVResBlock(in_ch)
        self.res2 = HVResBlock(in_ch)
        self.use_conv_down = use_conv_down
        if use_conv_down:
            self.conv_down = CausalConv3d(in_ch, out_ch, 3, stride=(2, 2, 2), padding=1)
        else:
            self.conv_resample = CausalConv3d(in_ch, out_ch, 3, 1, 1)

    def __call__(self, x):
        h = self.res1(x)
        h = self.res2(h)
        if self.use_conv_down:
            return self.conv_down(h)
        return self.conv_resample(h)


class HVUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_conv_up=True):
        super().__init__()
        self.res1 = HVResBlock(in_ch)
        self.res2 = HVResBlock(in_ch)
        self.use_conv_up = use_conv_up
        if use_conv_up:
            self.conv_up = CausalConv3d(in_ch, out_ch, 3, 1, 1)
        else:
            self.conv_resample = CausalConv3d(in_ch, out_ch, 3, 1, 1)

    def __call__(self, x):
        h = self.res1(x)
        h = self.res2(h)
        if self.use_conv_up:
            B, C, T, H, W = h.shape
            h = mx.broadcast_to(
                h.reshape(B, C, T, 1, H, 1, W, 1),
                (B, C, T, 2, H, 2, W, 2),
            )
            h = h.reshape(B, C, T * 2, H * 2, W * 2)
        return self.conv_up(h)


class HunyuanVideoVAE(nn.Module):
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
            down = i < len(ch_mult) - 1
            enc_blocks.append(HVDownBlock(prev_ch, cur_ch, use_conv_down=down))
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
            up = i > 0
            dec_blocks.append(HVUpBlock(prev_ch, cur_ch, use_conv_up=up))
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
            from safetensors import safe_open

            with safe_open(sf, framework="mlx") as f:
                for key in f.keys():
                    all_params[key] = f.get_tensor(key)
        mapped = _remap_vae_weights(all_params)
        flat = tree_flatten(vae.parameters())
        loaded = {}
        for k, v in flat:
            if k in mapped:
                loaded[k] = mapped[k]
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
        nk = nk.replace("norm1.", "norm1.")
        nk = nk.replace("norm2.", "norm2.")
        nk = nk.replace("conv1.", "conv1.")
        nk = nk.replace("conv2.", "conv2.")
        out[nk] = v
    return out
