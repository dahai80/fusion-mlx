# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX Oobleck audio VAE (issue #988): encode waveform -> latent,
# decode latent -> waveform. Ported from diffusers AutoencoderOobleck (Apache-2.0).
# Snake1d + weight-norm Conv1d/ConvTranspose1d + residual units.
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def conv1d(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
) -> mx.array:
    # x: (B, Cin, T). weight: (Cout, Cin, K). im2col.
    B, Cin, T = x.shape
    Cout, _, K = weight.shape
    if padding > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (padding, padding)], constant_values=0.0)
        T = x.shape[2]
    if T < K:
        out_T = 0
    else:
        out_T = (T - (dilation * (K - 1) + 1)) // stride + 1
    if out_T <= 0:
        return mx.zeros((B, Cout, 0), dtype=x.dtype) + (
            bias[:, None] if bias is not None else 0.0
        )
    # patch indices: (K, out_T) — base offsets gather dilated
    base = mx.arange(K)[:, None] * dilation  # (K, 1)
    shifts = mx.arange(out_T)[None, :] * stride  # (1, out_T)
    idxs = base + shifts  # (K, out_T)
    patches = x[:, :, idxs]  # (B, Cin, K, out_T)
    w_flat = mx.reshape(weight, (Cout, Cin * K))  # (Cout, Cin*K)
    patches_flat = mx.reshape(
        mx.swapaxes(patches, 1, 2), (B, out_T, Cin * K)
    )  # (B, out_T, Cin*K)
    out = mx.matmul(patches_flat, w_flat.T)  # (B, out_T, Cout)
    out = mx.swapaxes(out, 1, 2)  # (B, Cout, out_T)
    if bias is not None:
        out = out + bias[:, None]
    return out


def conv_transpose1d(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
) -> mx.array:
    # x: (B, Cin, T). weight: (Cin, Cout, K). Transpose conv with overlap.
    # out_T = (T-1)*stride + K (before padding crop).
    B, Cin, T = x.shape
    _, Cout, K = weight.shape
    out_T = (T - 1) * stride + K
    out = mx.zeros((B, Cout, out_T), dtype=x.dtype)
    for k in range(K):
        # contribution of kernel tap k: weight[:,:,k] is (Cin, Cout).
        wk = weight[:, :, k]  # (Cin, Cout)
        contrib = mx.einsum("bct,co->bot", x, wk)  # (B, Cout, T)
        pad_before = k
        pad_after = out_T - (T + k)
        c = contrib
        if pad_after < 0:
            c = contrib[:, :, : out_T - k]
            pad_after = 0
        padded = mx.pad(
            c, [(0, 0), (0, 0), (pad_before, pad_after)], constant_values=0.0
        )
        out = out + padded
    if padding > 0 and padding * 2 < out_T:
        out = out[:, :, padding : out_T - padding]
    if bias is not None:
        out = out + bias[:, None]
    return out


def wn_weight(weight_g: mx.array, weight_v: mx.array) -> mx.array:
    # Reconstruct weight-normed weight: g * v / ||v|| per output channel.
    # weight_g: (Cout, 1, 1). weight_v: (Cout, Cin, K).
    Cout = weight_v.shape[0]
    v_flat = mx.reshape(weight_v, (Cout, -1))  # (Cout, Cin*K)
    norm = mx.sqrt(mx.sum(v_flat * v_flat, axis=1, keepdims=True) + 1e-12)  # (Cout, 1)
    g = mx.reshape(weight_g, (Cout, 1))
    w = (g / norm) * v_flat  # (Cout, Cin*K)
    return mx.reshape(w, weight_v.shape)


class Snake1d(nn.Module):
    # x + (1/beta) * sin(alpha*x)^2. alpha/beta shape (1, C, 1).
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))
        self.beta = mx.ones((1, channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        return x + (1.0 / (self.beta + 1e-9)) * mx.sin(self.alpha * x) ** 2


class WNConv1d(nn.Module):
    # Weight-normed Conv1d. Stores weight_g/weight_v + bias.
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.weight_g = mx.ones((out_ch, 1, 1))
        self.weight_v = mx.random.normal((out_ch, in_ch, kernel_size))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        w = wn_weight(self.weight_g, self.weight_v)
        return conv1d(x, w, self.bias, self.stride, self.dilation, self.padding)


class WNConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.weight_g = mx.ones((in_ch, 1, 1))
        self.weight_v = mx.random.normal((in_ch, out_ch, kernel_size))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        w = wn_weight(self.weight_g, self.weight_v)
        return conv_transpose1d(x, w, self.bias, self.stride, 1, self.padding)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = WNConv1d(dim, dim, 7, dilation=dilation, padding=pad)
        self.snake2 = Snake1d(dim)
        self.conv2 = WNConv1d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(self.snake1(x))
        h = self.conv2(self.snake2(h))
        # center-crop x to match h length
        T = x.shape[-1]
        Th = h.shape[-1]
        if Th < T:
            pad = (T - Th) // 2
            x = x[:, :, pad : pad + Th]
        return x + h


class EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.res_unit1 = ResidualUnit(in_dim, dilation=1)
        self.res_unit2 = ResidualUnit(in_dim, dilation=3)
        self.res_unit3 = ResidualUnit(in_dim, dilation=9)
        self.snake1 = Snake1d(in_dim)
        self.conv1 = WNConv1d(
            in_dim, out_dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.snake1(self.res_unit3(x))
        x = self.conv1(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.snake1 = Snake1d(in_dim)
        self.conv_t1 = WNConvTranspose1d(
            in_dim, out_dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )
        self.res_unit1 = ResidualUnit(out_dim, dilation=1)
        self.res_unit2 = ResidualUnit(out_dim, dilation=3)
        self.res_unit3 = ResidualUnit(out_dim, dilation=9)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        audio_channels: int,
        downsampling_ratios: list,
        channel_multiples: list,
    ):
        super().__init__()
        strides = downsampling_ratios
        cm = [1] + channel_multiples
        self.conv1 = WNConv1d(audio_channels, encoder_hidden_size, 7, padding=3)
        self.block = [
            EncoderBlock(
                encoder_hidden_size * cm[i], encoder_hidden_size * cm[i + 1], strides[i]
            )
            for i in range(len(strides))
        ]
        d_model = encoder_hidden_size * cm[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = WNConv1d(d_model, encoder_hidden_size, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        for b in self.block:
            x = b(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list,
        channel_multiples: list,
    ):
        super().__init__()
        strides = upsampling_ratios
        cm = [1] + channel_multiples
        self.conv1 = WNConv1d(input_channels, channels * cm[-1], 7, padding=3)
        self.block = [
            DecoderBlock(
                channels * cm[len(strides) - i],
                channels * cm[len(strides) - i - 1],
                strides[i],
            )
            for i in range(len(strides))
        ]
        self.snake1 = Snake1d(channels)
        self.conv2 = WNConv1d(channels, audio_channels, 7, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        for b in self.block:
            x = b(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


class AutoencoderOobleckMLX(nn.Module):
    # MLX port of AutoencoderOobleck. encode: waveform (B, audio_ch, T) ->
    # latent mean/scale (B, latent_ch=encoder_hidden, T/hop). decode: latent
    # (B, decoder_input_channels, T/hop) -> waveform (B, audio_ch, T*hop).
    def __init__(self, config: dict):
        super().__init__()
        enc_h = config["encoder_hidden_size"]
        ds = config["downsampling_ratios"]
        cm = config["channel_multiples"]
        dc = config["decoder_channels"]
        dic = config["decoder_input_channels"]
        ac = config["audio_channels"]
        self.encoder_hidden_size = enc_h
        self.downsampling_ratios = ds
        self.upsampling_ratios = ds[::-1]
        self.hop_length = int(math.prod(ds))
        self.sampling_rate = config["sampling_rate"]
        self.audio_channels = ac
        self.encoder = OobleckEncoder(enc_h, ac, ds, cm)
        self.decoder = OobleckDecoder(dc, dic, ac, self.upsampling_ratios, cm)

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (B, audio_ch, T). returns (mean, std) each (B, enc_h, T/hop).
        h = self.encoder(x)  # (B, enc_h, T/hop)
        mean, scale = mx.split(h, 2, axis=1)
        std = mx.logaddexp(scale, mx.zeros_like(scale)) + 1e-4  # softplus
        return mean, std

    def decode(self, z: mx.array) -> mx.array:
        # z: (B, decoder_input_channels, T/hop) -> (B, audio_ch, T)
        return self.decoder(z)

    @classmethod
    def from_pretrained(cls, vae_dir: str | Path) -> AutoencoderOobleckMLX:
        vae_dir = Path(vae_dir)
        cfg = json.loads((vae_dir / "config.json").read_text())
        model = cls(cfg)
        weights = mx.load(str(vae_dir / "diffusion_pytorch_model.safetensors"))
        pairs = []
        for k, arr in weights.items():
            pairs.append((k, arr))
        model.load_weights(pairs, strict=False)
        mx.eval(model.parameters())
        logger.info(
            "oobleck VAE loaded: %d tensors, hop=%d sr=%d",
            len(weights),
            model.hop_length,
            model.sampling_rate,
        )
        return model


__all__ = ["AutoencoderOobleckMLX", "conv1d", "conv_transpose1d"]
