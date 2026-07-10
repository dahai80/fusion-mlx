# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 vocoder (vendored from mlx-video).
# Supports HiFi-GAN (resblock="1") and BigVGAN v2 (resblock="AMP1") + BWE.
# Phase 4 Stage E: audio_vae port.
import json
import logging
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..config import VocoderModelConfig
from .resnet import LRELU_SLOPE, ResBlock1, ResBlock2, leaky_relu

logger = logging.getLogger(__name__)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    def __init__(self, in_features: int, alpha_logscale: bool = True) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = (
            mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))
        )

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        return x + (1.0 / (alpha + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha_logscale: bool = True) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = (
            mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))
        )
        self.beta = (
            mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))
        )

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha
        beta = self.beta
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


def _sinc(x: mx.array) -> mx.array:
    return mx.where(
        x == 0,
        mx.ones_like(x),
        mx.sin(mx.array(math.pi) * x) / (mx.array(math.pi) * x),
    )


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> mx.array:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0

    window = mx.array(np.kaiser(kernel_size, beta).astype(np.float32))

    if even:
        time = mx.arange(-half_size, half_size).astype(mx.float32) + 0.5
    else:
        time = mx.arange(kernel_size).astype(mx.float32) - half_size

    if cutoff == 0:
        filter_ = mx.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ = filter_ / mx.sum(filter_)

    return filter_.reshape(1, 1, kernel_size)


def hann_sinc_filter1d(ratio: int) -> tuple[mx.array, int, int, int]:
    rolloff = 0.99
    lowpass_filter_width = 6
    width = math.ceil(lowpass_filter_width / rolloff)
    kernel_size = 2 * width * ratio + 1
    pad = width
    pad_left = 2 * width * ratio
    pad_right = kernel_size - ratio

    time = (np.arange(kernel_size) / ratio - width) * rolloff
    time_clamped = np.clip(time, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
    sinc_filter = np.sinc(time) * window * rolloff / ratio

    filter_ = mx.array(sinc_filter.astype(np.float32)).reshape(1, 1, kernel_size)
    return filter_, pad, pad_left, pad_right


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        n, l, c = x.shape

        first = mx.repeat(x[:, :1, :], self.pad_left, axis=1)
        last = mx.repeat(x[:, -1:, :], self.pad_right, axis=1)
        x = mx.concatenate([first, x, last], axis=1)

        filt = self.filter.astype(x.dtype)
        filt = mx.transpose(filt, (0, 2, 1))
        filt = mx.repeat(filt, c, axis=0)

        x = mx.transpose(x, (0, 2, 1))
        x = x.reshape(n * c, -1, 1)

        x = mx.conv1d(x, filt[:1], stride=self.stride, groups=1)

        x = x.reshape(n, c, -1)
        x = mx.transpose(x, (0, 2, 1))
        return x


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        window_type: str = "kaiser",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            filt, self.pad, self.pad_left, self.pad_right = hann_sinc_filter1d(ratio)
            self.kernel_size = filt.shape[2]
            self.filter = filt
        else:
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
            self.filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

    def __call__(self, x: mx.array) -> mx.array:
        n, l, c = x.shape

        first = mx.repeat(x[:, :1, :], self.pad, axis=1)
        last = mx.repeat(x[:, -1:, :], self.pad, axis=1)
        x = mx.concatenate([first, x, last], axis=1)

        x = mx.transpose(x, (0, 2, 1))
        x = x.reshape(n * c, -1, 1)

        filt = self.filter.astype(x.dtype)
        filt = mx.transpose(filt, (0, 2, 1))

        x = self.ratio * mx.conv_transpose1d(x, filt, stride=self.stride)

        x = x[:, self.pad_left : -self.pad_right, :]

        x = x.reshape(n, c, -1)
        x = mx.transpose(x, (0, 2, 1))
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class AMPBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        activation: str = "snakebeta",
    ) -> None:
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake

        self.convs1 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for i, d in enumerate(dilation)
        }

        self.convs2 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for i in range(len(dilation))
        }

        self.acts1 = {i: Activation1d(act_cls(channels)) for i in range(len(dilation))}
        self.acts2 = {i: Activation1d(act_cls(channels)) for i in range(len(dilation))}

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(len(self.convs1)):
            xt = self.acts1[i](x)
            xt = self.convs1[i](xt)
            xt = self.acts2[i](xt)
            xt = self.convs2[i](xt)
            x = x + xt
        return x


class STFTFn(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.forward_basis = mx.zeros((n_freqs * 2, 1, filter_length))
        self.inverse_basis = mx.zeros((n_freqs * 2, 1, filter_length))

    def __call__(self, y: mx.array) -> tuple[mx.array, mx.array]:
        if y.ndim == 2:
            y = mx.expand_dims(y, -1)

        left_pad = max(0, self.win_length - self.hop_length)
        if left_pad > 0:
            first = mx.repeat(y[:, :1, :], left_pad, axis=1)
            y = mx.concatenate([first, y], axis=1)

        basis = mx.transpose(self.forward_basis.astype(y.dtype), (0, 2, 1))

        spec = mx.conv1d(y, basis, stride=self.hop_length)

        n_freqs = spec.shape[-1] // 2
        real = spec[..., :n_freqs]
        imag = spec[..., n_freqs:]

        magnitude = mx.sqrt(real**2 + imag**2)
        phase = mx.arctan2(imag.astype(mx.float32), real.astype(mx.float32)).astype(
            real.dtype
        )

        return magnitude, phase


class MelSTFT(nn.Module):
    def __init__(
        self, filter_length: int, hop_length: int, win_length: int, n_mel_channels: int
    ) -> None:
        super().__init__()
        self.stft_fn = STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel_channels, n_freqs))

    def mel_spectrogram(self, y: mx.array) -> mx.array:
        magnitude, phase = self.stft_fn(y)
        mel = magnitude @ self.mel_basis.astype(magnitude.dtype).T
        log_mel = mx.log(mx.clip(mel, 1e-5, None))
        return mx.transpose(log_mel, (0, 2, 1))


class Vocoder(nn.Module):
    def __init__(self, config: VocoderModelConfig) -> None:
        super().__init__()

        self.output_sampling_rate = config.output_sample_rate
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.upsample_rates = config.upsample_rates
        self.is_amp = config.resblock == "AMP1"
        self.use_tanh_at_final = config.use_tanh_at_final
        self.apply_final_activation = config.apply_final_activation

        in_channels = 128 if config.stereo else 64
        self.conv_pre = nn.Conv1d(
            in_channels,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.ups = {}
        for i, (stride, kernel_size) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            in_ch = config.upsample_initial_channel // (2**i)
            out_ch = config.upsample_initial_channel // (2 ** (i + 1))
            self.ups[i] = nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )

        if self.is_amp:
            self.resblocks = {}
            block_idx = 0
            for i in range(len(self.ups)):
                ch = config.upsample_initial_channel // (2 ** (i + 1))
                for kernel_size, dilations in zip(
                    config.resblock_kernel_sizes, config.resblock_dilation_sizes
                ):
                    self.resblocks[block_idx] = AMPBlock1(
                        ch,
                        kernel_size,
                        tuple(dilations),
                        activation=config.activation,
                    )
                    block_idx += 1
        else:
            resblock_class = ResBlock1 if config.resblock == "1" else ResBlock2
            self.resblocks = {}
            block_idx = 0
            for i in range(len(self.ups)):
                ch = config.upsample_initial_channel // (2 ** (i + 1))
                for kernel_size, dilations in zip(
                    config.resblock_kernel_sizes, config.resblock_dilation_sizes
                ):
                    self.resblocks[block_idx] = resblock_class(
                        ch, kernel_size, tuple(dilations)
                    )
                    block_idx += 1

        final_channels = config.upsample_initial_channel // (
            2 ** len(config.upsample_rates)
        )

        if self.is_amp:
            act_cls = SnakeBeta if config.activation == "snakebeta" else Snake
            self.act_post = Activation1d(act_cls(final_channels))

        out_channels = 2 if config.stereo else 1
        self.conv_post = nn.Conv1d(
            final_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=config.use_bias_at_final,
        )

        self.upsample_factor = math.prod(config.upsample_rates)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 1, 3, 2))

        if x.ndim == 4:
            b, s, c, t = x.shape
            x = x.reshape(b, s * c, t)

        x = mx.transpose(x, (0, 2, 1))

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            if not self.is_amp:
                x = leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            start = i * self.num_kernels
            end = start + self.num_kernels

            block_outputs = mx.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                axis=0,
            )
            x = mx.mean(block_outputs, axis=0)

        if self.is_amp:
            x = self.act_post(x)
        else:
            x = nn.leaky_relu(x)

        x = self.conv_post(x)

        if self.apply_final_activation:
            x = mx.tanh(x) if self.use_tanh_at_final else mx.clip(x, -1, 1)

        x = mx.transpose(x, (0, 2, 1))
        return x


class VocoderWithBWE(nn.Module):
    def __init__(
        self,
        vocoder: Vocoder,
        bwe_generator: Vocoder,
        mel_stft: MelSTFT,
        input_sampling_rate: int = 16000,
        output_sampling_rate: int = 48000,
        hop_length: int = 80,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.resampler = UpSample1d(
            ratio=output_sampling_rate // input_sampling_rate,
            window_type="hann",
        )

    @property
    def output_sample_rate(self) -> int:
        return self.output_sampling_rate

    def _compute_mel(self, audio: mx.array) -> mx.array:
        batch, n_channels, _ = audio.shape
        flat = audio.reshape(batch * n_channels, -1)
        mel = self.mel_stft.mel_spectrogram(flat)
        return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])

    def __call__(self, mel_spec: mx.array) -> mx.array:
        x = self.vocoder(mel_spec)
        _, _, length_low_rate = x.shape
        output_length = (
            length_low_rate * self.output_sampling_rate // self.input_sampling_rate
        )

        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            pad_amount = self.hop_length - remainder
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_amount)])

        mel = self._compute_mel(x)

        mel_for_bwe = mx.transpose(mel, (0, 1, 3, 2))
        residual = self.bwe_generator(mel_for_bwe)

        x_for_resample = mx.transpose(x, (0, 2, 1))
        skip = self.resampler(x_for_resample)
        skip = mx.transpose(skip, (0, 2, 1))

        return mx.clip(residual + skip, -1, 1)[..., :output_length]


def load_vocoder(model_path: Path) -> nn.Module:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    weights = mx.load(str(model_path / "model.safetensors"))

    has_bwe = config_dict.get("has_bwe_generator", False)

    if has_bwe:
        logger.info("load_vocoder: VocoderWithBWE from %s", model_path)
        return _load_vocoder_with_bwe(config_dict, weights)
    else:
        config = VocoderModelConfig.from_dict(config_dict)
        model = Vocoder(config)
        model.load_weights(list(weights.items()), strict=True)
        logger.info(
            "load_vocoder: Vocoder from %s (%d weights)", model_path, len(weights)
        )
        return model


def _load_vocoder_with_bwe(config_dict: dict, weights: dict) -> VocoderWithBWE:
    vocoder_cfg = config_dict.get("vocoder", {})
    vocoder_config = VocoderModelConfig.from_dict(vocoder_cfg)
    vocoder = Vocoder(vocoder_config)

    bwe_cfg = config_dict.get("bwe", {})
    bwe_config = VocoderModelConfig.from_dict(bwe_cfg)
    bwe_config.apply_final_activation = False
    bwe_generator = Vocoder(bwe_config)

    stft_basis = weights.get("mel_stft.stft_fn.forward_basis")
    filter_length = stft_basis.shape[2] if stft_basis is not None else 512
    mel_basis = weights.get("mel_stft.mel_basis")
    n_mel_channels = mel_basis.shape[0] if mel_basis is not None else 64

    hop_length = bwe_cfg.get("hop_length", 80)
    input_sr = bwe_cfg.get("input_sampling_rate", 16000)
    output_sr = bwe_cfg.get("output_sampling_rate", 48000)

    mel_stft = MelSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=filter_length,
        n_mel_channels=n_mel_channels,
    )

    model = VocoderWithBWE(
        vocoder=vocoder,
        bwe_generator=bwe_generator,
        mel_stft=mel_stft,
        input_sampling_rate=input_sr,
        output_sampling_rate=output_sr,
        hop_length=hop_length,
    )

    model.load_weights(list(weights.items()), strict=False)
    return model
