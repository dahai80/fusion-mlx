# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE ops (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AudioLatentShape:
    batch: int
    channels: int
    frames: int
    mel_bins: int


class PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.std_of_means = mx.ones((latent_channels,))
        self.mean_of_means = mx.zeros((latent_channels,))

    def un_normalize(self, x: mx.array) -> mx.array:
        std = self.std_of_means.astype(x.dtype)
        mean = self.mean_of_means.astype(x.dtype)
        return (x * std) + mean

    def normalize(self, x: mx.array) -> mx.array:
        std = self.std_of_means.astype(x.dtype)
        mean = self.mean_of_means.astype(x.dtype)
        return (x - mean) / std


class AudioPatchifier:
    def __init__(
        self,
        patch_size: int = 1,
        audio_latent_downsample_factor: int = 4,
        sample_rate: int = 16000,
        hop_length: int = 160,
        is_causal: bool = True,
    ):
        self.patch_size = patch_size
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.is_causal = is_causal

    def patchify(self, x: mx.array) -> mx.array:
        b, t, f, c = x.shape
        x = mx.transpose(x, (0, 1, 3, 2))
        return x.reshape(b, t, c * f)

    def unpatchify(self, x: mx.array, latent_shape: AudioLatentShape) -> mx.array:
        b, t, cf = x.shape
        c = latent_shape.channels
        f = latent_shape.mel_bins
        x = x.reshape(b, t, c, f)
        return mx.transpose(x, (0, 1, 3, 2))
