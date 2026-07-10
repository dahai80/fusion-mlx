# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 audio VAE encoder/decoder (vendored from mlx-video).
# Phase 4 Stage E: audio_vae port.
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import check_array_shape

from ..config import AudioDecoderModelConfig, AudioEncoderModelConfig, CausalityAxis
from .attention import AttentionType, make_attn
from .causal_conv_2d import make_conv2d
from .downsample import build_downsampling_path
from .normalization import NormType, build_normalization_layer
from .ops import AudioLatentShape, AudioPatchifier, PerChannelStatistics
from .resnet import ResnetBlock
from .upsample import build_upsampling_path

if TYPE_CHECKING:
    from .vocoder import Vocoder

logger = logging.getLogger(__name__)

LATENT_DOWNSAMPLE_FACTOR = 4


def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> dict:
    mid = {}
    mid["block_1"] = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid["attn_1"] = (
        make_attn(channels, attn_type=attn_type, norm_type=norm_type)
        if add_attention
        else None
    )
    mid["block_2"] = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: dict, features: mx.array) -> mx.array:
    features = mid["block_1"](features, temb=None)
    if mid["attn_1"] is not None:
        features = mid["attn_1"](features)
    return mid["block_2"](features, temb=None)


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderModelConfig) -> None:
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=config.ch)
        self.sample_rate = config.sample_rate
        self.mel_hop_length = config.mel_hop_length
        self.is_causal = config.is_causal
        self.mel_bins = config.mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=config.sample_rate,
            hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
        )

        self.ch = config.ch
        self.temb_ch = 0
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.z_channels = config.z_channels
        self.double_z = config.double_z
        self.norm_type = config.norm_type
        self.causality_axis = config.causality_axis
        self.attn_type = config.attn_type

        self.conv_in = make_conv2d(
            config.in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.down, block_in = build_downsampling_path(
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=config.num_res_blocks,
            resolution=config.resolution,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=config.attn_resolutions or set(),
            resamp_with_conv=config.resamp_with_conv,
        )

        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=config.mid_block_add_attention,
        )

        self.norm_out = build_normalization_layer(block_in, normtype=self.norm_type)
        out_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = make_conv2d(
            block_in,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            new_key = key
            if key.startswith("audio_vae.encoder."):
                new_key = key.replace("audio_vae.encoder.", "")
            elif key.startswith("encoder."):
                new_key = key.replace("encoder.", "")
            elif key.startswith("audio_vae.per_channel_statistics."):
                if "mean-of-means" in key:
                    new_key = "per_channel_statistics.mean_of_means"
                elif "std-of-means" in key:
                    new_key = "per_channel_statistics.std_of_means"
                else:
                    continue
            elif "per_channel_statistics" in key:
                if "mean-of-means" in key or "latents_mean" in key:
                    new_key = "per_channel_statistics.mean_of_means"
                elif "std-of-means" in key or "latents_std" in key:
                    new_key = "per_channel_statistics.std_of_means"
                else:
                    continue
            elif key == "latents_mean":
                new_key = "per_channel_statistics.mean_of_means"
            elif key == "latents_std":
                new_key = "per_channel_statistics.std_of_means"
            else:
                continue

            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
                value = (
                    value
                    if check_array_shape(value)
                    else mx.transpose(value, (0, 2, 3, 1))
                )

            sanitized[new_key] = value
        return sanitized

    @classmethod
    def from_pretrained(cls, model_path: Path) -> "AudioEncoder":
        model_path = Path(model_path)
        config = AudioEncoderModelConfig.from_dict(
            json.load(open(model_path / "config.json"))
        )
        encoder = cls(config)
        weights = mx.load(str(model_path / "model.safetensors"))
        encoder.load_weights(list(weights.items()), strict=True)
        logger.info(
            "AudioEncoder loaded from %s (%d weights)", model_path, len(weights)
        )
        return encoder

    def __call__(self, spectrogram: mx.array) -> mx.array:
        if spectrogram.ndim == 4 and spectrogram.shape[1] == self.in_channels:
            spectrogram = mx.transpose(spectrogram, (0, 2, 3, 1))

        h = self.conv_in(spectrogram)
        h = self._run_downsampling_path(h)
        h = run_mid_block(self.mid, h)
        h = self._finalize_output(h)
        return self._normalize_latents(h)

    def _run_downsampling_path(self, h: mx.array) -> mx.array:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage["block"][block_idx](h, temb=None)
                if block_idx in stage["attn"]:
                    h = stage["attn"][block_idx](h)
            if level != self.num_resolutions - 1 and "downsample" in stage:
                h = stage["downsample"](h)
        return h

    def _finalize_output(self, h: mx.array) -> mx.array:
        h = self.norm_out(h)
        h = nn.silu(h)
        return self.conv_out(h)

    def _normalize_latents(self, h: mx.array) -> mx.array:
        z_channels = self.z_channels
        means = h[..., :z_channels]

        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[3],
            frames=means.shape[1],
            mel_bins=means.shape[2],
        )

        patched = self.patchifier.patchify(means)
        normalized = self.per_channel_statistics.normalize(patched)
        return self.patchifier.unpatchify(normalized, latent_shape)


class AudioDecoder(nn.Module):
    def __init__(
        self,
        config: AudioDecoderModelConfig,
    ) -> None:
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=config.ch)
        self.sample_rate = config.sample_rate
        self.mel_hop_length = config.mel_hop_length
        self.is_causal = config.is_causal
        self.mel_bins = config.mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=config.sample_rate,
            hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
        )

        self.ch = config.ch
        self.temb_ch = 0
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.out_ch = config.out_ch
        self.give_pre_end = config.give_pre_end
        self.tanh_out = config.tanh_out
        self.norm_type = config.norm_type
        self.z_channels = config.z_channels
        self.channel_multipliers = config.ch_mult
        self.attn_resolutions = config.attn_resolutions
        self.causality_axis = config.causality_axis
        self.attn_type = config.attn_type

        base_block_channels = config.ch * self.channel_multipliers[-1]
        base_resolution = config.resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, config.z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            config.z_channels,
            base_block_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=config.mid_block_add_attention,
        )

        self.up, final_block_channels = build_upsampling_path(
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=config.num_res_blocks,
            resolution=config.resolution,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=config.attn_resolutions,
            resamp_with_conv=config.resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        self.norm_out = build_normalization_layer(
            final_block_channels, normtype=self.norm_type
        )
        self.conv_out = make_conv2d(
            final_block_channels,
            config.out_ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}

        for key, value in weights.items():
            new_key = key

            if key.startswith("audio_vae.decoder."):
                new_key = key.replace("audio_vae.decoder.", "")
            elif key.startswith("audio_vae.per_channel_statistics."):
                if "mean-of-means" in key:
                    new_key = "per_channel_statistics.mean_of_means"
                elif "std-of-means" in key:
                    new_key = "per_channel_statistics.std_of_means"
                else:
                    continue
            else:
                continue

            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
                value = (
                    value
                    if check_array_shape(value)
                    else mx.transpose(value, (0, 2, 3, 1))
                )

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def from_pretrained(cls, model_path: Path) -> "AudioDecoder":
        config = AudioDecoderModelConfig.from_dict(
            json.load(open(model_path / "config.json"))
        )
        decoder = cls(config)
        weights = mx.load(str(model_path / "model.safetensors"))
        decoder.load_weights(list(weights.items()), strict=True)
        logger.info(
            "AudioDecoder loaded from %s (%d weights)", model_path, len(weights)
        )
        return decoder

    def __call__(self, sample: mx.array) -> mx.array:
        if sample.shape[1] == self.z_channels and sample.ndim == 4:
            sample = mx.transpose(sample, (0, 2, 3, 1))

        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(
        self, sample: mx.array
    ) -> tuple[mx.array, AudioLatentShape]:
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[3],
            frames=sample.shape[1],
            mel_bins=sample.shape[2],
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=(
                self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins
            ),
        )

        return sample, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: mx.array,
        target_shape: AudioLatentShape,
    ) -> mx.array:
        _, current_time, current_freq, _ = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        decoded_output = decoded_output[
            :,
            : min(current_time, target_time),
            : min(current_freq, target_freq),
            :target_channels,
        ]

        time_padding_needed = target_time - decoded_output.shape[1]
        freq_padding_needed = target_freq - decoded_output.shape[2]

        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = [
                (0, 0),
                (0, max(time_padding_needed, 0)),
                (0, max(freq_padding_needed, 0)),
                (0, 0),
            ]
            decoded_output = mx.pad(decoded_output, padding)

        decoded_output = decoded_output[:, :target_time, :target_freq, :target_channels]

        decoded_output = mx.transpose(decoded_output, (0, 3, 1, 2))

        return decoded_output

    def _run_upsampling_path(self, h: mx.array) -> mx.array:
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx in range(len(stage["block"])):
                h = stage["block"][block_idx](h, temb=None)
                if block_idx in stage["attn"]:
                    h = stage["attn"][block_idx](h)

            if level != 0 and "upsample" in stage:
                h = stage["upsample"](h)

        return h

    def _finalize_output(self, h: mx.array) -> mx.array:
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return mx.tanh(h) if self.tanh_out else h


def decode_audio(
    latent: mx.array, audio_decoder: AudioDecoder, vocoder: "Vocoder"
) -> mx.array:
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio)
    if decoded_audio.shape[0] == 1:
        decoded_audio = decoded_audio[0]
    return decoded_audio.astype(mx.float32)
