import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..config import VideoEncoderModelConfig
from .convolution import CausalConv3d, PaddingModeType
from .ops import PerChannelStatistics, PixelNorm, patchify, unpatchify
from .resnet import NormLayerType, ResnetBlock3D, UNetMidBlock3D
from .sampling import DepthToSpaceUpsample, SpaceToDepthDownsample

logger = logging.getLogger(__name__)


class LogVarianceType(Enum):
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> tuple[nn.Module, int]:
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Unknown encoder block: {block_name}")

    return block, out_channels


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> tuple[nn.Module, int]:
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Unknown decoder block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(self, config: "VideoEncoderModelConfig"):
        super().__init__()

        self.patch_size = config.patch_size
        self.norm_layer = config.norm_layer
        self.latent_channels = config.out_channels
        self.latent_log_var = config.latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        encoder_blocks = config.encoder_blocks if config.encoder_blocks else []
        encoder_spatial_padding_mode = config.encoder_spatial_padding_mode

        self.per_channel_statistics = PerChannelStatistics(
            latent_channels=config.out_channels
        )

        in_channels = config.in_channels * config.patch_size**2
        feature_channels = config.out_channels

        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = {}
        for idx, (block_name, block_params) in enumerate(encoder_blocks):
            block_config = (
                {"num_layers": block_params}
                if isinstance(block_params, int)
                else block_params
            )

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=config.convolution_dimensions,
                norm_layer=config.norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )
            self.down_blocks[idx] = block

        if config.norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_groups=self._norm_num_groups,
                dims=feature_channels,
                eps=1e-6,
            )
        elif config.norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = config.out_channels
        if config.latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif config.latent_log_var in {
            LogVarianceType.UNIFORM,
            LogVarianceType.CONSTANT,
        }:
            conv_out_channels += 1

        self.conv_out = CausalConv3d(
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def __call__(self, sample: mx.array) -> mx.array:
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                f"(e.g., 1, 9, 17, ...). Got {frames_count} frames."
            )

        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample, causal=True)

        for i in range(len(self.down_blocks)):
            down_block = self.down_blocks[i]
            if isinstance(down_block, (UNetMidBlock3D, ResnetBlock3D)):
                sample = down_block(sample, causal=True)
            else:
                sample = down_block(sample, causal=True)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=True)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            means = sample[:, :-1, ...]
            logvar = sample[:, -1:, ...]
            num_channels = means.shape[1]
            repeated_logvar = mx.tile(logvar, (1, num_channels, 1, 1, 1))
            sample = mx.concatenate([means, repeated_logvar], axis=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = mx.concatenate(
                [
                    sample,
                    mx.full_like(sample, approx_ln_0),
                ],
                axis=1,
            )

        means = sample[:, : self.latent_channels, ...]
        return self.per_channel_statistics.normalize(means)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        if "per_channel_statistics.mean" in weights:
            return weights

        for key, value in weights.items():
            new_key = key

            if "position_ids" in key:
                continue

            if not key.startswith("vae."):
                continue

            if "vae.per_channel_statistics" in key:
                if key == "vae.per_channel_statistics.mean-of-means":
                    new_key = "per_channel_statistics.mean"
                elif key == "vae.per_channel_statistics.std-of-means":
                    new_key = "per_channel_statistics.std"
                else:
                    continue
            elif key.startswith("vae.encoder."):
                new_key = key.replace("vae.encoder.", "")
            else:
                continue

            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
                value = mx.transpose(value, (0, 2, 3, 4, 1))

            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
                value = mx.transpose(value, (0, 2, 3, 1))

            sanitized[new_key] = value
        return sanitized

    @classmethod
    def from_pretrained(cls, model_path) -> "VideoEncoder":
        from ..config import VideoEncoderModelConfig

        model_path = Path(model_path)
        logger.info("VideoEncoder.from_pretrained path=%s", model_path)

        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = VideoEncoderModelConfig.from_dict(config_dict)
            logger.info(
                "VideoEncoder config loaded in_channels=%d out_channels=%d patch=%d blocks=%d",
                config.in_channels,
                config.out_channels,
                config.patch_size,
                len(config.encoder_blocks or []),
            )
        else:
            logger.warning("No config.json in %s, using defaults", model_path)
            config = VideoEncoderModelConfig()

        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            if model_path.is_file():
                weights = mx.load(str(model_path))
            else:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
        else:
            weights = {}
            for wf in weight_files:
                weights.update(mx.load(str(wf)))
        logger.info(
            "VideoEncoder weights loaded files=%d total_keys=%d",
            len(weight_files),
            len(weights),
        )

        model = cls(config)
        sanitized = model.sanitize(weights)

        from mlx.utils import tree_flatten

        model_keys = {k for k, _ in tree_flatten(model.parameters())}
        sanitized_keys = set(sanitized.keys())
        unmatched = sanitized_keys - model_keys
        missing = model_keys - sanitized_keys
        if unmatched:
            logger.warning(
                "VideoEncoder unmatched (sanitized not in model) count=%d sample=%s",
                len(unmatched),
                list(unmatched)[:30],
            )
        if missing:
            logger.warning(
                "VideoEncoder missing (model not in sanitized) count=%d sample=%s",
                len(missing),
                list(missing)[:30],
            )

        model.load_weights(list(sanitized.items()), strict=False)
        mx.eval(model.parameters())
        model.eval()
        logger.info("VideoEncoder loaded and evaluated OK")
        return model


class VideoDecoder(nn.Module):

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: list[tuple[str, Any]] = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        super().__init__()

        if decoder_blocks is None:
            decoder_blocks = []

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = {}
        for idx, (block_name, block_params) in enumerate(reversed(decoder_blocks)):
            block_config = (
                {"num_layers": block_params}
                if isinstance(block_params, int)
                else block_params
            )

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )
            self.up_blocks[idx] = block

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_groups=self._norm_num_groups,
                dims=feature_channels,
                eps=1e-6,
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array | None = None,
    ) -> mx.array:
        batch_size = sample.shape[0]

        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample

        sample = self.per_channel_statistics.un_normalize(sample)

        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)

        sample = self.conv_in(sample, causal=self.causal)

        for i in range(len(self.up_blocks)):
            up_block = self.up_blocks[i]
            sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample
