import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .convolution import CausalConv3d, PaddingModeType
from .ops import PerChannelStatistics, unpatchify
from .sampling import DepthToSpaceUpsample
from .tiling import TilingConfig, decode_with_tiling

logger = logging.getLogger(__name__)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> mx.array:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb

    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb


class TimestepEmbedding(nn.Module):

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.act = nn.SiLU()

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaTimestepEmbedder(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def __call__(
        self, timestep: mx.array, hidden_dtype: mx.Dtype = mx.float32
    ) -> mx.array:
        timesteps_proj = get_timestep_embedding(
            timestep, embedding_dim=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(hidden_dtype))
        return timesteps_emb


class ResnetBlock3DSimple(nn.Module):

    def __init__(
        self,
        channels: int,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning

        self.conv1 = self._make_conv_wrapper(channels, channels, spatial_padding_mode)
        self.conv2 = self._make_conv_wrapper(channels, channels, spatial_padding_mode)

        self.act = nn.SiLU()

        if timestep_conditioning:
            self.scale_shift_table = mx.zeros((4, channels))

    def _make_conv_wrapper(self, in_ch, out_ch, padding_mode):

        class ConvWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = CausalConv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=padding_mode,
                )

            def __call__(self, x, causal=False):
                return self.conv(x, causal=causal)

        return ConvWrapper()

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        return x / mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep_embed: mx.array | None = None,
    ) -> mx.array:
        residual = x
        batch_size = x.shape[0]

        x = self.pixel_norm(x)

        if self.timestep_conditioning and timestep_embed is not None:
            ada_values = self.scale_shift_table[None, :, :, None, None, None]
            channels = self.scale_shift_table.shape[1]
            ts_reshaped = timestep_embed.reshape(batch_size, 4, channels, 1, 1, 1)
            ada_values = ada_values + ts_reshaped

            shift1 = ada_values[:, 0]
            scale1 = ada_values[:, 1]
            shift2 = ada_values[:, 2]
            scale2 = ada_values[:, 3]

            x = x * (1 + scale1) + shift1

        x = self.act(x)
        x = self.conv1(x, causal=causal)

        x = self.pixel_norm(x)

        if self.timestep_conditioning and timestep_embed is not None:
            x = x * (1 + scale2) + shift2

        x = self.act(x)
        x = self.conv2(x, causal=causal)

        return x + residual


class ResBlockGroup(nn.Module):

    def __init__(
        self,
        channels: int,
        num_layers: int = 5,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaTimestepEmbedder(embedding_dim=channels * 4)

        self.res_blocks = {
            i: ResnetBlock3DSimple(
                channels,
                spatial_padding_mode,
                timestep_conditioning=timestep_conditioning,
            )
            for i in range(num_layers)
        }

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep: mx.array | None = None,
    ) -> mx.array:
        timestep_embed = None

        if self.timestep_conditioning and timestep is not None:
            batch_size = x.shape[0]
            timestep_embed = self.time_embedder(
                timestep.flatten(), hidden_dtype=x.dtype
            )
            timestep_embed = timestep_embed.reshape(batch_size, -1, 1, 1, 1)

        for res_block in self.res_blocks.values():
            x = res_block(x, causal=causal, timestep_embed=timestep_embed)
        return x


class LTX2VideoDecoder(nn.Module):

    DEFAULT_BLOCKS = [
        ("res", 1024, 5),
        ("d2s", 1024, 2, (2, 2, 2)),
        ("res", 512, 5),
        ("d2s", 512, 2, (2, 2, 2)),
        ("res", 256, 5),
        ("d2s", 256, 2, (2, 2, 2)),
        ("res", 128, 5),
    ]

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        num_layers_per_block: int = 5,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = True,
        decoder_blocks: list = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.timestep_conditioning = timestep_conditioning

        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        blocks = decoder_blocks or self.DEFAULT_BLOCKS
        first_ch = blocks[0][1]
        last_ch = blocks[-1][1]

        class ConvInWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = CausalConv3d(
                    in_channels=in_channels,
                    out_channels=first_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=spatial_padding_mode,
                )

            def __call__(self, x, causal=False):
                return self.conv(x, causal=causal)

        self.conv_in = ConvInWrapper()

        self.up_blocks = {}
        for idx, block_def in enumerate(blocks):
            block_type = block_def[0]
            ch = block_def[1]
            if block_type == "res":
                num_layers = (
                    block_def[2] if len(block_def) > 2 else num_layers_per_block
                )
                self.up_blocks[idx] = ResBlockGroup(
                    ch, num_layers, spatial_padding_mode, timestep_conditioning
                )
            elif block_type == "d2s":
                reduction = block_def[2] if len(block_def) > 2 else 2
                stride = block_def[3] if len(block_def) > 3 else (2, 2, 2)
                residual = block_def[4] if len(block_def) > 4 else True
                self.up_blocks[idx] = DepthToSpaceUpsample(
                    dims=3,
                    in_channels=ch,
                    stride=stride,
                    residual=residual,
                    out_channels_reduction_factor=reduction,
                    spatial_padding_mode=spatial_padding_mode,
                )

        final_out_channels = out_channels * patch_size * patch_size

        class ConvOutWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = CausalConv3d(
                    in_channels=last_ch,
                    out_channels=final_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=spatial_padding_mode,
                )

            def __call__(self, x, causal=False):
                return self.conv(x, causal=causal)

        self.conv_out = ConvOutWrapper()

        self.act = nn.SiLU()

        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = PixArtAlphaTimestepEmbedder(
                embedding_dim=last_ch * 2
            )
            self.last_scale_shift_table = mx.zeros((2, last_ch))

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        if "per_channel_statistics.mean" in weights:
            return weights
        for key, value in weights.items():
            new_key = key

            if not key.startswith("vae.") or key.startswith("vae.encoder."):
                continue

            if key.startswith("vae.per_channel_statistics."):
                if key == "vae.per_channel_statistics.mean-of-means":
                    new_key = "per_channel_statistics.mean"
                elif key == "vae.per_channel_statistics.std-of-means":
                    new_key = "per_channel_statistics.std"
                else:
                    continue

            if key.startswith("vae.decoder."):
                new_key = key.replace("vae.decoder.", "")

            if ".conv.weight" in key and value.ndim == 5:
                value = mx.transpose(value, (0, 2, 3, 4, 1))

            if ".conv.bias" in key:
                pass

            if ".conv.weight" in new_key or ".conv.bias" in new_key:

                if (
                    ".conv.conv.weight" not in new_key
                    and ".conv.conv.bias" not in new_key
                ):
                    new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                    new_key = new_key.replace(".conv.bias", ".conv.conv.bias")

            sanitized[new_key] = value
        return sanitized

    @classmethod
    def from_pretrained(cls, model_path, strict: bool = True) -> "LTX2VideoDecoder":
        model_path = Path(model_path)
        config_dict = {}

        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(str(wf)))
        logger.info(
            "LTX2VideoDecoder.from_pretrained: model_path=%s weight_files=%d total_keys=%d",
            model_path,
            len(weight_files),
            len(weights),
        )

        decoder_blocks = cls._infer_blocks(weights)

        spatial_padding_mode_str = config_dict.get("spatial_padding_mode", "reflect")
        spatial_padding_mode = PaddingModeType(spatial_padding_mode_str)

        model = cls(
            timestep_conditioning=config_dict.get("timestep_conditioning", False),
            decoder_blocks=decoder_blocks,
            spatial_padding_mode=spatial_padding_mode,
        )
        weights = model.sanitize(weights)

        try:
            from mlx.utils import tree_flatten

            model_params = dict(tree_flatten(model.parameters()))
            model_keys = set(model_params.keys())
            sanitized_keys = set(weights.keys())
            unmatched = sorted(sanitized_keys - model_keys)
            missing = sorted(model_keys - sanitized_keys)
            logger.info(
                "LTX2VideoDecoder audit: model_params=%d sanitized=%d unmatched=%d missing=%d",
                len(model_keys),
                len(sanitized_keys),
                len(unmatched),
                len(missing),
            )
            if unmatched:
                logger.warning(
                    "LTX2VideoDecoder unmatched (first 30): %s", unmatched[:30]
                )
            if missing:
                logger.warning("LTX2VideoDecoder missing (first 30): %s", missing[:30])
        except Exception as audit_err:
            logger.warning("LTX2VideoDecoder audit skipped: %s", audit_err)

        model.load_weights(list(weights.items()), strict=strict)
        mx.eval(model.parameters())
        model.eval()
        return model

    @staticmethod
    def _infer_blocks(weights: dict) -> list:
        block_indices = set()
        for k in weights:
            if "up_blocks." in k:
                idx_str = k.split("up_blocks.")[1].split(".")[0]
                if idx_str.isdigit():
                    block_indices.add(int(idx_str))

        if not block_indices:
            return None

        raw_blocks = []
        for idx in sorted(block_indices):
            has_conv = any(f"up_blocks.{idx}.conv." in k for k in weights)
            res_indices = set()
            for k in weights:
                prefix = f"up_blocks.{idx}.res_blocks."
                if prefix in k:
                    res_idx = k.split(prefix)[1].split(".")[0]
                    if res_idx.isdigit():
                        res_indices.add(int(res_idx))

            if has_conv and not res_indices:
                for k, v in weights.items():
                    if f"up_blocks.{idx}.conv." in k and "weight" in k:
                        in_ch = v.shape[-1] if v.ndim == 5 else v.shape[1]
                        conv_out_ch = v.shape[0]
                        raw_blocks.append(("d2s", in_ch, conv_out_ch))
                        break
            elif res_indices:
                num_res = max(res_indices) + 1
                for k, v in weights.items():
                    if f"up_blocks.{idx}.res_blocks.0.conv1" in k and "weight" in k:
                        ch = v.shape[0]
                        raw_blocks.append(("res", ch, num_res))
                        break

        blocks = []
        d2s_strides = []
        for i, block in enumerate(raw_blocks):
            if block[0] == "res":
                blocks.append(block)
            elif block[0] == "d2s":
                in_ch, conv_out_ch = block[1], block[2]
                next_ch = None
                for j in range(i + 1, len(raw_blocks)):
                    if raw_blocks[j][0] == "res":
                        next_ch = raw_blocks[j][1]
                        break

                if next_ch is None:
                    next_ch = in_ch // 2

                reduction = in_ch // next_ch if next_ch > 0 else 2

                multiplier = conv_out_ch // next_ch if next_ch > 0 else 8

                if multiplier == 8:
                    stride = (2, 2, 2)
                elif multiplier == 4:
                    stride = (1, 2, 2)
                elif multiplier == 2:
                    stride = (2, 1, 1)
                else:
                    stride = (2, 2, 2)

                d2s_strides.append(stride)
                blocks.append(("d2s", in_ch, reduction, stride))

        if not blocks:
            return None

        has_mixed_strides = len(set(d2s_strides)) > 1
        has_non_standard_reduction = any(b[2] != 2 for b in blocks if b[0] == "d2s")
        use_residual = not has_mixed_strides and not has_non_standard_reduction

        final_blocks = []
        for block in blocks:
            if block[0] == "d2s":
                final_blocks.append(("d2s", block[1], block[2], block[3], use_residual))
            else:
                final_blocks.append(block)

        return final_blocks

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        return x / mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        sample: mx.array,
        causal: bool = False,
        timestep: mx.array | None = None,
        debug: bool = False,
        chunked_conv: bool = False,
    ) -> mx.array:

        batch_size = sample.shape[0]

        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample

        sample = self.per_channel_statistics.un_normalize(sample)

        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)

        scaled_timestep = None
        if self.timestep_conditioning and timestep is not None:
            scaled_timestep = timestep * self.timestep_scale_multiplier

        x = self.conv_in(sample, causal=causal)

        for i, block in self.up_blocks.items():
            if isinstance(block, ResBlockGroup):
                x = block(x, causal=causal, timestep=scaled_timestep)
            elif isinstance(block, DepthToSpaceUpsample):
                x = block(x, causal=causal, chunked_conv=chunked_conv)
            else:
                x = block(x, causal=causal)

        x = self.pixel_norm(x)

        if self.timestep_conditioning and scaled_timestep is not None:
            embedded_timestep = self.last_time_embedder(
                scaled_timestep.flatten(), hidden_dtype=x.dtype
            )
            embedded_timestep = embedded_timestep.reshape(batch_size, -1, 1, 1, 1)

            ada_values = self.last_scale_shift_table[None, :, :, None, None, None]
            # last_ch is blocks[-1][1] (128 for DEFAULT_BLOCKS); hardcoding
            # 128 breaks non-default decoder_blocks inferred from weights.
            # last_scale_shift_table == mx.zeros((2, last_ch)), so its
            # shape[1] tracks the real config.
            ts_reshaped = embedded_timestep.reshape(
                batch_size, 2, self.last_scale_shift_table.shape[1], 1, 1, 1
            )
            ada_values = ada_values + ts_reshaped

            shift = ada_values[:, 0]
            scale = ada_values[:, 1]

            x = x * (1 + scale) + shift

        x = self.act(x)

        x = self.conv_out(x, causal=causal)

        x = unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

        return x

    def decode_tiled(
        self,
        sample: mx.array,
        tiling_config: TilingConfig | None = None,
        tiling_mode: str = "auto",
        causal: bool = False,
        timestep: mx.array | None = None,
        debug: bool = False,
        on_frames_ready: Callable | None = None,
    ) -> mx.array:
        if tiling_config is None:
            tiling_config = TilingConfig.default()

        _, _, f, h, w = sample.shape
        needs_spatial_tiling = False
        needs_temporal_tiling = False

        spatial_scale = 32
        temporal_scale = 8

        if tiling_config.spatial_config is not None:
            s_cfg = tiling_config.spatial_config
            tile_size_latent = s_cfg.tile_size_in_pixels // spatial_scale
            if h > tile_size_latent or w > tile_size_latent:
                needs_spatial_tiling = True

        if tiling_config.temporal_config is not None:
            t_cfg = tiling_config.temporal_config
            tile_size_latent = t_cfg.tile_size_in_frames // temporal_scale
            if f > tile_size_latent:
                needs_temporal_tiling = True

        use_chunked_conv = tiling_mode in (
            "conservative",
            "none",
            "auto",
            "default",
            "spatial",
        )

        if not needs_spatial_tiling and not needs_temporal_tiling:
            return self(
                sample,
                causal=causal,
                timestep=timestep,
                debug=debug,
                chunked_conv=use_chunked_conv,
            )

        return decode_with_tiling(
            decoder_fn=self,
            latents=sample,
            tiling_config=tiling_config,
            spatial_scale=32,
            temporal_scale=8,
            causal=causal,
            timestep=timestep,
            chunked_conv=use_chunked_conv,
            on_frames_ready=on_frames_ready,
        )


VideoDecoder = LTX2VideoDecoder
