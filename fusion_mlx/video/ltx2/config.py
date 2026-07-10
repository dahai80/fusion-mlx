# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 model config (vendored from mlx-video).
# Phase 4 LTX-2 direct-MLX port: model-layer foundation.
# Video VAE configs (CausalityAxis, VideoEncoder/DecoderModelConfig) added
# in Stage B (video_vae port). Audio configs (AudioDecoder/EncoderModelConfig,
# VocoderModelConfig) added in Stage E (audio_vae port).
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"
    TWO_D = "2d"


class AttentionType(Enum):
    DEFAULT = "default"


@dataclass
class BaseModelConfig:

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "BaseModelConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if isinstance(v, Enum):
                    result[k] = v.value
                elif hasattr(v, "to_dict"):
                    result[k] = v.to_dict()
                else:
                    result[k] = v
        return result


@dataclass
class TransformerConfig(BaseModelConfig):
    dim: int
    heads: int
    d_head: int
    context_dim: int


@dataclass
class VideoVAEConfig(BaseModelConfig):
    convolution_dimensions: int = 3
    in_channels: int = 3
    out_channels: int = 128
    latent_channels: int = 128
    patch_size: int = 4
    encoder_blocks: list[tuple] = field(
        default_factory=lambda: [
            ("res_x", {"num_layers": 4}),
            ("compress_space_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 6}),
            ("compress_time_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 6}),
            ("compress_all_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 2}),
            ("compress_all_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 2}),
        ]
    )
    decoder_blocks: list[tuple] = field(
        default_factory=lambda: [
            ("res_x", {"num_layers": 5, "inject_noise": False}),
            ("compress_all", {"residual": True, "multiplier": 2}),
            ("res_x", {"num_layers": 5, "inject_noise": False}),
            ("compress_all", {"residual": True, "multiplier": 2}),
            ("res_x", {"num_layers": 5, "inject_noise": False}),
            ("compress_all", {"residual": True, "multiplier": 2}),
            ("res_x", {"num_layers": 5, "inject_noise": False}),
        ]
    )


@dataclass
class LTXModelConfig(BaseModelConfig):
    model_type: LTXModelType = LTXModelType.AudioVideo
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    caption_channels: int = 3840
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_caption_channels: int = 3840
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: list[int] | None = None
    audio_positional_embedding_max_pos: list[int] | None = None
    use_middle_indices_grid: bool = True
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED
    double_precision_rope: bool = False
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1000
    norm_eps: float = 1e-6
    attention_type: AttentionType = AttentionType.DEFAULT
    has_prompt_adaln: bool = False
    vae_config: VideoVAEConfig | None = None

    def __post_init__(self):
        if self.positional_embedding_max_pos is None:
            self.positional_embedding_max_pos = [20, 2048, 2048]
        if self.audio_positional_embedding_max_pos is None:
            self.audio_positional_embedding_max_pos = [20]
        if not self.has_prompt_adaln:
            self.double_precision_rope = False
        if isinstance(self.model_type, str):
            self.model_type = LTXModelType(self.model_type)
        if isinstance(self.rope_type, str):
            self.rope_type = LTXRopeType(self.rope_type)
        if isinstance(self.attention_type, str):
            self.attention_type = AttentionType(self.attention_type)

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    def get_video_config(self) -> TransformerConfig | None:
        if not self.model_type.is_video_enabled():
            return None
        return TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=self.attention_head_dim,
            context_dim=self.cross_attention_dim,
        )

    def get_audio_config(self) -> TransformerConfig | None:
        if not self.model_type.is_audio_enabled():
            return None
        return TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=self.audio_attention_head_dim,
            context_dim=self.audio_cross_attention_dim,
        )


class CausalityAxis(Enum):
    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"


@dataclass
class VideoDecoderModelConfig(BaseModelConfig):
    ch: int = 128
    out_ch: int = 2
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: list[int] | None = None
    resolution: int = 256
    z_channels: int = 8
    norm_type: Enum = None
    causality_axis: Enum = None
    dropout: float = 0.0
    timestep_conditioning: bool = False


@dataclass
class VideoEncoderModelConfig(BaseModelConfig):
    convolution_dimensions: int = 3
    in_channels: int = 3
    out_channels: int = 128
    patch_size: int = 4
    norm_layer: Enum = None
    latent_log_var: Enum = None
    encoder_spatial_padding_mode: Enum = None
    encoder_blocks: list[tuple] = field(
        default_factory=lambda: [
            ("res_x", {"num_layers": 4}),
            ("compress_space_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 6}),
            ("compress_time_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 6}),
            ("compress_all_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 2}),
            ("compress_all_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 2}),
        ]
    )

    def __post_init__(self):
        from .video_vae.convolution import PaddingModeType
        from .video_vae.resnet import NormLayerType
        from .video_vae.video_vae import LogVarianceType

        if self.norm_layer is None:
            self.norm_layer = NormLayerType.PIXEL_NORM
        if self.latent_log_var is None:
            self.latent_log_var = LogVarianceType.UNIFORM
        if self.encoder_spatial_padding_mode is None:
            self.encoder_spatial_padding_mode = PaddingModeType.ZEROS

        if isinstance(self.norm_layer, str):
            self.norm_layer = NormLayerType(self.norm_layer)
        if isinstance(self.latent_log_var, str):
            self.latent_log_var = LogVarianceType(self.latent_log_var)
        if isinstance(self.encoder_spatial_padding_mode, str):
            self.encoder_spatial_padding_mode = PaddingModeType(
                self.encoder_spatial_padding_mode
            )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.encoder_blocks is not None:
            result["encoder_blocks"] = [list(block) for block in self.encoder_blocks]
        return result


@dataclass
class AudioDecoderModelConfig(BaseModelConfig):
    ch: int = 128
    out_ch: int = 2
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: list[int] | None = None
    resolution: int = 256
    z_channels: int = 8
    norm_type: Enum = None
    causality_axis: Enum = None
    dropout: float = 0.0
    mid_block_add_attention: bool = True
    sample_rate: int = 16000
    mel_hop_length: int = 160
    is_causal: bool = True
    mel_bins: int | None = None
    resamp_with_conv: bool = True
    attn_type: str = None
    give_pre_end: bool = False
    tanh_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.attn_resolutions is not None:
            result["attn_resolutions"] = list(self.attn_resolutions)
        return result

    def __post_init__(self):
        from .audio_vae.attention import AttentionType
        from .audio_vae.normalization import NormType

        if isinstance(self.causality_axis, str):
            self.causality_axis = CausalityAxis(self.causality_axis)
        if isinstance(self.norm_type, str):
            self.norm_type = NormType(self.norm_type)
        if isinstance(self.attn_type, str):
            self.attn_type = AttentionType(self.attn_type)


@dataclass
class AudioEncoderModelConfig(BaseModelConfig):
    ch: int = 128
    in_channels: int = 2
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: list[int] | None = None
    resolution: int = 256
    z_channels: int = 8
    double_z: bool = True
    n_fft: int = 1024
    norm_type: Enum = None
    causality_axis: Enum = None
    dropout: float = 0.0
    mid_block_add_attention: bool = True
    sample_rate: int = 16000
    mel_hop_length: int = 160
    is_causal: bool = True
    mel_bins: int = 64
    resamp_with_conv: bool = True
    attn_type: str = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.attn_resolutions is not None:
            result["attn_resolutions"] = list(self.attn_resolutions)
        return result

    def __post_init__(self):
        from .audio_vae.attention import AttentionType
        from .audio_vae.normalization import NormType

        if isinstance(self.causality_axis, str):
            self.causality_axis = CausalityAxis(self.causality_axis)
        if isinstance(self.norm_type, str):
            self.norm_type = NormType(self.norm_type)
        if isinstance(self.attn_type, str):
            self.attn_type = AttentionType(self.attn_type)


@dataclass
class VocoderModelConfig(BaseModelConfig):
    resblock_kernel_sizes: list[int] | None = None
    upsample_rates: list[int] | None = None
    upsample_kernel_sizes: list[int] | None = None
    resblock_dilation_sizes: list[list[int]] | None = None
    upsample_initial_channel: int = 1024
    stereo: bool = True
    resblock: str = "1"
    output_sample_rate: int = 24000
    activation: str = "snake"
    use_tanh_at_final: bool = True
    apply_final_activation: bool = True
    use_bias_at_final: bool = True

    def __post_init__(self):
        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11]
        if self.upsample_rates is None:
            self.upsample_rates = [6, 5, 2, 2, 2]
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
