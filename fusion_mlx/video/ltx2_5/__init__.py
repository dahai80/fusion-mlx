# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 (22B video+audio). Reuses fusion_mlx.video.ltx2
# skeleton (transformer block / VAE / scheduler / denoise / conditioning /
# positions / rope / adaln / feed_forward / attention / guidance / lora).
# New vs LTX-2/2.3: Gemma4-12b text encoder, duration-head, two-stage
# spatial+temporal upsampler, 48-layer transformer, caption_channels=3840.
from .config import (
    LTX2_5ModelConfig,
    LTX2_5Variant,
    default_ltx2_5_config,
)
from .duration_head import (
    DurationHead,
    duration_to_num_frames,
    infer_num_frames,
    load_duration_head,
)
from .ltx2_5_model import LTX2_5Model, LTX2_5X0Model
from .scheduler import (
    DISTILLED_STAGE_1_SIGMAS,
    DISTILLED_STAGE_2_SIGMAS,
    denoise_dev_av,
    denoise_distilled,
    denoise_res2s_av,
    ltx2_5_scheduler,
    resolve_distilled_sigmas,
)
from .text_encoder import (
    LTX2_5TextEncoder,
    LTX2_5TextProjection,
    load_text_encoder,
)
from .upsampler import (
    LatentTemporalUpsampler,
    TemporalPixelShuffle,
    TemporalUpsampler2x,
    load_spatial_upsampler_2_5,
    load_temporal_upsampler,
)
from .utils import (
    component_keys,
    get_model_path,
    resolve_component,
)
from .video_vae import (
    LTX2VideoDecoder,
    VideoEncoder,
    load_video_decoder,
    load_video_encoder,
)

__all__ = [
    "LTX2_5ModelConfig",
    "LTX2_5Variant",
    "default_ltx2_5_config",
    "LTX2_5Model",
    "LTX2_5X0Model",
    "ltx2_5_scheduler",
    "denoise_distilled",
    "denoise_dev_av",
    "denoise_res2s_av",
    "DISTILLED_STAGE_1_SIGMAS",
    "DISTILLED_STAGE_2_SIGMAS",
    "resolve_distilled_sigmas",
    "LTX2_5TextEncoder",
    "LTX2_5TextProjection",
    "load_text_encoder",
    "DurationHead",
    "duration_to_num_frames",
    "infer_num_frames",
    "load_duration_head",
    "LatentTemporalUpsampler",
    "TemporalPixelShuffle",
    "TemporalUpsampler2x",
    "load_spatial_upsampler_2_5",
    "load_temporal_upsampler",
    "VideoEncoder",
    "LTX2VideoDecoder",
    "load_video_encoder",
    "load_video_decoder",
    "get_model_path",
    "resolve_component",
    "component_keys",
]
