import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SD3Config:
    inner_dim: int = 1536
    num_layers: int = 24
    num_attention_heads: int = 24
    attention_head_dim: int = 64
    joint_attention_dim: int = 4096
    caption_projection_dim: int = 1152
    pooled_projection_dim: int = 2048
    in_channels: int = 16
    out_channels: int = 16
    patch_size: int = 2
    pos_embed_max_size: int = 192
    sample_size: int = 128
    vae_latent_channels: int = 16
    vae_scaling_factor: float = 1.5305
    vae_shift_factor: float = 0.0609
    qk_norm: str | None = None
    dual_attention_layers: tuple = ()
    max_t5_token_len: int = 256
    max_clip_token_len: int = 77
    num_train_timesteps: int = 1000
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096


@dataclass
class SD3ModelPaths:
    transformer_ckpt: str = "argmaxinc/mlx-stable-diffusion-3-medium"
    transformer_file: str = "sd3_medium.safetensors"
    encoders_repo: str = "frankjoshua/stable-diffusion-3-medium-diffusers"
    clip_l_subfolder: str = "text_encoder"
    clip_g_subfolder: str = "text_encoder_2"
    t5_subfolder: str = "text_encoder_3"
    clip_l_tokenizer_subfolder: str = "tokenizer"
    clip_g_tokenizer_subfolder: str = "tokenizer_2"
    t5_tokenizer_subfolder: str = "tokenizer_3"

    def __post_init__(self) -> None:
        env = os.environ
        if v := env.get("SD3_ENCODERS_REPO"):
            self.encoders_repo = v
        if v := env.get("SD3_CLIP_L_SUBFOLDER"):
            self.clip_l_subfolder = v
        if v := env.get("SD3_CLIP_G_SUBFOLDER"):
            self.clip_g_subfolder = v
        if v := env.get("SD3_T5_SUBFOLDER"):
            self.t5_subfolder = v
        logger.info(
            "SD3 paths encoders_repo=%s clip_l=%s clip_g=%s t5=%s",
            self.encoders_repo,
            self.clip_l_subfolder,
            self.clip_g_subfolder,
            self.t5_subfolder,
        )


@dataclass
class ClipLConfig:
    dims: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate: int = 3072
    act: str = "quick_gelu"
    vocab: int = 49408
    max_pos: int = 77


@dataclass
class ClipGConfig:
    dims: int = 1280
    num_layers: int = 32
    num_heads: int = 20
    intermediate: int = 5120
    act: str = "gelu"
    vocab: int = 49408
    max_pos: int = 77


VARIANTS = {
    "sd3": SD3Config(),
    "sd3-medium": SD3Config(),
    "sd3-medium-turbo": SD3Config(base_shift=0.95, max_shift=0.95),
}


def get_config(variant: str) -> SD3Config:
    key = (variant or "sd3").lower()
    if key not in VARIANTS:
        logger.warning("SD3 variant '%s' unknown, fallback to sd3", variant)
        key = "sd3"
    cfg = VARIANTS[key]
    logger.info(
        "SD3 config variant=%s inner_dim=%d layers=%d heads=%d head_dim=%d",
        key,
        cfg.inner_dim,
        cfg.num_layers,
        cfg.num_attention_heads,
        cfg.attention_head_dim,
    )
    return cfg
