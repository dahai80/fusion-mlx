import logging
from dataclasses import dataclass, field

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
    clip_l_repo: str = "openai/clip-vit-large-patch14"
    clip_l_subfolder: str = ""
    clip_g_repo: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    clip_g_subfolder: str = ""
    t5_repo: str = "google/t5-v1_1-xxl"
    t5_subfolder: str = ""


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
        key, cfg.inner_dim, cfg.num_layers,
        cfg.num_attention_heads, cfg.attention_head_dim,
    )
    return cfg
