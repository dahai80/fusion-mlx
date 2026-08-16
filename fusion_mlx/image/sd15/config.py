import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SD15VAEConfig:
    # SD1.5 VAE is architecturally identical to SDXL VAE (AutoencoderKL)
    # but uses a different scaling_factor: 0.18215 (SD1.5) vs 0.13025 (SDXL).
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215
    act_fn: str = "silu"


@dataclass
class SD15UNetConfig:
    # diffusers UNet2DConditionModel for SD1.5 (runwayml/stable-diffusion-v1-5).
    # block_out_channels=[320,640,1280,1280], attention_head_dim=8 (scalar ->
    # heads = ch // 8), cross_attention_dim=768 (CLIP-L), transformer_layers
    # _per_block=1. NO addition_embed_type / class_embed_type (unlike SDXL).
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple = (320, 640, 1280, 1280)
    attention_head_dim: int = 8
    layers_per_block: int = 2
    cross_attention_dim: int = 768
    transformer_layers_per_block: int = 1
    act_fn: str = "silu"
    norm_eps: float = 1e-5
    norm_num_groups: int = 32
    sample_size: int = 64


@dataclass
class SD15TextEncoderConfig:
    # SD1.5 uses CLIP-L only (single text encoder). SDXL's SDXLCLIPTextModel
    # is reused verbatim with projection_dim=None (no text_projection).
    clip_l_dims: int = 768
    clip_l_layers: int = 12
    clip_l_heads: int = 12
    clip_l_intermediate: int = 3072
    clip_l_act: str = "quick_gelu"
    vocab: int = 49408
    max_pos: int = 77


@dataclass
class SD15ModelPaths:
    repo: str = "runwayml/stable-diffusion-v1-5"
    unet_subfolder: str = "unet"
    unet_file: str = "diffusion_pytorch_model.safetensors"
    vae_subfolder: str = "vae"
    vae_file: str = "diffusion_pytorch_model.safetensors"
    clip_l_subfolder: str = "text_encoder"
    clip_l_file: str = "model.safetensors"
    tokenizer_subfolder: str = "tokenizer"

    def __post_init__(self):
        # env overrides for offline / mirror layouts (same pattern as SDXL/SD3).
        repo = os.environ.get("SD15_REPO")
        if repo:
            self.repo = repo
            logger.info("SD15 repo override: %s", repo)
        for attr, env in (
            ("unet_subfolder", "SD15_UNET_SUBFOLDER"),
            ("vae_subfolder", "SD15_VAE_SUBFOLDER"),
            ("clip_l_subfolder", "SD15_CLIP_L_SUBFOLDER"),
            ("tokenizer_subfolder", "SD15_TOKENIZER_SUBFOLDER"),
            ("unet_file", "SD15_UNET_FILE"),
            ("vae_file", "SD15_VAE_FILE"),
            ("clip_l_file", "SD15_CLIP_L_FILE"),
        ):
            val = os.environ.get(env)
            if val:
                setattr(self, attr, val)
                logger.info("SD15 path override %s=%s", env, val)


@dataclass
class SD15Config:
    unet: SD15UNetConfig = field(default_factory=SD15UNetConfig)
    vae: SD15VAEConfig = field(default_factory=SD15VAEConfig)
    text: SD15TextEncoderConfig = field(default_factory=SD15TextEncoderConfig)
    paths: SD15ModelPaths = field(default_factory=SD15ModelPaths)
