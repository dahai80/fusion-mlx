import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SDXLVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.13025
    act_fn: str = "silu"


@dataclass
class SDXLUNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple = (320, 640, 1280)
    attention_head_dim: tuple = (5, 10, 20)
    layers_per_block: int = 2
    cross_attention_dim: int = 2048
    transformer_layers_per_block: tuple = (1, 2, 10)
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int = 2816
    addition_embed_type_num_heads: int = 64
    act_fn: str = "silu"
    norm_eps: float = 1e-5
    norm_num_groups: int = 32
    sample_size: int = 128


@dataclass
class SDXLTextEncoderConfig:
    # CLIP-L (text_encoder)
    clip_l_dims: int = 768
    clip_l_layers: int = 12
    clip_l_heads: int = 12
    clip_l_intermediate: int = 3072
    clip_l_act: str = "quick_gelu"
    # OpenCLIP-G (text_encoder_2)
    clip_g_dims: int = 1280
    clip_g_layers: int = 32
    clip_g_heads: int = 20
    clip_g_intermediate: int = 5120
    clip_g_act: str = "gelu"
    # CLIP-G also has a text_projection (Linear) + pooled projection_dim=1280
    clip_g_projection_dim: int = 1280
    vocab: int = 49408
    max_pos: int = 77


@dataclass
class SDXLModelPaths:
    repo: str = "stabilityai/stable-diffusion-xl-base-1.0"
    unet_subfolder: str = "unet"
    unet_file: str = "diffusion_pytorch_model.fp16.safetensors"
    vae_subfolder: str = "vae"
    vae_file: str = "diffusion_pytorch_model.fp16.safetensors"
    clip_l_subfolder: str = "text_encoder"
    clip_l_file: str = "model.fp16.safetensors"
    clip_g_subfolder: str = "text_encoder_2"
    clip_g_file: str = "model.fp16.safetensors"
    tokenizer_subfolder: str = "tokenizer"
    tokenizer_2_subfolder: str = "tokenizer_2"

    def __post_init__(self):
        # env overrides for offline / mirror layouts (same pattern as SD3).
        repo = os.environ.get("SDXL_REPO")
        if repo:
            self.repo = repo
            logger.info("SDXL repo override: %s", repo)
        for attr, env in (
            ("unet_subfolder", "SDXL_UNET_SUBFOLDER"),
            ("vae_subfolder", "SDXL_VAE_SUBFOLDER"),
            ("clip_l_subfolder", "SDXL_CLIP_L_SUBFOLDER"),
            ("clip_g_subfolder", "SDXL_CLIP_G_SUBFOLDER"),
            ("tokenizer_subfolder", "SDXL_TOKENIZER_SUBFOLDER"),
            ("tokenizer_2_subfolder", "SDXL_TOKENIZER_2_SUBFOLDER"),
            ("unet_file", "SDXL_UNET_FILE"),
            ("vae_file", "SDXL_VAE_FILE"),
            ("clip_l_file", "SDXL_CLIP_L_FILE"),
            ("clip_g_file", "SDXL_CLIP_G_FILE"),
        ):
            val = os.environ.get(env)
            if val:
                setattr(self, attr, val)
                logger.info("SDXL path override %s=%s", env, val)


@dataclass
class SDXLConfig:
    unet: SDXLUNetConfig = field(default_factory=SDXLUNetConfig)
    vae: SDXLVAEConfig = field(default_factory=SDXLVAEConfig)
    text: SDXLTextEncoderConfig = field(default_factory=SDXLTextEncoderConfig)
    paths: SDXLModelPaths = field(default_factory=SDXLModelPaths)
