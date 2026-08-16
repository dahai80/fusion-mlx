import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SD2VAEConfig:
    # SD2.1 VAE is architecturally identical to SD1.5/SDXL VAE (AutoencoderKL):
    # block_out_channels=(128,256,512,512), latent_channels=4, layers_per_block=2.
    # scaling_factor 0.18215 (same as SD1.5). Verified against the diffusers
    # checkpoint vae/config.json (scaling_factor=None -> default 0.18215).
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215
    act_fn: str = "silu"


@dataclass
class SD2UNetConfig:
    # diffusers UNet2DConditionModel for SD2.1-768 (sd2-community/
    # stable-diffusion-2-1, weights-verified):
    #   block_out_channels=[320,640,1280,1280]
    #   cross_attention_dim=1024 (ViT-H text context)
    #   attention_head_dim=[5,10,20,20] -> per-BLOCK NUMBER OF HEADS
    #     (diffusers "num_attention_heads or attention_head_dim" convention,
    #     weights-verified: block0 heads=5 head_dim=64). head_dim = ch//heads
    #     = 64 uniformly. Up-block heads = reversed([5,10,20,20]) = [20,20,10,5].
    #   use_linear_projection=True (proj_in/proj_out are nn.Linear, not Conv1x1)
    #   transformer_layers_per_block=1
    #   NO add_embedding / class_embed (like SD1.5, unlike SDXL)
    #   time_embedding.linear_1/linear_2 (diffusers default naming)
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple = (320, 640, 1280, 1280)
    attention_head_dim: tuple = (5, 10, 20, 20)
    layers_per_block: int = 2
    cross_attention_dim: int = 1024
    transformer_layers_per_block: int = 1
    use_linear_projection: bool = True
    act_fn: str = "silu"
    norm_eps: float = 1e-5
    norm_num_groups: int = 32
    sample_size: int = 96


@dataclass
class SD2TextEncoderConfig:
    # SD2.1 text encoder = ViT-H/14 in CLIPTextModel layout (sd2-community
    # weights-verified): hidden=1024, 23 layers, 16 heads, intermediate=4096,
    # hidden_act=gelu, NO text_projection (pooled via EOS-row, CLIP-style,
    # projection_dim=512 but unused for txt2img conditioning which uses the
    # full last_hidden_state). vocab 49408 (same CLIP vocab as SD1.5).
    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    vocab: int = 49408
    max_pos: int = 77


@dataclass
class SD2ModelPaths:
    # sd2-community/stable-diffusion-2-1 is the only mirror-accessible SD2.1
    # repo (canonical stabilityai/stable-diffusion-2-1 is gated, 404 on
    # hf-mirror.com). Weights-verified to match the config here. The repo
    # also carries v2-1_768-ema-pruned.ckpt (the ComfyUI 2_pass_txt2img
    # example checkpoint name).
    repo: str = "sd2-community/stable-diffusion-2-1"
    unet_subfolder: str = "unet"
    unet_file: str = "diffusion_pytorch_model.fp16.safetensors"
    vae_subfolder: str = "vae"
    vae_file: str = "diffusion_pytorch_model.fp16.safetensors"
    text_subfolder: str = "text_encoder"
    text_file: str = "model.fp16.safetensors"
    tokenizer_subfolder: str = "tokenizer"

    def __post_init__(self):
        repo = os.environ.get("SD2_REPO")
        if repo:
            self.repo = repo
            logger.info("SD2 repo override: %s", repo)
        for attr, env in (
            ("unet_subfolder", "SD2_UNET_SUBFOLDER"),
            ("vae_subfolder", "SD2_VAE_SUBFOLDER"),
            ("text_subfolder", "SD2_TEXT_SUBFOLDER"),
            ("tokenizer_subfolder", "SD2_TOKENIZER_SUBFOLDER"),
            ("unet_file", "SD2_UNET_FILE"),
            ("vae_file", "SD2_VAE_FILE"),
            ("text_file", "SD2_TEXT_FILE"),
        ):
            val = os.environ.get(env)
            if val:
                setattr(self, attr, val)
                logger.info("SD2 path override %s=%s", env, val)


@dataclass
class SD2Config:
    unet: SD2UNetConfig = field(default_factory=SD2UNetConfig)
    vae: SD2VAEConfig = field(default_factory=SD2VAEConfig)
    text: SD2TextEncoderConfig = field(default_factory=SD2TextEncoderConfig)
    paths: SD2ModelPaths = field(default_factory=SD2ModelPaths)
