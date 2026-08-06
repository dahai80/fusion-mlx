import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PriorConfig:
    in_channels: int = 16
    out_channels: int = 16
    timestep_ratio_embedding_dim: int = 64
    patch_size: int = 1
    conditioning_dim: int = 2048
    block_out_channels: tuple = (2048, 2048)
    num_attention_heads: tuple = (32, 32)
    down_num_layers_per_block: tuple = (8, 24)
    up_num_layers_per_block: tuple = (24, 8)
    down_blocks_repeat_mappers: tuple = (1, 1)
    up_blocks_repeat_mappers: tuple = (1, 1)
    block_types_per_layer: tuple = (
        ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
        ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
    )
    clip_text_in_channels: int = 1280
    clip_text_pooled_in_channels: int = 1280
    clip_image_in_channels: int = 768
    clip_seq: int = 4
    effnet_in_channels: int | None = None
    pixel_mapper_in_channels: int | None = None
    kernel_size: int = 3
    dropout: tuple = (0.1, 0.1)
    self_attn: bool = True
    timestep_conditioning_type: tuple = ("sca", "crp")
    switch_level: tuple = (False,)


@dataclass
class DecoderConfig:
    in_channels: int = 4
    out_channels: int = 4
    timestep_ratio_embedding_dim: int = 64
    patch_size: int = 2
    conditioning_dim: int = 1280
    block_out_channels: tuple = (320, 640, 1280, 1280)
    num_attention_heads: tuple = (0, 0, 20, 20)
    down_num_layers_per_block: tuple = (2, 6, 28, 6)
    up_num_layers_per_block: tuple = (6, 28, 6, 2)
    down_blocks_repeat_mappers: tuple = (1, 1, 1, 1)
    up_blocks_repeat_mappers: tuple = (3, 3, 2, 2)
    block_types_per_layer: tuple = (
        ("SDCascadeResBlock", "SDCascadeTimestepBlock"),
        ("SDCascadeResBlock", "SDCascadeTimestepBlock"),
        ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
        ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
    )
    clip_text_in_channels: int | None = None
    clip_text_pooled_in_channels: int = 1280
    clip_image_in_channels: int | None = None
    clip_seq: int = 4
    effnet_in_channels: int = 16
    pixel_mapper_in_channels: int = 3
    kernel_size: int = 3
    dropout: tuple = (0, 0, 0.1, 0.1)
    self_attn: bool = True
    timestep_conditioning_type: tuple = ("sca",)
    switch_level: tuple | None = None


@dataclass
class VQGANConfig:
    in_channels: int = 3
    out_channels: int = 3
    up_down_scale_factor: int = 2
    levels: int = 2
    bottleneck_blocks: int = 12
    embed_dim: int = 384
    latent_channels: int = 4
    num_vq_embeddings: int = 8192
    scale_factor: float = 0.3764


@dataclass
class EffnetConfig:
    c_hidden: tuple = (24, 48, 96, 192, 384, 768, 1024)
    blwidth: tuple = (1, 1, 2, 4, 4)
    blocktype: tuple = ("C", "C", "T", "T", "A")
    c_effnet: int = 16
    c_latent: int = 16
    in_channels: int = 3
    patch_size: int = 16


@dataclass
class TextEncoderConfig:
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    max_position_embeddings: int = 77
    vocab_size: int = 49408
    hidden_act: str = "gelu"
    projection_dim: int = 1280


@dataclass
class CascadeModelPaths:
    prior_repo: str = "stabilityai/stable-cascade-prior"
    prior_subfolder: str = "prior"
    prior_file: str = "diffusion_pytorch_model.bf16.safetensors"
    decoder_repo: str = "stabilityai/stable-cascade"
    decoder_subfolder: str = "decoder"
    decoder_file: str = "diffusion_pytorch_model.bf16.safetensors"
    vqgan_repo: str = "stabilityai/stable-cascade"
    vqgan_subfolder: str = "vqgan"
    vqgan_file: str = "diffusion_pytorch_model.safetensors"
    text_encoder_repo: str = "stabilityai/stable-cascade-prior"
    text_encoder_subfolder: str = "text_encoder"
    text_encoder_file: str = "model.bf16.safetensors"
    tokenizer_repo: str = "stabilityai/stable-cascade-prior"
    tokenizer_subfolder: str = "tokenizer"
    effnet_repo: str = "stabilityai/stable-cascade"
    effnet_file: str = "effnet_encoder.safetensors"

    def __post_init__(self):
        repo = os.environ.get("CASCADE_PRIOR_REPO")
        if repo:
            self.prior_repo = repo
            logger.info("Cascade prior repo override: %s", repo)
        repo = os.environ.get("CASCADE_DECODER_REPO")
        if repo:
            self.decoder_repo = repo
            logger.info("Cascade decoder repo override: %s", repo)
        repo = os.environ.get("CASCADE_VQGAN_REPO")
        if repo:
            self.vqgan_repo = repo
            logger.info("Cascade vqgan repo override: %s", repo)
        repo = os.environ.get("CASCADE_TEXT_REPO")
        if repo:
            self.text_encoder_repo = repo
            logger.info("Cascade text repo override: %s", repo)
        for attr, env in (
            ("prior_file", "CASCADE_PRIOR_FILE"),
            ("decoder_file", "CASCADE_DECODER_FILE"),
            ("vqgan_file", "CASCADE_VQGAN_FILE"),
            ("text_encoder_file", "CASCADE_TEXT_FILE"),
            ("effnet_file", "CASCADE_EFFNET_FILE"),
        ):
            val = os.environ.get(env)
            if val:
                setattr(self, attr, val)
                logger.info("Cascade path override %s=%s", env, val)


@dataclass
class CascadeConfig:
    prior: PriorConfig = field(default_factory=PriorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    vqgan: VQGANConfig = field(default_factory=VQGANConfig)
    effnet: EffnetConfig = field(default_factory=EffnetConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    paths: CascadeModelPaths = field(default_factory=CascadeModelPaths)
    resolution_multiple: float = 42.67
    latent_dim_scale: float = 10.67
    scheduler_s: float = 0.008
    scheduler_scaler: float = 1.0
