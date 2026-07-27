# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OpenSoraConfig:
    in_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: list = field(default_factory=lambda: [16, 56, 56])
    theta: int = 10000
    qkv_bias: bool = True
    guidance_embed: bool = False
    cond_embed: bool = True
    fused_qkv: bool = False
    patch_size: int = 2
    # VAE
    vae_in_channels: int = 3
    vae_out_channels: int = 3
    vae_latent_channels: int = 16
    vae_layers_per_block: int = 2
    vae_block_out_channels: tuple = (128, 256, 512, 512)
    vae_time_compression_ratio: int = 4
    vae_spatial_compression_ratio: int = 8
    # Text encoder
    t5_model: str = "google/t5-v1_1-xxl"
    t5_max_length: int = 512
    clip_model: str = "openai/clip-vit-large-patch14"
    clip_max_length: int = 77
    # Sampling defaults
    default_num_steps: int = 50
    default_guidance: float = 7.5
    default_guidance_img: float = 3.0
    temporal_reduction: int = 4
    is_causal_vae: bool = True

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
