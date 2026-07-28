# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)


@dataclass
class CogVideoXConfig:
    num_attention_heads: int = 30
    attention_head_dim: int = 64
    in_channels: int = 16
    out_channels: int = 16
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    time_embed_dim: int = 512
    ofs_embed_dim: int | None = None
    text_embed_dim: int = 4096
    num_layers: int = 30
    dropout: float = 0.0
    attention_bias: bool = True
    sample_width: int = 90
    sample_height: int = 60
    sample_frames: int = 49
    patch_size: int = 2
    patch_size_t: int | None = None
    temporal_compression_ratio: int = 4
    max_text_seq_length: int = 226
    activation_fn: str = "gelu-approximate"
    timestep_activation_fn: str = "silu"
    norm_elementwise_affine: bool = True
    norm_eps: float = 1e-5
    spatial_interpolation_scale: float = 1.875
    temporal_interpolation_scale: float = 1.0
    use_rotary_positional_embeddings: bool = False
    use_learned_positional_embeddings: bool = False
    patch_bias: bool = True
    sample_steps: int = 50
    sample_fps: int = 8
    sample_guide_scale: float = 6.0
    sample_shift: float = 3.0
    num_train_timesteps: int = 1000
    model_type: str = "t2v"
    scaling_factor: float = 1.15258426
    vae_temporal_compression_ratio: int = 4
    vae_block_out_channels: tuple = (128, 256, 256, 512)
    vae_layers_per_block: int = 3
    vae_latent_channels: int = 16

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def ff_inner_dim(self) -> int:
        return self.inner_dim * 4

    @classmethod
    def cogvideox_2b(cls) -> "CogVideoXConfig":
        return cls(
            num_attention_heads=30,
            attention_head_dim=64,
            num_layers=30,
            text_embed_dim=4096,
            time_embed_dim=512,
            use_rotary_positional_embeddings=False,
            use_learned_positional_embeddings=False,
            sample_steps=50,
            sample_guide_scale=6.0,
            sample_shift=3.0,
            model_type="t2v",
        )

    @classmethod
    def cogvideox_5b(cls) -> "CogVideoXConfig":
        return cls(
            num_attention_heads=48,
            attention_head_dim=64,
            num_layers=42,
            text_embed_dim=4096,
            time_embed_dim=512,
            use_rotary_positional_embeddings=True,
            use_learned_positional_embeddings=False,
            sample_steps=50,
            sample_guide_scale=6.0,
            sample_shift=3.0,
            model_type="t2v",
        )

    @classmethod
    def cogvideox_5b_i2v(cls) -> "CogVideoXConfig":
        cfg = cls.cogvideox_5b()
        cfg.model_type = "i2v"
        return cfg

    @classmethod
    def from_dict(cls, d: dict) -> "CogVideoXConfig":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid}
        for key in ("vae_block_out_channels",):
            if key in filtered and isinstance(filtered[key], list):
                filtered[key] = tuple(filtered[key])
        return cls(**filtered)

    def to_dict(self) -> dict:
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, tuple):
                v = list(v)
            d[f.name] = v
        return d
