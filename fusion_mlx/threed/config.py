# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 config parsing. Three config.json files (shape / paint / unirig)
# shipped by ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit. Parse into dataclasses so the
# port modules read a single typed object, not raw dict lookups.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ShapeConfig:
    model_type: str = "hunyuan3d_2_1"
    quant: str = "8bit"
    hidden_size: int = 2048
    depth: int = 21
    num_heads: int = 16
    context_dim: int = 1024
    num_latents: int = 4096
    embed_dim: int = 64
    vae_width: int = 1024
    vae_heads: int = 16
    vae_decoder_layers: int = 16
    num_freqs: int = 8
    scale_factor: float = 1.0039506158752403
    num_moe_layers: int = 6
    num_experts: int = 8
    moe_top_k: int = 2
    dino_hidden: int = 1024
    dino_layers: int = 24
    dino_heads: int = 16
    dino_patch: int = 14
    dino_image_size: int = 518

    @property
    def dino_group_size(self) -> int:
        # 8bit quant, group 64 (in=1024 -> 16 groups, verified from
        # conditioner.safetensors scales shape [1024, 16]).
        return 64

    @property
    def dino_num_tokens(self) -> int:
        # 518//14 = 37 patches per side -> 37*37 + 1 cls = 1370.
        return (self.dino_image_size // self.dino_patch) ** 2 + 1


@dataclass
class PaintUNetConfig:
    in_channels: int = 12
    out_channels: int = 4
    cross_attention_dim: int = 1024
    block_out_channels: list = field(default_factory=lambda: [320, 640, 1280, 1280])
    layers_per_block: int = 2
    attention_head_dim: list = field(default_factory=lambda: [5, 10, 20, 20])
    sample_size: int = 64
    norm_num_groups: int = 32
    norm_eps: float = 1e-05
    use_linear_projection: bool = True
    transformer_layers_per_block: int = 1


@dataclass
class PaintVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: list = field(default_factory=lambda: [128, 256, 512, 512])
    layers_per_block: int = 2
    norm_num_groups: int = 32
    sample_size: int = 768
    norm_eps: float = 1e-06
    scaling_factor: float = 0.18215


@dataclass
class PaintDinoConfig:
    # DINOv2-Giant ViT backbone for the paint pipeline. Feeds image_proj_model_dino
    # (1536 -> 4096 reshape 4x1024 + LayerNorm(1024)) in the paint UNet.
    # mlp is SwiGLU: w_in dim->mlp_hidden (fused gate+up, 2*intermediate),
    # gelu(gate)*up -> intermediate, w_out intermediate->dim.
    hidden: int = 1536
    layers: int = 40
    heads: int = 24
    head_dim: int = 64
    patch: int = 14
    image_size: int = 518
    mlp_hidden: int = 8192  # w_in out (fused gate+up = 2*intermediate)
    group_size: int = 64  # 1536/64 = 24 groups (scales shape [out, 24])

    @property
    def intermediate(self) -> int:
        return self.mlp_hidden // 2  # 4096 (w_out in_dim)

    @property
    def num_tokens(self) -> int:
        return (self.image_size // self.patch) ** 2 + 1  # 37*37 + 1 = 1370


@dataclass
class PaintConfig:
    model_type: str = "hunyuan3d_2_1_paint"
    quant: str = "8bit"
    pbr_settings: list = field(default_factory=lambda: ["albedo", "mr"])
    num_views: int = 6
    view_resolution: int = 512
    guidance_scale: float = 3.0
    num_inference_steps: int = 30
    unet: PaintUNetConfig = field(default_factory=PaintUNetConfig)
    vae: PaintVAEConfig = field(default_factory=PaintVAEConfig)
    dino: PaintDinoConfig = field(default_factory=PaintDinoConfig)


@dataclass
class UniRigConfig:
    model_type: str = "unirig_skeleton"
    quant: str = "8bit"


def load_shape_config(model_dir: str | Path) -> ShapeConfig:
    p = Path(model_dir) / "config.json"
    with open(p) as f:
        d = json.load(f)
    return ShapeConfig(**{k: d[k] for k in d if k in ShapeConfig.__dataclass_fields__})


def load_paint_config(model_dir: str | Path) -> PaintConfig:
    p = Path(model_dir) / "paint" / "config.json"
    with open(p) as f:
        d = json.load(f)
    unet_d = d.get("unet", {})
    unet = PaintUNetConfig(
        **{k: unet_d[k] for k in unet_d if k in PaintUNetConfig.__dataclass_fields__}
    )
    vae_d = d.get("vae", {})
    vae = PaintVAEConfig(
        **{k: vae_d[k] for k in vae_d if k in PaintVAEConfig.__dataclass_fields__}
    )
    dino_d = d.get("dino", {})
    # map paint config.json dino field names -> PaintDinoConfig fields.
    dino_map = {
        "hidden_size": "hidden",
        "num_layers": "layers",
        "num_heads": "heads",
        "head_dim": "head_dim",
        "patch_size": "patch",
        "image_size": "image_size",
        "intermediate_size": "mlp_hidden",  # 4096 -> but mlp_hidden is 8192 (fused)
    }
    dino_kwargs: dict = {}
    for src, dst in dino_map.items():
        if src in dino_d:
            if src == "intermediate_size":
                # SwiGLU: w_in fuses gate+up = 2 * intermediate_size.
                dino_kwargs[dst] = int(dino_d[src]) * 2
            else:
                dino_kwargs[dst] = dino_d[src]
    dino = PaintDinoConfig(**dino_kwargs)
    return PaintConfig(
        unet=unet,
        vae=vae,
        dino=dino,
        **{
            k: d[k]
            for k in d
            if k in PaintConfig.__dataclass_fields__
            and k not in ("unet", "vae", "dino")
        },
    )


def load_unirig_config(model_dir: str | Path) -> UniRigConfig:
    p = Path(model_dir) / "unirig" / "config.json"
    with open(p) as f:
        d = json.load(f)
    return UniRigConfig(
        **{k: d[k] for k in d if k in UniRigConfig.__dataclass_fields__}
    )
