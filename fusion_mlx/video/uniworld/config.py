# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UniWorldConfig:
    vlm_model: str = "Qwen2.5-VL-7B-Instruct"
    vlm_hidden_size: int = 3584
    vlm_num_attention_heads: int = 28
    vlm_num_hidden_layers: int = 28
    vlm_intermediate_size: int = 18944
    vlm_num_key_value_heads: int = 4
    vlm_vocab_size: int = 152064
    vlm_rope_theta: float = 1000000.0

    siglip_model: str = "siglip2-so400m-patch16-512"
    siglip_hidden_size: int = 1152
    siglip_num_attention_heads: int = 16
    siglip_num_hidden_layers: int = 27
    siglip_intermediate_size: int = 4304
    siglip_patch_size: int = 16
    siglip_image_size: int = 512

    flux_hidden_size: int = 3072
    flux_num_heads: int = 24
    flux_num_layers: int = 19
    flux_num_single_layers: int = 38
    flux_mlp_ratio: float = 4.0
    flux_guidance_embed: bool = True
    flux_in_channels: int = 64
    flux_out_channels: int = 4
    flux_vae_scale_factor: int = 16

    denoise_projector_input: int = 3584
    denoise_projector_hidden: int = 9216
    denoise_projector_output: int = 3072

    vae_projector_input: int = 64
    vae_projector_hidden: int = 3072
    vae_projector_output: int = 4096

    siglip_projector_input: int = 1152
    siglip_projector_hidden: int = 12288
    siglip_projector_output: int = 4096

    task_head_input: int = 3584
    task_head_hidden: int = 10240
    task_head_output: int = 2
    task_head_dropout: float = 0.3

    shortcut_scale: float = 0.5
    vlm_residual_image_factor: float = 0.3

    no_joint_with_t5: bool = False
    denoise_steps: int = 50
    guidance_scale: float = 3.5

    model_path: str = ""
    dtype: str = "float16"

    @property
    def model_dir(self) -> Path:
        if self.model_path:
            return Path(self.model_path)
        base = Path.home() / ".fusion-mlx" / "models"
        return base / "uniworld-v1"

    @property
    def vlm_dir(self) -> Path:
        return self.model_dir / "vlm"

    @property
    def siglip_dir(self) -> Path:
        return self.model_dir / "siglip"

    @property
    def flux_dir(self) -> Path:
        return self.model_dir / "flux"

    @property
    def projectors_dir(self) -> Path:
        return self.model_dir / "projectors"

    @property
    def mx_dtype(self) -> Any:
        import mlx.core as mx

        return mx.float16 if self.dtype == "float16" else mx.bfloat16

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: Any) -> "UniWorldConfig":
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            import json

            with open(config_path) as f:
                data = json.load(f)
            merged = {**data, **kwargs, "model_path": model_path}
            known = {k: v for k, v in merged.items() if k in cls.__dataclass_fields__}
            unknown = {
                k: v for k, v in merged.items() if k not in cls.__dataclass_fields__
            }
            if unknown:
                logger.warning(
                    "UniWorldConfig: ignoring unknown keys: %s", list(unknown.keys())
                )
            return cls(**known)
        logger.info("No config.json found at %s, using defaults", model_path)
        return cls(model_path=model_path, **kwargs)
