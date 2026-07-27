# SPDX-License-Identifier: Apache-2.0
# UniWorld-V1 projectors and task_head as pure MLX nn.Modules.
# denoise_projector: hidden_size -> output*3 -> output (SiLU)
# vae_projector: 64 -> 3072 -> 4096 (SiLU)
# siglip_projector: 1152 -> 4096*3 -> 4096 (SiLU)
# task_head: 3584 -> 10240 -> 2 (SiLU + Dropout)

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class DenoiseProjector(nn.Module):
    def __init__(self, input_dim: int = 3584, hidden_dim: int = 9216,
                 output_dim: int = 3072):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.silu(x)
        x = self.fc2(x)
        return x


class VAEProjector(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 3072,
                 output_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.silu(x)
        x = self.fc2(x)
        return x


class SigLIPProjector(nn.Module):
    def __init__(self, input_dim: int = 1152, hidden_dim: int = 12288,
                 output_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.silu(x)
        x = self.fc2(x)
        return x


class TaskHead(nn.Module):
    def __init__(self, input_dim: int = 3584, hidden_dim: int = 10240,
                 output_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        x = self.fc1(x)
        x = nn.silu(x)
        if training and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)
        x = self.fc2(x)
        return x


class UniWorldProjectors(nn.Module):
    def __init__(self, denoise_in: int = 3584, denoise_hidden: int = 9216,
                 denoise_out: int = 3072, vae_in: int = 64,
                 vae_hidden: int = 3072, vae_out: int = 4096,
                 siglip_in: int = 1152, siglip_hidden: int = 12288,
                 siglip_out: int = 4096):
        super().__init__()
        self.denoise_projector = DenoiseProjector(denoise_in, denoise_hidden, denoise_out)
        self.vae_projector = VAEProjector(vae_in, vae_hidden, vae_out)
        self.siglip_projector = SigLIPProjector(siglip_in, siglip_hidden, siglip_out)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, **kwargs) -> "UniWorldProjectors":
        model_dir = Path(model_dir)
        logger.info("Loading UniWorld projectors from %s", model_dir)
        proj = cls(**kwargs)
        weight_file = model_dir / "projectors.safetensors"
        if not weight_file.exists():
            candidates = list(model_dir.glob("*.safetensors"))
            if candidates:
                weight_file = candidates[0]
        if weight_file.exists():
            weights = mx.load(str(weight_file))
            filtered = _remap_projector_weights(weights)
            proj.load_weights(list(filtered.items()))
            mx.eval(proj.parameters())
            logger.info("Projectors loaded %d weights from %s", len(filtered), weight_file.name)
        else:
            logger.warning("No projector weights found in %s", model_dir)
        return proj

    def __call__(self, vlm_hidden: mx.array, vae_latent: mx.array,
                 siglip_hidden: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        denoise_embeds = self.denoise_projector(vlm_hidden)
        vae_embeds = self.vae_projector(vae_latent)
        siglip_embeds = self.siglip_projector(siglip_hidden)
        return denoise_embeds, vae_embeds, siglip_embeds


def _remap_projector_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    remapped = {}
    for k, v in weights.items():
        new_k = k
        if k.startswith("model.denoise_tower.denoise_projector."):
            new_k = k.replace("model.denoise_tower.denoise_projector.", "denoise_projector.")
        elif k.startswith("denoise_projector."):
            pass
        if k.startswith("model.denoise_tower.vae_projector."):
            new_k = k.replace("model.denoise_tower.vae_projector.", "vae_projector.")
        elif k.startswith("vae_projector."):
            pass
        if k.startswith("model.denoise_tower.siglip_projector."):
            new_k = k.replace("model.denoise_tower.siglip_projector.", "siglip_projector.")
        elif k.startswith("siglip_projector."):
            pass
        new_k = new_k.replace(".0.weight", ".fc1.weight")
        new_k = new_k.replace(".0.bias", ".fc1.bias")
        new_k = new_k.replace(".2.weight", ".fc2.weight")
        new_k = new_k.replace(".2.bias", ".fc2.bias")
        remapped[new_k] = v
    return remapped
