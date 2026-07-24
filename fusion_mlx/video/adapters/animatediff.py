# SPDX-License-Identifier: Apache-2.0
# AnimateDiff: LoRA-based temporal adaptation for Wan DiT.
# cnforge/wan2_2_AnimateDiff_Lora: rank-32 LoRA applied to self_attn/cross_attn/ffn
# of each DiT block. inject() merges LoRA deltas into DiT weights; remove() restores.
# Two variants: high_noise (strong motion) and low_noise (subtle motion).

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import VideoAdapter, register_adapter

logger = logging.getLogger(__name__)

LORA_RANK = 32
DEFAULT_REPO = "cnforge/wan2_2_AnimateDiff_Lora"
DEFAULT_VARIANT = "high_noise"

_VARIANT_FILES = {
    "high_noise": "wan22_14B_AD_REDIFF_Lora_high_noise.safetensors",
    "low_noise": "wan22_14B_AD_REDIFF_Lora_low_noise.safetensors",
}


def remap_animatediff_lora_weights(
    raw_weights: dict, num_layers: int
) -> dict[str, dict[str, mx.array]]:
    """Remap HF AnimateDiff LoRA weights into per-target {lora_A, lora_B} dicts.

    HF naming: diffusion_model.blocks.{N}.{sublayer}.lora_{A,B}.weight
    MLX DiT:   blocks.{N}.{sublayer}.weight
      ffn.0 -> ffn.fc1, ffn.2 -> ffn.fc2

    Returns: {mlx_target_key: {"lora_A": array, "lora_B": array}}
    """
    lora_map: dict[str, dict[str, mx.array]] = {}

    for hf_key, value in raw_weights.items():
        clean = hf_key
        if clean.startswith("diffusion_model."):
            clean = clean[len("diffusion_model."):]

        m = re.match(r"blocks\.(\d+)\.(.*)", clean)
        if not m:
            continue

        block_idx = int(m.group(1))
        if block_idx >= num_layers:
            continue

        rest = m.group(2)
        lora_match = re.match(r"(.*)\.lora_([AB])\.weight", rest)
        if not lora_match:
            continue

        sublayer = lora_match.group(1)
        ab = lora_match.group(2)

        sublayer = sublayer.replace("ffn.0", "ffn.fc1").replace("ffn.2", "ffn.fc2")

        target_key = f"blocks.{block_idx}.{sublayer}.weight"

        if target_key not in lora_map:
            lora_map[target_key] = {}
        lora_map[target_key][f"lora_{ab}"] = value

    return lora_map


@register_adapter("animatediff")
class AnimateDiff(VideoAdapter):
    """AnimateDiff: LoRA-based temporal adaptation for Wan DiT.

    inject() merges LoRA weight deltas into the DiT's Linear layers:
        W_merged = W_base + scale * lora_B @ lora_A
    remove() restores the original DiT weights.
    """

    name = "animatediff"

    def __init__(
        self,
        *,
        scale: float = 1.0,
        image: str | None = None,
        config: dict | None = None,
    ):
        self.scale = scale
        self.image = image
        self.config = config or {}
        self.num_layers = self.config.get("num_layers", 40)
        self.variant = self.config.get("animatediff_variant", DEFAULT_VARIANT)
        self._lora_map: dict[str, dict[str, mx.array]] = {}
        self._original_weights: dict[str, mx.array] = {}
        self._loaded = False
        self._injected = False

    def load(self, model_path: str | None = None) -> None:
        if self._loaded:
            logger.debug("AnimateDiff: already loaded, skipping")
            return

        loaded = False
        if model_path is not None:
            loaded = self._load_weights(model_path)
        if not loaded:
            loaded = self._load_from_hf()
        if not loaded:
            logger.warning(
                "AnimateDiff: LoRA weights not loaded, inject will be no-op"
            )

        self._loaded = True
        logger.info(
            "AnimateDiff: loaded (weights=%s scale=%.2f variant=%s layers=%d)",
            loaded,
            self.scale,
            self.variant,
            self.num_layers,
        )

    def _load_weights(self, model_path: str) -> bool:
        import glob

        path = Path(model_path)
        search_dirs = [path]
        for subdir in ("animatediff", "lora", ""):
            candidate = path / subdir
            if candidate.is_dir():
                search_dirs.append(candidate)

        for search_dir in search_dirs:
            safetensors = sorted(glob.glob(str(search_dir / "*.safetensors")))
            if not safetensors:
                continue

            for sf in safetensors:
                fname = Path(sf).name.lower()
                if "animatediff" not in fname and "ad_re" not in fname:
                    continue
                try:
                    raw = mx.load(sf)
                    lora_map = remap_animatediff_lora_weights(raw, self.num_layers)
                    if lora_map:
                        self._lora_map.update(lora_map)
                        logger.info(
                            "AnimateDiff: LoRA loaded from %s (%d targets)",
                            sf,
                            len(lora_map),
                        )
                        return True
                except Exception as exc:
                    logger.warning(
                        "AnimateDiff: failed to load from %s: %s", sf, exc
                    )

        return False

    def _load_from_hf(self) -> bool:
        try:
            import os

            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from huggingface_hub import hf_hub_download

            repo = self.config.get("animatediff_repo", DEFAULT_REPO)
            variant = self.variant
            target_file = _VARIANT_FILES.get(variant)
            if target_file is None:
                logger.warning(
                    "AnimateDiff: unknown variant '%s', available: %s",
                    variant,
                    list(_VARIANT_FILES),
                )
                return False

            logger.info("AnimateDiff: downloading %s/%s", repo, target_file)
            sf = hf_hub_download(repo, target_file)

            raw = {}
            try:
                raw = mx.load(sf)
            except Exception:
                from safetensors import safe_open

                try:
                    with safe_open(sf, framework="pt") as f:
                        import numpy as np

                        for k in f.keys():
                            t = f.get_tensor(k)
                            arr = t.detach().cpu().float().numpy()
                            raw[k] = mx.array(arr)
                except Exception as exc2:
                    logger.warning("AnimateDiff: safetensors load failed: %s", exc2)
                    return False

            lora_map = remap_animatediff_lora_weights(raw, self.num_layers)
            if lora_map:
                self._lora_map = lora_map
                logger.info(
                    "AnimateDiff: LoRA loaded from HF (%d targets from %s)",
                    len(lora_map),
                    target_file,
                )
                return True
            else:
                logger.warning("AnimateDiff: no compatible LoRA keys in %s", target_file)
        except ImportError:
            logger.debug("AnimateDiff: huggingface_hub not installed, skipping HF")
        except Exception as exc:
            logger.warning("AnimateDiff: HF load failed: %s", exc)
        return False

    def unload(self) -> None:
        if self._injected:
            logger.warning("AnimateDiff: unload called while injected, removing first")
        self._lora_map = {}
        self._original_weights = {}
        self._loaded = False
        self._injected = False
        logger.info("AnimateDiff: unloaded")

    def inject(self, dit: Any) -> None:
        if self._injected:
            logger.debug("AnimateDiff: already injected, skipping")
            return

        if not self._loaded:
            self.load()

        if not self._lora_map:
            logger.warning("AnimateDiff: no LoRA weights loaded, inject is no-op")
            self._injected = True
            return

        flat = nn.utils.tree_flatten(dit.parameters())
        param_map = {k: v for k, v in flat}

        merged_count = 0
        for target_key, lora_dict in self._lora_map.items():
            if target_key not in param_map:
                logger.debug("AnimateDiff: target %s not in DiT, skipping", target_key)
                continue

            lora_a = lora_dict.get("lora_A")
            lora_b = lora_dict.get("lora_B")
            if lora_a is None or lora_b is None:
                logger.debug("AnimateDiff: incomplete LoRA for %s, skipping", target_key)
                continue

            base_weight = param_map[target_key]

            delta = lora_b.astype(base_weight.dtype) @ lora_a.astype(base_weight.dtype)
            merged = base_weight + self.scale * delta

            self._original_weights[target_key] = base_weight
            param_map[target_key] = merged
            merged_count += 1

        if merged_count > 0:
            new_flat = [(k, param_map.get(k, v)) for k, v in flat]
            dit.update(nn.utils.tree_unflatten(new_flat))
            mx.eval(dit.parameters())

        self._injected = True
        logger.info(
            "AnimateDiff: injected %d/%d LoRA targets (scale=%.2f)",
            merged_count,
            len(self._lora_map),
            self.scale,
        )

    def remove(self, dit: Any) -> None:
        if not self._injected:
            return

        if not self._original_weights:
            self._injected = False
            return

        flat = nn.utils.tree_flatten(dit.parameters())
        param_map = {k: v for k, v in flat}

        restored_count = 0
        for target_key, orig_weight in self._original_weights.items():
            if target_key in param_map:
                param_map[target_key] = orig_weight
                restored_count += 1

        if restored_count > 0:
            new_flat = [(k, param_map.get(k, v)) for k, v in flat]
            dit.update(nn.utils.tree_unflatten(new_flat))
            mx.eval(dit.parameters())

        self._original_weights = {}
        self._injected = False
        logger.info("AnimateDiff: removed %d LoRA deltas from DiT", restored_count)

    def modify_denoise_step(
        self,
        dit: Any,
        latents: mx.array,
        t: mx.array,
        context: mx.array,
        **kw: Any,
    ) -> mx.array:
        return latents


__all__ = [
    "AnimateDiff",
    "remap_animatediff_lora_weights",
]
