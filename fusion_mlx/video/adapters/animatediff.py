# SPDX-License-Identifier: Apache-2.0
# AnimateDiff: temporal motion modules injected into DiT blocks.
# Importers: adapters/__init__.py eager import triggers @register_adapter("animatediff").
# Callers: SkyReelsPipelineConfig._denoise_sample() via create_adapter("animatediff"),
# videos_routes.py forwards animatediff_scale.
# Schema: motion_module input/output [B, L, dim]; zero-init output_proj = identity at start.
# User instruction: "Phase-3: AnimateDiff (temporal attention motion modules)"

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import VideoAdapter, register_adapter

logger = logging.getLogger(__name__)


class MotionModule(nn.Module):
    """Temporal motion module: self-attention across video frames.

    Reshapes [B, L, dim] -> [B*S, T, dim] where L = T * S,
    runs temporal self-attention, then reshapes back.
    Zero-initialized output projection ensures identity at init.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim, eps=eps)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.output_proj = nn.Linear(dim, dim)
        self.output_proj.weight = mx.zeros_like(self.output_proj.weight)
        self.output_proj.bias = mx.zeros_like(self.output_proj.bias)

    def __call__(self, x: mx.array, temporal_len: int) -> mx.array:
        B, L, D = x.shape
        if temporal_len is None or temporal_len <= 1:
            return mx.zeros_like(x)

        S = L // temporal_len
        if S * temporal_len != L:
            logger.warning(
                "MotionModule: L=%d not divisible by temporal_len=%d, skipping",
                L,
                temporal_len,
            )
            return mx.zeros_like(x)

        x_norm = self.norm(x)

        # Reshape: [B, T*S, D] -> [B*S, T, D]
        x_t = (
            x_norm.reshape(B, temporal_len, S, D)
            .transpose(0, 2, 1, 3)
            .reshape(B * S, temporal_len, D)
        )

        # Temporal self-attention
        q = self.q_proj(x_t)
        k = self.k_proj(x_t)
        v = self.v_proj(x_t)

        n = self.num_heads
        d = self.head_dim
        q = q.reshape(B * S, temporal_len, n, d).transpose(0, 2, 1, 3)
        k = k.reshape(B * S, temporal_len, n, d).transpose(0, 2, 1, 3)
        v = v.reshape(B * S, temporal_len, n, d).transpose(0, 2, 1, 3)

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
        attn = attn.transpose(0, 2, 1, 3).reshape(B * S, temporal_len, D)
        attn = self.o_proj(attn)

        # Reshape back: [B*S, T, D] -> [B, T*S, D]
        attn = (
            attn.reshape(B, S, temporal_len, D).transpose(0, 2, 1, 3).reshape(B, L, D)
        )

        out = self.output_proj(attn)
        return out


def remap_animatediff_weights(weights: dict, num_layers: int) -> dict:
    """Remap AnimateDiff weights from HF naming to MLX naming.

    HF naming patterns:
      motion_modules.{N}.norm.weight -> block_{N}.norm.weight
      motion_modules.{N}.temporal_attn.to_q.weight -> block_{N}.q_proj.weight
    """
    remapped = {}
    for key, value in weights.items():
        clean = key
        for prefix in ("motion_modules.", "animatediff.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]

        m = re.match(r"(\d+)\.(.*)", clean)
        if m:
            idx, rest = m.group(1), m.group(2)
            idx_int = int(idx)
            if idx_int >= num_layers:
                continue

            name_map = {
                "norm.weight": "norm.weight",
                "norm.bias": "norm.bias",
                "temporal_attn.to_q.weight": "q_proj.weight",
                "temporal_attn.to_q.bias": "q_proj.bias",
                "temporal_attn.to_k.weight": "k_proj.weight",
                "temporal_attn.to_k.bias": "k_proj.bias",
                "temporal_attn.to_v.weight": "v_proj.weight",
                "temporal_attn.to_v.bias": "v_proj.bias",
                "temporal_attn.to_out.0.weight": "o_proj.weight",
                "temporal_attn.to_out.0.bias": "o_proj.bias",
                "output_proj.weight": "output_proj.weight",
                "output_proj.bias": "output_proj.bias",
            }
            if rest in name_map:
                remapped[f"block_{idx}.{name_map[rest]}"] = value
            else:
                remapped[f"block_{idx}.{rest}"] = value
            continue

    return remapped


@register_adapter("animatediff")
class AnimateDiff(VideoAdapter):
    """AnimateDiff: temporal motion modules injected into DiT blocks.

    inject(): adds motion_module attr to each DiT block
    remove(): restores original DiT blocks
    animatediff_scale controls motion contribution strength
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
        self.dim = self.config.get("dim", 5120)
        self.num_heads = self.config.get("num_heads", 8)
        self.num_layers = self.config.get("num_layers", 40)
        self.eps = self.config.get("eps", 1e-5)
        self._modules: list[MotionModule] = []
        self._loaded = False
        self._injected = False

    def load(self, model_path: str | None = None) -> None:
        if self._loaded:
            logger.debug("AnimateDiff: already loaded, skipping")
            return

        for i in range(self.num_layers):
            self._modules.append(MotionModule(self.dim, self.num_heads, self.eps))

        loaded = False
        if model_path is not None:
            loaded = self._load_weights(model_path)
        if not loaded:
            loaded = self._load_from_hf_cache()
        if not loaded:
            logger.warning(
                "AnimateDiff: weights not loaded, zero-init (identity at start)"
            )

        self._loaded = True
        logger.info(
            "AnimateDiff: loaded (weights=%s scale=%.2f layers=%d dim=%d)",
            loaded,
            self.scale,
            self.num_layers,
            self.dim,
        )

    def _load_weights(self, model_path: str) -> bool:
        import glob

        path = Path(model_path)
        search_dirs = [path]
        for subdir in ("animatediff", "motion_modules", ""):
            candidate = path / subdir
            if candidate.is_dir():
                search_dirs.append(candidate)

        for search_dir in search_dirs:
            safetensors = sorted(glob.glob(str(search_dir / "*.safetensors")))
            if not safetensors:
                continue
            try:
                raw_weights = {}
                for wf in safetensors:
                    raw_weights.update(mx.load(wf))
                ad_keys = {
                    k: v
                    for k, v in raw_weights.items()
                    if "motion_module" in k or "animatediff" in k
                }
                if not ad_keys:
                    ad_keys = raw_weights
                remapped = remap_animatediff_weights(ad_keys, self.num_layers)
                if remapped:
                    for i, mod in enumerate(self._modules):
                        block_weights = {
                            k.replace(f"block_{i}.", ""): v
                            for k, v in remapped.items()
                            if k.startswith(f"block_{i}.")
                        }
                        if block_weights:
                            mod.load_weights(list(block_weights.items()))
                    logger.info(
                        "AnimateDiff: weights loaded from %s (%d params)",
                        search_dir,
                        len(remapped),
                    )
                    return True
            except Exception as exc:
                logger.warning(
                    "AnimateDiff: failed to load from %s: %s", search_dir, exc
                )
        return False

    def _load_from_hf_cache(self) -> bool:
        try:
            from huggingface_hub import hf_hub_download

            ad_repo = self.config.get(
                "animatediff_repo",
                "guoyww/AnimateDiff",
            )
            logger.info("AnimateDiff: trying HF cache: %s", ad_repo)

            sf = hf_hub_download(ad_repo, "model.safetensors")
            raw = mx.load(sf)
            ad_keys = {
                k: v
                for k, v in raw.items()
                if "motion_module" in k or "animatediff" in k
            }
            if not ad_keys:
                ad_keys = raw
            remapped = remap_animatediff_weights(ad_keys, self.num_layers)
            if remapped:
                for i, mod in enumerate(self._modules):
                    block_weights = {
                        k.replace(f"block_{i}.", ""): v
                        for k, v in remapped.items()
                        if k.startswith(f"block_{i}.")
                    }
                    if block_weights:
                        mod.load_weights(list(block_weights.items()))
                logger.info(
                    "AnimateDiff: weights loaded from HF (%d params)", len(remapped)
                )
                return True
        except ImportError:
            logger.debug(
                "AnimateDiff: huggingface_hub not installed, skipping HF download"
            )
        except Exception as exc:
            logger.warning("AnimateDiff: HF load failed: %s", exc)
        return False

    def unload(self) -> None:
        self._modules = []
        self._loaded = False
        self._injected = False
        logger.info("AnimateDiff: unloaded")

    def inject(self, dit: Any) -> None:
        if self._injected:
            logger.debug("AnimateDiff: already injected, skipping")
            return

        if not self._loaded:
            self.load()

        num_blocks = self._get_num_blocks(dit)
        inject_count = min(num_blocks, len(self._modules))
        if inject_count < num_blocks:
            logger.warning(
                "AnimateDiff: %d motion modules for %d DiT blocks, injecting first %d",
                len(self._modules),
                num_blocks,
                inject_count,
            )

        for i in range(inject_count):
            block = self._get_block(dit, i)
            if block is not None:
                block.motion_module = self._modules[i]
                block.animatediff_scale = self.scale

        self._injected = True
        logger.info("AnimateDiff: injected %d motion modules", inject_count)

    def remove(self, dit: Any) -> None:
        if not self._injected:
            return

        num_blocks = self._get_num_blocks(dit)
        for i in range(num_blocks):
            block = self._get_block(dit, i)
            if block is not None and hasattr(block, "motion_module"):
                delattr(block, "motion_module")
                if hasattr(block, "animatediff_scale"):
                    delattr(block, "animatediff_scale")

        self._injected = False
        logger.info("AnimateDiff: removed motion modules from %d blocks", num_blocks)

    def _get_num_blocks(self, dit: Any) -> int:
        if hasattr(dit, "_num_blocks"):
            return dit._num_blocks
        count = 0
        while hasattr(dit, f"block_{count}"):
            count += 1
        return count

    def _get_block(self, dit: Any, idx: int) -> Any | None:
        if hasattr(dit, f"block_{idx}"):
            return getattr(dit, f"block_{idx}")
        return None

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
    "MotionModule",
    "remap_animatediff_weights",
]
