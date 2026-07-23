# SPDX-License-Identifier: Apache-2.0
"""ControlNet adapter for video DiT (SkyReels-V3 / Wan2.2).

ControlNet adds structural guidance to the denoising process by:
  1. Encoding a control image (Canny edge, depth map, pose, etc.) into latents
  2. Running a parallel smaller DiT that processes control latents + same t/context
  3. Injecting per-block residuals into the main DiT's forward pass

Architecture:
  - ControlNetDiT: mirrors main DiT block structure but with fewer layers/dim
  - Each block outputs a residual added to the corresponding main DiT block output
  - Zero-convolution: each block output passes through a zero-initialized Linear
    so that at init, ControlNet contributes nothing (training starts from identity)

Weight layout (HF naming):
  controlnet.blocks.{N}.<same as main DiT block>
  controlnet.zero_convs.{N}.weight / bias
  controlnet.input_hint_block.* (control image preprocessing conv stack)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import VideoAdapter, register_adapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zero convolution: zero-initialized Linear for stable training start
# ---------------------------------------------------------------------------


class ZeroConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.linear.weight = mx.zeros_like(self.linear.weight)
        self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Hint block: preprocess control image (Canny/depth/pose) to latent space
# Conv stack: Conv2d -> Silu -> Conv2d -> Silu -> Conv2d
# ---------------------------------------------------------------------------


class HintBlock(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 5120):
        super().__init__()
        mid = dim // 4
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid, dim, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.silu(self.conv1(x))
        x = nn.silu(self.conv2(x))
        x = self.conv3(x)
        return x


# ---------------------------------------------------------------------------
# ControlNet DiT Block (lighter than main DiT block)
# Reuses same structure as SkyReelsDiTBlock but with reduced ffn_dim
# ---------------------------------------------------------------------------


class ControlNetDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = nn.MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.cross_attn = nn.MultiHeadAttention(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.zero_conv = ZeroConv(dim)

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        context: mx.array,
    ) -> tuple[mx.array, mx.array]:
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)

        x_norm2 = self.norm2(x)
        x = x + self.cross_attn(x_norm2, context, context)

        x_norm3 = self.norm3(x)
        x = x + self.ffn(x_norm3)

        residual = self.zero_conv(x)
        return x, residual


# ---------------------------------------------------------------------------
# ControlNet DiT: parallel smaller DiT producing per-block residuals
# ---------------------------------------------------------------------------


class ControlNetDiT(nn.Module):
    def __init__(
        self,
        dim: int = 5120,
        ffn_dim: int = 6912,
        num_heads: int = 40,
        num_layers: int = 20,
        in_channels: int = 3,
        text_dim: int = 4096,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.hint_block = HintBlock(in_channels, dim)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(256, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        for i in range(num_layers):
            setattr(
                self, f"block_{i}", ControlNetDiTBlock(dim, ffn_dim, num_heads, eps)
            )

        self.mid_zero_conv = ZeroConv(dim)

    def forward(
        self,
        control_latent: mx.array,
        t: mx.array,
        context: mx.array,
    ) -> list[mx.array]:
        """Run ControlNet forward, return per-block residuals.

        Args:
            control_latent: [B, C, H, W] control image (Canny/depth etc)
            t: [B] timestep
            context: [B, L, text_dim] text context

        Returns:
            List of num_layers + 1 residuals, each [B, L, dim]
        """
        x = self.hint_block(control_latent)
        B = x.shape[0]
        x = x.reshape(B, self.dim, -1).transpose(0, 2, 1)

        ctx = self.text_embedding(context)
        e = self.time_embedding(t)

        residuals = []
        for i in range(self.num_layers):
            block = getattr(self, f"block_{i}")
            x, res = block(x, e, ctx)
            residuals.append(res)

        residuals.append(self.mid_zero_conv(x))

        return residuals


# ---------------------------------------------------------------------------
# Control image preprocessing utilities
# ---------------------------------------------------------------------------


def preprocess_canny(
    image_path: str, low_threshold: int = 100, high_threshold: int = 200
) -> mx.array:
    """Load image and apply Canny edge detection. Returns [1, 3, H, W]."""
    import numpy as np
    from PIL import Image

    try:
        import cv2

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        edges = cv2.Canny(img_np, low_threshold, high_threshold)
        edges = np.stack([edges] * 3, axis=-1).astype(np.float32) / 255.0
        return mx.array(edges[np.newaxis].transpose(0, 3, 1, 2))
    except ImportError:
        logger.warning("OpenCV not installed, falling back to grayscale edge detection")
        img = Image.open(image_path).convert("L")
        from PIL import ImageFilter

        edges = img.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges).astype(np.float32) / 255.0
        arr = np.stack([arr] * 3, axis=-1)
        return mx.array(arr[np.newaxis].transpose(0, 3, 1, 2))


def preprocess_depth(image_path: str) -> mx.array:
    """Load depth map image. Returns [1, 3, H, W]."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return mx.array(arr[np.newaxis].transpose(0, 3, 1, 2))


def preprocess_control_image(
    image_path: str,
    control_type: str = "canny",
) -> mx.array:
    """Preprocess control image based on type.

    Args:
        image_path: path to control image
        control_type: "canny", "depth", "pose", or "raw"

    Returns:
        [1, C, H, W] control latent
    """
    if control_type == "canny":
        return preprocess_canny(image_path)
    elif control_type == "depth":
        return preprocess_depth(image_path)
    else:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return mx.array(arr[np.newaxis].transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Weight remapping (HF -> MLX)
# ---------------------------------------------------------------------------


def remap_controlnet_weights(weights: dict, num_layers: int = 20) -> dict:
    """Remap ControlNet weights from HF naming to MLX ControlNetDiT naming.

    HF naming patterns:
      controlnet.input_hint_block.{N}.weight -> hint_block.conv{N}.weight
      controlnet.blocks.{N}.* -> block_{N}.*
      controlnet.zero_convs.{N}.* -> block_{N}.zero_conv.*
      controlnet.middle_block.* -> mid_zero_conv.*
    """
    remapped = {}
    for key, value in weights.items():
        clean = key
        for prefix in ("controlnet.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]

        hint_map = {
            "input_hint_block.0": "hint_block.conv1",
            "input_hint_block.2": "hint_block.conv2",
            "input_hint_block.4": "hint_block.conv3",
        }
        for old, new in hint_map.items():
            if clean.startswith(old):
                param = clean[len(old) :]
                remapped[f"{new}{param}"] = value
                break
        else:
            m = re.match(r"zero_convs\.(\d+)\.(.*)", clean)
            if m:
                idx, param = m.group(1), m.group(2)
                remapped[f"block_{idx}.zero_conv.{param}"] = value
                continue

            m = re.match(r"blocks\.(\d+)\.(.*)", clean)
            if m:
                idx, rest = m.group(1), m.group(2)
                block_map = {
                    "norm1.weight": "norm1.weight",
                    "norm1.bias": "norm1.bias",
                    "attn1.to_q.weight": "self_attn.query_proj.weight",
                    "attn1.to_k.weight": "self_attn.key_proj.weight",
                    "attn1.to_v.weight": "self_attn.value_proj.weight",
                    "attn1.to_out.0.weight": "self_attn.out_proj.weight",
                    "attn1.to_out.0.bias": "self_attn.out_proj.bias",
                    "norm2.weight": "norm2.weight",
                    "norm2.bias": "norm2.bias",
                    "attn2.to_q.weight": "cross_attn.query_proj.weight",
                    "attn2.to_k.weight": "cross_attn.key_proj.weight",
                    "attn2.to_v.weight": "cross_attn.value_proj.weight",
                    "attn2.to_out.0.weight": "cross_attn.out_proj.weight",
                    "attn2.to_out.0.bias": "cross_attn.out_proj.bias",
                    "norm3.weight": "norm3.weight",
                    "norm3.bias": "norm3.bias",
                    "ff.net.0.proj.weight": "ffn.0.weight",
                    "ff.net.0.proj.bias": "ffn.0.bias",
                    "ff.net.2.weight": "ffn.2.weight",
                    "ff.net.2.bias": "ffn.2.bias",
                }
                if rest in block_map:
                    remapped[f"block_{idx}.{block_map[rest]}"] = value
                else:
                    remapped[f"block_{idx}.{rest}"] = value
                continue

            if clean.startswith("middle_block."):
                param = clean[len("middle_block.") :]
                remapped[f"mid_zero_conv.{param}"] = value
                continue

            for emb_prefix, emb_target in [
                ("time_embedding.", "time_embedding."),
                ("text_embedding.", "text_embedding."),
            ]:
                if clean.startswith(emb_prefix):
                    remapped[f"{emb_target}{clean[len(emb_prefix):]}"] = value
                    break

    return remapped


# ---------------------------------------------------------------------------
# ControlNet: main adapter class
# ---------------------------------------------------------------------------


@register_adapter("controlnet")
class ControlNet(VideoAdapter):
    """ControlNet: structural guidance via parallel DiT + per-block residual injection.

    Workflow:
      1. Control image (Canny/depth/pose) -> hint_block -> control latents
      2. Control latents + t + context -> ControlNetDiT -> per-block residuals
      3. Residuals added to main DiT block outputs during denoising
      4. controlnet_strength scales the residual contribution

    Integration with main DiT:
      - modify_denoise_step() pre-computes residuals from ControlNet forward
      - Residuals are stored in self._residuals for the pipeline to inject
      - Pipeline must call get_residuals() and pass to _run_blocks(controlnet_residuals=...)
    """

    name = "controlnet"

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
        self.ffn_dim = self.config.get("ffn_dim", 6912)
        self.num_heads = self.config.get("num_heads", 40)
        self.num_layers = self.config.get("num_layers", 20)
        self.text_dim = self.config.get("text_dim", 4096)
        self.control_type = self.config.get("control_type", "canny")
        self.in_channels = self.config.get("in_channels", 3)
        self.dit: ControlNetDiT | None = None
        self._loaded = False
        self._residuals: list[mx.array] | None = None

    def load(self, model_path: str | None = None) -> None:
        if self._loaded:
            logger.debug("ControlNet: already loaded, skipping")
            return

        self.dit = ControlNetDiT(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            in_channels=self.in_channels,
            text_dim=self.text_dim,
        )
        logger.info(
            "ControlNet: created DiT (dim=%d ffn=%d layers=%d heads=%d)",
            self.dim,
            self.ffn_dim,
            self.num_layers,
            self.num_heads,
        )

        loaded = False
        if model_path is not None:
            loaded = self._load_weights(model_path)
        if not loaded:
            loaded = self._load_from_hf_cache()
        if not loaded:
            logger.warning(
                "ControlNet: weights not loaded, random init (zero_conv ensures identity)"
            )

        self._loaded = True
        logger.info(
            "ControlNet: loaded (weights=%s scale=%.2f type=%s)",
            loaded,
            self.scale,
            self.control_type,
        )

    def _load_weights(self, model_path: str) -> bool:
        import glob

        path = Path(model_path)
        search_dirs = [path]
        for subdir in ("controlnet", "control_net", ""):
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
                cn_keys = {
                    k: v
                    for k, v in raw_weights.items()
                    if "controlnet" in k or "control" in k
                }
                if not cn_keys:
                    cn_keys = raw_weights
                remapped = remap_controlnet_weights(cn_keys, self.num_layers)
                if remapped:
                    self.dit.load_weights(list(remapped.items()))
                    logger.info(
                        "ControlNet: weights loaded from %s (%d params)",
                        search_dir,
                        len(remapped),
                    )
                    return True
            except Exception as exc:
                logger.warning(
                    "ControlNet: failed to load from %s: %s", search_dir, exc
                )
        return False

    def _load_from_hf_cache(self) -> bool:
        try:
            from huggingface_hub import hf_hub_download

            cn_repo = self.config.get(
                "controlnet_repo",
                "Kwai-Kolors/Kolors-ControlNet-Canny",
            )
            logger.info("ControlNet: trying HF cache: %s", cn_repo)

            sf = hf_hub_download(cn_repo, "model.safetensors")
            raw = mx.load(sf)
            cn_keys = {
                k: v for k, v in raw.items() if "controlnet" in k or "control" in k
            }
            if not cn_keys:
                cn_keys = raw
            remapped = remap_controlnet_weights(cn_keys, self.num_layers)
            if remapped:
                self.dit.load_weights(list(remapped.items()))
                logger.info(
                    "ControlNet: weights loaded from HF (%d params)", len(remapped)
                )
                return True
        except ImportError:
            logger.debug(
                "ControlNet: huggingface_hub not installed, skipping HF download"
            )
        except Exception as exc:
            logger.warning("ControlNet: HF load failed: %s", exc)
        return False

    def unload(self) -> None:
        self.dit = None
        self._residuals = None
        self._loaded = False
        logger.info("ControlNet: unloaded")

    def encode_control(
        self,
        image_path: str | None = None,
        control_type: str | None = None,
    ) -> mx.array | None:
        """Preprocess control image to latent representation."""
        path = image_path or self.image
        if path is None:
            logger.warning("ControlNet: no control image provided")
            return None

        ctype = control_type or self.control_type
        try:
            return preprocess_control_image(path, ctype)
        except Exception as exc:
            logger.error("ControlNet: control image preprocessing failed: %s", exc)
            return None

    def compute_residuals(
        self,
        control_latent: mx.array,
        t: mx.array,
        context: mx.array,
    ) -> list[mx.array] | None:
        """Run ControlNet forward to compute per-block residuals.

        Args:
            control_latent: [B, C, H, W] preprocessed control image
            t: [B] timestep
            context: [B, L, text_dim] text context

        Returns:
            List of residuals (num_layers + 1), each [B, L, dim], or None
        """
        if not self._loaded:
            self.load()

        if self.dit is None:
            logger.warning("ControlNet: DiT not loaded, cannot compute residuals")
            return None

        try:
            residuals = self.dit.forward(control_latent, t, context)
            if self.scale != 1.0:
                residuals = [r * self.scale for r in residuals]
            self._residuals = residuals
            logger.debug(
                "ControlNet: computed %d residuals (scale=%.2f)",
                len(residuals),
                self.scale,
            )
            return residuals
        except Exception as exc:
            logger.error("ControlNet: residual computation failed: %s", exc)
            return None

    def get_residuals(self) -> list[mx.array] | None:
        """Get cached residuals from last compute_residuals() call."""
        return self._residuals

    def modify_denoise_step(
        self,
        dit: Any,
        latents: mx.array,
        t: mx.array,
        context: mx.array,
        **kw: Any,
    ) -> mx.array:
        """Pre-compute ControlNet residuals for this denoising step.

        The actual residual injection happens inside _run_blocks via
        controlnet_residuals parameter. This method prepares the residuals
        and stores them for the pipeline to retrieve.

        Returns:
            latents unchanged (residuals stored in self._residuals)
        """
        control_image = kw.get("control_image", self.image)
        if control_image is None:
            return latents

        control_latent = self.encode_control(
            control_image,
            control_type=kw.get("control_type", self.control_type),
        )
        if control_latent is None:
            return latents

        residuals = self.compute_residuals(control_latent, t, context)
        if residuals is None:
            logger.warning("ControlNet: residual computation failed, step unchanged")

        return latents


__all__ = [
    "ControlNet",
    "ControlNetDiT",
    "ControlNetDiTBlock",
    "ZeroConv",
    "HintBlock",
    "preprocess_control_image",
    "preprocess_canny",
    "preprocess_depth",
    "remap_controlnet_weights",
]
