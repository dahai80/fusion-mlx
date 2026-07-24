# SPDX-License-Identifier: Apache-2.0
# IP-Adapter: CLIP-Vision image encoder + projection MLP -> prepend to text context.
# Reference: https://github.com/tencent-ailab/IP-Adapter
# Architecture: image -> CLIP-ViT -> projection MLP -> image_tokens prepended to text context
# Zero DiT architecture change: only modifies the context tensor fed to DiT.
#
# Callers: SkyReelsBasePipeline._encode_context(), create_adapter("ip_adapter") factory,
# videos_routes.py passes ip_adapter_image/ip_adapter_scale.
# Schema: [B, 257, text_dim] image tokens prepended to [B, L_text, text_dim] -> [B, 257+L_text, text_dim]
# User instruction: "若缺少，需要先用 MLX 移植这些模块（可参考已有开源实现如 mlx-examples/stable_diffusion 进行扩展）"

import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import VideoAdapter, register_adapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP Vision Encoder (ported from mlx-examples wan2.1/wan/clip.py)
# ViT-H/14: image_size=224, patch_size=14, dim=1280, heads=16, layers=32
# ---------------------------------------------------------------------------


class _CLIPSelfAttention(nn.Module):
    def __init__(self, dim: int = 1280, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.q_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, n * d)
        return self.out_proj(x)


class _CLIPMLP(nn.Module):
    def __init__(self, dim: int, mid_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class _CLIPAttentionBlock(nn.Module):
    def __init__(self, dim: int = 1280, num_heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = _CLIPSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _CLIPMLP(dim, int(dim * mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class IPAdapterClipVisionEncoder(nn.Module):
    """ViT-H/14 vision encoder returning patch + CLS token features.

    Returns [B, 257, 1280] (1 CLS + 256 patch tokens).
    Only first (num_layers - 1) blocks are evaluated (matching reference use_31_block=True).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        num_heads: int = 16,
        num_layers: int = 32,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.num_layers = num_layers

        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.cls_embedding = mx.zeros((1, 1, dim))
        self.position_embedding = mx.zeros((1, self.num_patches + 1, dim))
        self.pre_norm = nn.LayerNorm(dim)

        for i in range(num_layers):
            setattr(self, f"block_{i}", _CLIPAttentionBlock(dim, num_heads, mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        x = self.patch_embedding(x)
        x = x.reshape(B, -1, self.dim)
        cls = mx.broadcast_to(self.cls_embedding, (B, 1, self.dim))
        x = mx.concatenate([cls, x], axis=1)
        x = x + self.position_embedding
        x = self.pre_norm(x)
        for i in range(self.num_layers - 1):
            block = getattr(self, f"block_{i}")
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# IP-Adapter Projection MLP
# Projects CLIP image features from clip_dim (1280) to text_dim (4096).
# Pattern from Wan2.1 _embed_image: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm
# ---------------------------------------------------------------------------


class IPAdapterProjection(nn.Module):
    """Project CLIP image features to text context dimension."""

    def __init__(self, clip_dim: int = 1280, text_dim: int = 4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(clip_dim)
        self.linear1 = nn.Linear(clip_dim, clip_dim)
        self.linear2 = nn.Linear(clip_dim, text_dim)
        self.norm2 = nn.LayerNorm(text_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm1(x)
        x = nn.gelu(self.linear1(x))
        x = self.linear2(x)
        x = self.norm2(x)
        return x


# ---------------------------------------------------------------------------
# Image preprocessing for CLIP ViT-H/14
# ---------------------------------------------------------------------------


def preprocess_clip_image(image_path: str) -> mx.array:
    """Load and preprocess image for CLIP ViT-H/14.

    Returns [1, 224, 224, 3] float32 (channels-last for MLX).
    """
    import numpy as np
    from PIL import Image

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return mx.array(arr[np.newaxis])


# ---------------------------------------------------------------------------
# CLIP weight remapping (HF -> MLX)
# Handles OpenCLIP naming and Wan2.1 HF naming.
# Extracts only visual.* keys. Splits fused QKV into q/k/v.
# ---------------------------------------------------------------------------


def remap_clip_weights(weights: dict) -> dict:
    """Remap CLIP .pth/safetensors checkpoint keys to IPAdapterClipVisionEncoder format.

    Handles two naming conventions:
    1. OpenCLIP: visual.conv1.weight, visual.transformer.resblocks.N.*
    2. HF Transformers: vision_model.embeddings.*, vision_model.encoder.layers.N.*
    """
    remapped = {}
    for key, value in weights.items():
        # --- HF Transformers naming: vision_model.* ---
        if key.startswith("vision_model."):
            if key == "vision_model.embeddings.patch_embedding.weight":
                if value.ndim == 4:
                    value = mx.transpose(value, (0, 2, 3, 1))
                remapped["patch_embedding.weight"] = value
                continue

            if key == "vision_model.embeddings.class_embedding":
                remapped["cls_embedding"] = value.reshape(1, 1, -1)
                continue

            if key == "vision_model.embeddings.position_embedding.weight":
                if value.ndim == 2:
                    value = value.reshape(1, value.shape[0], value.shape[1])
                remapped["position_embedding"] = value
                continue

            if key.startswith("vision_model.pre_layrnorm.") or key.startswith(
                "vision_model.pre_layernorm."
            ):
                param = key.split(".")[-1]
                remapped[f"pre_norm.{param}"] = value
                continue

            if "post_layernorm" in key or "position_ids" in key:
                continue

            m = re.match(r"vision_model\.encoder\.layers\.(\d+)\.(.*)", key)
            if m:
                block_idx = m.group(1)
                rest = m.group(2)

                for attn_name, mlx_name in [
                    ("self_attn.q_proj.", "self_attn.q_proj."),
                    ("self_attn.k_proj.", "self_attn.k_proj."),
                    ("self_attn.v_proj.", "self_attn.v_proj."),
                    ("self_attn.out_proj.", "self_attn.out_proj."),
                ]:
                    if rest.startswith(attn_name):
                        param = rest.split(".")[-1]
                        remapped[f"block_{block_idx}.{mlx_name}{param}"] = value
                        break
                else:
                    for old, new in [
                        ("layer_norm1.", "norm1."),
                        ("layer_norm2.", "norm2."),
                    ]:
                        if rest.startswith(old):
                            param = rest.split(".")[-1]
                            remapped[f"block_{block_idx}.{new}{param}"] = value
                            break
                    else:
                        for old, new in [
                            ("mlp.fc1.", "mlp.fc1."),
                            ("mlp.fc2.", "mlp.fc2."),
                        ]:
                            if rest.startswith(old):
                                param = rest.split(".")[-1]
                                remapped[f"block_{block_idx}.{new}{param}"] = value
                                break
            continue

        # --- OpenCLIP naming: visual.* ---
        if not key.startswith("visual."):
            continue
        if "post_norm" in key or "ln_post" in key or key == "visual.head":
            continue

        if key in ("visual.conv1.weight", "visual.patch_embedding.weight"):
            if value.ndim == 4:
                value = mx.transpose(value, (0, 2, 3, 1))
            remapped["patch_embedding.weight"] = value
            continue

        if key in ("visual.class_embedding", "visual.cls_embedding"):
            remapped["cls_embedding"] = value.reshape(1, 1, -1)
            continue

        if key in ("visual.positional_embedding", "visual.pos_embedding"):
            if value.ndim == 2:
                value = value.reshape(1, value.shape[0], value.shape[1])
            remapped["position_embedding"] = value
            continue

        if key.startswith("visual.ln_pre.") or key.startswith("visual.pre_norm."):
            param = key.split(".")[-1]
            remapped[f"pre_norm.{param}"] = value
            continue

        m = re.match(r"visual\.transformer\.(?:resblocks\.)?(\d+)\.(.*)", key)
        if m:
            block_idx = m.group(1)
            rest = m.group(2)

            if rest in ("attn.in_proj_weight", "attn.to_qkv.weight"):
                dim = value.shape[0] // 3
                q, k, v = value[:dim], value[dim : 2 * dim], value[2 * dim :]
                remapped[f"block_{block_idx}.self_attn.q_proj.weight"] = q
                remapped[f"block_{block_idx}.self_attn.k_proj.weight"] = k
                remapped[f"block_{block_idx}.self_attn.v_proj.weight"] = v
                continue

            if rest in ("attn.in_proj_bias", "attn.to_qkv.bias"):
                dim = value.shape[0] // 3
                q, k, v = value[:dim], value[dim : 2 * dim], value[2 * dim :]
                remapped[f"block_{block_idx}.self_attn.q_proj.bias"] = q
                remapped[f"block_{block_idx}.self_attn.k_proj.bias"] = k
                remapped[f"block_{block_idx}.self_attn.v_proj.bias"] = v
                continue

            for prefix in ("attn.out_proj.", "attn.proj."):
                if rest.startswith(prefix):
                    param = rest.split(".")[-1]
                    remapped[f"block_{block_idx}.self_attn.out_proj.{param}"] = value
                    break
            else:
                for old, new in [
                    ("ln_1.", "norm1."),
                    ("norm1.", "norm1."),
                    ("ln_2.", "norm2."),
                    ("norm2.", "norm2."),
                ]:
                    if rest.startswith(old):
                        param = rest.split(".")[-1]
                        remapped[f"block_{block_idx}.{new}{param}"] = value
                        break
                else:
                    for old, new in [
                        ("mlp.c_fc.", "mlp.fc1."),
                        ("mlp.0.", "mlp.fc1."),
                        ("mlp.c_proj.", "mlp.fc2."),
                        ("mlp.2.", "mlp.fc2."),
                    ]:
                        if rest.startswith(old):
                            param = rest.split(".")[-1]
                            remapped[f"block_{block_idx}.{new}{param}"] = value
                            break

    return remapped


def remap_ip_adapter_weights(weights: dict) -> dict:
    """Remap IP-Adapter projection weights from HF naming to MLX naming.

    HF IP-Adapter uses keys like:
      image_proj_model.norm1.weight -> norm1.weight
      image_proj_model.kv_proj.weight -> linear1.weight (first half) + linear2.weight (second half)
    """
    remapped = {}
    for key, value in weights.items():
        clean = key
        for prefix in ("image_proj_model.", "ip_adapter.", "adapter."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]

        mapping = {
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "linear1.weight": "linear1.weight",
            "linear1.bias": "linear1.bias",
            "linear2.weight": "linear2.weight",
            "linear2.bias": "linear2.bias",
            "ff.weight": "linear1.weight",
            "ff.bias": "linear1.bias",
        }

        if clean in mapping:
            remapped[mapping[clean]] = value
        elif clean == "kv_proj.weight":
            mid = value.shape[0] // 2
            remapped["linear1.weight"] = value[:mid]
            remapped["linear2.weight"] = value[mid:]

    return remapped


# ---------------------------------------------------------------------------
# IP-Adapter: main adapter class
# ---------------------------------------------------------------------------


@register_adapter("ip_adapter")
class IPAdapter(VideoAdapter):
    """IP-Adapter: CLIP-Vision image encoder + projection MLP.

    Workflow:
      1. Input image -> CLIP ViT-H/14 -> [B, 257, 1280] image features
      2. Image features -> projection MLP -> [B, 257, text_dim] projected tokens
      3. Prepend projected tokens to text context: [img_tokens | text_tokens]
      4. Scale contribution by ip_adapter_scale
    """

    name = "ip_adapter"

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
        self.clip_dim = self.config.get("clip_dim", 1280)
        self.text_dim = self.config.get("text_dim", 4096)
        self.clip_vision: IPAdapterClipVisionEncoder | None = None
        self.projection: IPAdapterProjection | None = None
        self._loaded = False

    def load(self, model_path: str | None = None) -> None:
        if self._loaded:
            logger.debug("IP-Adapter already loaded, skipping")
            return

        self.clip_vision = IPAdapterClipVisionEncoder(dim=self.clip_dim)
        logger.info("IP-Adapter: created CLIP vision encoder (dim=%d)", self.clip_dim)

        self.projection = IPAdapterProjection(
            clip_dim=self.clip_dim,
            text_dim=self.text_dim,
        )
        logger.info(
            "IP-Adapter: created projection MLP (%d -> %d)",
            self.clip_dim,
            self.text_dim,
        )

        clip_loaded = False
        if model_path is not None:
            clip_loaded = self._load_clip_weights(model_path)
        if not clip_loaded:
            clip_loaded = self._load_clip_from_hf_cache()
        if not clip_loaded:
            logger.warning("IP-Adapter: CLIP weights not loaded, random init")

        proj_loaded = False
        if model_path is not None:
            proj_loaded = self._load_projection_weights(model_path)
        if not proj_loaded:
            logger.warning("IP-Adapter: projection weights not loaded, random init")

        self._loaded = True
        logger.info(
            "IP-Adapter: loaded (clip=%s proj=%s scale=%.2f)",
            clip_loaded,
            proj_loaded,
            self.scale,
        )

    def _load_clip_weights(self, model_path: str) -> bool:
        import glob

        path = Path(model_path)
        clip_dirs = [path]
        for subdir in ("clip_vision", "ip_adapter_clip_vision", "image_encoder"):
            candidate = path / subdir
            if candidate.is_dir():
                clip_dirs.append(candidate)

        for clip_dir in clip_dirs:
            safetensors = sorted(glob.glob(str(clip_dir / "*.safetensors")))
            if not safetensors:
                continue
            try:
                raw_weights = {}
                for wf in safetensors:
                    raw_weights.update(mx.load(wf))
                remapped = remap_clip_weights(raw_weights)
                if remapped:
                    self.clip_vision.load_weights(list(remapped.items()))
                    logger.info(
                        "IP-Adapter: CLIP weights loaded from %s (%d params)",
                        clip_dir,
                        len(remapped),
                    )
                    return True
            except Exception as exc:
                logger.warning(
                    "IP-Adapter: failed to load CLIP from %s: %s", clip_dir, exc
                )
        return False

    def _load_clip_from_hf_cache(self) -> bool:
        try:
            from huggingface_hub import hf_hub_download

            clip_repo = self.config.get(
                "clip_repo",
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            )
            logger.info("IP-Adapter: trying HF cache for CLIP: %s", clip_repo)

            for fname in (
                "open_clip_pytorch_model.safetensors",
                "open_clip_model.safetensors",
                "model.safetensors",
            ):
                try:
                    sf = hf_hub_download(clip_repo, fname)
                    raw = mx.load(sf)
                    break
                except Exception:
                    continue
            else:
                logger.warning("IP-Adapter: no safetensors found for %s", clip_repo)
                return False

            remapped = remap_clip_weights(raw)
            if remapped:
                self.clip_vision.load_weights(list(remapped.items()))
                logger.info(
                    "IP-Adapter: CLIP weights loaded from HF (%d params)", len(remapped)
                )
                return True
        except ImportError:
            logger.debug(
                "IP-Adapter: huggingface_hub not installed, skipping HF download"
            )
        except Exception as exc:
            logger.warning("IP-Adapter: HF CLIP load failed: %s", exc)
        return False

    def _load_projection_weights(self, model_path: str) -> bool:
        import glob

        path = Path(model_path)
        search_dirs = [path]
        for subdir in ("ip_adapter", "ip-adapter", ""):
            search_dirs.append(path / subdir)

        for search_dir in search_dirs:
            safetensors = sorted(glob.glob(str(search_dir / "*.safetensors")))
            for sf in safetensors:
                try:
                    raw = mx.load(sf)
                    proj_keys = {
                        k
                        for k in raw
                        if any(
                            p in k
                            for p in (
                                "image_proj_model",
                                "ip_adapter",
                                "adapter",
                                "norm1",
                                "norm2",
                                "linear1",
                                "linear2",
                                "kv_proj",
                                "ff",
                            )
                        )
                        and not k.startswith("visual.")
                    }
                    if not proj_keys:
                        continue
                    proj_weights = {k: raw[k] for k in proj_keys}
                    remapped = remap_ip_adapter_weights(proj_weights)
                    if remapped:
                        self.projection.load_weights(list(remapped.items()))
                        logger.info(
                            "IP-Adapter: projection weights loaded from %s (%d params)",
                            sf,
                            len(remapped),
                        )
                        return True
                except Exception as exc:
                    logger.warning(
                        "IP-Adapter: failed to load projection from %s: %s", sf, exc
                    )
        return False

    def unload(self) -> None:
        self.clip_vision = None
        self.projection = None
        self._loaded = False
        logger.info("IP-Adapter: unloaded")

    def encode_image(self, image_path: str | None = None) -> mx.array | None:
        path = image_path or self.image
        if path is None:
            logger.warning("IP-Adapter: no image provided")
            return None

        if not self._loaded:
            self.load()

        if self.clip_vision is None or self.projection is None:
            logger.warning("IP-Adapter: models not loaded, cannot encode image")
            return None

        try:
            pixel_values = preprocess_clip_image(path)
            clip_features = self.clip_vision(pixel_values)
            logger.debug("IP-Adapter: CLIP features shape=%s", clip_features.shape)

            projected = self.projection(clip_features)
            logger.debug("IP-Adapter: projected features shape=%s", projected.shape)

            if self.scale != 1.0:
                projected = projected * self.scale

            return projected
        except Exception as exc:
            logger.error("IP-Adapter: image encoding failed: %s", exc)
            return None

    def modify_context(
        self,
        context: mx.array,
        **kw: Any,
    ) -> mx.array:
        """Prepend IP-Adapter image tokens to text context.

        Args:
            context: [B, L_text, text_dim] text context from UMT5/CLIP text encoder.
            **kw: 'image' (str path) or uses self.image.

        Returns:
            [B, L_img + L_text, text_dim] context with image tokens prepended.
        """
        image_path = kw.get("image", self.image)
        if image_path is None:
            logger.debug("IP-Adapter: no image, returning context unchanged")
            return context

        image_tokens = self.encode_image(image_path)
        if image_tokens is None:
            logger.warning("IP-Adapter: image encoding failed, context unchanged")
            return context

        augmented = mx.concatenate([image_tokens, context], axis=1)
        logger.info(
            "IP-Adapter: context augmented %s + %s -> %s (scale=%.2f)",
            list(image_tokens.shape),
            list(context.shape),
            list(augmented.shape),
            self.scale,
        )
        return augmented
