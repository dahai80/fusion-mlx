# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 文本编码器 MLX 端口.

复用底座 fusion_mlx.video.t5_encoder.T5Encoder (T5-v1_1-xxl 纯 MLX 端口),
适配 SkyReels-V3 的 UMT5EncoderModel.

UMT5 (Universal Multilingual T5) 与标准 T5 差异:
  - 相对位置偏置共享 (所有层共用一个 rel_pos_bias)
  - gated-gelu FFN 一致
  - d_model=4096, num_layers=24, num_heads=64, d_kv=64, d_ff=10240

本封装:
  - UMT5Encoder: 复用底座 T5Encoder, 调整配置
  - CLIPTextEncoder: CLIP 文本编码器 (短 Prompt 走 CPU 预处理兜底)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UMT5 配置 (SkyReels-V3 用 google/umt5-xxl)
# ---------------------------------------------------------------------------
@dataclass
class UMT5Config:
    """UMT5-XXL 配置, 对齐 transformers.UMT5EncoderModel 默认值."""
    d_model: int = 4096
    num_layers: int = 24
    num_heads: int = 64
    d_kv: int = 64
    d_ff: int = 10240
    vocab_size: int = 256384  # UMT5 词表更大
    rel_num_buckets: int = 32
    rel_max_distance: int = 128
    layer_norm_eps: float = 1e-6
    feed_forward_proj: str = "gated-gelu"
    model_type: str = "umt5"

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.d_kv

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "UMT5Config":
        eps = cfg.get("layer_norm_epsilon", cfg.get("layer_norm_eps", 1e-6))
        return cls(
            d_model=cfg.get("d_model", 4096),
            num_layers=cfg.get("num_layers", 24),
            num_heads=cfg.get("num_heads", 64),
            d_kv=cfg.get("d_kv", 64),
            d_ff=cfg.get("d_ff", 10240),
            vocab_size=cfg.get("vocab_size", 256384),
            rel_num_buckets=cfg.get("relative_attention_num_buckets", 32),
            rel_max_distance=cfg.get("relative_attention_max_distance", 128),
            layer_norm_eps=eps,
        )


# ---------------------------------------------------------------------------
# UMT5 编码器 (复用底座 T5Encoder)
# ---------------------------------------------------------------------------
class UMT5Encoder(nn.Module):
    """UMT5 文本编码器 MLX 端口.

    复用 fusion_mlx.video.t5_encoder.T5Encoder 底座实现,
    仅调整配置以匹配 UMT5-XXL.
    """

    def __init__(self, config: UMT5Config | None = None):
        super().__init__()
        self.config = config or UMT5Config()

        # 复用底座 T5Encoder
        try:
            from fusion_mlx.video.t5_encoder import T5Encoder, T5EncoderConfig

            t5_cfg = T5EncoderConfig(
                d_model=self.config.d_model,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                d_kv=self.config.d_kv,
                d_ff=self.config.d_ff,
                vocab_size=self.config.vocab_size,
                rel_num_buckets=self.config.rel_num_buckets,
                rel_max_distance=self.config.rel_max_distance,
                layer_norm_eps=self.config.layer_norm_eps,
            )
            self.encoder = T5Encoder(t5_cfg)
            self._uses_base = True
        except Exception as exc:  # pragma: no cover - 底座不可用时降级
            logger.warning("T5Encoder base unavailable (%s), using stub", exc)
            self.encoder = None
            self._uses_base = False

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """前向: token ids -> text embedding.

        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] 可选掩码

        Returns:
            [B, L, d_model] 文本 embedding
        """
        if self.encoder is None:
            # Stub: 返回零张量
            b, l = input_ids.shape
            return mx.zeros((b, l, self.config.d_model))

        return self.encoder(input_ids, attention_mask)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "UMT5Encoder":
        """从 HuggingFace 权重目录加载 UMT5 编码器.

        Args:
            path: 权重目录 (含 config.json + *.safetensors)

        Returns:
            UMT5Encoder 实例
        """
        path = Path(path)
        cfg_path = path / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {path}")

        import json
        with open(cfg_path) as f:
            hf_cfg = json.load(f)

        config = UMT5Config.from_hf_config(hf_cfg)
        instance = cls(config)

        if instance._uses_base:
            try:
                from fusion_mlx.video.t5_encoder import load_t5_encoder
                instance.encoder = load_t5_encoder(str(path), config=instance.encoder.config)
            except Exception as exc:
                logger.warning("Failed to load UMT5 weights: %s", exc)

        return instance


# ---------------------------------------------------------------------------
# CLIP 文本编码器 (短 Prompt CPU 兜底)
# ---------------------------------------------------------------------------
class CLIPTextEncoder:
    """CLIP 文本编码器.

    两种方案:
      方案1 (最优): 完整迁移 CLIP 到 MLX, 全局推理链路闭环
      方案2 (兜底): 短 Prompt 用 CPU 预处理, 大批量生成用 MLX

    本实现走方案2 (CPU 预处理兜底),
    完整 MLX CLIP 端口可后续基于 fusion_mlx/mlx_clip 实现.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model_name = model_name
        self._clip_model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """惰性加载 CLIP 模型 + tokenizer."""
        if self._clip_model is not None:
            return

        try:
            # 优先尝试 mlx_clip (底座 MLX CLIP)
            import mlx_clip
            self._clip_model = mlx_clip.load(self.model_name)
            logger.info("CLIP loaded via mlx_clip")
        except Exception:
            try:
                # 兜底: transformers CLIPModel (CPU)
                from transformers import CLIPModel, CLIPTokenizer
                self._clip_model = CLIPModel.from_pretrained(self.model_name)
                self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                logger.info("CLIP loaded via transformers (CPU)")
            except Exception as exc:
                logger.warning("CLIP unavailable (%s), using stub", exc)
                self._clip_model = "stub"

    def encode_text(
        self,
        text: str | list[str],
        *,
        max_length: int = 77,
    ) -> mx.array:
        """编码文本为 CLIP embedding.

        Args:
            text: 单条文本或文本列表
            max_length: 最大 token 长度

        Returns:
            [B, L, dim] CLIP 文本 embedding
        """
        self._ensure_loaded()

        if isinstance(text, str):
            text = [text]

        if self._clip_model == "stub":
            # Stub: 返回零张量
            b = len(text)
            return mx.zeros((b, max_length, 768))

        if self._tokenizer is not None:
            # transformers CPU 路径
            inputs = self._tokenizer(
                text, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            import torch
            with torch.no_grad():
                outputs = self._clip_model.get_text_features(**inputs)
            return mx.array(outputs.numpy())
        else:
            # mlx_clip 路径
            embeddings = []
            for t in text:
                emb = self._clip_model.encode_text(t)
                if isinstance(emb, mx.array):
                    embeddings.append(emb)
                else:
                    embeddings.append(mx.array(emb))
            return mx.stack(embeddings, axis=0)


__all__ = [
    "UMT5Config",
    "UMT5Encoder",
    "CLIPTextEncoder",
]
