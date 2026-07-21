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

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from fusion_mlx.cache.radix_diffusion_cache import DiffusionRadixCache

logger = logging.getLogger(__name__)


def _text_cache_enabled() -> bool:
    return os.getenv("FUSION_DIFFUSION_TEXT_CACHE", "1") == "1"


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
    def from_hf_config(cls, cfg: dict) -> UMT5Config:
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

    内置 tokenizer (transformers.AutoTokenizer from "google/umt5-xxl"),
    encode_text(prompt) 直接返回 embedding, 无需外部 tokenize.
    """

    # 默认 tokenizer repo (与底座 wan2/generate.py 一致)
    DEFAULT_TOKENIZER_REPO = "google/umt5-xxl"

    def __init__(self, config: UMT5Config | None = None):
        super().__init__()
        self.config = config or UMT5Config()
        self._tokenizer = None  # 惰性加载
        # #178: per-instance radix text-embedding cache (multi-shot reuse)
        self._text_cache = (
            DiffusionRadixCache(max_mb=512) if _text_cache_enabled() else None
        )

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

    # ------------------------------------------------------------------
    # tokenizer 接入 (任务 #2 遗留修复)
    # ------------------------------------------------------------------
    def _ensure_tokenizer(self) -> None:
        """惰性加载 UMT5 tokenizer (transformers.AutoTokenizer)."""
        if self._tokenizer is not None:
            return

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.DEFAULT_TOKENIZER_REPO,
                legacy=False,
                padding_side="right",
            )
            logger.info(
                "UMT5 tokenizer loaded: %s",
                self.DEFAULT_TOKENIZER_REPO,
            )
        except Exception as exc:
            logger.warning(
                "UMT5 tokenizer 加载失败 (%s); encode_text 将退回 stub",
                exc,
            )
            self._tokenizer = "stub"  # 哨兵, 避免重复尝试

    def tokenize(
        self,
        prompt: str,
        *,
        max_length: int = 512,
    ) -> tuple[mx.array, mx.array]:
        """tokenize 文本 -> (input_ids, attention_mask).

        Args:
            prompt: 文本 prompt
            max_length: 最大 token 长度

        Returns:
            (input_ids [1, L], attention_mask [1, L])
        """
        self._ensure_tokenizer()

        if self._tokenizer == "stub":
            # stub: 返回零张量
            return mx.zeros((1, max_length), dtype=mx.int32), mx.zeros((1, max_length))

        tokens = self._tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        ids = mx.array(tokens["input_ids"])
        mask = mx.array(tokens["attention_mask"])
        return ids, mask

    def encode_text(
        self,
        prompt: str,
        *,
        max_length: int = 512,
    ) -> mx.array:
        """端到端文本编码: prompt -> embedding.

        内部 tokenize + 前向, 返回非 padding 部分的 embedding.

        Args:
            prompt: 文本 prompt
            max_length: 最大 token 长度

        Returns:
            [1, L_valid, d_model] 文本 embedding (L_valid = 非 padding token 数)
        """
        # #178: radix text-embedding cache (skip stub mode)
        cache = self._text_cache
        key = None
        if cache is not None:
            key = f"umt5:{max_length}:{_prompt_hash(prompt)}"
            cached = cache.get(key)
            if cached is not None:
                logger.debug("umt5 text cache hit: %s", key)
                return cached

        ids, mask = self.tokenize(prompt, max_length=max_length)

        if self._tokenizer == "stub" or self.encoder is None:
            # stub: 返回零张量
            return mx.zeros((1, max_length, self.config.d_model))

        embeddings = self.encoder(ids, attention_mask=mask)

        # 截断到非 padding 部分 (与底座 wan2/utils.py encode_text 一致)
        seq_len = int(mask.sum().item())
        result = embeddings[0, :seq_len][None]  # [1, L_valid, d_model]

        if key is not None:
            cache.put(key, result)
            logger.debug("umt5 text cache miss+insert: %s", key)

        return result

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """前向: token ids -> text embedding (底层 encoder 调用).

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
    def from_pretrained(cls, path: str | Path) -> UMT5Encoder:
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

                instance.encoder = load_t5_encoder(
                    str(path), config=instance.encoder.config
                )
            except Exception as exc:
                logger.warning("Failed to load UMT5 weights: %s", exc)

        return instance


# ---------------------------------------------------------------------------
# CLIP 文本编码器 (短 Prompt CPU 兜底)
# ---------------------------------------------------------------------------
class CLIPTextEncoder:
    """CLIP 文本编码器 (三级兜底链路).

    优先级:
      1. mlx_clip (底座 MLX CLIP, 纯 MLX 端口, 最优)
      2. transformers CLIPModel (CPU 兜底, 需 torch)
      3. stub 零张量 (无依赖环境, 测试用)

    与原版差异:
      - mlx_clip 路径: 批量 encode_text, 避免逐条调用
      - transformers 路径: 支持 torch 与 numpy 双后端 (无 torch 时退 numpy)
      - stub 路径: 返回正确维度的零张量, 不报错
    """

    # 默认 CLIP repo (SkyReels-V3 用 openai/clip-vit-large-patch14)
    DEFAULT_REPO = "openai/clip-vit-large-patch14"
    # CLIP text embedding 维度 (ViT-L/14)
    EMBED_DIM = 768

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_REPO
        self._clip_model = None  # "stub" 哨兵或实际模型
        self._tokenizer = None
        self._backend: str | None = None  # "mlx_clip" / "transformers" / "stub"

    def _ensure_loaded(self) -> None:
        """惰性加载 CLIP, 按三级兜底链路尝试."""
        if self._clip_model is not None:
            return

        # 1. 优先 mlx_clip (纯 MLX)
        try:
            import mlx_clip

            self._clip_model = mlx_clip.load(self.model_name)
            self._backend = "mlx_clip"
            logger.info("CLIP loaded via mlx_clip: %s", self.model_name)
            return
        except ImportError:
            pass  # mlx_clip 未安装, 继续兜底
        except Exception as exc:
            logger.warning("mlx_clip 加载失败 (%s), 尝试 transformers", exc)

        # 2. transformers CPU 兜底
        try:
            from transformers import CLIPModel, CLIPTokenizer

            self._clip_model = CLIPModel.from_pretrained(self.model_name)
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self._backend = "transformers"
            logger.info("CLIP loaded via transformers (CPU): %s", self.model_name)
            return
        except ImportError:
            pass  # transformers 未安装
        except Exception as exc:
            logger.warning("transformers CLIP 加载失败 (%s), 退 stub", exc)

        # 3. stub 兜底
        self._clip_model = "stub"
        self._backend = "stub"
        logger.warning(
            "CLIP unavailable (无 mlx_clip/transformers), 使用 stub 零张量",
        )

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
            [B, L, dim] 或 [B, dim] CLIP 文本 embedding
            (mlx_clip/transformers 返回 [B, dim], stub 返回 [B, L, dim])
        """
        self._ensure_loaded()

        if isinstance(text, str):
            text = [text]
        b = len(text)

        # stub 兜底
        if self._backend == "stub":
            return mx.zeros((b, max_length, self.EMBED_DIM))

        # mlx_clip 路径 (纯 MLX)
        if self._backend == "mlx_clip":
            try:
                # mlx_clip.encode_text 支持批量
                emb = self._clip_model.encode_text(text)
                if isinstance(emb, list):
                    return mx.stack([mx.array(e) for e in emb], axis=0)
                return mx.array(emb) if not isinstance(emb, mx.array) else emb
            except Exception as exc:
                # mlx_clip.encode_text 可能不支持批量, 退逐条
                logger.warning(
                    "mlx_clip.encode_text 批量失败 (%s), 退逐条",
                    exc,
                )
                embeddings = []
                for t in text:
                    e = self._clip_model.encode_text(t)
                    embeddings.append(mx.array(e) if not isinstance(e, mx.array) else e)
                return mx.stack(embeddings, axis=0)

        # transformers CPU 路径 (支持 torch 与 numpy 双后端)
        if self._backend == "transformers":
            inputs = self._tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np",  # numpy 后端, 无需 torch
            )
            try:
                # 优先 torch (更快)
                import torch

                inputs_pt = {k: torch.from_numpy(v) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._clip_model.get_text_features(**inputs_pt)
                return mx.array(outputs.numpy())
            except ImportError:
                # 无 torch: 用 transformers 的 numpy 前向
                outputs = self._clip_model.get_text_features(**inputs)
                if hasattr(outputs, "numpy"):
                    outputs = outputs.numpy()
                return mx.array(outputs)

        # 理论不可达
        return mx.zeros((b, max_length, self.EMBED_DIM))

    def is_stub(self) -> bool:
        """是否处于 stub 模式 (无真实 CLIP 可用)."""
        return self._backend == "stub"


__all__ = [
    "UMT5Config",
    "UMT5Encoder",
    "CLIPTextEncoder",
]
