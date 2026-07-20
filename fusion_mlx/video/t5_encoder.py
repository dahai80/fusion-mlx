# SPDX-License-Identifier: Apache-2.0
# Pure-MLX T5 encoder port (t5-v1_1-xxl) for LTX-Video 0.9.x caption embedding.
# No torch dependency. Mirrors transformers/models/t5/modeling_t5.py math:
# T5LayerNorm (RMSNorm: no bias, no mean-sub, fp32 reduce, eps=1e-6),
# gated-gelu feed-forward (wi_0/wi_1/wo), bidirectional relative-position
# bias self-attention, pre-norm residuals. Output = last hidden state of the
# full sequence, fed directly into the 0.9.x caption_projection (in=4096).

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

logger = logging.getLogger(__name__)

_MASK_NEG = -1e9


def _gelu_erf(x):
    # Exact GELU (erf), matches transformers get_activation("gelu") used by
    # feed_forward_proj="gated-gelu".
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


@dataclass
class T5EncoderConfig:
    d_model: int = 4096
    num_layers: int = 24
    num_heads: int = 64
    d_kv: int = 64
    d_ff: int = 10240
    vocab_size: int = 32128
    rel_num_buckets: int = 32
    rel_max_distance: int = 128
    layer_norm_eps: float = 1e-6

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.d_kv

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "T5EncoderConfig":
        ffp = cfg.get("feed_forward_proj", "gated-gelu")
        if "gated" not in ffp:
            raise ValueError(f"t5_encoder: only gated-gelu supported, got {ffp!r}")
        eps = cfg.get("layer_norm_epsilon", cfg.get("layer_norm_eps", 1e-6))
        return cls(
            d_model=cfg.get("d_model", 4096),
            num_layers=cfg.get("num_layers", 24),
            num_heads=cfg.get("num_heads", 64),
            d_kv=cfg.get("d_kv", 64),
            d_ff=cfg.get("d_ff", 10240),
            vocab_size=cfg.get("vocab_size", 32128),
            rel_num_buckets=cfg.get("relative_attention_num_buckets", 32),
            rel_max_distance=cfg.get("relative_attention_max_distance", 128),
            layer_norm_eps=eps,
        )


class T5LayerNorm(nn.Module):
    # T5 RMSNorm: variance only (no mean subtraction), no bias, fp32 reduction.

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, x):
        orig = x.dtype
        xf = x.astype(mx.float32)
        variance = mx.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(variance + self.variance_epsilon)
        return (self.weight * xf).astype(orig)


class T5DenseGatedActDense(nn.Module):
    """对齐真真实权重 ffn.{fc1,fc2,gate.0} (非 DenseReluDense.{wi_0,wi_1,wo})."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # ffn.gate.0: Linear(d_model → d_ff) 门控 (gelu)
        # ffn.fc1:    Linear(d_model → d_ff) 主路 (linear)
        # ffn.fc2:    Linear(d_ff → d_model) 输出
        # MLX nn 无 ModuleList, gate 用命名属性 gate_0 对齐真实权重 gate.0.weight
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.gate_0 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        h_gelu = _gelu_erf(self.gate_0(x))
        h_lin = self.fc1(x)
        return self.fc2(h_gelu * h_lin)


class T5LayerFF(nn.Module):
    """对齐真真实权重 ffn.* + norm2."""

    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-6):
        super().__init__()
        self.ffn = T5DenseGatedActDense(d_model, d_ff)
        self.norm2 = T5LayerNorm(d_model, eps)

    def __call__(self, x):
        h = self.norm2(x)
        return x + self.ffn(h)


class T5Attention(nn.Module):
    def __init__(self, cfg: T5EncoderConfig, has_bias: bool = False):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.d_kv = cfg.d_kv
        self.inner_dim = cfg.inner_dim
        self.q = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, cfg.d_model, bias=False)
        self.relative_attention_bias = (
            nn.Embedding(cfg.rel_num_buckets, cfg.num_heads) if has_bias else None
        )
        self._rel_num_buckets = cfg.rel_num_buckets
        self._rel_max_distance = cfg.rel_max_distance

    @staticmethod
    def _relative_position_bucket(
        rel_pos, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = mx.zeros(rel_pos.shape, dtype=mx.int32)
        n = num_buckets
        if bidirectional:
            n = n // 2
            ret = ret + (rel_pos > 0).astype(mx.int32) * n
            rel_pos = mx.abs(rel_pos)
        else:
            rel_pos = -mx.minimum(rel_pos, mx.zeros_like(rel_pos))
        max_exact = n // 2
        is_small = rel_pos < max_exact
        rp = mx.maximum(rel_pos.astype(mx.float32), mx.array(1.0))
        large = max_exact + (
            mx.log(rp / max_exact)
            / math.log(max_distance / max_exact)
            * (n - max_exact)
        ).astype(mx.int32)
        large = mx.minimum(large, mx.full(large.shape, n - 1))
        return ret + mx.where(is_small, rel_pos.astype(mx.int32), large)

    def compute_bias(self, q_len: int, k_len: int):
        ctx_pos = mx.arange(q_len)[:, None]
        mem_pos = mx.arange(k_len)[None, :]
        rel_pos = mem_pos - ctx_pos
        bucket = self._relative_position_bucket(
            rel_pos,
            bidirectional=True,
            num_buckets=self._rel_num_buckets,
            max_distance=self._rel_max_distance,
        )
        values = self.relative_attention_bias(bucket)  # (q, k, n_heads)
        values = mx.transpose(values, [2, 0, 1])  # (n_heads, q, k)
        return mx.expand_dims(values, 0)  # (1, n_heads, q, k)

    def __call__(self, hidden, position_bias, mask=None):
        b, s, _ = hidden.shape
        q = self.q(hidden).reshape(b, s, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        k = self.k(hidden).reshape(b, s, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        v = self.v(hidden).reshape(b, s, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2))  # (b, h, s, s)
        scores = scores + position_bias
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = mx.matmul(attn, v)  # (b, h, s, d_kv)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o(out)


class T5LayerSelfAttention(nn.Module):
    """对齐真真实权重 attn.* + norm1 + pos_embedding."""

    def __init__(self, cfg: T5EncoderConfig, has_bias: bool = False):
        super().__init__()
        self.attn = T5Attention(cfg, has_bias)
        self.norm1 = T5LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        # pos_embedding: 真实权重 blocks.N.pos_embedding.embedding.weight (32, 64)
        # 用于相对位置偏置桶映射 (T5Attention.relative_attention_bias 的等价暴露)
        self.pos_embedding = nn.Module()
        self.pos_embedding.embedding = (
            nn.Embedding(cfg.rel_num_buckets, cfg.num_heads)
            if has_bias
            else nn.Embedding(cfg.rel_num_buckets, cfg.num_heads)
        )

    def __call__(self, hidden, position_bias, mask=None):
        h = self.norm1(hidden)
        return hidden + self.attn(h, position_bias, mask)


class T5Block(nn.Module):
    """对齐真真实权重 blocks.N.{attn,ffn,norm1,norm2,pos_embedding}."""

    def __init__(self, cfg: T5EncoderConfig, has_bias: bool = False):
        super().__init__()
        # 用命名属性 替 普通 list (MLX nn.Module 不收录 list 属性)
        self.layer0 = T5LayerSelfAttention(cfg, has_bias)
        self.layer1 = T5LayerFF(cfg.d_model, cfg.d_ff, cfg.layer_norm_eps)

    def __call__(self, hidden, position_bias, mask=None):
        hidden = self.layer0(hidden, position_bias, mask)
        hidden = self.layer1(hidden)
        return hidden


class T5Encoder(nn.Module):
    def __init__(self, cfg: T5EncoderConfig):
        super().__init__()
        self.cfg = cfg
        # 命名严格对齐 SkyReels-V3 真实权重: token_embedding.weight / blocks.N.* / norm.weight
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # MLX nn.Module 不收录普通 list 属性, 用 block_0/block_1/... 命名属性替 blocks list
        # (真真实权重 key 是 blocks.N.*, 加载时映射 block_{N} → blocks.{N})
        for i in range(cfg.num_layers):
            setattr(self, f"block_{i}", T5Block(cfg, has_bias=(i == 0)))
        self._num_blocks = cfg.num_layers
        self.norm = T5LayerNorm(cfg.d_model, cfg.layer_norm_eps)

    @property
    def blocks(self):
        """兼容旧代码 self.blocks[i] 访问 (代理到 block_0/block_1/...)."""
        return [getattr(self, f"block_{i}") for i in range(self._num_blocks)]

    def __call__(self, input_ids, attention_mask=None):
        hidden = self.token_embedding(input_ids)
        s = hidden.shape[1]
        # layer0 是 SelfAttention (含 compute_bias), layer1 是 FF
        position_bias = self.blocks[0].layer0.attn.compute_bias(s, s)
        mask = None
        if attention_mask is not None:
            ext = attention_mask[:, None, None, :].astype(mx.float32)
            mask = ((1.0 - ext) * _MASK_NEG).astype(hidden.dtype)
        for blk in self.blocks:
            hidden = blk(hidden, position_bias, mask)
        return self.norm(hidden)

    @classmethod
    def from_pretrained(cls, model_path, dtype=mx.float32) -> "T5Encoder":
        t0 = time.time()
        model_path = Path(model_path)
        cfg = json.loads((model_path / "config.json").read_text())
        config = T5EncoderConfig.from_hf_config(cfg)
        model = cls(config)
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"t5_encoder: no safetensors in {model_path}")
        logger.info(
            "t5: load path=%s layers=%d dtype=%s shards=%d",
            model_path.name,
            config.num_layers,
            dtype,
            len(shards),
        )
        raw = {}
        for shard in shards:
            with safe_open(str(shard), framework="numpy") as f:
                for k in list(f.keys()):
                    raw[k] = f.get_tensor(k)
        mapped = _map_t5_weights(raw)
        pairs = [(k, mx.array(v).astype(dtype)) for k, v in mapped.items()]
        n_params = sum(int(p[1].size) for p in pairs)
        model.load_weights(pairs, strict=False)
        mx.eval(model.parameters())
        logger.info(
            "t5: ready params=%.2fB dt=%.2fs",
            n_params / 1e9,
            time.time() - t0,
        )
        return model

    def encode(self, prompt, tokenizer, max_length: int = 256):
        t0 = time.time()
        enc = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="np",
        )
        input_ids = mx.array(np.asarray(enc["input_ids"], dtype=np.int32))
        attn = mx.array(np.asarray(enc["attention_mask"], dtype=np.int32))
        out = self.__call__(input_ids, attn)
        mx.eval(out)
        n_tok = int(attn.sum())
        logger.info(
            "t5: encode prompt_len=%d tokens=%d/%d out=%s dt=%.2fs",
            len(prompt),
            n_tok,
            max_length,
            tuple(out.shape),
            time.time() - t0,
        )
        return out


def _map_t5_weights(raw: dict) -> dict:
    # Keep encoder + shared embeddings only; drop decoder/lm_head.
    out = {}
    for k, v in raw.items():
        if k in ("shared.weight", "encoder.embed_tokens.weight"):
            out["embed_tokens.weight"] = v
        elif k.startswith("encoder."):
            out[k[len("encoder.") :]] = v
    return out


def load_t5_encoder(model_path, dtype=mx.float32) -> T5Encoder:
    return T5Encoder.from_pretrained(model_path, dtype=dtype)


def load_t5_tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))
