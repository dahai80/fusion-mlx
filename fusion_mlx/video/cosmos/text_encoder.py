# SPDX-License-Identifier: Apache-2.0
# Cosmos text encoder: T5-11B (d_model=1024, relu FFN, 24 layers).

import json
import logging
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

COSMOS_T5_CONFIG = {
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 128,
    "d_kv": 128,
    "d_ff": 65536,
    "vocab_size": 32128,
    "rel_num_buckets": 32,
    "rel_max_distance": 128,
    "layer_norm_eps": 1e-6,
    "feed_forward_proj": "relu",
}


class _T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, x):
        orig = x.dtype
        xf = x.astype(mx.float32)
        variance = mx.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(variance + self.variance_epsilon)
        return (self.weight * xf).astype(orig)


class _T5DenseReluDense(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        return self.wo(nn.relu(self.wi(x)))


class _T5Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv, rel_num_buckets=32,
                 rel_max_distance=128, has_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_kv
        inner_dim = num_heads * d_kv
        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, d_model, bias=False)
        self.relative_attention_bias = (
            nn.Embedding(rel_num_buckets, num_heads) if has_bias else None
        )
        self._rel_num_buckets = rel_num_buckets
        self._rel_max_distance = rel_max_distance

    @staticmethod
    def _relative_position_bucket(rel_pos, num_buckets=32, max_distance=128):
        ret = mx.zeros(rel_pos.shape, dtype=mx.int32)
        n = num_buckets // 2
        ret = ret + (rel_pos > 0).astype(mx.int32) * n
        rel_pos = mx.abs(rel_pos)
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

    def compute_bias(self, q_len, k_len):
        ctx_pos = mx.arange(q_len)[:, None]
        mem_pos = mx.arange(k_len)[None, :]
        rel_pos = mem_pos - ctx_pos
        bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self._rel_num_buckets,
            max_distance=self._rel_max_distance,
        )
        values = self.relative_attention_bias(bucket)
        values = mx.transpose(values, [2, 0, 1])
        return mx.expand_dims(values, 0)

    def __call__(self, hidden, position_bias, mask=None):
        b, s, _ = hidden.shape
        q = self.q(hidden).reshape(b, s, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        k = self.k(hidden).reshape(b, s, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        v = self.v(hidden).reshape(b, s, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2))
        scores = scores + position_bias
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o(out)


class _T5Block(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, d_kv, eps=1e-6,
                 has_bias=False, rel_num_buckets=32, rel_max_distance=128):
        super().__init__()
        self.attn = _T5Attention(
            d_model, num_heads, d_kv, rel_num_buckets, rel_max_distance, has_bias,
        )
        self.norm1 = _T5LayerNorm(d_model, eps)
        self.ffn = _T5DenseReluDense(d_model, d_ff)
        self.norm2 = _T5LayerNorm(d_model, eps)

    def __call__(self, hidden, position_bias, mask=None):
        h = self.norm1(hidden)
        hidden = hidden + self.attn(h, position_bias, mask)
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden


class CosmosT5Encoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = config or COSMOS_T5_CONFIG
        self.d_model = cfg["d_model"]
        self.num_layers = cfg["num_layers"]
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.blocks = nn.Module()
        for i in range(cfg["num_layers"]):
            block = _T5Block(
                cfg["d_model"],
                cfg["d_ff"],
                cfg["num_heads"],
                cfg["d_kv"],
                eps=cfg.get("layer_norm_eps", 1e-6),
                has_bias=(i == 0),
                rel_num_buckets=cfg.get("rel_num_buckets", 32),
                rel_max_distance=cfg.get("rel_max_distance", 128),
            )
            setattr(self.blocks, str(i), block)
        self.norm = _T5LayerNorm(cfg["d_model"], cfg.get("layer_norm_eps", 1e-6))

    def __call__(self, input_ids, attention_mask=None):
        hidden = self.token_embedding(input_ids)
        s = hidden.shape[1]
        block0 = getattr(self.blocks, "0")
        position_bias = block0.attn.compute_bias(s, s)
        mask = None
        if attention_mask is not None:
            ext = attention_mask[:, None, None, :].astype(mx.float32)
            mask = ((1.0 - ext) * -1e9).astype(hidden.dtype)
        for i in range(self.num_layers):
            block = getattr(self.blocks, str(i))
            hidden = block(hidden, position_bias, mask)
        return self.norm(hidden)

    @classmethod
    def from_pretrained(cls, model_path, dtype=mx.float32):
        t0 = time.time()
        model_path = Path(model_path)
        config_json = model_path / "config.json"
        if config_json.exists():
            with open(config_json) as f:
                cfg = json.load(f)
            config = {
                "d_model": cfg.get("d_model", 1024),
                "num_layers": cfg.get("num_layers", 24),
                "num_heads": cfg.get("num_heads", 128),
                "d_kv": cfg.get("d_kv", 128),
                "d_ff": cfg.get("d_ff", 65536),
                "vocab_size": cfg.get("vocab_size", 32128),
                "rel_num_buckets": cfg.get("relative_attention_num_buckets", 32),
                "rel_max_distance": cfg.get("relative_attention_max_distance", 128),
                "layer_norm_eps": cfg.get("layer_norm_epsilon", 1e-6),
            }
        else:
            config = COSMOS_T5_CONFIG
            logger.info("cosmos t5: no config.json, using defaults")
        model = cls(config)
        if model_path.is_file() and str(model_path).endswith(".safetensors"):
            shards = [model_path]
        else:
            shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"cosmos t5: no safetensors in {model_path}")
        logger.info("cosmos t5: loading %d shards from %s", len(shards), model_path.name)
        raw = {}
        for shard in shards:
            raw.update(mx.load(str(shard)))
        mapped = _map_cosmos_t5_weights(raw)
        _update_module(model, mapped)
        mx.eval(model.parameters())
        elapsed = time.time() - t0
        logger.info("cosmos t5: ready d_model=%d layers=%d dt=%.2fs", config["d_model"], config["num_layers"], elapsed)
        return model

    def encode(self, prompt, tokenizer, max_length=512):
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
        logger.info("cosmos t5: encoded %d tokens dt=%.2fs", n_tok, time.time() - t0)
        return out


def _map_cosmos_t5_weights(raw):
    import re
    out = {}
    for k, v in raw.items():
        if k in ("shared.weight", "encoder.embed_tokens.weight"):
            out["token_embedding.weight"] = v
        elif k == "encoder.final_layer_norm.weight":
            out["norm.weight"] = v
        elif k.startswith("encoder.block."):
            m = re.match(r"encoder\.block\.(\d+)\.layer\.(\d+)\.(.+)", k)
            if m:
                block_n, layer_n, rest = m.group(1), m.group(2), m.group(3)
                prefix = f"blocks.{block_n}"
                if layer_n == "0":
                    if rest == "SelfAttention.relative_attention_bias.weight":
                        out[f"{prefix}.attn.relative_attention_bias.weight"] = v
                    elif rest.startswith("SelfAttention."):
                        attr = rest[len("SelfAttention."):]
                        out[f"{prefix}.attn.{attr}"] = v
                    elif rest.startswith("layer_norm."):
                        attr = rest[len("layer_norm."):]
                        out[f"{prefix}.norm1.{attr}"] = v
                    else:
                        out[f"{prefix}.{rest}"] = v
                elif layer_n == "1":
                    if rest == "DenseReluDense.wi.weight":
                        out[f"{prefix}.ffn.wi.weight"] = v
                    elif rest == "DenseReluDense.wo.weight":
                        out[f"{prefix}.ffn.wo.weight"] = v
                    elif rest.startswith("layer_norm."):
                        attr = rest[len("layer_norm."):]
                        out[f"{prefix}.norm2.{attr}"] = v
                    else:
                        out[f"{prefix}.{rest}"] = v
                else:
                    out[f"{prefix}.{rest}"] = v
            else:
                out[k[len("encoder."):]] = v
        elif not k.startswith(("decoder.", "lm_head.")):
            out[k] = v
    return out


def _flatten_params(params, prefix=""):
    out = {}
    if isinstance(params, dict):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_params(v, key))
    elif isinstance(params, mx.array):
        out[prefix] = params
    return out


def _update_module(module, flat_params):
    children = {}
    for k, v in flat_params.items():
        parts = k.split(".")
        if len(parts) == 1:
            if isinstance(v, mx.array):
                setattr(module, k, v)
            continue
        child_name = parts[0]
        rest = ".".join(parts[1:])
        if child_name not in children:
            children[child_name] = {}
        children[child_name][rest] = v
    for child_name, child_params in children.items():
        child = getattr(module, child_name, None)
        if child is not None and isinstance(child, nn.Module):
            _update_module(child, child_params)
        elif child is not None and isinstance(child, mx.array):
            if len(child_params) == 1 and "" in child_params:
                setattr(module, child_name, child_params[""])
