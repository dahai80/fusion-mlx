# SPDX-License-Identifier: Apache-2.0
import logging
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class Eagle3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        in_dim = input_size or hidden_size
        self.q_proj = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=500000.0)

    def __call__(self, x, mask=None, cache=None, position_offset=0):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        )

        offset = position_offset
        if cache is not None:
            offset = cache.offset
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Eagle3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3FirstLayer(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, num_heads, num_kv_heads, rms_eps
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.hidden_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.self_attn = Eagle3Attention(
            hidden_size, num_heads, num_kv_heads, input_size=2 * hidden_size
        )
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.mlp = Eagle3MLP(hidden_size, intermediate_size)

    def __call__(self, embeds, hidden, mask=None, cache=None, position_offset=0):
        embeds_normed = self.input_layernorm(embeds)
        hidden_normed = self.hidden_norm(hidden)
        h = mx.concatenate([embeds_normed, hidden_normed], axis=-1)
        h = self.self_attn(h, mask=mask, cache=cache, position_offset=position_offset)
        h = hidden + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class Eagle3DecoderLayer(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, num_heads, num_kv_heads, rms_eps
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.self_attn = Eagle3Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.mlp = Eagle3MLP(hidden_size, intermediate_size)

    def __call__(self, x, mask=None, cache=None, position_offset=0):
        h = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
            position_offset=position_offset,
        )
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class Eagle3Model(nn.Module):
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
        num_heads=32,
        num_kv_heads=8,
        num_layers=1,
        draft_vocab_size=32000,
        target_vocab_size=128256,
        rms_eps=1e-5,
        target_hidden_size=4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.draft_vocab_size = draft_vocab_size
        self.target_vocab_size = target_vocab_size
        self.uses_draft_vocab = draft_vocab_size != target_vocab_size
        self.target_hidden_size = target_hidden_size

        self.embed_tokens = nn.Embedding(target_vocab_size, hidden_size)
        self.fc = nn.Linear(3 * target_hidden_size, hidden_size, bias=False)
        self.layers = [
            Eagle3FirstLayer(
                hidden_size, intermediate_size, num_heads, num_kv_heads, rms_eps
            ),
            *[
                Eagle3DecoderLayer(
                    hidden_size, intermediate_size, num_heads, num_kv_heads, rms_eps
                )
                for _ in range(1, num_layers)
            ],
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)
        self.d2t = mx.zeros((draft_vocab_size,), dtype=mx.int32)

    def draft_to_target(self, draft_ids):
        draft_ids = draft_ids.astype(mx.int32)
        if self.uses_draft_vocab and self.d2t is not None:
            draft_ids = draft_ids + self.d2t[draft_ids]
        return draft_ids

    def forward_standalone(self, tokens, cache=None, hidden_state=None):
        x = self.embed_tokens(tokens)
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            offset = layer_cache.offset if layer_cache else 0
            if i == 0:
                if hidden_state is not None:
                    h = hidden_state
                else:
                    h = mx.zeros_like(x)
                x = layer(x, h, cache=layer_cache, position_offset=offset)
            else:
                x = layer(x, cache=layer_cache, position_offset=offset)
        x = self.norm(x)
        logits = self.lm_head(x)
        if self.uses_draft_vocab:
            draft_ids = mx.argmax(logits, axis=-1)
            target_ids = self.draft_to_target(draft_ids)
            B, L = target_ids.shape
            target_logits = mx.zeros((B, L, self.target_vocab_size), dtype=logits.dtype)
            rows = mx.arange(B)[:, None]
            cols = mx.arange(L)[None, :]
            target_logits[rows, cols, target_ids] = logits[rows, cols, draft_ids]
            return target_logits
        return logits


WEIGHT_REMAP = {
    "midlayer.input_layernorm.weight": "layers.0.input_layernorm.weight",
    "midlayer.hidden_norm.weight": "layers.0.hidden_norm.weight",
    "midlayer.self_attn.q_proj.weight": "layers.0.self_attn.q_proj.weight",
    "midlayer.self_attn.k_proj.weight": "layers.0.self_attn.k_proj.weight",
    "midlayer.self_attn.v_proj.weight": "layers.0.self_attn.v_proj.weight",
    "midlayer.self_attn.o_proj.weight": "layers.0.self_attn.o_proj.weight",
    "midlayer.post_attention_layernorm.weight": "layers.0.post_attention_layernorm.weight",
    "midlayer.mlp.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
    "midlayer.mlp.up_proj.weight": "layers.0.mlp.up_proj.weight",
    "midlayer.mlp.down_proj.weight": "layers.0.mlp.down_proj.weight",
}


def load_eagle3_weights(model_dir: str) -> tuple[dict, dict]:
    import glob
    import json

    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    weights = {}
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_files:
        pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(pytorch_path):
            import torch

            state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
            for k, v in state_dict.items():
                weights[k] = mx.array(v.numpy())
        else:
            raise FileNotFoundError(f"No weights found in {model_dir}")
    else:
        from safetensors import safe_open

        for sf in safetensor_files:
            with safe_open(sf, framework="numpy") as f:
                for k in f:
                    weights[k] = mx.array(f.get_tensor(k))

    remapped = {}
    for k, v in weights.items():
        if k == "t2d":
            continue
        new_key = WEIGHT_REMAP.get(k, k)
        if new_key == "d2t":
            v = v.astype(mx.int32)
        remapped[new_key] = v

    return cfg, remapped


def create_eagle3_model(model_dir: str) -> Eagle3Model:
    cfg, weights = load_eagle3_weights(model_dir)

    hidden_size = cfg.get("hidden_size", 4096)
    intermediate_size = cfg.get("intermediate_size", 14336)
    num_heads = cfg.get("num_attention_heads", 32)
    num_kv_heads = cfg.get("num_key_value_heads", 8)
    num_layers = cfg.get("num_hidden_layers", 1)
    draft_vocab_size = cfg.get("draft_vocab_size", 32000)
    target_vocab_size = cfg.get("vocab_size", 128256)
    rms_eps = cfg.get("rms_norm_eps", 1e-5)
    target_hidden_size = hidden_size

    model = Eagle3Model(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        draft_vocab_size=draft_vocab_size,
        target_vocab_size=target_vocab_size,
        rms_eps=rms_eps,
        target_hidden_size=target_hidden_size,
    )

    has_embed = "embed_tokens.weight" in weights
    if has_embed:
        model.load_weights(list(weights.items()))
    else:
        filtered = {k: v for k, v in weights.items() if k != "embed_tokens.weight"}
        model.load_weights(list(filtered.items()), strict=False)
        _init_embed_from_lm_head(model, weights)
    mx.eval(model.parameters())

    logger.info(
        "eagle3: model created hidden=%d heads=%d/%d layers=%d draft_vocab=%d target_vocab=%d has_embed=%s",
        hidden_size,
        num_heads,
        num_kv_heads,
        num_layers,
        draft_vocab_size,
        target_vocab_size,
        has_embed,
    )
    return model


def bind_target_embedding(model: Eagle3Model, embed_weight: mx.array):
    model.embed_tokens.weight = embed_weight
    mx.eval(model.embed_tokens.weight)
    logger.info("eagle3: bound target embed_tokens, shape=%s", embed_weight.shape)


def _init_embed_from_lm_head(model: Eagle3Model, weights: dict):
    lm_head_w = weights.get("lm_head.weight")
    d2t_w = weights.get("d2t")
    if lm_head_w is None:
        logger.info("eagle3: no lm_head weight, embed_tokens stays zero init")
        return
    target_vocab = model.target_vocab_size
    hidden = model.hidden_size
    embed = mx.zeros((target_vocab, hidden), dtype=lm_head_w.dtype)
    if d2t_w is not None:
        d2t_np = np.array(d2t_w, dtype=np.int32)
    else:
        d2t_np = np.zeros(model.draft_vocab_size, dtype=np.int32)
    n_bound = 0
    for i in range(model.draft_vocab_size):
        target_id = int(d2t_np[i]) + i
        if 0 <= target_id < target_vocab:
            embed[target_id] = lm_head_w[i]
            n_bound += 1
    model.embed_tokens.weight = embed
    logger.info(
        "eagle3: embed_tokens init from lm_head, bound %d/%d rows",
        n_bound,
        target_vocab,
    )
