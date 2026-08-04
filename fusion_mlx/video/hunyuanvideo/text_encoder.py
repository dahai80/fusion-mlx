# SPDX-License-Identifier: Apache-2.0
# HunyuanVideo dual text encoder: CLIP-L (pooled) + Llama3-8B (sequence).
# Returns sequence embeddings (4096d) + pooled embeddings (768d) for DiT.

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _silu(x):
    return x * mx.sigmoid(x)


def _dequantize_fp8(weight, scale_weight):
    return weight.astype(mx.float32) * scale_weight.astype(mx.float32)


class _CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.self_attn.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.mlp.fc2 = nn.Linear(hidden_size * 4, hidden_size, bias=True)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

    def __call__(self, x):
        B, L, _ = x.shape
        h = self.layer_norm1(x)
        q = self.self_attn.q_proj(h).reshape(
            B, L, self._num_heads, self._head_dim
        ).transpose(0, 2, 1, 3)
        k = self.self_attn.k_proj(h).reshape(
            B, L, self._num_heads, self._head_dim
        ).transpose(0, 2, 1, 3)
        v = self.self_attn.v_proj(h).reshape(
            B, L, self._num_heads, self._head_dim
        ).transpose(0, 2, 1, 3)
        scale = self._head_dim**-0.5
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        x = x + self.self_attn.out_proj(out)
        x = x + self.mlp.fc2(nn.gelu(self.mlp.fc1(self.layer_norm2(x))))
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Module()
        self.embeddings.token_embedding = nn.Embedding(49408, hidden_size)
        self.embeddings.position_embedding = nn.Module()
        self.embeddings.position_embedding.weight = mx.zeros(
            (77, hidden_size), dtype=mx.float32
        )
        self.encoder = nn.Module()
        for i in range(num_layers):
            setattr(
                self.encoder, str(i), _CLIPEncoderLayer(hidden_size, num_heads)
            )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self._num_layers = num_layers

    def __call__(self, input_ids):
        x = self.embeddings.token_embedding(input_ids)
        x = x + self.embeddings.position_embedding.weight
        for i in range(self._num_layers):
            x = getattr(self.encoder, str(i))(x)
        x = self.final_layer_norm(x)
        eos_idx = (input_ids != 0).sum(axis=-1) - 1
        pooled = x[mx.arange(x.shape[0]), eos_idx]
        return x, pooled


class _LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.float32)
        self._eps = eps

    def __call__(self, x):
        x_type = x.dtype
        x = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self._eps)
        return (x * rrms).astype(x_type) * self.weight


class _LlamaAttention(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32, num_kv_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(
            hidden_size, num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim, hidden_size, bias=False
        )

    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(
            B, L, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(
            B, L, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(
            B, L, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)
        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class _LlamaMLP(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=14336):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class _LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
    ):
        super().__init__()
        self.self_attn = _LlamaAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = _LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = _LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = _LlamaRMSNorm(hidden_size)

    def __call__(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama3TextEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        num_layers=32,
        intermediate_size=14336,
        vocab_size=128320,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Module()
        for i in range(num_layers):
            setattr(
                self.layers,
                str(i),
                _LlamaDecoderLayer(
                    hidden_size, num_heads, num_kv_heads, intermediate_size
                ),
            )
        self.norm = _LlamaRMSNorm(hidden_size)
        self._num_layers = num_layers

    def __call__(self, input_ids):
        x = self.embed_tokens(input_ids)
        T = input_ids.shape[1]
        mask = mx.triu(mx.full((T, T), -1e9, dtype=mx.float32), k=1)
        mask = mask[None, None, :, :]
        for i in range(self._num_layers):
            x = getattr(self.layers, str(i))(x, mask)
        x = self.norm(x)
        return x


class HunyuanDualTextEncoder(nn.Module):
    def __init__(
        self,
        clip_l_hidden=768,
        clip_l_heads=12,
        clip_l_layers=12,
        llama_hidden=4096,
        llama_heads=32,
        llama_kv_heads=8,
        llama_layers=32,
        llama_intermediate=14336,
        llama_vocab=128320,
    ):
        super().__init__()
        self.clip_l = CLIPTextEncoder(
            hidden_size=clip_l_hidden,
            num_heads=clip_l_heads,
            num_layers=clip_l_layers,
        )
        self.llama3 = Llama3TextEncoder(
            hidden_size=llama_hidden,
            num_heads=llama_heads,
            num_kv_heads=llama_kv_heads,
            num_layers=llama_layers,
            intermediate_size=llama_intermediate,
            vocab_size=llama_vocab,
        )

    def __call__(self, input_ids_clip, input_ids_llama=None):
        _, pooled = self.clip_l(input_ids_clip)
        if input_ids_llama is None:
            input_ids_llama = input_ids_clip
        seq_emb = self.llama3(input_ids_llama)
        return seq_emb, pooled

    @property
    def output_dim(self):
        return self.llama3.hidden_size

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        import glob
        import os

        enc = cls(**kwargs)
        clip_files = sorted(
            glob.glob(os.path.join(model_path, "clip_l*.safetensors"))
        )
        llama_files = sorted(
            glob.glob(os.path.join(model_path, "llava*.safetensors"))
        )
        if not clip_files and not llama_files:
            safetensor_files = sorted(
                glob.glob(os.path.join(model_path, "*.safetensors"))
            )
            if not safetensor_files:
                logger.warning("no safetensors at %s, random init", model_path)
                return enc
            clip_files = [
                f for f in safetensor_files if "clip" in f.lower()
            ]
            llama_files = [
                f
                for f in safetensor_files
                if "llava" in f.lower() or "llama" in f.lower()
            ]

        from mlx.utils import tree_flatten

        if clip_files:
            clip_params = {}
            for sf in clip_files:
                clip_params.update(mx.load(sf))
            mapped_clip = _remap_clip_weights(clip_params)
            flat = tree_flatten(enc.clip_l.parameters())
            loaded = {}
            matched = 0
            for k, v in flat:
                if k in mapped_clip:
                    loaded[k] = (
                        mapped_clip[k].astype(mx.float16)
                        if mapped_clip[k].dtype != mx.float16
                        else mapped_clip[k]
                    )
                    matched += 1
                else:
                    loaded[k] = v
            logger.info(
                "hunyuan text enc clip_l: %d/%d matched", matched, len(flat)
            )
            _update_module(enc.clip_l, loaded)

        if llama_files:
            llama_params = {}
            for sf in llama_files:
                llama_params.update(mx.load(sf))
            mapped_llama = _remap_llama_weights(llama_params)
            flat = tree_flatten(enc.llama3.parameters())
            loaded = {}
            matched = 0
            for k, v in flat:
                if k in mapped_llama:
                    loaded[k] = (
                        mapped_llama[k].astype(mx.float16)
                        if mapped_llama[k].dtype != mx.float16
                        else mapped_llama[k]
                    )
                    matched += 1
                else:
                    loaded[k] = v
            logger.info(
                "hunyuan text enc llama3: %d/%d matched", matched, len(flat)
            )
            _update_module(enc.llama3, loaded)

        return enc


def _update_module(module, flat_params):
    nested = {}
    for key, val in flat_params.items():
        parts = key.split(".")
        d = nested
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val
    module.update(nested)


def _remap_clip_weights(params):
    out = {}
    for k, v in params.items():
        nk = k
        if nk.startswith("text_model."):
            nk = nk[len("text_model."):]
        nk = nk.replace("encoder.layers.", "encoder.")
        out[nk] = v
    return out


def _remap_llama_weights(params):
    out = {}
    for k, v in params.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk in ("scaled_fp8", "tokenizer") or nk.startswith("lm_head"):
            continue
        if nk.endswith(".scale_weight"):
            continue
        if nk.endswith(".weight"):
            base = k[:-7]
            scale_key = base + ".scale_weight"
            if scale_key in params and params[scale_key].size == 1:
                v = _dequantize_fp8(v, params[scale_key])
        out[nk] = v
    return out
