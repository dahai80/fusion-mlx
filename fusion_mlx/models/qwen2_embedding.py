# SPDX-License-Identifier: Apache-2.0
"""Native Qwen2 decoder embedding model (last-token pool + L2 normalize)."""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .base_model import (
    BaseModelArgs,
    BaseModelOutput,
    last_token_pool,
    normalize_embeddings,
)

logger = __import__("logging").getLogger(__name__)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen2"
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    intermediate_size: int = 8960
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int = 32768
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    is_causal: bool = True
    tie_word_embeddings: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    architectures: list[str] = field(default_factory=lambda: ["Qwen2ForCausalLM"])

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads


class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rotary_emb = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        queries = queries.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        queries = self.rotary_emb(queries)
        keys = self.rotary_emb(keys)
        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=attention_mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _build_attention_mask(
        self,
        attention_mask: mx.array | None,
        seq_length: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        if self.config.is_causal:
            keep = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
            keep = keep[None, None]
        else:
            keep = mx.ones((1, 1, seq_length, seq_length), dtype=mx.bool_)
        if attention_mask is not None:
            key_keep = (attention_mask != 0)[:, None, None, :]
            keep = keep & key_keep
        return mx.where(keep, 0.0, mx.finfo(dtype).min).astype(dtype)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        _, seq_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is not None and attention_mask.ndim != 2:
            mask = attention_mask
        else:
            mask = self._build_attention_mask(
                attention_mask, seq_length, hidden_states.dtype
            )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=mask)
        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        elif attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} doesn't match "
                f"input_ids shape {input_ids.shape}"
            )
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = last_token_pool(last_hidden_state, attention_mask)
        text_embeds = normalize_embeddings(pooled_output)
        return BaseModelOutput(
            text_embeds=text_embeds, last_hidden_state=last_hidden_state
        )

    def sanitize(self, weights: dict) -> dict:
        sanitized_weights = {}
        for key, value in weights.items():
            if "lm_head.weight" in key:
                continue
            if "rotary_emb.inv_freq" in key:
                continue
            if key.startswith("transformer."):
                new_key = key.replace("transformer.", "model.", 1)
            elif not key.startswith("model.") and "." in key:
                new_key = f"model.{key}"
            else:
                new_key = key
            sanitized_weights[new_key] = value
        return sanitized_weights
