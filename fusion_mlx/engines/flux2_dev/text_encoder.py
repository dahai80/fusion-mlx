import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_MISTRAL_PREFIX = "model."


class _MistralAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rope = nn.RoPE(head_dim, base=rope_theta)

    def __call__(self, hidden_states, mask, position_ids=None):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        q = self.rope(q)
        k = self.rope(k)
        if self.num_kv_heads < self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, reps, axis=1)
            v = mx.repeat(v, reps, axis=1)
        q_f32 = q.astype(mx.float32)
        k_f32 = k.astype(mx.float32)
        v_f32 = v.astype(mx.float32)
        out = mx.fast.scaled_dot_product_attention(
            q_f32, k_f32, v_f32, scale=self.scale, mask=mask
        )
        out = out.astype(q.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


class _MistralMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states):
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class _MistralDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rope_theta,
        rms_norm_eps,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _MistralAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = _MistralMLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size
        )

    def __call__(self, hidden_states, mask):
        r = self.self_attn(self.input_layernorm(hidden_states), mask)
        hidden_states = hidden_states + r
        r = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + r
        return hidden_states


class Mistral3TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=131072,
        hidden_size=5120,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=32768,
        rope_theta=1_000_000_000.0,
        rms_norm_eps=1e-5,
        head_dim=128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _MistralDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        logger.info(
            "Mistral3TextEncoder init: hidden_size=%d layers=%d heads=%d kv_heads=%d "
            "head_dim=%d inter=%d vocab=%d",
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
        )

    def _build_attention_mask(self, attention_mask, batch_size, seq_len, dtype):
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        if seq_len == 1:
            causal_tri_mask = mx.zeros((batch_size, 1, 1, 1), dtype=dtype)
        else:
            idx = mx.arange(seq_len, dtype=mx.int32)
            j = mx.expand_dims(idx, axis=0)
            i = mx.expand_dims(idx, axis=1)
            tri_bool = j > i
            zeros_2d = mx.zeros((seq_len, seq_len), dtype=dtype)
            neginf_2d = mx.full((seq_len, seq_len), -float("inf"), dtype=dtype)
            causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
            causal_tri_mask = mx.expand_dims(
                mx.expand_dims(causal_tri_mask, axis=0), axis=0
            )
            causal_tri_mask = mx.broadcast_to(
                causal_tri_mask, (batch_size, 1, seq_len, seq_len)
            )
        return causal_tri_mask + padding_mask

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        mask_dtype = hidden_states.dtype
        attention_mask_4d = self._build_attention_mask(
            attention_mask, batch_size, seq_len, mask_dtype
        )
        hidden_states_list = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_4d)
            if output_hidden_states:
                hidden_states_list.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states, hidden_states_list

    def get_prompt_embeds(
        self,
        input_ids,
        attention_mask=None,
        hidden_state_layers=(9, 18, 27),
    ):
        _, hidden_states_list = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hidden_states_list is None:
            raise RuntimeError("Hidden states not available for prompt embedding.")
        stacked = mx.stack([hidden_states_list[i] for i in hidden_state_layers], axis=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = mx.transpose(stacked, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )
        logger.info(
            "Mistral3 get_prompt_embeds: layers=%s out_dim=%d",
            hidden_state_layers,
            num_layers * hidden_dim,
        )
        return prompt_embeds

    def sanitize(self, weights):
        sanitized = {}
        dropped = 0
        for key, value in weights.items():
            if key.startswith(_MISTRAL_PREFIX):
                nk = key[len(_MISTRAL_PREFIX) :]
                sanitized[nk] = value
            elif (
                key.startswith("vision_tower.")
                or key.startswith("multi_modal_projector.")
                or key.startswith("tekken_model")
            ):
                dropped += 1
            else:
                sanitized[key] = value
        logger.info(
            "Mistral3 sanitize: kept=%d dropped=%d",
            len(sanitized),
            dropped,
        )
        return sanitized
