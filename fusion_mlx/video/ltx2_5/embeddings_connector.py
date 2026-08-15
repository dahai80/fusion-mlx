# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 Embeddings1DConnector (NEW vs LTX-2).
# 上游 ec.py：Embeddings1DConnector 消费 Gemma4 caption embedding，经 N 个 1D
# transformer block（gated self-attn + FF）处理后，产出 per-token hidden state，
# 既是 cross-attn 的 context 也是 duration-head 的输入。learnable_registers 替换
# padded token。键树：video/audio_embeddings_connector.{learnable_registers,
# transformer_1d_blocks.<0..7>.{attn1.*,ff.*}}（258 keys，已代码验证）。
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .config import LTXRopeType
from .feed_forward import FeedForward
from .rope import precompute_freqs_cis
from .utils import rms_norm

logger = logging.getLogger(__name__)


class _BasicTransformerBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
        ff_bias: bool = True,
    ):
        super().__init__()
        self.norm_eps = norm_eps
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=norm_eps,
            has_gate_logits=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim, bias=ff_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        additive_attention_mask: mx.array | None = None,
        pe: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        # 1. norm + self-attn（gated）
        norm_hidden_states = rms_norm(hidden_states, eps=self.norm_eps)
        attn_output = self.attn1(
            norm_hidden_states, mask=additive_attention_mask, pe=pe
        )
        hidden_states = attn_output + hidden_states

        # 2. norm + FF
        norm_hidden_states = rms_norm(hidden_states, eps=self.norm_eps)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        return hidden_states


class Embeddings1DConnector(nn.Module):
    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        num_layers: int = 8,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = True,
        ff_bias: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos
            if positional_embedding_max_pos is not None
            else [1]
        )
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.norm_eps = norm_eps

        self.transformer_1d_blocks = {
            idx: _BasicTransformerBlock1D(
                dim=self.inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=apply_gated_attention,
                ff_bias=ff_bias,
            )
            for idx in range(num_layers)
        }

        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            # learnable registers 初始化为 [-1,1] 均匀分布（对齐上游 torch.rand*2-1）。
            self.learnable_registers = mx.random.uniform(
                -1.0, 1.0, (self.num_learnable_registers, self.inner_dim)
            ).astype(mx.bfloat16)

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: mx.array,
        additive_attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # additive mask 形状 (B,1,1,S)：valid=0.0，padding=大负数。
        # binary_mask = (mask >= 0) → 1.0 valid / 0.0 padded。
        batch_size, seq_len, _ = hidden_states.shape
        assert (
            seq_len % self.num_learnable_registers == 0
        ), f"seq_len {seq_len} must be divisible by num_learnable_registers {self.num_learnable_registers}"

        repeats = seq_len // self.num_learnable_registers
        registers = mx.repeat(self.learnable_registers, repeats, axis=0)
        registers = mx.broadcast_to(
            mx.expand_dims(registers, axis=0),
            (batch_size, seq_len, self.inner_dim),
        )
        registers = registers.astype(hidden_states.dtype)

        binary_mask = additive_attention_mask[:, 0, 0, :]
        binary_mask = (binary_mask >= 0).astype(hidden_states.dtype)
        binary_mask = mx.expand_dims(binary_mask, axis=-1)

        hidden_states = binary_mask * hidden_states + (1 - binary_mask) * registers
        zero_mask = mx.zeros_like(additive_attention_mask)
        return hidden_states, zero_mask

    def __call__(
        self,
        hidden_states: mx.array,
        additive_attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if self.num_learnable_registers:
            if additive_attention_mask is None:
                b, s, _ = hidden_states.shape
                additive_attention_mask = mx.zeros(
                    (b, 1, 1, s), dtype=hidden_states.dtype
                )
            hidden_states, additive_attention_mask = (
                self._replace_padded_with_learnable_registers(
                    hidden_states, additive_attention_mask
                )
            )

        # 1D 位置：arange(seq_len) → (B,1,S) 作为 indices_grid。
        seq_len = hidden_states.shape[1]
        indices_grid = mx.arange(seq_len, dtype=mx.float32)
        indices_grid = mx.broadcast_to(
            mx.expand_dims(mx.expand_dims(indices_grid, axis=0), axis=0),
            (hidden_states.shape[0], 1, seq_len),
        )

        pe = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=False,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            double_precision=self.double_precision_rope,
        )

        for block in self.transformer_1d_blocks.values():
            hidden_states = block(
                hidden_states,
                additive_attention_mask=additive_attention_mask,
                pe=pe,
            )

        hidden_states = rms_norm(hidden_states, eps=self.norm_eps)
        return hidden_states, additive_attention_mask
