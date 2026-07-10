# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2 text encoder (vendored from mlx-video).
# Phase 4 LTX-2 direct-MLX port: Gemma3-based text encoder + connectors.
import functools
import logging
import math
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.gemma3.config import TextConfig
from mlx_vlm.models.gemma3.language import Gemma3Model

from .utils import apply_quantization, rms_norm

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_system_prompt(prompt_name: str) -> str:
    prompt_path = PROMPTS_DIR / prompt_name
    if prompt_path.exists():
        with open(prompt_path) as f:
            return f.read()
    raise FileNotFoundError(f"System prompt not found: {prompt_path}")


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(self.config)

    def _create_causal_mask_with_padding(
        self,
        seq_len: int,
        attention_mask: mx.array | None,
        dtype: mx.Dtype,
    ) -> mx.array:
        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))

        if attention_mask is not None:
            padding_mask = attention_mask.astype(mx.bool_)
            combined = causal_mask[None, :, :] & padding_mask[:, None, :]
            min_val = (
                mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            )
            mask = mx.where(
                combined,
                mx.zeros(combined.shape, dtype=dtype),
                mx.full(combined.shape, min_val, dtype=dtype),
            )
            return mask[:, None, :, :]
        else:
            min_val = (
                mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            )
            mask = mx.where(
                causal_mask,
                mx.zeros((seq_len, seq_len), dtype=dtype),
                mx.full((seq_len, seq_len), min_val, dtype=dtype),
            )
            return mask[None, None, :, :]

    def __call__(
        self,
        inputs: mx.array,
        input_embeddings: mx.array | None = None,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
        cache: list[mx.array] | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        batch_size, seq_len = inputs.shape

        h = (
            input_embeddings
            if input_embeddings is not None
            else self.model.embed_tokens(inputs)
        )
        h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        mx.eval(h)

        all_hidden_states = [h] if output_hidden_states else []

        if cache is None:
            cache = [None] * len(self.model.layers)

        full_causal_mask = self._create_causal_mask_with_padding(
            seq_len, attention_mask, h.dtype
        )

        sliding_mask = full_causal_mask

        num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )
            local_mask = full_causal_mask if is_global else sliding_mask

            h = layer(h, local_mask, cache[i])
            mx.eval(h)

            if output_hidden_states and i < num_layers - 1:
                all_hidden_states.append(h)

        hidden_states = self.model.norm(h)
        mx.eval(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        else:
            return self.model.embed_tokens.as_linear(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        prefix = "language_model."
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                if hasattr(value, "dtype") and value.dtype == mx.float32:
                    sanitized[key[len(prefix) :]] = value.astype(mx.bfloat16)
                else:
                    sanitized[key[len(prefix) :]] = value
        return sanitized

    @property
    def layers(self) -> list[nn.Module]:
        return self.model.layers

    def make_cache(self):
        from mlx_vlm.models.cache import KVCache, RotatingKVCache

        caches = []
        for i in range(len(self.layers)):
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window))
        return caches

    @classmethod
    def from_pretrained(cls, model_path: str):
        import json

        weight_files = sorted(Path(model_path).glob("*.safetensors"))
        config_file = Path(model_path) / "config.json"
        config_dict = {}
        if config_file.exists():
            with open(config_file) as f:
                config_dict = json.load(f)
            language_model = cls(
                config=TextConfig.from_dict(config_dict["text_config"])
            )
        else:
            raise ValueError(f"Config file not found at {model_path}")

        quantization = config_dict.get("quantization", None)
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(str(wf)))

        if hasattr(language_model, "sanitize"):
            weights = language_model.sanitize(weights=weights)

        apply_quantization(
            model=language_model, weights=weights, quantization=quantization
        )

        language_model.load_weights(list(weights.items()), strict=False)

        logger.info(
            "LanguageModel.from_pretrained: loaded %d weight files from %s, "
            "quantization=%s",
            len(weight_files),
            model_path,
            quantization is not None,
        )
        return language_model


class ConnectorAttention(nn.Module):
    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        has_gate_logits: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)

        self.q_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(inner_dim, eps=1e-6)

        if has_gate_logits:
            self.to_gate_logits = nn.Linear(dim, num_heads, bias=True)

    def __call__(
        self,
        x: mx.array,
        attention_mask: mx.array | None = None,
        pe: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        gate = None
        if hasattr(self, "to_gate_logits"):
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = mx.reshape(
            q, (batch_size, seq_len, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        k = mx.reshape(
            k, (batch_size, seq_len, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        v = mx.reshape(
            v, (batch_size, seq_len, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)

        if pe is not None:
            q = self._apply_split_rope(q, pe[0], pe[1])
            k = self._apply_split_rope(k, pe[0], pe[1])

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        if gate is not None:
            out = mx.reshape(out, (batch_size, seq_len, self.num_heads, self.head_dim))
            out = out * gate[..., None]
            out = mx.reshape(out, (batch_size, seq_len, -1))

        return self.to_out(out)

    def _apply_split_rope(
        self,
        x: mx.array,
        cos_freq: mx.array,
        sin_freq: mx.array,
    ) -> mx.array:
        input_dtype = x.dtype

        x = x.astype(mx.float32)
        cos_freq = cos_freq.astype(mx.float32)
        sin_freq = sin_freq.astype(mx.float32)

        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        out1 = x1 * cos_freq - x2 * sin_freq
        out2 = x2 * cos_freq + x1 * sin_freq

        return mx.concatenate([out1, out2], axis=-1).astype(input_dtype)


class GEGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.proj(x))


class ConnectorFeedForward(nn.Module):
    def __init__(self, dim: int = 3840, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.proj_in = nn.Linear(dim, inner_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu_approx(self.proj_in(x))
        x = self.dropout(x)
        x = self.proj_out(x)
        return x


class ConnectorTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        has_gate_logits: bool = False,
    ):
        super().__init__()
        self.attn1 = ConnectorAttention(
            dim, num_heads, head_dim, has_gate_logits=has_gate_logits
        )
        self.ff = ConnectorFeedForward(dim)

    def __call__(
        self,
        x: mx.array,
        attention_mask: mx.array | None = None,
        pe: mx.array | None = None,
    ) -> mx.array:
        norm_x = rms_norm(x)
        if norm_x.ndim == 4:
            norm_x = mx.squeeze(norm_x, axis=1)
        attn_out = self.attn1(norm_x, attention_mask, pe)
        x = x + attn_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)

        norm_x = rms_norm(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)

        return x


class Embeddings1DConnector(nn.Module):
    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: int = 128,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list | None = None,
        has_gate_logits: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_learnable_registers = num_learnable_registers
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [1]

        self.transformer_1d_blocks = {
            i: ConnectorTransformerBlock(
                dim, num_heads, head_dim, has_gate_logits=has_gate_logits
            )
            for i in range(num_layers)
        }

        if num_learnable_registers > 0:
            self.learnable_registers = mx.zeros((num_learnable_registers, dim))

    def _precompute_freqs_cis(
        self, seq_len: int, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        import numpy as np

        dim = self.num_heads * self.head_dim
        theta = self.positional_embedding_theta
        max_pos = self.positional_embedding_max_pos
        n_elem = 2 * len(max_pos)

        start = 1.0
        end = theta
        num_indices = dim // n_elem

        log_start = np.log(start) / np.log(theta)
        log_end = np.log(end) / np.log(theta)
        lin_space = np.linspace(log_start, log_end, num_indices, dtype=np.float64)
        indices = (np.power(theta, lin_space) * (np.pi / 2)).astype(np.float32)

        positions = np.arange(seq_len, dtype=np.float32)
        fractional_positions = positions / max_pos[0]
        scaled_positions = fractional_positions * 2 - 1

        freqs = scaled_positions[:, None] * indices[None, :]

        expected_freqs = dim // 2
        pad_size = expected_freqs - freqs.shape[-1]

        cos_freq = np.cos(freqs)
        sin_freq = np.sin(freqs)

        if pad_size > 0:
            cos_padding = np.ones((seq_len, pad_size), dtype=np.float32)
            sin_padding = np.zeros((seq_len, pad_size), dtype=np.float32)
            cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)

        cos_freq = cos_freq.reshape(seq_len, self.num_heads, self.head_dim // 2)
        sin_freq = sin_freq.reshape(seq_len, self.num_heads, self.head_dim // 2)

        cos_freq = np.transpose(cos_freq, (1, 0, 2))[np.newaxis, ...]
        sin_freq = np.transpose(sin_freq, (1, 0, 2))[np.newaxis, ...]

        cos_full = mx.array(cos_freq)
        sin_full = mx.array(sin_freq)

        return cos_full.astype(dtype), sin_full.astype(dtype)

    def _replace_padded_with_registers(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        batch_size, seq_len, dim = hidden_states.shape
        dtype = hidden_states.dtype

        mask_binary = (attention_mask.squeeze(1).squeeze(1) >= -9000.0).astype(mx.int32)

        num_tiles = seq_len // self.num_learnable_registers
        registers = mx.tile(self.learnable_registers, (num_tiles, 1)).astype(dtype)

        result_list = []
        for b in range(batch_size):
            mask_b = mask_binary[b]
            hs_b = hidden_states[b]

            num_valid = int(mx.sum(mask_b))

            valid_tokens = hs_b[seq_len - num_valid :]

            pad_length = seq_len - num_valid
            if pad_length > 0:
                padding = mx.zeros((pad_length, dim), dtype=dtype)
                adjusted = mx.concatenate([valid_tokens, padding], axis=0)
            else:
                adjusted = valid_tokens

            flipped_mask = mx.concatenate(
                [
                    mx.ones((num_valid,), dtype=mx.int32),
                    mx.zeros((pad_length,), dtype=mx.int32),
                ],
                axis=0,
            )

            flipped_mask_expanded = flipped_mask[:, None].astype(dtype)
            combined = (
                flipped_mask_expanded * adjusted
                + (1 - flipped_mask_expanded) * registers
            )
            result_list.append(combined)

        hidden_states = mx.stack(result_list, axis=0)
        attention_mask = mx.zeros_like(attention_mask)

        return hidden_states, attention_mask

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if self.num_learnable_registers > 0 and attention_mask is not None:
            hidden_states, attention_mask = self._replace_padded_with_registers(
                hidden_states, attention_mask
            )

        seq_len = hidden_states.shape[1]
        freqs_cis = self._precompute_freqs_cis(seq_len, hidden_states.dtype)

        for i in range(len(self.transformer_1d_blocks)):
            hidden_states = self.transformer_1d_blocks[i](
                hidden_states, attention_mask, freqs_cis
            )

        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask


def norm_and_concat_hidden_states(
    hidden_states: list[mx.array],
    attention_mask: mx.array,
    padding_side: str = "left",
) -> mx.array:
    stacked = mx.stack(hidden_states, axis=-1)
    dtype = stacked.dtype
    b, t, d, num_layers = stacked.shape

    sequence_lengths = mx.sum(attention_mask, axis=-1)

    token_indices = mx.arange(t)[None, :]

    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    else:
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices

    mask = mask[:, :, None, None]
    eps = mx.array(1e-6, dtype=dtype)

    masked = mx.where(mask, stacked, mx.zeros_like(stacked))
    denom = (sequence_lengths * d).reshape(b, 1, 1, 1).astype(dtype)
    mean = mx.sum(masked, axis=(1, 2), keepdims=True) / (denom + eps)

    x_for_min = mx.where(
        mask, stacked, mx.full(stacked.shape, float("inf"), dtype=dtype)
    )
    x_for_max = mx.where(
        mask, stacked, mx.full(stacked.shape, float("-inf"), dtype=dtype)
    )
    x_min = mx.min(x_for_min, axis=(1, 2), keepdims=True)
    x_max = mx.max(x_for_max, axis=(1, 2), keepdims=True)
    range_val = x_max - x_min

    normed = 8 * (stacked - mean) / (range_val + eps)

    normed = mx.reshape(normed, (b, t, -1))

    mask_flat = mx.broadcast_to(mask[:, :, :, 0], (b, t, d * num_layers))
    normed = mx.where(mask_flat, normed, mx.zeros_like(normed))

    return normed


def norm_and_concat_per_token_rms(
    encoded_text: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    b, t, d, num_layers = encoded_text.shape
    dtype = encoded_text.dtype

    variance = mx.mean(encoded_text.astype(mx.float32) ** 2, axis=2, keepdims=True)
    normed = encoded_text.astype(mx.float32) * mx.rsqrt(variance + 1e-6)
    normed = normed.astype(dtype)

    normed = mx.reshape(normed, (b, t, d * num_layers))

    mask_3d = attention_mask[:, :, None].astype(mx.bool_)
    normed = mx.where(mask_3d, normed, mx.zeros_like(normed))

    return normed


def _rescale_norm(x: mx.array, target_dim: int, source_dim: int) -> mx.array:
    return x * math.sqrt(target_dim / source_dim)


class GemmaFeaturesExtractor(nn.Module):
    def __init__(
        self, input_dim: int = 188160, output_dim: int = 3840, bias: bool = False
    ):
        super().__init__()
        self.aggregate_embed = nn.Linear(input_dim, output_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.aggregate_embed(x)


class GemmaFeaturesExtractorV2(nn.Module):
    def __init__(
        self,
        flat_dim: int,
        embedding_dim: int,
        video_output_dim: int,
        audio_output_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.video_aggregate_embed = nn.Linear(flat_dim, video_output_dim, bias=bias)
        self.audio_aggregate_embed = nn.Linear(flat_dim, audio_output_dim, bias=bias)

    def __call__(
        self,
        hidden_states: list[mx.array],
        attention_mask: mx.array,
        mode: str = "video",
    ) -> mx.array:
        encoded = mx.stack(hidden_states, axis=-1)

        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.astype(encoded.dtype)

        if mode == "video":
            target_dim = self.video_aggregate_embed.weight.shape[0]
            return self.video_aggregate_embed(
                _rescale_norm(normed, target_dim, self.embedding_dim)
            )
        else:
            target_dim = self.audio_aggregate_embed.weight.shape[0]
            return self.audio_aggregate_embed(
                _rescale_norm(normed, target_dim, self.embedding_dim)
            )


class AudioEmbeddingsConnector(nn.Module):
    def __init__(self, input_dim: int = 3840, output_dim: int = 2048):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class LTX2TextEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 3840,
        audio_dim: int = 2048,
        num_layers: int = 49,
        has_prompt_adaln: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_dim = audio_dim
        self.num_layers = num_layers
        self.has_prompt_adaln = has_prompt_adaln
        self.language_model = None

        feature_input_dim = hidden_dim * num_layers

        if has_prompt_adaln:
            video_output_dim = 4096
            audio_output_dim = 2048
            self.feature_extractor_v2 = GemmaFeaturesExtractorV2(
                flat_dim=feature_input_dim,
                embedding_dim=hidden_dim,
                video_output_dim=video_output_dim,
                audio_output_dim=audio_output_dim,
                bias=True,
            )

            self.video_embeddings_connector = Embeddings1DConnector(
                dim=video_output_dim,
                num_heads=32,
                head_dim=128,
                num_layers=8,
                num_learnable_registers=128,
                positional_embedding_max_pos=[4096],
                has_gate_logits=True,
            )
            self.audio_embeddings_connector = Embeddings1DConnector(
                dim=audio_output_dim,
                num_heads=32,
                head_dim=64,
                num_layers=8,
                num_learnable_registers=128,
                positional_embedding_max_pos=[4096],
                has_gate_logits=True,
            )
        else:
            self.feature_extractor = GemmaFeaturesExtractor(
                feature_input_dim, hidden_dim
            )

            self.video_embeddings_connector = Embeddings1DConnector(
                dim=hidden_dim,
                num_heads=30,
                head_dim=128,
                num_layers=2,
                num_learnable_registers=128,
                positional_embedding_max_pos=[1],
            )
            self.audio_embeddings_connector = Embeddings1DConnector(
                dim=hidden_dim,
                num_heads=30,
                head_dim=128,
                num_layers=2,
                num_learnable_registers=128,
                positional_embedding_max_pos=[1],
            )

        self.processor = None

    def load(
        self,
        model_path: str | None = None,
        text_encoder_path: str | None = "google/gemma-3-12b-it",
    ):
        if model_path is None:
            raise ValueError("model_path is required for LTX2TextEncoder.load")
        model_path = Path(model_path)

        if Path(str(text_encoder_path)).joinpath("text_encoder").is_dir():
            text_encoder_path = str(Path(text_encoder_path) / "text_encoder")

        self.language_model = LanguageModel.from_pretrained(text_encoder_path)

        transformer_weights = {}
        is_reformatted = False

        text_proj_dir = model_path / "text_projections"
        if text_proj_dir.is_dir():
            is_reformatted = True
            for sf in text_proj_dir.glob("*.safetensors"):
                transformer_weights.update(mx.load(str(sf)))

        if not transformer_weights:
            transformer_files = list(model_path.glob("ltx-2-19*.safetensors"))
            if transformer_files:
                transformer_weights = mx.load(str(transformer_files[0]))

        if transformer_weights:
            self._load_feature_extractors(transformer_weights, is_reformatted)
            self._load_connector(
                "video_embeddings_connector", transformer_weights, is_reformatted
            )
            self._load_connector(
                "audio_embeddings_connector", transformer_weights, is_reformatted
            )
        else:
            logger.warning(
                "No transformer weights found for text projection connectors; "
                "text conditioning will use uninitialized weights"
            )

        from transformers import AutoTokenizer

        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            self.processor = AutoTokenizer.from_pretrained(
                str(tokenizer_path), trust_remote_code=True
            )
        else:
            try:
                self.processor = AutoTokenizer.from_pretrained(
                    text_encoder_path, trust_remote_code=True
                )
            except Exception:
                self.processor = AutoTokenizer.from_pretrained(
                    "google/gemma-3-12b-it", trust_remote_code=True
                )
        self.processor.padding_side = "left"
        logger.info("Text encoder loaded successfully")

    def _load_feature_extractors(self, weights: dict, is_reformatted: bool):
        if self.has_prompt_adaln:
            for attr, prefix in [
                ("video_aggregate_embed", "video_aggregate_embed"),
                ("audio_aggregate_embed", "audio_aggregate_embed"),
            ]:
                w_key = f"{prefix}.weight"
                b_key = f"{prefix}.bias"
                if w_key in weights:
                    submodule = getattr(self.feature_extractor_v2, attr)
                    submodule.weight = weights[w_key]
                    if b_key in weights:
                        submodule.bias = weights[b_key]
        else:
            agg_key = (
                "aggregate_embed.weight"
                if is_reformatted
                else "text_embedding_projection.aggregate_embed.weight"
            )
            if agg_key in weights:
                self.feature_extractor.aggregate_embed.weight = weights[agg_key]

    def _load_connector(self, name: str, weights: dict, is_reformatted: bool):
        connector = getattr(self, name)

        connector_weights = {}
        if is_reformatted:
            prefix = f"{name}."
            for key, value in weights.items():
                if key.startswith(prefix):
                    connector_weights[key[len(prefix) :]] = value
        else:
            mono_prefix = f"model.diffusion_model.{name}."
            for key, value in weights.items():
                if key.startswith(mono_prefix):
                    connector_weights[key[len(mono_prefix) :]] = value

        if not connector_weights:
            return

        mapped = {}
        for key, value in connector_weights.items():
            new_key = key
            if not is_reformatted:
                new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
                new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
                new_key = new_key.replace(".to_out.0.", ".to_out.")
            mapped[new_key] = value

        connector.load_weights(list(mapped.items()), strict=False)

        if "learnable_registers" in connector_weights:
            connector.learnable_registers = connector_weights["learnable_registers"]

    def encode(
        self,
        prompt: str,
        max_length: int = 1024,
        return_audio_embeddings: bool = True,
    ) -> tuple[mx.array, mx.array]:
        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.processor(
            prompt,
            return_tensors="np",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        _, all_hidden_states = self.language_model(
            inputs=input_ids,
            input_embeddings=None,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if self.has_prompt_adaln:
            video_features = self.feature_extractor_v2(
                all_hidden_states, attention_mask, mode="video"
            )
            additive_mask = (attention_mask - 1).astype(video_features.dtype)
            additive_mask = (
                additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
            )

            video_embeddings, _ = self.video_embeddings_connector(
                video_features, additive_mask
            )

            if return_audio_embeddings:
                audio_features = self.feature_extractor_v2(
                    all_hidden_states, attention_mask, mode="audio"
                )
                audio_mask = (attention_mask - 1).astype(audio_features.dtype)
                audio_mask = audio_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
                audio_embeddings, _ = self.audio_embeddings_connector(
                    audio_features, audio_mask
                )
                return video_embeddings, audio_embeddings
            else:
                return video_embeddings, attention_mask
        else:
            concat_hidden = norm_and_concat_hidden_states(
                all_hidden_states, attention_mask, padding_side="left"
            )

            video_features = self.feature_extractor(concat_hidden)
            additive_mask = (attention_mask - 1).astype(video_features.dtype)
            additive_mask = (
                additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
            )

            video_embeddings, _ = self.video_embeddings_connector(
                video_features, additive_mask
            )

            if return_audio_embeddings:
                audio_embeddings, _ = self.audio_embeddings_connector(
                    video_features, additive_mask
                )
                return video_embeddings, audio_embeddings
            else:
                return video_embeddings, attention_mask

    def __call__(
        self,
        prompt: str,
        max_length: int = 1024,
        return_audio_embeddings: bool = True,
    ) -> tuple[mx.array, mx.array]:
        return self.encode(prompt, max_length, return_audio_embeddings)

    @functools.cached_property
    def default_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")

    @functools.cached_property
    def default_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    def _clean_response(self, response: str) -> str:
        response = response.strip()
        response = re.sub(r"^[^\w\s]+", "", response)
        return response

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "user":
                if isinstance(content, str):
                    formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif isinstance(content, list):
                    text_parts = [c["text"] for c in content if c.get("type") == "text"]
                    formatted += (
                        f"<start_of_turn>user\n{' '.join(text_parts)}"
                        f"<end_of_turn>\n"
                    )
            elif role == "assistant":
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        formatted += "<start_of_turn>model\n"
        return formatted

    def enhance_t2v(
        self,
        prompt: str,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ) -> str:
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ImportError:
            logger.warning(
                "mlx-lm not available for prompt enhancement; using original prompt"
            )
            return prompt

        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        system_prompt = system_prompt or self.default_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        formatted = self._apply_chat_template(messages)

        mx.random.seed(seed)

        inputs = self.processor(
            formatted,
            return_tensors="np",
            add_special_tokens=False,
        )
        input_ids = mx.array(inputs["input_ids"])

        sampler = make_sampler(
            kwargs.get("temperature", 0.7),
            kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
        )
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias"),
            kwargs.get("repetition_penalty", 1.3),
            kwargs.get("repetition_context_size", 20),
        )

        generated_tokens = []

        generator = stream_generate(
            self.language_model,
            tokenizer=self.processor,
            prompt=input_ids.squeeze(0),
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )

        for i, response in enumerate(generator):
            next_token = mx.array([response.token])
            input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
            generated_tokens.append(response.token)

            if verbose and i % 50 == 0:
                logger.info("enhance_t2v: generated %d tokens", len(generated_tokens))

            if i % 50 == 0:
                mx.clear_cache()

            if response.token == 1 or response.token == 107:
                break

        mx.clear_cache()

        enhanced_prompt = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        )

        enhanced_prompt = self._clean_response(enhanced_prompt)
        logger.info("Enhanced prompt: %s", enhanced_prompt)

        return enhanced_prompt

    def enhance_i2v(
        self,
        prompt: str,
        image: mx.array | None = None,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ) -> str:
        return self.enhance_t2v(
            prompt,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt or self.default_i2v_system_prompt,
            seed=seed,
        )


def load_text_encoder(model_path: str = "/tmp/ltx2") -> LTX2TextEncoder:
    encoder = LTX2TextEncoder()
    encoder.load(model_path=model_path)
    return encoder
