# SPDX-License-Identifier: Apache-2.0
"""DFly drafter — MLX implementation of the AngelSpec DFly draft model.

DFly = DFlash + HiddenStatesCorrection.  The inference path:
  1. Target model forward -> extract target_layer_ids hidden states (5 layers)
  2. DFly context FC: concat(target_hiddens) -> FC -> RMSNorm -> layer context
  3. Hidden correction: h' = h + SwiGLU(norm(h) :: norm(prev_embed))
  4. Per-position logit via target lm_head (weight tying)
  5. Block-diffusion verify: batch K draft positions -> accept/reject

Weight loading: mx.load(safetensors) + remap keys from PyTorch naming
to MLX convention.

Released drafter models:
  - AngelSlim/Hy3-DFly-Block8 (no-think, 5 DFlash decoder layers)
  - AngelSlim/Hy3-DFly-Block8-Think-High (high-think mode)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LAYER_IDS = [1, 20, 39, 58, 77]
DEFAULT_NUM_DRAFT_LAYERS = 5
DEFAULT_HIDDEN_SIZE = 4096


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rrms).astype(dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HiddenStatesCorrection(nn.Module):
    """h'_{t+i} = h_{t+i} + SwiGLU(norm(h_{t+i}) :: norm(e_{t+i-1}))

    Zero-initialized down_proj so correction starts as identity.
    """

    def __init__(self, hidden_size: int, intermediate_size: int | None = None):
        super().__init__()
        inter = intermediate_size or hidden_size * 2
        self.input_norm = RMSNorm(hidden_size)
        self.embed_norm = RMSNorm(hidden_size)
        self.correction = SwiGLU(hidden_size * 2, inter, hidden_size)
        nn.init.zeros_(self.correction.down_proj.weight)

    def __call__(self, hidden: mx.array, prev_embed: mx.array) -> mx.array:
        concat_input = mx.concatenate(
            [self.input_norm(hidden), self.embed_norm(prev_embed)], axis=-1
        )
        return hidden + self.correction(concat_input)


class DFlyContextFC(nn.Module):
    def __init__(
        self,
        num_target_layers: int,
        hidden_size: int,
        output_size: int | None = None,
    ):
        super().__init__()
        out = output_size or hidden_size
        self.fc = nn.Linear(num_target_layers * hidden_size, out, bias=False)
        self.norm = RMSNorm(out)

    def __call__(self, target_hiddens: list[mx.array]) -> mx.array:
        concat = mx.concatenate(target_hiddens, axis=-1)
        return self.norm(self.fc(concat))


@dataclass
class DFlyConfig:
    num_draft_layers: int = DEFAULT_NUM_DRAFT_LAYERS
    target_layer_ids: list[int] = field(default_factory=lambda: list(DEFAULT_TARGET_LAYER_IDS))
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    enable_hidden_correction: bool = True
    hidden_correction_intermediate_size: int | None = None
    vocab_size: int = 152064
    rope_theta: float = 1000000.0


class DFlyDraftModel(nn.Module):
    """DFly draft model: DFlash decoder layers + context FC + hidden correction."""

    def __init__(self, config: DFlyConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.context_fc = DFlyContextFC(
            num_target_layers=len(config.target_layer_ids),
            hidden_size=config.hidden_size,
        )
        self.layers = [
            nn.TransformerDecoderLayer(
                dims=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_draft_layers)
        ]
        self.norm = RMSNorm(config.hidden_size)
        if config.enable_hidden_correction:
            self.hidden_correction = HiddenStatesCorrection(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_correction_intermediate_size,
            )
        else:
            self.hidden_correction = None
        self.lm_head_weight: mx.array | None = None

    def set_lm_head_weight(self, weight: mx.array) -> None:
        self.lm_head_weight = weight

    def __call__(
        self,
        input_ids: mx.array,
        target_hiddens: list[mx.array],
        prev_embed: mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)
        ctx = self.context_fc(target_hiddens)
        h = h + ctx

        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            h = layer(h, cache=c)

        h = self.norm(h)

        if self.hidden_correction is not None and prev_embed is not None:
            h = self.hidden_correction(h, prev_embed)

        if self.lm_head_weight is not None:
            logits = h @ self.lm_head_weight.T
            return logits

        return h


class DFlyDrafter:
    """High-level DFly drafter interface for speculative decoding."""

    def __init__(
        self,
        model_path: str,
        target_model: Any | None = None,
        config: DFlyConfig | None = None,
        block_size: int = 16,
    ):
        self.model_path = model_path
        self.target_model = target_model
        self.config = config or DFlyConfig()
        self.block_size = block_size
        self._draft_model: DFlyDraftModel | None = None
        self._cache: list[Any] | None = None
        self._loaded = False
        logger.info(
            "DFlyDrafter: path=%s block_size=%d target_layers=%s",
            model_path,
            block_size,
            self.config.target_layer_ids,
        )

    def load(self) -> None:
        if self._loaded:
            return
        self._draft_model = DFlyDraftModel(self.config)
        weights = self._load_weights(self.model_path)
        if weights:
            self._draft_model.load_weights(list(weights.items()))
        if self.target_model is not None:
            embed = getattr(self.target_model, "embed_tokens", None)
            if embed is not None:
                self._draft_model.set_lm_head_weight(embed.weight)
        self._loaded = True
        logger.info("DFlyDrafter: loaded from %s", self.model_path)

    def _load_weights(self, path: str) -> dict[str, mx.array]:
        resolved = Path(os.path.expanduser(path))
        if not resolved.exists():
            logger.warning("DFlyDrafter: path %s does not exist, using random init", path)
            return {}
        weights: dict[str, mx.array] = {}
        for safetensor_file in sorted(resolved.glob("*.safetensors")):
            loaded = mx.load(str(safetensor_file), stream=mx.cpu)
            for k, v in loaded.items():
                mapped = self._remap_key(k)
                weights[mapped] = v
        logger.info("DFlyDrafter: loaded %d weight tensors from %s", len(weights), path)
        return weights

    @staticmethod
    def _remap_key(key: str) -> str:
        if key.startswith("model."):
            return key[len("model."):]
        return key

    def draft(
        self,
        input_ids: mx.array,
        target_hiddens: list[mx.array],
        prev_embed: mx.array | None = None,
    ) -> mx.array:
        if not self._loaded:
            self.load()
        assert self._draft_model is not None
        logits = self._draft_model(
            input_ids,
            target_hiddens,
            prev_embed=prev_embed,
            cache=self._cache,
        )
        return logits

    def reset_cache(self) -> None:
        self._cache = None

    @property
    def target_layer_ids(self) -> list[int]:
        return self.config.target_layer_ids
