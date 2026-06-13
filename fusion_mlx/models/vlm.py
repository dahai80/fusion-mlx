# SPDX-License-Identifier: Apache-2.0
"""VLM (Vision-Language Model) adapter for BatchGenerator integration.

Wraps an mlx-vlm model to present a standard model interface compatible
with mlx-lm's BatchGenerator. Handles vision embedding injection during
prefill, then becomes transparent for autoregressive decode.
"""

import logging
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class VLMModelAdapter(nn.Module):
    """Adapter wrapping a VLM's language_model for BatchGenerator compatibility.

    During prefill: substitutes pre-computed vision+text embeddings.
    During decode: passes token IDs directly (vision context lives in KV cache).
    """

    def __init__(self, vlm_model: nn.Module):
        super().__init__()
        self._vlm_model = vlm_model
        self._language_model = vlm_model.language_model
        self._uses_mrope = self._detect_mrope(vlm_model)

        # Pending vision embeddings (set before prefill, cleared after)
        self._pending_embeds: Optional[mx.array] = None
        self._pending_kwargs: Dict[str, Any] = {}
        self._embed_offset: int = 0

        # Batch mRoPE state
        self._batch_rope_deltas: Optional[mx.array] = None

    @property
    def layers(self):
        """Expose language model layers for cache creation."""
        lm = self._language_model
        for parent in (getattr(lm, "model", None), lm):
            if parent is None:
                continue
            for attr in ("layers", "blocks"):
                v = getattr(parent, attr, None)
                if v is not None:
                    return v
        raise AttributeError(f"{type(lm).__name__} has no .layers/.blocks")

    @property
    def model_type(self) -> str:
        if hasattr(self._vlm_model, "config") and hasattr(self._vlm_model.config, "model_type"):
            return self._vlm_model.config.model_type
        return "vlm"

    @property
    def config(self):
        return self._vlm_model.config

    @property
    def args(self):
        if hasattr(self._language_model, "args"):
            return self._language_model.args
        return self.config

    def make_cache(self) -> List[Any]:
        """Create KV cache using the language model's make_cache()."""
        if hasattr(self._language_model, "make_cache"):
            return self._language_model.make_cache()
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(len(self.layers))]

    def set_pending_embeddings(
        self,
        inputs_embeds: mx.array,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        start_offset: int = 0,
    ) -> None:
        """Register pre-computed embeddings for the next prefill."""
        self._pending_embeds = inputs_embeds
        self._pending_kwargs = extra_kwargs or {}
        self._embed_offset = start_offset

    def clear_pending_embeddings(self) -> None:
        self._pending_embeds = None
        self._pending_kwargs = {}
        self._embed_offset = 0

    @staticmethod
    def _detect_mrope(vlm_model) -> bool:
        config = getattr(vlm_model, "config", None)
        if not config:
            return False
        text_config = getattr(config, "text_config", None)
        if not text_config:
            return False
        rope_cfg = getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None)
        if not isinstance(rope_cfg, dict):
            return False
        return "mrope_section" in rope_cfg or rope_cfg.get("type") == "mrope"

    def clear_vlm_position_state(self) -> None:
        """Clear stale mRoPE state from previous VLM requests."""
        if hasattr(self._language_model, "_position_ids"):
            self._language_model._position_ids = None
        if hasattr(self._language_model, "_rope_deltas"):
            self._language_model._rope_deltas = None

    def set_batch_rope_deltas(self, deltas: mx.array) -> None:
        self._batch_rope_deltas = deltas

    def get_last_rope_deltas(self) -> float:
        rd = getattr(self._language_model, "_rope_deltas", None)
        if rd is None:
            return 0.0
        return float(rd.item()) if hasattr(rd, "item") else float(rd)

    @property
    def has_pending_embeddings(self) -> bool:
        return self._pending_embeds is not None

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> Any:
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        vlm_extra = kwargs.pop("vlm_extra_kwargs", None) or {}
        vlm_extra.pop("_captured_rope_deltas", None)

        if inputs_embeds is not None:
            result = self._language_model(input_ids, inputs_embeds=inputs_embeds, cache=cache, **vlm_extra, **kwargs)
        elif self._pending_embeds is not None:
            result = self._forward_with_embeddings(input_ids, cache, **kwargs)
        else:
            if self._uses_mrope and self._batch_rope_deltas is not None and cache is not None:
                offsets = None
                for c in cache:
                    if hasattr(c, "offset"):
                        offsets = c.offset
                        break
                B, L = input_ids.shape
                deltas = self._batch_rope_deltas
                if offsets is not None and isinstance(offsets, mx.array) and deltas.shape[-1] == B:
                    positions = offsets + deltas
                    position_ids = mx.broadcast_to(positions[None, :, None], (3, B, L))
                    result = self._language_model(input_ids, cache=cache, position_ids=position_ids, **kwargs)
                else:
                    result = self._language_model(input_ids, cache=cache, **kwargs)
            elif self._uses_mrope and cache is not None:
                offsets = None
                for c in cache:
                    if hasattr(c, "offset") and isinstance(c.offset, mx.array) and c.offset.ndim > 0:
                        offsets = c.offset
                        break
                if offsets is not None:
                    B, L = input_ids.shape
                    position_ids = mx.broadcast_to(offsets[None, :, None], (3, B, L))
                    result = self._language_model(input_ids, cache=cache, position_ids=position_ids, **kwargs)
                else:
                    result = self._language_model(input_ids, cache=cache, **kwargs)
            else:
                if hasattr(self._vlm_model, "_set_position_state"):
                    self._vlm_model._set_position_state(input_ids)
                result = self._language_model(input_ids, cache=cache, **kwargs)

        return result.logits if hasattr(result, "logits") else result

    def _forward_with_embeddings(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> Any:
        chunk_len = input_ids.shape[1]
        total_len = self._pending_embeds.shape[1]
        end_offset = min(self._embed_offset + chunk_len, total_len)
        chunk_embeds = self._pending_embeds[:, self._embed_offset:end_offset, :]

        result = self._language_model(input_ids, inputs_embeds=chunk_embeds, cache=cache, **self._pending_kwargs, **kwargs)
        self._embed_offset = end_offset
        if self._embed_offset >= total_len:
            self.clear_pending_embeddings()
        return result

    def get_input_embeddings(
        self, input_ids: mx.array, pixel_values: Optional[mx.array] = None, **kwargs
    ) -> Any:
        """Compute vision+text merged embeddings via the VLM model."""
        return self._vlm_model.get_input_embeddings(input_ids, pixel_values, **kwargs)
