# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of LTX-2.5 text encoder: Gemma4-12b + projection head.
# LTX-2/2.3 use Gemma3 (mlx-vlm gemma3); LTX-2.5 upgrades to Gemma4-12b. The
# projection head maps Gemma4 hidden_size -> caption_channels=3840 and lives in
# the same single-file checkpoint (gemma4-12b-with-proj-ltx-2.5-bf16.safetensors)
# under key prefixes "projection.*". mlx-vlm 0.5.0 supports gemma4 (verified),
# so no upstream issue is needed.
#
# Gemma4TextModel.__call__ already supports skip_final_norm + hidden_sink +
# capture_layer_ids natively, so we reuse the upstream forward rather than
# hand-rolling a layer loop (unlike the MiniMax-H3 Qwen3VL wrapper).
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# LTX-2.5 caption projection target dim (AR doc §2.1).
LTX2_5_CAPTION_CHANNELS = 3840
# Gemma4 projection weight key prefix in the with-proj checkpoint.
_PROJ_KEY_PREFIX = "projection."


class LTX2_5TextProjection(nn.Module):
    # PixArtAlpha-style 2-layer MLP projection: gemma4 hidden -> caption 3840.
    def __init__(
        self,
        in_features: int,
        out_features: int = LTX2_5_CAPTION_CHANNELS,
        hidden_size: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        hidden = hidden_size or out_features
        self.linear1 = nn.Linear(in_features, hidden, bias=bias)
        self.act = nn.GELU(approx="tanh")
        self.linear2 = nn.Linear(hidden, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class LTX2_5TextEncoder(nn.Module):
    # Wraps a Gemma4 language model + optional projection head. The language
    # model is injected (loaded elsewhere from the with-proj checkpoint) so the
    # encoder stays weight-agnostic and unit-testable without 12B weights.
    def __init__(
        self,
        language_model: nn.Module,
        projection: nn.Module | None = None,
        *,
        caption_channels: int = LTX2_5_CAPTION_CHANNELS,
    ):
        super().__init__()
        self.language_model = language_model
        self.projection = projection
        self.caption_channels = caption_channels

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        *,
        skip_final_norm: bool = True,
    ) -> mx.array:
        # Run Gemma4 forward; skip_final_norm=True returns the last decoder
        # layer's pre-norm hidden state (matches LTX caption convention).
        kwargs: dict = {"skip_final_norm": skip_final_norm}
        if attention_mask is not None:
            kwargs["mask"] = attention_mask
        hidden = self.language_model(input_ids, **kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if self.projection is not None:
            hidden = self.projection(hidden)
        logger.debug(
            "LTX2_5TextEncoder.encode: hidden shape=%s projected=%s",
            hidden.shape,
            self.projection is not None,
        )
        return hidden

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        return self.encode(input_ids, attention_mask)


def _split_projection_weights(weights: dict) -> tuple[dict, dict]:
    # Separate projection.* keys from Gemma4 language-model keys.
    proj: dict = {}
    lang: dict = {}
    for k, v in weights.items():
        if k.startswith(_PROJ_KEY_PREFIX):
            proj[k[len(_PROJ_KEY_PREFIX) :]] = v
        else:
            lang[k] = v
    logger.info(
        "LTX2_5TextEncoder weights split: lang=%d proj=%d", len(lang), len(proj)
    )
    return lang, proj


def load_text_encoder(
    weights_path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> LTX2_5TextEncoder:
    # Load Gemma4-12b + projection from the single-file with-proj checkpoint.
    # Requires the Gemma4 TextConfig (text_config block of the checkpoint's
    # config.json). Raises FileNotFoundError if config/weights absent (fail
    # visible per Rule 12) — no silent zero-init.
    weights_path = Path(weights_path)
    config_path = (
        Path(config_path) if config_path else weights_path.parent / "config.json"
    )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"LTX-2.5 text encoder weights not found: {weights_path}"
        )
    if not config_path.exists():
        raise FileNotFoundError(f"LTX-2.5 text encoder config not found: {config_path}")

    from mlx_vlm.models.gemma4.config import TextConfig
    from mlx_vlm.models.gemma4.language import Gemma4TextModel

    with open(config_path) as f:
        config_dict = json.load(f)
    text_config = TextConfig.from_dict(config_dict["text_config"])

    logger.info(
        "load_text_encoder: hidden_size=%d layers=%d caption=%d",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        LTX2_5_CAPTION_CHANNELS,
    )

    weights = mx.load(str(weights_path))
    lang_weights, proj_weights = _split_projection_weights(weights)

    language_model = Gemma4TextModel(text_config)
    if hasattr(language_model, "sanitize"):
        lang_weights = language_model.sanitize(weights=lang_weights)
    language_model.load_weights(list(lang_weights.items()), strict=False)

    projection = None
    if proj_weights:
        in_features = text_config.hidden_size
        projection = LTX2_5TextProjection(in_features=in_features)
        projection.load_weights(list(proj_weights.items()), strict=False)
        logger.info(
            "LTX2_5TextEncoder: loaded projection head (%d keys)", len(proj_weights)
        )
    else:
        logger.warning("LTX2_5TextEncoder: no projection keys found in checkpoint")

    return LTX2_5TextEncoder(language_model, projection)
