# SPDX-License-Identifier: Apache-2.0
# ACE-Step config (MLX port, issue #988). Mirrors transformers AceStepConfig
# but plain dataclass — no PretrainedConfig base (fusion-mlx loads weights
# directly via safetensors, not HF AutoConfig).
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_layer_types(n: int) -> list[str]:
    return ["sliding_attention" if (i + 1) % 2 else "full_attention" for i in range(n)]


@dataclass
class AceStepConfig:
    vocab_size: int = 64003
    fsq_dim: int = 2048
    fsq_input_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    fsq_input_num_quantizers: int = 1
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    use_cache: bool = True
    rope_theta: float = 1000000.0
    use_sliding_window: bool = True
    sliding_window: int = 128
    layer_types: list[str] = field(default_factory=lambda: _default_layer_types(24))
    attention_dropout: float = 0.0
    num_lyric_encoder_hidden_layers: int = 8
    audio_acoustic_hidden_dim: int = 64
    pool_window_size: int = 5
    text_hidden_dim: int = 1024
    in_channels: int = 192
    data_proportion: float = 0.5
    timestep_mu: float = -0.4
    timestep_sigma: float = 1.0
    timbre_hidden_dim: int = 64
    num_timbre_encoder_hidden_layers: int = 4
    timbre_fix_frame: int = 750
    patch_size: int = 2
    num_attention_pooler_hidden_layers: int = 2
    num_audio_decoder_hidden_layers: int = 24
    model_version: str = "turbo"
    is_turbo: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> AceStepConfig:
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if raw.get("layer_types") and isinstance(raw["layer_types"], list):
            kwargs["layer_types"] = raw["layer_types"]
        cfg = cls(**kwargs)
        logger.info(
            "AceStepConfig loaded: hidden=%d layers=%d heads=%d kv=%d hd=%d "
            "sliding=%s(%d) fsq_dim=%d levels=%s patch=%d pool=%d",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.use_sliding_window,
            cfg.sliding_window,
            cfg.fsq_dim,
            cfg.fsq_input_levels,
            cfg.patch_size,
            cfg.pool_window_size,
        )
        return cfg
