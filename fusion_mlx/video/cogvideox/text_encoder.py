# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def load_t5_encoder(t5_path: str | Path, config) -> nn.Module:
    from fusion_mlx.video.wan2.utils import load_t5_encoder as _wan2_load_t5

    return _wan2_load_t5(Path(t5_path), config)


def encode_text(t5_encoder, tokenizer, prompt: str, max_length: int = 226) -> mx.array:
    from fusion_mlx.video.wan2.utils import encode_text as _wan2_encode

    return _wan2_encode(t5_encoder, tokenizer, prompt, max_length)
