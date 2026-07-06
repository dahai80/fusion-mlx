# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any

import mlx.nn as nn

logger = logging.getLogger(__name__)

_MTP_MODULE_BUILT = False
_MTPDecoderLayer: type | None = None
_MTPModule: type | None = None


def build_mtp_module(args: Any, num_layers: int) -> tuple[type, type]:
    global _MTP_MODULE_BUILT, _MTPDecoderLayer, _MTPModule
    if _MTP_MODULE_BUILT and _MTPDecoderLayer is not None and _MTPModule is not None:
        return _MTPDecoderLayer, _MTPModule
    try:
        from mlx_lm.models.qwen3_5 import Attention, MLP
    except ImportError:
        try:
            from mlx_lm.models.gemma4_unified import Attention, MLP
        except ImportError:
            logger.warning("mtp/head: cannot import Attention/MLP from mlx-lm")
            _MTPDecoderLayer = type("_MTPDecoderLayer", (nn.Module,), {})
            _MTPModule = type("_MTPModule", (nn.Module,), {})
            _MTP_MODULE_BUILT = True
            return _MTPDecoderLayer, _MTPModule
    try:
        from mlx_lm.models.qwen3_5 import SparseMoeBlock
    except ImportError:
        SparseMoeBlock = None
    class _BuiltMTPDecoderLayer(nn.Module):
        def __init__(self, args: Any) -> None:
            super().__init__()
            self.hidden_size = args.hidden_size
            self.input_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.self_attn = Attention(args=args)
            if (
                SparseMoeBlock is not None
                and getattr(args, "num_experts", 0) > 0
            ):
                self.mlp = SparseMoeBlock(args=args)
            else:
                self.mlp = MLP(args=args)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.args = args

        def __call__(self, x: Any, cache: Any = None) -> Any:
            r = self.self_attn(self.input_layernorm(x), cache=cache)
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            return h + r

    class _BuiltMTPModule(nn.Module):
        def __init__(self, args: Any, num_layers: int) -> None:
            super().__init__()
            self.embed_tokens = nn.Identity()
            self.en_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.layers = [_BuiltMTPDecoderLayer(args) for _ in range(num_layers)]
            for i, layer in enumerate(self.layers):
                setattr(self, f"layer_{i}", layer)
            self.num_layers = num_layers

        def __call__(self, *a: Any, **kw: Any) -> Any:
            raise NotImplementedError("use mtp_forward on the main model")

    _MTPDecoderLayer = _BuiltMTPDecoderLayer
    _MTPModule = _BuiltMTPModule
    _MTP_MODULE_BUILT = True
    logger.info("mtp/head: built MTP module classes (%d layers)", num_layers)
    return _MTPDecoderLayer, _MTPModule
