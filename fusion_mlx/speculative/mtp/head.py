# SPDX-License-Identifier: Apache-2.0
import inspect
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
        from mlx_lm.models.qwen3_5 import Attention
    except ImportError:
        try:
            from mlx_lm.models.gemma4_unified import Attention
        except ImportError:
            logger.warning("mtp/head: cannot import Attention from mlx-lm")
            _MTPDecoderLayer = type("_MTPDecoderLayer", (nn.Module,), {})
            _MTPModule = type("_MTPModule", (nn.Module,), {})
            _MTP_MODULE_BUILT = True
            return _MTPDecoderLayer, _MTPModule

    try:
        from mlx_lm.models.qwen3_5 import MLP as _MLP
        from mlx_lm.models.qwen3_5 import SparseMoeBlock
    except ImportError:
        try:
            from mlx_lm.models.gemma4_unified import MLP as _MLP
            from mlx_lm.models.gemma4_unified import SparseMoeBlock
        except ImportError:
            _MLP = None
            SparseMoeBlock = None

    if _MLP is None:
        logger.warning("mtp/head: cannot import MLP from mlx-lm")
        _MTPDecoderLayer = type("_MTPDecoderLayer", (nn.Module,), {})
        _MTPModule = type("_MTPModule", (nn.Module,), {})
        _MTP_MODULE_BUILT = True
        return _MTPDecoderLayer, _MTPModule

    def _make_mlp(mlp_args: Any) -> nn.Module:
        mlp_sig = inspect.signature(_MLP.__init__)
        if "args" in mlp_sig.parameters:
            return _MLP(args=mlp_args)
        return _MLP(dim=mlp_args.hidden_size, hidden_dim=mlp_args.intermediate_size)

    class _BuiltMTPDecoderLayer(nn.Module):
        def __init__(self, args: Any) -> None:
            super().__init__()
            self.hidden_size = args.hidden_size
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.self_attn = Attention(args=args)
            if SparseMoeBlock is not None and getattr(args, "num_experts", 0) > 0:
                self.mlp = SparseMoeBlock(args=args)
            else:
                self.mlp = _make_mlp(args)
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
            self.hidden_size = args.hidden_size
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [_BuiltMTPDecoderLayer(args) for _ in range(num_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.num_layers = num_layers

        def __call__(
            self,
            hidden_states: Any,
            next_token_ids: Any,
            embed_tokens: Any,
            mtp_cache: list | None = None,
        ) -> Any:
            h_normed = self.pre_fc_norm_hidden(hidden_states)
            next_embed = embed_tokens(next_token_ids)
            e_normed = self.pre_fc_norm_embedding(next_embed)
            combined = mx.concatenate([h_normed, e_normed], axis=-1)
            x = self.fc(combined)
            for i, layer in enumerate(self.layers):
                c = mtp_cache[i] if mtp_cache is not None else None
                x = layer(x, cache=c)
            return self.norm(x)

    import mlx.core as mx

    _MTPDecoderLayer = _BuiltMTPDecoderLayer
    _MTPModule = _BuiltMTPModule
    _MTP_MODULE_BUILT = True
    logger.info("mtp/head: built MTP module classes (%d layers)", num_layers)
    return _MTPDecoderLayer, _MTPModule
