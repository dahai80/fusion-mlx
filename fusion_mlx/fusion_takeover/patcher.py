from __future__ import annotations

import logging
from typing import Any

import mlx.nn as nn

from .config import FusionConfig

logger = logging.getLogger(__name__)

_TAKEOVER_ATTR = "_fusion_takeover_applied"
_FUSED_DECODE_MODEL_FAMILIES = (
    "llama",
    "qwen2",
    "qwen3",
    "gemma2",
    "gemma3",
    "mistral3",
    "mistral",
)

try:
    from mlx.nn.layers.quantized import QuantizedLinear
except Exception:  # pragma: no cover
    QuantizedLinear = None


def _is_linear_like(val) -> bool:
    if isinstance(val, nn.Linear):
        return True
    if QuantizedLinear is not None and isinstance(val, QuantizedLinear):
        return True
    return False


def _detect_model_type(model: nn.Module) -> str | None:
    model_type = getattr(model, "model_type", None)
    if model_type:
        return str(model_type)
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type:
            return str(model_type)
    cls_name = type(model).__name__.lower()
    for candidate in ("qwen2", "qwen3", "llama", "phi", "mistral", "gemma"):
        if candidate in cls_name:
            return candidate
    return None


def _iter_linear(parent: nn.Module, prefix: str = ""):
    for key in list(parent.keys()):
        val = parent[key]
        if (
            val is None
            or isinstance(val, (int, float, str, bool, tuple))
            or type(val) is dict
        ):
            continue
        name = f"{prefix}.{key}" if prefix else key
        if _is_linear_like(val):
            yield (parent, key, name, val, None)
        elif isinstance(val, nn.Module):
            yield from _iter_linear(val, name)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                item_name = f"{name}.{i}"
                if _is_linear_like(item):
                    yield (val, i, item_name, item, "list")
                elif isinstance(item, nn.Module):
                    yield from _iter_linear(item, item_name)


def _resolve_sliding_softcap(layer, attn, model) -> tuple[int, float]:
    is_sliding = bool(getattr(attn, "is_sliding", False))
    use_sliding = bool(getattr(layer, "use_sliding", False))
    softcap = float(getattr(attn, "attn_logit_softcapping", 0.0) or 0.0)
    text_model = getattr(model, "model", None)
    window = (
        int(getattr(model, "window_size", 0) or 0)
        or int(getattr(text_model, "window_size", 0) or 0)
        or int(getattr(model, "sliding_window", 0) or 0)
        or int(getattr(text_model, "sliding_window", 0) or 0)
        or 0
    )
    if (is_sliding or use_sliding) and window > 0:
        resolved_sw = window
    else:
        resolved_sw = 0
    logger.debug(
        "resolve sliding: is_sliding=%s use_sliding=%s window=%d softcap=%s -> sw=%d",
        is_sliding,
        use_sliding,
        window,
        softcap,
        resolved_sw,
    )
    return resolved_sw, softcap


def _wrap_attention(layer, attn, model):
    from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache

    base_call = type(attn).__call__
    has_qnorm = hasattr(attn, "q_norm")
    sliding_window, softcap = _resolve_sliding_softcap(layer, attn, model)
    logger.debug(
        "paged_kv sliding layer window=%d softcap=%s (memory-cap follow-up)",
        sliding_window,
        softcap,
    )

    def fused_call(self, x, mask=None, cache=None):
        B, L, D = x.shape
        if not (
            isinstance(cache, FusionPagedKVCache)
            and cache.fused_decode_available(num_new=L)
        ):
            return base_call(self, x, mask=mask, cache=cache)
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        if has_qnorm:
            queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
                0, 2, 1, 3
            )
            keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
                0, 2, 1, 3
            )
        else:
            queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
        head_dim = queries.shape[-1]
        output = cache.fused_decode_attention(
            queries,
            self.scale,
            self.n_heads,
            head_dim,
            sliding_window=sliding_window,
            softcap=softcap,
        )
        logger.info(
            "paged_kv fused decode attention path taken offset=%d sw=%d sc=%s",
            cache.offset,
            sliding_window,
            softcap,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    wrapped_cls = type(
        type(attn).__name__ + "FusedDecode",
        (type(attn),),
        {"__call__": fused_call},
    )
    attn.__class__ = wrapped_cls
    logger.debug(
        "paged_kv fused decode wrap installed on attn=%r sw=%d sc=%s",
        attn,
        sliding_window,
        softcap,
    )


def _install_fused_decode(model):
    for layer in getattr(model, "layers", []) or []:
        attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attention", None)
            or getattr(layer, "attn", None)
        )
        if attn is None:
            continue
        _wrap_attention(layer, attn, model)


class FusionModulePatcher:
    @staticmethod
    def patch_model(model: nn.Module, config: FusionConfig) -> nn.Module:
        if getattr(model, _TAKEOVER_ATTR, False):
            logger.debug("fusion takeover already applied, skipping")
            return model
        model_type = _detect_model_type(model)
        if not config.is_supported_model_type(model_type):
            logger.info(
                "fusion takeover: model_type=%s not in target set %s, passthrough",
                model_type,
                config.target_model_types,
            )
            return model
        logger.info(
            "fusion takeover patching model_type=%s quant=%s paged_kv=%s",
            model_type,
            config.quant,
            config.paged_kv_enabled,
        )
        linear_count = 0
        for parent, key, name, module, container_kind in _iter_linear(model):
            module._fusion_quant = config.quant
            linear_count += 1
        if config.paged_kv_enabled:
            try:
                from ..custom_kernels.fusion_paged_kv import install_paged_kv

                install_paged_kv(model, config)
                logger.info(
                    "fusion takeover: paged_kv installed on model_type=%s", model_type
                )
            except Exception as e:
                logger.warning(
                    "fusion takeover: paged_kv install failed (%s), passthrough", e
                )
        if config.fused_decode_enabled and model_type in _FUSED_DECODE_MODEL_FAMILIES:
            _install_fused_decode(model)
            logger.info(
                "fusion takeover: fused decode wrap installed on model_type=%s",
                model_type,
            )
        try:
            setattr(model, _TAKEOVER_ATTR, True)
        except Exception:
            pass
        logger.info(
            "fusion takeover complete: model_type=%s %d linear layers tagged",
            model_type,
            linear_count,
        )
        return model


def apply_fusion_takeover(model: Any, model_settings: Any | None = None) -> Any:
    if model_settings is None:
        return model
    config = FusionConfig.from_model_settings(model_settings)
    if not config.enabled:
        return model
    if not isinstance(model, nn.Module):
        logger.debug(
            "fusion takeover: model is not nn.Module (%s), passthrough",
            type(model).__name__,
        )
        return model
    try:
        return FusionModulePatcher.patch_model(model, config)
    except Exception as e:
        logger.warning("fusion takeover failed (%s), returning model unchanged", e)
        return model
