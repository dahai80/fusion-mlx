from __future__ import annotations

import logging
from typing import Any

import mlx.nn as nn

from .config import FusionConfig

logger = logging.getLogger(__name__)

_TAKEOVER_ATTR = "_fusion_takeover_applied"

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
