from __future__ import annotations

import logging
from typing import Any

import mlx_lm

logger = logging.getLogger(__name__)

__version__ = getattr(mlx_lm, "__version__", "unknown")

convert = mlx_lm.convert
generate = mlx_lm.generate
stream_generate = mlx_lm.stream_generate
batch_generate = mlx_lm.batch_generate
tokenizer_utils = mlx_lm.tokenizer_utils
sample_utils = mlx_lm.sample_utils
utils = mlx_lm.utils
models = mlx_lm.models

__all__ = [
    "__version__",
    "load",
    "generate",
    "stream_generate",
    "batch_generate",
    "convert",
    "tokenizer_utils",
    "sample_utils",
    "utils",
    "models",
    "set_fusion_model_settings",
]

_active_model_settings: Any | None = None


def set_fusion_model_settings(model_settings: Any | None = None) -> None:
    global _active_model_settings
    _active_model_settings = model_settings
    logger.debug(
        "fusion_mlx_lm: active model_settings set (%s)",
        "enabled" if model_settings is not None else "None",
    )


def load(
    path_or_hf_repo: str,
    tokenizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    adapter_path: str | None = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: str | None = None,
):
    result = mlx_lm.load(
        path_or_hf_repo,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        adapter_path=adapter_path,
        lazy=lazy,
        return_config=return_config,
        revision=revision,
    )
    if return_config:
        model, tokenizer, config = result
    else:
        model, tokenizer = result
    settings = _active_model_settings
    if settings is not None:
        try:
            from ..utils.model_loading import apply_post_load_transforms

            model = apply_post_load_transforms(model, settings)
            logger.debug("fusion_mlx_lm.load: post-load transforms applied")
        except Exception as e:
            logger.warning("fusion_mlx_lm.load: post-load transform failed: %s", e)
    if return_config:
        return model, tokenizer, config
    return model, tokenizer
