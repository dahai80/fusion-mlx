from __future__ import annotations

import logging
from typing import Any

import mlx_vlm

logger = logging.getLogger(__name__)

version = getattr(mlx_vlm, "version", "unknown")

generate = mlx_vlm.generate
stream_generate = mlx_vlm.stream_generate
batch_generate = mlx_vlm.batch_generate
prepare_inputs = mlx_vlm.prepare_inputs
apply_chat_template = mlx_vlm.apply_chat_template
process_image = mlx_vlm.process_image
get_message_json = mlx_vlm.get_message_json
models = mlx_vlm.models
utils = mlx_vlm.utils
tokenizer_utils = mlx_vlm.tokenizer_utils
prompt_utils = mlx_vlm.prompt_utils
turboquant = mlx_vlm.turboquant
vision_cache = mlx_vlm.vision_cache
apc = mlx_vlm.apc
speculative = mlx_vlm.speculative
convert = mlx_vlm.convert
trainer = mlx_vlm.trainer

GenerationResult = mlx_vlm.GenerationResult
BatchResponse = mlx_vlm.BatchResponse
BatchStats = mlx_vlm.BatchStats
PromptCacheState = mlx_vlm.PromptCacheState
VisionFeatureCache = mlx_vlm.VisionFeatureCache

__all__ = [
    "version",
    "load",
    "generate",
    "stream_generate",
    "batch_generate",
    "prepare_inputs",
    "apply_chat_template",
    "process_image",
    "get_message_json",
    "models",
    "utils",
    "tokenizer_utils",
    "prompt_utils",
    "turboquant",
    "vision_cache",
    "apc",
    "speculative",
    "convert",
    "trainer",
    "GenerationResult",
    "BatchResponse",
    "BatchStats",
    "PromptCacheState",
    "VisionFeatureCache",
    "set_fusion_model_settings",
]

_active_model_settings: Any | None = None


def set_fusion_model_settings(model_settings: Any | None = None) -> None:
    global _active_model_settings
    _active_model_settings = model_settings
    logger.debug(
        "fusion_mlx_vlm: active model_settings set (%s)",
        "enabled" if model_settings is not None else "None",
    )


def load(
    path_or_hf_repo: str,
    adapter_path: str | None = None,
    lazy: bool = False,
    revision: str | None = None,
    **kwargs: Any,
):
    result = mlx_vlm.load(
        path_or_hf_repo,
        adapter_path=adapter_path,
        lazy=lazy,
        revision=revision,
        **kwargs,
    )
    model, processor = result
    settings = _active_model_settings
    if settings is not None:
        try:
            from ..utils.model_loading import apply_post_load_transforms

            model = apply_post_load_transforms(model, settings)
            logger.debug("fusion_mlx_vlm.load: post-load transforms applied")
        except Exception as e:
            logger.warning("fusion_mlx_vlm.load: post-load transform failed: %s", e)
    return model, processor
