import logging
from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

VIDEO_ADAPTERS: dict[str, type] = {}


def register_adapter(name: str):
    def decorator(cls):
        VIDEO_ADAPTERS[name] = cls
        logger.debug("Registered video adapter: %s -> %s", name, cls.__name__)
        return cls

    return decorator


class VideoAdapter(ABC):
    name: str = ""

    def modify_context(self, context: mx.array, **kw: Any) -> mx.array:
        return context

    def modify_denoise_step(
        self,
        dit: Any,
        latents: mx.array,
        t: mx.array,
        context: mx.array,
        **kw: Any,
    ) -> mx.array:
        return latents

    @abstractmethod
    def load(self, model_path: str | None = None) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...


def create_adapter(
    adapter_type: str,
    *,
    scale: float = 1.0,
    image: str | None = None,
    config: dict | None = None,
) -> VideoAdapter | None:
    cls = VIDEO_ADAPTERS.get(adapter_type)
    if cls is None:
        logger.warning(
            "Unknown adapter type: %s (available: %s)",
            adapter_type,
            list(VIDEO_ADAPTERS),
        )
        return None
    adapter = cls(scale=scale, image=image, config=config or {})
    logger.info("Created adapter: %s scale=%.2f", adapter_type, scale)
    return adapter


# Eager import to trigger @register_adapter decorators
from fusion_mlx.video.adapters import animatediff as _animatediff  # noqa: E402, F401
from fusion_mlx.video.adapters import controlnet as _controlnet  # noqa: E402, F401
from fusion_mlx.video.adapters import ip_adapter as _ip_adapter  # noqa: E402, F401
