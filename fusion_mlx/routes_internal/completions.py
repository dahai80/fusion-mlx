"""Compatibility shim: re-exports router + restores deleted helpers."""

import inspect
import logging
import time

from ..api.openai_routes import router  # noqa: F401
from ..service.helpers import (
    enforce_context_length_for_prompt,  # noqa: F401
    get_engine,  # noqa: F401
)

logger = logging.getLogger(__name__)


def _engine_supports_completion_logprobs(engine) -> bool:
    structural_support = getattr(engine, "tokenizer", None) is not None and callable(
        getattr(engine, "stream_generate", None)
    )
    capability = getattr(engine, "supports_completion_logprobs", None)
    if capability is not None:
        if callable(capability):
            try:
                value = capability()
            except Exception as exc:
                logger.debug(
                    "supports_completion_logprobs capability probe failed: %s",
                    exc,
                )
                return False
            if inspect.isawaitable(value):
                close = getattr(value, "close", None)
                if callable(close):
                    close()
                return False
            return value if isinstance(value, bool) else False
        return capability if isinstance(capability, bool) else False
    return structural_support
