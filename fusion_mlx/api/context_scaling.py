import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_CONTEXT = 128000


def is_claude_code_request(headers: dict) -> bool:
    if os.environ.get("FUSION_MLX_CONTEXT_SCALING", "").strip() in ("1", "true", "yes"):
        return True
    user_agent = (headers.get("user-agent") or "").lower()
    if "claude-code" in user_agent:
        return True
    x_client = (headers.get("x-client") or "").lower()
    if "claude-code" in x_client:
        return True
    return False


def get_context_scaling_settings(global_settings: dict) -> tuple[bool, int]:
    cc = (
        global_settings.get("claude_code", {})
        if isinstance(global_settings, dict)
        else {}
    )
    enabled = bool(cc.get("context_scaling_enabled", False))
    target = cc.get("target_context_size", _DEFAULT_TARGET_CONTEXT)
    try:
        target = int(target)
    except (TypeError, ValueError):
        target = _DEFAULT_TARGET_CONTEXT
    if target <= 0:
        target = _DEFAULT_TARGET_CONTEXT
    return enabled, target


def compute_scale_factor(model_context: int, target_context: int) -> float | None:
    if model_context <= 0 or target_context <= 0:
        return None
    if model_context >= target_context:
        return None
    factor = model_context / target_context
    logger.debug(
        "Context scaling: model_context=%d target=%d factor=%.4f",
        model_context,
        target_context,
        factor,
    )
    return factor


def scale_usage(usage: dict, factor: float) -> dict:
    if factor <= 0 or factor >= 1:
        return usage
    scaled = dict(usage)
    for key in (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "prompt_tokens",
    ):
        val = scaled.get(key)
        if isinstance(val, (int, float)) and val > 0:
            scaled[key] = int(val * factor)
    return scaled
