# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MAX_TOOL_SCHEMA_DEPTH_ENV_PRIMARY = "FUSION_MLX_MAX_TOOL_SCHEMA_DEPTH"
MAX_TOOL_SCHEMA_DEPTH_ENV_LEGACY = "RAPID_MLX_MAX_TOOL_SCHEMA_DEPTH"
MAX_BODY_DEPTH_ENV_PRIMARY = "FUSION_MLX_MAX_BODY_DEPTH"
MAX_BODY_DEPTH_ENV_LEGACY = "RAPID_MLX_MAX_BODY_DEPTH"
DEFAULT_MAX_TOOL_SCHEMA_DEPTH = 64
DEFAULT_MAX_BODY_DEPTH = 64

_legacy_warned: set[str] = set()


def _resolve_env_int(primary: str, legacy: str, default: int) -> int:
    raw = os.environ.get(primary, "").strip()
    if raw:
        return _parse_int_or(raw, default)
    raw = os.environ.get(legacy, "").strip()
    if raw:
        if legacy not in _legacy_warned:
            _legacy_warned.add(legacy)
            logger.warning(
                "env var %s is deprecated, use %s instead", legacy, primary
            )
        return _parse_int_or(raw, default)
    return default


def _parse_int_or(raw: str, default: int) -> int:
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_max_tool_schema_depth() -> int:
    return _resolve_env_int(
        MAX_TOOL_SCHEMA_DEPTH_ENV_PRIMARY,
        MAX_TOOL_SCHEMA_DEPTH_ENV_LEGACY,
        DEFAULT_MAX_TOOL_SCHEMA_DEPTH,
    )


def resolve_max_body_depth() -> int:
    return _resolve_env_int(
        MAX_BODY_DEPTH_ENV_PRIMARY,
        MAX_BODY_DEPTH_ENV_LEGACY,
        DEFAULT_MAX_BODY_DEPTH,
    )


def json_nesting_depth_exceeds(obj: Any, max_depth: int) -> bool:
    if max_depth <= 0:
        return False
    if not isinstance(obj, (dict, list, tuple)):
        return False
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            return True
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, depth + 1))
        elif isinstance(node, (list, tuple)):
            for v in node:
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, depth + 1))
    return False
