# SPDX-License-Identifier: Apache-2.0
"""PydanticAI launch adapter — writes ~/.pydantic-ai/config.json."""

from __future__ import annotations

from pathlib import Path

from . import _common

_CONFIG_PATH = Path.home() / ".pydantic-ai" / "config.json"


def detect() -> bool:
    return _common.which("pydantic-ai") is not None or _CONFIG_PATH.exists()


def current_config_path() -> Path | None:
    return _CONFIG_PATH


def write_or_patch_config(
    server_url: str, model: str, api_key: str = "fusion-mlx", config_path=None
) -> Path:
    path = config_path or current_config_path()
    assert path is not None

    existing = {}
    if path.exists():
        _common.backup_existing(path)
        existing = _common.load_json_lenient(path)

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    existing["openai_base_url"] = base_url
    existing["openai_api_key"] = api_key
    if model:
        existing["default_model"] = f"openai:{model}"

    _common.atomic_write_json(path, existing)
    return path
