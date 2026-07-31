# SPDX-License-Identifier: Apache-2.0
"""OpenCode launch adapter.

Patches ``~/.config/opencode/opencode.json`` so OpenCode routes to
the local fusion-mlx server.  OpenCode reads a JSON config with a
``provider`` map (keyed by provider name) and a ``model`` top-level key.
"""

from __future__ import annotations

from pathlib import Path

from . import _common

_CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"


def detect() -> bool:
    return _common.which("opencode") is not None or _CONFIG_PATH.exists()


def current_config_path() -> Path | None:
    return _CONFIG_PATH


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "fusion-mlx",
    config_path: Path | None = None,
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

    existing.setdefault("provider", {})
    provider_config = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Fusion-MLX",
        "options": {
            "baseURL": base_url,
        },
    }
    if api_key:
        provider_config["options"]["apiKey"] = api_key
    if model:
        model_entry: dict = {
            "name": model,
            "modalities": {"input": ["text"], "output": ["text"]},
        }
        provider_config["models"] = {model: model_entry}
    existing["provider"]["fusion-mlx"] = provider_config

    if model:
        existing["model"] = f"fusion-mlx/{model}"

    _common.atomic_write_json(path, existing)
    return path
