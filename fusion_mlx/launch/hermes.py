# SPDX-License-Identifier: Apache-2.0
"""Hermes Agent launch adapter.

Patches ``~/.hermes/config.yaml`` so Hermes routes to the local
fusion-mlx server.  Hermes reads a YAML config with a ``providers``
map and ``model`` top-level key.
"""

from __future__ import annotations

from pathlib import Path

from . import _common

_CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"


def detect() -> bool:
    return _common.which("hermes") is not None or _CONFIG_PATH.exists()


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

    import yaml

    existing: dict = {}
    if path.exists():
        _common.backup_existing(path)
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    providers = existing.setdefault("providers", {})
    if not isinstance(providers, dict):
        providers = {}
        existing["providers"] = providers

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    provider_config = providers.get("fusion-mlx", {})
    if not isinstance(provider_config, dict):
        provider_config = {}
    provider_config.update(
        {
            "name": "Fusion-MLX",
            "base_url": base_url,
            "api_key": api_key,
            "api_mode": "chat_completions",
        }
    )
    if model:
        provider_config["default_model"] = model
    providers["fusion-mlx"] = provider_config

    model_config = existing.get("model", {})
    if not isinstance(model_config, dict):
        model_config = {}
    for stale_key in ("base_url", "api_key", "api", "api_mode", "transport"):
        model_config.pop(stale_key, None)
    model_config["provider"] = "fusion-mlx"
    if model:
        model_config["default"] = model
    existing["model"] = model_config

    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True)
    path.write_text(yaml_content.rstrip() + "\n", encoding="utf-8")
    return path
