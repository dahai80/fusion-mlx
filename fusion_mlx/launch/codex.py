# SPDX-License-Identifier: Apache-2.0
"""Codex (OpenAI Codex CLI) launch adapter.

Patches ``~/.codex/config.toml`` so Codex routes to the local
fusion-mlx server.  Codex reads a TOML config with top-level
``model`` / ``model_provider`` keys plus a
``[model_providers.<name>]`` section for custom endpoints.
"""

from __future__ import annotations

from pathlib import Path

from . import _common

_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def detect() -> bool:
    return _common.which("codex") is not None or _CONFIG_PATH.exists()


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

    existing_content = ""
    if path.exists():
        _common.backup_existing(path)
        existing_content = path.read_text(encoding="utf-8")

    lines = existing_content.splitlines()
    new_lines = []
    in_any_section = False
    in_fusion_section = False

    top_level_overrides = {
        "model": f'"{model}"',
        "model_provider": '"fusion-mlx"',
    }

    managed_keys = {"model_reasoning_effort"} - set(top_level_overrides.keys())
    seen_keys = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_any_section = True
            in_fusion_section = stripped == "[model_providers.fusion-mlx]"

        if not in_any_section and "=" in stripped:
            key = stripped.split("=")[0].strip()
            if key in top_level_overrides:
                new_lines.append(f"{key} = {top_level_overrides[key]}")
                seen_keys.add(key)
                continue
            if key in managed_keys:
                continue

        if in_fusion_section:
            continue

        new_lines.append(line)

    for key, val in top_level_overrides.items():
        if key not in seen_keys:
            new_lines.insert(0, f"{key} = {val}")

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    new_lines.append("\n[model_providers.fusion-mlx]")
    new_lines.append('name = "Fusion-MLX"')
    new_lines.append(f'base_url = "{base_url}"')
    new_lines.append('env_key = "FUSION_API_KEY"')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return path
