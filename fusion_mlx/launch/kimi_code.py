# SPDX-License-Identifier: Apache-2.0
"""Kimi Code launch adapter — env-var based, no config file to patch."""

from __future__ import annotations

from . import _common


def detect() -> bool:
    return _common.which("kimi-code") is not None


def current_config_path() -> None:
    return None


def write_or_patch_config(
    server_url: str, model: str, api_key: str = "fusion-mlx", config_path=None
) -> None:
    print("Kimi Code uses environment variables — no config file to patch.")
    print(f"  Set OPENAI_BASE_URL={server_url.rstrip('/')}/v1")
    print(f"  Set OPENAI_API_KEY={api_key}")
    print(f"  Set KIMI_CODE_MODEL={model}")
