# SPDX-License-Identifier: Apache-2.0
"""``fusion-mlx launch <client>`` — one-shot bootstrap.

Detects whether the named client is installed on this machine, then
writes/patches the client's local config so it routes traffic at the
local fusion-mlx OpenAI-compatible server.
"""

from . import (
    claude_code,
    cline,
    codex,
    continue_dev,
    cursor,
    factory_droid,
    hermes,
    kilo_code,
    kimi_code,
    openhands,
    opencode,
    pydantic_ai,
    qwen_code,
    smolagents,
)

ADAPTERS: dict[str, object] = {
    "cline": cline,
    "claude": claude_code,
    "claude-code": claude_code,
    "codex": codex,
    "continue-dev": continue_dev,
    "cursor": cursor,
    "hermes": hermes,
    "opencode": opencode,
    "qwen-code": qwen_code,
    "openhands": openhands,
    "kilo-code": kilo_code,
    "factory-droid": factory_droid,
    "kimi-code": kimi_code,
    "pydantic-ai": pydantic_ai,
    "smolagents": smolagents,
}

__all__ = ["ADAPTERS"]
