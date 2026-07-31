"""Integration registry for external coding tools."""

from fusion_mlx.integrations.base import Integration
from fusion_mlx.integrations.claude import ClaudeCodeIntegration
from fusion_mlx.integrations.codex import CodexIntegration
from fusion_mlx.integrations.comfyui import ComfyUIIntegration
from fusion_mlx.integrations.copilot import CopilotIntegration
from fusion_mlx.integrations.factory_droid import FactoryDroidIntegration
from fusion_mlx.integrations.hermes import HermesIntegration
from fusion_mlx.integrations.kilo_code import KiloCodeIntegration
from fusion_mlx.integrations.kimi_code import KimiCodeIntegration
from fusion_mlx.integrations.openclaw import OpenClawIntegration
from fusion_mlx.integrations.opencode import OpenCodeIntegration
from fusion_mlx.integrations.openhands import OpenHandsIntegration
from fusion_mlx.integrations.pi import PiIntegration
from fusion_mlx.integrations.pydantic_ai import PydanticAIIntegration
from fusion_mlx.integrations.qwen_code import QwenCodeIntegration
from fusion_mlx.integrations.smolagents import SmolagentsIntegration

INTEGRATIONS: dict[str, Integration] = {
    "claude": ClaudeCodeIntegration(),
    "codex": CodexIntegration(),
    "opencode": OpenCodeIntegration(),
    "openclaw": OpenClawIntegration(),
    "hermes": HermesIntegration(),
    "pi": PiIntegration(),
    "copilot": CopilotIntegration(),
    "comfyui": ComfyUIIntegration(),
    "qwen-code": QwenCodeIntegration(),
    "openhands": OpenHandsIntegration(),
    "kilo-code": KiloCodeIntegration(),
    "factory-droid": FactoryDroidIntegration(),
    "kimi-code": KimiCodeIntegration(),
    "pydantic-ai": PydanticAIIntegration(),
    "smolagents": SmolagentsIntegration(),
}


def get_integration(name: str) -> Integration | None:
    return INTEGRATIONS.get(name)


def list_integrations() -> list[Integration]:
    return list(INTEGRATIONS.values())


__all__ = [
    "Integration",
    "INTEGRATIONS",
    "get_integration",
    "list_integrations",
]
