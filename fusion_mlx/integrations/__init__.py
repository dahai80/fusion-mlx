"""Integration registry for external coding tools."""

from fusion_mlx.integrations.base import Integration
from fusion_mlx.integrations.claude import ClaudeCodeIntegration
from fusion_mlx.integrations.codex import CodexIntegration
from fusion_mlx.integrations.comfyui import ComfyUIIntegration
from fusion_mlx.integrations.copilot import CopilotIntegration
from fusion_mlx.integrations.hermes import HermesIntegration
from fusion_mlx.integrations.openclaw import OpenClawIntegration
from fusion_mlx.integrations.opencode import OpenCodeIntegration
from fusion_mlx.integrations.pi import PiIntegration

INTEGRATIONS: dict[str, Integration] = {
    "claude": ClaudeCodeIntegration(),
    "codex": CodexIntegration(),
    "opencode": OpenCodeIntegration(),
    "openclaw": OpenClawIntegration(),
    "hermes": HermesIntegration(),
    "pi": PiIntegration(),
    "copilot": CopilotIntegration(),
     "comfyui": ComfyUIIntegration(),
}


def get_integration(name: str) -> Integration | None:
    """Get an integration by name."""
    return INTEGRATIONS.get(name)


def list_integrations() -> list[Integration]:
    """List all available integrations."""
    return list(INTEGRATIONS.values())


__all__ = [
    "Integration",
    "ClaudeCodeIntegration",
    "CopilotIntegration",
    "HermesIntegration",
    "INTEGRATIONS",
    "get_integration",
    "list_integrations",
]
