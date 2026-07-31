"""PydanticAI integration."""

from __future__ import annotations

from pathlib import Path

from fusion_mlx.integrations.base import Integration
from fusion_mlx.utils.install import get_cli_prefix


class PydanticAIIntegration(Integration):
    """PydanticAI integration using OPENAI_API_KEY/OPENAI_BASE_URL env vars."""

    def __init__(self):
        super().__init__(
            name="pydantic-ai",
            display_name="PydanticAI",
            type="env_var",
            install_check="pydantic-ai",
            install_hint="pip install pydantic-ai",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"{get_cli_prefix()} launch pydantic-ai --model {model or 'select-a-model'}"
        )

    def configure(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> None:
        config_path = Path.home() / ".pydantic-ai" / "config.json"

        def updater(config: dict) -> None:
            config["openai_base_url"] = f"http://{host}:{port}/v1"
            config["openai_api_key"] = api_key or "fusion-mlx"
            if model:
                config["default_model"] = f"openai:{model}"

        self._write_json_config(config_path, updater)

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        context_window: int | None = None,
        **kwargs,
    ) -> None:
        self.configure(port, api_key, model, host=host)

        env = self._scrubbed_env()
        env["OPENAI_BASE_URL"] = f"http://{host}:{port}/v1"
        env["OPENAI_API_KEY"] = api_key or "fusion-mlx"

        if model:
            env["PYDANTIC_AI_MODEL"] = f"openai:{model}"
        if context_window:
            env["PYDANTIC_AI_MAX_CONTEXT"] = str(context_window)

        import sys

        print(f"PydanticAI configured with model openai:{model}")
        print(f"  OPENAI_BASE_URL={env['OPENAI_BASE_URL']}")
        print("  Use 'from pydantic_ai import Agent' with model='openai:{model}'")
        sys.exit(0)
