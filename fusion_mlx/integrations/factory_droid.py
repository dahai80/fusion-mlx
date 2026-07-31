"""Factory Droid integration."""

from __future__ import annotations

import os
from pathlib import Path

from fusion_mlx.integrations.base import Integration
from fusion_mlx.utils.install import get_cli_prefix


class FactoryDroidIntegration(Integration):
    """Factory Droid integration using OPENAI_API_KEY/OPENAI_BASE_URL env vars."""

    def __init__(self):
        super().__init__(
            name="factory-droid",
            display_name="Factory Droid",
            type="env_var",
            install_check="factory-droid",
            install_hint="pip install factory-droid",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return f"{get_cli_prefix()} launch factory-droid --model {model or 'select-a-model'}"

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        context_window: int | None = None,
        extra_args: list[str] | None = None,
        **kwargs,
    ) -> None:
        env = self._scrubbed_env()
        env["OPENAI_BASE_URL"] = f"http://{host}:{port}/v1"
        env["OPENAI_API_KEY"] = api_key or "fusion-mlx"

        if model:
            env["FACTORY_DROID_MODEL"] = model

        if context_window:
            env["FACTORY_DROID_MAX_CONTEXT"] = str(context_window)

        args = ["factory-droid"]
        if model:
            args.extend(["--model", model])
        args.extend(extra_args or [])

        print(f"Launching Factory Droid with model {model}...")
        os.execvpe("factory-droid", args, env)
