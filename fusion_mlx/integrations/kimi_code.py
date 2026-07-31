"""Moonshot Kimi Code integration."""

from __future__ import annotations

import os
from pathlib import Path

from fusion_mlx.integrations.base import Integration
from fusion_mlx.utils.install import get_cli_prefix


class KimiCodeIntegration(Integration):
    """Kimi Code integration using OPENAI_API_KEY/OPENAI_BASE_URL env vars."""

    def __init__(self):
        super().__init__(
            name="kimi-code",
            display_name="Kimi Code",
            type="env_var",
            install_check="kimi-code",
            install_hint="pip install kimi-code",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return f"{get_cli_prefix()} launch kimi-code --model {model or 'select-a-model'}"

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
            env["KIMI_CODE_MODEL"] = model

        if context_window:
            env["KIMI_CODE_MAX_CONTEXT"] = str(context_window)

        args = ["kimi-code"]
        if model:
            args.extend(["--model", model])
        args.extend(extra_args or [])

        print(f"Launching Kimi Code with model {model}...")
        os.execvpe("kimi-code", args, env)
