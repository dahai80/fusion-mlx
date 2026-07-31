"""Kilo Code integration."""

from __future__ import annotations

import os

from fusion_mlx.integrations.base import Integration
from fusion_mlx.utils.install import get_cli_prefix


class KiloCodeIntegration(Integration):
    """Kilo Code integration using OPENAI_API_KEY/OPENAI_BASE_URL env vars."""

    def __init__(self):
        super().__init__(
            name="kilo-code",
            display_name="Kilo Code",
            type="env_var",
            install_check="kilo-code",
            install_hint="pip install kilo-code",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"{get_cli_prefix()} launch kilo-code --model {model or 'select-a-model'}"
        )

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
            env["KILO_CODE_MODEL"] = model

        if context_window:
            env["KILO_CODE_MAX_CONTEXT"] = str(context_window)

        args = ["kilo-code"]
        if model:
            args.extend(["--model", model])
        args.extend(extra_args or [])

        print(f"Launching Kilo Code with model {model}...")
        os.execvpe("kilo-code", args, env)
