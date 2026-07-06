# SPDX-License-Identifier: Apache-2.0
"""Security banner shown before printing share URL+key."""

from __future__ import annotations

import json
import shlex
import sys


def _supports_color() -> bool:
    return sys.stdout.isatty() and not sys.platform.startswith("win")


def render(
    url: str,
    api_key: str,
    model: str,
    tunnel_id: str,
    chat_frontend: str | None,
) -> str:
    red = "\033[1;31m" if _supports_color() else ""
    yellow = "\033[1;33m" if _supports_color() else ""
    reset = "\033[0m" if _supports_color() else ""
    bold = "\033[1m" if _supports_color() else ""

    safe_url = shlex.quote(f"{url}/v1/chat/completions")
    safe_body = shlex.quote(
        json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    )
    if chat_frontend:
        chat_link = f"{chat_frontend}/#k={tunnel_id}.{api_key}"
        chat_line = f"  {bold}Chat:{reset}   {yellow}{chat_link}{reset}\n"
    else:
        chat_line = ""

    return (
        f"\n  🔥 {bold}Fusion-MLX share{reset}\n"
        f"\n{red}╔══════════════════════════════════════════════════════════════════╗{reset}\n"
        f"{red}║  ⚠  PUBLIC INTERNET — read this before sharing                   ║{reset}\n"
        f"{red}╠══════════════════════════════════════════════════════════════════╣{reset}\n"
        f"{red}║{reset} fusion-mlx share is now exposing this machine to the public    {red} ║{reset}\n"
        f"{red}║{reset} internet. Anyone who has the API key below can:                {red} ║{reset}\n"
        f"{red}║{reset}   • use your compute (free inference on your bill)              {red} ║{reset}\n"
        f"{red}║{reset}   • see every prompt and response that goes through              {red}║{reset}\n"
        f"{red}║{reset}                                                                  {red}║{reset}\n"
        f"{red}║{reset} Do NOT screenshot, paste, or commit this key. Ctrl-C stops it.  {red} ║{reset}\n"
        f"{red}╚══════════════════════════════════════════════════════════════════╝{reset}\n"
        f"\n"
        f"  {bold}Model:{reset}  {model}\n"
        f"{chat_line}"
        f"  {bold}URL:{reset}    {url}\n"
        f"  {bold}Key:{reset}    {yellow}{api_key}{reset}\n"
        f"\n"
        f"  Test it (key stays out of shell history via env-var):\n"
        f"    export FUSION_MLX_SHARE_KEY={yellow}<paste-key>{reset}\n"
        f"    curl -sS {safe_url} \\\n"
        f'      -H "Authorization: Bearer $FUSION_MLX_SHARE_KEY" \\\n'
        f"      -H 'Content-Type: application/json' \\\n"
        f"      -d {safe_body}\n"
        f"\n"
        f"  Press Ctrl-C to stop sharing.\n"
    )
