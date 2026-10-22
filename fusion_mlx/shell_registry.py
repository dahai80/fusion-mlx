# SPDX-License-Identifier: Apache-2.0
"""#1220: union-web shell module registration + heartbeat (PRD §5.2).

The server registers web/manifest.json with the fusion-gateway shell
registry (POST /api/v1/shell/manifests, admin key auth) and keeps it
alive with periodic heartbeats so the module stays "online" in the shell
menu (heartbeat_timeout marks silent modules unreachable/greyed).

Env-gated — requires BOTH:
- FUSION_SHELL_GATEWAY_URL: gateway base, e.g. http://127.0.0.1:11432
- FUSION_GATEWAY_ADMIN_KEY: a gateway api_key authorized for /api/v1/*

Optional:
- FUSION_SHELL_MANIFEST: manifest path override (default web/manifest.json
  next to the repo root).

The /shell/fusion-mlx/ proxy itself is static gateway config
(web_shell.modules.fusion-mlx.upstream/api_key) — dynamic manifests only
drive the menu/entry; see web/README.md for the wiring steps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GATEWAY_URL_ENV = "FUSION_SHELL_GATEWAY_URL"
ADMIN_KEY_ENV = "FUSION_GATEWAY_ADMIN_KEY"
MANIFEST_PATH_ENV = "FUSION_SHELL_MANIFEST"
REGISTER_INTERVAL_SECONDS = 15.0
HTTP_TIMEOUT_SECONDS = 5.0
# #1323: the heartbeat fires every 15s — a dead/rejecting gateway must not
# produce a per-tick warning flood (same class as the #1314 scrape flood).
FAILURE_LOG_INTERVAL_SECONDS = 300.0


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parent.parent / "web" / "manifest.json"


def shell_registration_enabled() -> bool:
    gateway = os.environ.get(GATEWAY_URL_ENV, "").strip()
    admin_key = os.environ.get(ADMIN_KEY_ENV, "").strip()
    return bool(gateway and admin_key)


def _load_manifest() -> dict[str, Any]:
    path = Path(
        os.environ.get(MANIFEST_PATH_ENV, "").strip() or _default_manifest_path()
    )
    if not path.exists():
        raise FileNotFoundError(f"shell manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("id"):
        raise ValueError(f"shell manifest missing id: {path}")
    return manifest


async def register_once() -> tuple[bool, str]:
    """One registration/heartbeat round-trip. (True, "") when the gateway
    accepted; otherwise (False, reason)."""
    gateway = os.environ.get(GATEWAY_URL_ENV, "").strip().rstrip("/")
    admin_key = os.environ.get(ADMIN_KEY_ENV, "").strip()
    manifest = _load_manifest()
    url = f"{gateway}/api/v1/shell/manifests"
    headers = {"Authorization": f"Bearer {admin_key}"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, json=manifest, headers=headers)
    if resp.status_code in (200, 201, 204):
        return True, ""
    return False, f"registration failed {url} -> {resp.status_code} {resp.text[:200]}"


async def run_shell_registry() -> None:
    """Lifespan worker: register + heartbeat until cancelled.

    Never raises — a dead gateway must not take the inference server down;
    each failed round-trip is logged and retried on the next interval.
    """
    logger.info(
        "shell registry: enabled, gateway=%s manifest=%s",
        os.environ.get(GATEWAY_URL_ENV, ""),
        os.environ.get(MANIFEST_PATH_ENV, "") or _default_manifest_path(),
    )
    last_failure_log = 0.0
    consecutive_failures = 0
    while True:
        exc: Exception | None = None
        try:
            ok, reason = await register_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            ok = False
            exc = e
            reason = f"round-trip failed: {type(e).__name__}: {e}"
        if ok:
            if consecutive_failures:
                logger.info(
                    "shell registry: recovered after %d failed round-trip(s)",
                    consecutive_failures,
                )
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            now = time.monotonic()
            if now - last_failure_log >= FAILURE_LOG_INTERVAL_SECONDS:
                last_failure_log = now
                logger.warning("shell registry: %s", reason, exc_info=exc is not None)
        await asyncio.sleep(REGISTER_INTERVAL_SECONDS)
