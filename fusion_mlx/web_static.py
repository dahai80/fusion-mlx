# SPDX-License-Identifier: Apache-2.0
"""#1220: static hosting of the web/ monitor SPA (union-web shell module).

The gateway /shell/fusion-mlx/ proxy strips the module prefix and forwards
the remainder to this server's root, so the SPA index must be served at "/"
and its hashed assets at /assets/* (relative base + hash router keep every
other path untouched — the API surface below /v1, /admin, /metrics, /health
is unaffected). Source installs without web/dist keep the legacy JSON
probe response at "/".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)


def web_dist_dir() -> Path:
    """Built SPA directory. FUSION_MLX_WEB_DIST overrides (packaging/tests)."""
    override = os.environ.get("FUSION_MLX_WEB_DIST", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "web" / "dist"


def mount_web_static(app: FastAPI) -> bool:
    """Serve /assets from web/dist. Returns True when mounted.

    Called once at app setup. A missing dist (source install) skips the
    mount with a single info log — the JSON probe root stays in place.
    """
    dist = web_dist_dir()
    assets = dist / "assets"
    index = dist / "index.html"
    if not index.exists() or not assets.is_dir():
        logger.info("web dist not present at %s — '/' serves JSON probe", dist)
        return False
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=str(assets)), name="web_assets")
    logger.info("web dist mounted: /assets from %s", assets)
    return True


async def web_root_or_json() -> FileResponse | JSONResponse:
    """GET / handler body: SPA index when built, legacy probe JSON otherwise.

    HEAD is auto-derived by Starlette. The {"status": "ok"} shape remains
    available at /health, /healthz, /livez for every probe client.
    """
    index = web_dist_dir() / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return JSONResponse({"status": "ok"})
