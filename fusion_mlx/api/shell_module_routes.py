# SPDX-License-Identifier: Apache-2.0
"""#1489: shell-module manifest endpoints served by the engine itself.

The web module bridge (admin/webserve.py, opt-in :11471) previously owned
/api/search/models and /api/notifications, but the gateway shell proxy
always dials the STATIC upstream (engine :11434) — so with the bridge off
these manifest-driven endpoints died with connection-refused 503s, and the
self-registered upstream (:11471) permanently disagreed with the static
config, tripping the gateway drift guard (fusion-gateway#478) on every
heartbeat.

Fix: serve both endpoints on the engine, reading the same in-process data
the bridge aggregates (/v1/health oom_risk + admin model list). The
manifest now declares upstream=11434 (= static), registration is truthful
in every bridge state, and the endpoints degrade explicitly (200 with
empty items + degraded flag) instead of 503-from-a-dead-port. The bridge
keeps its own copies for direct :11471 access — no behavior change there.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..middleware.auth import verify_api_key_or_x_api_key

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key_or_x_api_key)])

_SEARCH_KEYS = ("id", "engine_type", "model_type", "config_model_type")


async def _engine_models() -> list[dict[str, Any]]:
    from ..admin.models_route import list_models

    data = await list_models(is_admin=True)
    return list(data.get("models", []) or [])


@router.get("/api/search/models")
async def shell_search_models(q: str = ""):
    query = (q or "").strip().lower()
    items: list[dict[str, Any]] = []
    degraded = False
    try:
        models = await _engine_models()
    except Exception as e:
        logger.warning("shell search: engine model list failed: %s", e)
        models = []
        degraded = True
    for m in models:
        hay = " ".join(str(m.get(k) or "") for k in _SEARCH_KEYS).lower()
        if query and query not in hay:
            continue
        items.append({"title": m.get("id"), "route": "/models"})
        if len(items) >= 8:
            break
    return JSONResponse({"items": items, "degraded": degraded})


@router.get("/api/notifications")
async def shell_notifications():
    items: list[dict[str, Any]] = []
    degraded = False
    risk = None
    try:
        from ..routes_internal.health import v1_health

        health = await v1_health(_auth=True)
        risk = health.get("oom_risk")
        if risk == "imminent":
            items.append(
                {
                    "id": "mlx-oom-imminent",
                    "title": "memory pressure: OOM imminent",
                    "level": "error",
                    "route": "/metrics",
                }
            )
        elif risk == "high":
            items.append(
                {
                    "id": "mlx-oom-high",
                    "title": "memory pressure high",
                    "level": "warning",
                    "route": "/metrics",
                }
            )
    except Exception as e:
        logger.warning("shell notifications: health poll failed: %s", e)
        degraded = True
    try:
        models = await _engine_models()
        for m in models:
            if m.get("loaded"):
                items.append(
                    {
                        "id": f"mlx-model-loaded:{m.get('id')}",
                        "title": f"model loaded: {m.get('id')}",
                        "level": "info",
                        "route": "/models",
                    }
                )
    except Exception as e:
        logger.warning("shell notifications: model list failed: %s", e)
        degraded = True
    return JSONResponse({"count": len(items), "items": items, "degraded": degraded})
