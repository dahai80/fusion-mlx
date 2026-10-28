# SPDX-License-Identifier: Apache-2.0
"""fusion-mlx web module bridge — issue #1276.

Serves the built web/dist (web monitor UI) plus a reverse proxy of the
engine API surface. Shell-gateway registration is NOT duplicated here:
fusion_mlx/shell_registry.py (engine, #1220) already registers
web/manifest.json with the gateway and heartbeats it. This module adds
the upstream endpoint that manifest.upstream points at, following the
cowork v0.5.20 pattern so gateway #232 dynamic proxy can resolve it:

- manifest declares upstream (:11471) + api_key_env (FUSION_MLX_WEB_API_KEY)
- FUSION_MLX_WEB_API_KEY gates API-prefix paths on this bridge (unset =
  open, loopback posture, same as cowork). Forwarded requests attach the
  engine key (FUSION_MLX_API_KEY env, else settings.json auth.api_key via
  the same resolution the engine uses) so the embedded UI ships no secret.
- FUSION_SHELL_GATEWAY_URL + FUSION_GATEWAY_ADMIN_KEY enable registration
  (run_shell_registry); unconfigured = skipped (default OFF).
"""

import asyncio
import logging
import os
from pathlib import Path

from fusion_mlx import shell_registry

logger = logging.getLogger(__name__)

WEB_PORT_DEFAULT = 11471
ENGINE_URL_DEFAULT = "http://127.0.0.1:11434"

UPSTREAM_OVERRIDE_ENV = "FUSION_MLX_WEB_UPSTREAM"
WEB_DIST_ENV = "FUSION_MLX_WEB_DIST"
WEB_API_KEY_ENV = "FUSION_MLX_WEB_API_KEY"
ENGINE_URL_ENV = "FUSION_MLX_WEB_ENGINE_URL"

# Engine API prefixes proxied through the bridge (same set the vite dev
# server proxies — see web/vite.config.ts apiPaths).
API_PREFIXES = ("/v1", "/admin", "/metrics", "/stats")

# hop-by-hop headers never forwarded in either direction
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def resolve_dist_dir() -> Path:
    d = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
    return Path(os.environ.get(WEB_DIST_ENV, "").strip() or d)


def resolve_engine_url() -> str:
    return os.environ.get(ENGINE_URL_ENV, "").strip().rstrip("/") or ENGINE_URL_DEFAULT


def resolve_web_api_key() -> str:
    return os.environ.get(WEB_API_KEY_ENV, "").strip()


def resolve_engine_api_key() -> str:
    from .config_seam import api_key as seam_api_key

    key = seam_api_key().strip()
    if key:
        return key
    try:
        from ..middleware.auth import _get_configured_api_key

        return (_get_configured_api_key() or "").strip()
    except Exception as e:
        logger.warning(
            "engine api_key resolution failed (forwarding unauthenticated): %s", e
        )
        return ""


def _registration_enabled() -> bool:
    return shell_registry.shell_registration_enabled()


def create_web_app(dist_dir=None):
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="fusion-mlx-web", docs_url=None, redoc_url=None, openapi_url=None
    )
    dist = Path(dist_dir) if dist_dir else resolve_dist_dir()
    index_file = dist / "index.html"
    if not index_file.exists():
        logger.warning(
            "web dist missing (%s) — UI unavailable, run `cd web && bun install && "
            "bun run build`; /health and the API proxy still work",
            dist,
        )
        dist = None

    engine_url = resolve_engine_url()
    web_key = resolve_web_api_key()
    engine_key = resolve_engine_api_key()
    client = httpx.AsyncClient(base_url=engine_url, timeout=None)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "module": "fusion-mlx",
            "dist": bool(dist),
            "engine": engine_url,
        }

    # --- manifest ext endpoints (#1281): ⌘K search + notification center ---
    # Both are plain same-origin fetches by the shell host through the
    # gateway /shell proxy, served here off data that already exists on the
    # engine (no new engine surface).

    @app.get("/api/search/models")
    async def search_models(request: Request, q: str = ""):
        if not _client_auth_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        query = (q or "").strip().lower()
        headers = _upstream_headers(request)
        try:
            resp = await client.get("/admin/api/models", headers=headers)
            models = resp.json().get("models", [])
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("search models upstream failed: %s", e)
            return JSONResponse({"items": []})
        items = []
        for m in models:
            hay = " ".join(
                str(m.get(k) or "")
                for k in ("id", "engine_type", "model_type", "config_model_type")
            ).lower()
            if query and query not in hay:
                continue
            items.append({"title": m.get("id"), "route": "/models"})
            if len(items) >= 8:
                break
        return JSONResponse({"items": items})

    @app.get("/api/notifications")
    async def notifications(request: Request):
        if not _client_auth_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        headers = _upstream_headers(request)
        items = []
        try:
            health_resp = await client.get("/v1/health", headers=headers)
            health = health_resp.json()
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
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("notifications health poll failed: %s", e)
        try:
            models_resp = await client.get("/admin/api/models", headers=headers)
            for m in models_resp.json().get("models", []):
                if m.get("loaded"):
                    items.append(
                        {
                            "id": f"mlx-model-loaded:{m.get('id')}",
                            "title": f"model loaded: {m.get('id')}",
                            "level": "info",
                            "route": "/models",
                        }
                    )
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("notifications models poll failed: %s", e)
        return JSONResponse({"count": len(items), "items": items})

    def _client_auth_ok(request):
        if not web_key:
            return True
        return request.headers.get("authorization", "") == f"Bearer {web_key}"

    def _forward_headers(headers):
        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in _HOP_HEADERS and k.lower() != "authorization"
        }

    def _upstream_headers(request):
        hdrs = _forward_headers(request.headers)
        if engine_key:
            hdrs["Authorization"] = f"Bearer {engine_key}"
        return hdrs

    async def _proxy(request: Request):
        if not _client_auth_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        path = request.url.path
        url = httpx.URL(path=path, query=request.url.query.encode("utf-8"))
        upstream_req = client.build_request(
            request.method,
            url,
            headers=_upstream_headers(request),
            content=request.stream(),
        )
        try:
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as e:
            logger.error("engine proxy %s %s failed: %s", request.method, path, e)
            return JSONResponse(
                {"error": "engine_unavailable", "detail": str(e)}, status_code=502
            )

        async def _stream():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            _stream(),
            status_code=upstream_resp.status_code,
            headers=_forward_headers(upstream_resp.headers),
        )

    for prefix in API_PREFIXES:
        app.add_api_route(
            prefix + "/{path:path}",
            _proxy,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
            include_in_schema=False,
        )
    for prefix in API_PREFIXES:
        app.add_api_route(
            prefix.rstrip("/") or "/",
            _proxy,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
            include_in_schema=False,
        )

    if dist is not None:
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


async def _serve(app, host, port, register):
    import uvicorn

    registry_task = None
    if register:
        registry_task = asyncio.create_task(shell_registry.run_shell_registry())
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        if registry_task:
            registry_task.cancel()


def serve_web(port=WEB_PORT_DEFAULT, host=None):
    host = host or os.environ.get("FUSION_BIND_HOST", "127.0.0.1")
    app = create_web_app()
    register = _registration_enabled()
    if not register:
        logger.info(
            "shell gateway registration disabled (%s / %s unset)",
            shell_registry.GATEWAY_URL_ENV,
            shell_registry.ADMIN_KEY_ENV,
        )
    logger.info("fusion-mlx web module bridge serving on %s:%s", host, port)
    asyncio.run(_serve(app, host, port, register))
