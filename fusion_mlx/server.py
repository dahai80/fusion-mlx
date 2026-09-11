"""FastAPI server for fusion-mlx.

Wires together all API routes:
- OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models
- Resumable streaming: /v1/stream, /v1/streams/lookup (#801)
- Anthropic-compatible: /v1/messages, /v1/count_tokens
- Audio: /v1/audio/transcriptions, /v1/audio/speech, /v1/audio/process
- Images: /v1/images/generate
- MCP: /v1/mcp/tools, /v1/mcp/servers, /v1/mcp/execute
- OpenClaw Agent: /v1/openclaw/agent/*
- JSON-RPC: /rpc (mlx.set_model, mlx.status)
- Admin: /admin/*
- GC: /api/v1/gc (post-compact KV cache release)
- GUI compatibility: /v1/manager/*, /v1/discover/*, /v1/settings, /admin
"""

import asyncio
import logging
import threading
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ._version import __version__
from .admin.auth import require_admin
from .admin.routes import router as admin_router
from .api.agent_routes import router as agent_router
from .api.anthropic_routes import router as anthropic_router
from .api.anthropic_routes import set_anthropic_context
from .api.audio_routes import router as audio_router
from .api.audio_routes import set_audio_context
from .api.convert_routes import router as convert_router
from .api.distributed_routes import router as distributed_router
from .api.images import router as images_router
from .api.images import set_images_context
from .api.images_sr import router as images_sr_router
from .api.images_sr import set_images_sr_context
from .api.layered_quantize_routes import router as layered_quantize_router
from .api.mcp_routes import router as mcp_router
from .api.mcp_routes import set_mcp_manager_getter
from .api.watermark_routes import router as watermark_router
from .exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelTooLargeError,
)
from .middleware import (
    install_auth_precheck_middleware,
    install_exception_handlers,
    install_probe_fastpath_middleware,
    install_request_body_depth_middleware,
    install_request_body_limit_middleware,
    install_request_id_middleware,
    install_route_guard_middleware,
    require_model_hub_source,
    scheduler_queue_full_handler,  # noqa: F401  re-exported for handler tests
)

# GUI compatibility layer
try:
    from fusion_mlx.gui_compat.database import close_database, get_database_manager
    from fusion_mlx.gui_compat.server import get_gui_compat_router
except (ImportError, AttributeError):
    # ImportError: gui_compat or its transitive deps missing
    # AttributeError: mlx_whisper→tiktoken chain can raise this
    #   (SwigPy interference in pytest). gui_compat is optional.
    get_gui_compat_router = None
    get_database_manager = None
    close_database = None

# Import route modules
from .admin.helpers import (
    set_admin_getters,
    set_hf_downloader,
    set_hf_uploader,
    set_ms_downloader,
    set_oq_manager,
)
from .api.bench_routes import router as bench_router
from .api.embeddings_routes import router as embeddings_router
from .api.embeddings_routes import set_embeddings_context
from .api.flywheel_routes import router as flywheel_router
from .api.ner_routes import router as ner_router
from .api.ner_routes import set_ner_context
from .api.ocr_routes import router as ocr_router
from .api.ocr_routes import set_ocr_context
from .api.ollama_routes import router as ollama_router
from .api.ollama_routes import set_ollama_context
from .api.openai_routes import router as openai_router
from .api.openai_routes import set_openai_context
from .api.openclaw_routes import router as openclaw_router
from .api.openclaw_routes import set_openclaw_agent_pool
from .api.reasoning_routes import router as reasoning_router
from .api.reasoning_routes import set_reasoning_context
from .api.recommend_batch_routes import router as recommend_batch_router
from .api.recommend_routes import router as recommend_router
from .api.rerank_routes import router as rerank_router
from .api.rerank_routes import set_rerank_context
from .api.session_routes import router as sessions_router
from .api.session_routes import set_sessions_context
from .api.spec_routes import router as spec_router
from .api.videos_routes import router as videos_router
from .api.videos_routes import set_videos_context
from .cluster.routes import router as cluster_router
from .config import ServerConfig
from .dispatch import CloudRouter, RequestRouter
from .engine_core import AsyncEngineCore
from .pool import EnginePool, ProcessMemoryEnforcer
from .routes_internal.cache import router as cache_router
from .routes_internal.config_reload import router as config_reload_router
from .routes_internal.gc import router as gc_router
from .routes_internal.health import admin_router as health_admin_router
from .routes_internal.health import probe_router as health_probe_router
from .routes_internal.health import router as health_router
from .routes_internal.metrics import router as metrics_router
from .routes_internal.models import set_models_context
from .routes_internal.responses import router as responses_router
from .routes_internal.responses import set_responses_context
from .server_metrics import get_server_metrics
from .settings import Settings

logger = logging.getLogger(__name__)


def _install_sighup_reload() -> None:
    # OPS-P4-6 (#0907 audit): SIGHUP hot-reload. Re-reads settings.json and
    # applies the safe reloadable subset (log level, idle timeout, memory
    # tier/ceiling, prefill guard, chunked prefill, route-guard toggles)
    # without restart. Loaded models stay resident. Reload errors are logged
    # but never crash the server (fail-visible, not fatal). Windows lacks
    # SIGHUP — skip silently there.
    import asyncio
    import signal

    try:
        signum = signal.SIGHUP
    except AttributeError:
        logger.debug("SIGHUP not available on this platform — hot-reload disabled")
        return

    loop = asyncio.get_running_loop()

    async def _on_sighup() -> None:
        logger.info("SIGHUP received — hot-reloading settings.json")
        try:
            from .routes_internal.config_reload import reload_config

            result = await reload_config(source="sighup")
            if result.get("errors"):
                logger.warning(
                    "SIGHUP reload completed with errors: %s", result["errors"]
                )
        except Exception:
            logger.error("SIGHUP reload failed — keeping live config", exc_info=True)

    try:
        loop.add_signal_handler(signum, lambda: asyncio.ensure_future(_on_sighup()))
        logger.info("SIGHUP hot-reload handler installed")
    except (NotImplementedError, RuntimeError):
        # add_signal_handler unsupported (e.g. non-main thread) — fall back to
        # a plain signal.signal handler that schedules the coroutine.
        import threading

        def _sync_handler(signum, frame):
            if threading.main_thread() != threading.current_thread():
                return
            logger.info("SIGHUP received (fallback handler) — scheduling reload")
            asyncio.ensure_future(_on_sighup())

        try:
            signal.signal(signum, _sync_handler)
            logger.info("SIGHUP hot-reload handler installed (fallback)")
        except (ValueError, OSError):
            logger.warning(
                "SIGHUP handler could not be installed — hot-reload unavailable"
            )


class _ServerState(dict):
    """Dict subclass that also supports attribute access for admin helpers.

    A-P2-3: Thread-safety contract — startup writes are single-threaded
    (pre-serve, safe). At runtime, simple __setitem__/__getitem__ are
    GIL-atomic. Read-modify-write sequences on mutable fields (api_key,
    engine_pool) MUST take _server_state_lock. Concurrent iteration over
    engine_pool._entries from request threads while pool.shutdown() runs
    is NOT protected by _server_state_lock — callers must snapshot via
    list() before iterating. This is a known limitation; a typed singleton
    with __slots__ would be stronger but would break existing attribute
    access patterns in admin helpers.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


_server_state = _ServerState()
_server_instance: "Server | None" = None
# AS-7 (#0907 audit): serialize read-modify-write on mutable _server_state
# fields (notably api_key setup) so two concurrent admin requests cannot
# both pass the "not yet configured" check and clobber each other. Startup
# writes (single-threaded, pre-serve) and simple __setitem__ updates are
# GIL-atomic and do not need this lock; only RMW admin paths take it.
_server_state_lock = asyncio.Lock()
# AS-11 (#0907 audit): serialize first-call Server() construction in
# get_app(). Without this, two concurrent calls both observe
# _server_instance is None, each builds a Server + FastAPI app, and the
# second overwrites the globals while the first's lifespan keeps running —
# two apps, one orphaned. threading.Lock (get_app is sync) with a
# double-checked fast path so warm calls pay no lock cost.
_app_init_lock = threading.Lock()

app = None

# Module-level server state — cli_serve.py reads/writes these directly
_api_key: str | None = None
# Staged single-model request from ``serve --model <X>``. ``load_model``
# populates this before uvicorn starts; ``Server._startup`` loads + registers
# the engine into the pool once the pool exists. None on the multi-model
# ``--model-dir`` path (which discovers into the pool directly).
_pending_single_model: dict | None = None
# Effective parser state - cli_serve / model_auto_config set these from explicit
# --tool-call-parser / --reasoning-parser flags OR auto-detect. routes_internal.
# models reads them to surface LIVE parsers on /v1/models (not just static
# alias-profile defaults). embedding_model_locked pins the embed model.


def _sync_config() -> None:
    # Propagate CLI-staged api_key to ServerConfig + admin.auth.
    # Other fields (sampling, gc_control, etc.) are written directly
    # to ServerConfig by cli_serve — no longer staged through globals (#50).
    try:
        from .config import get_config

        cfg = get_config()
        cfg.api_key = _api_key
    except Exception:
        logger.warning("_sync_config: failed to set api_key on config", exc_info=True)
    if _api_key:
        try:
            from .admin.auth import set_api_key

            set_api_key(_api_key)
        except Exception:
            logger.debug("set_api_key propagation failed (non-fatal)", exc_info=True)


def configure_logging(log_level: str) -> str:
    """Configure console logging and return the level name for uvicorn.

    Delegates to ``fusion_mlx.logging_config.configure_logging`` (colored
    stderr output, request-id filter, admin-polling access-log suppression,
    third-party noise taming) while preserving the released
    ``-> str`` contract that cli_serve relies on when wiring uvicorn.

    OP-1 (#0907 audit): ``FUSION_LOG_JSON=1`` switches the console formatter
    to the JSON structured formatter so a log aggregator (Loki/ELK/Datadog)
    can ingest records without regex parsing. Default stays the colored
    human-readable formatter for local dev.
    """
    import os

    from .logging_config import configure_logging as _configure_logging

    _json = os.environ.get("FUSION_LOG_JSON", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _configure_logging(
        level=log_level,
        format_style="json" if _json else "standard",
        # Disable ANSI colors under JSON — color codes corrupt structured
        # parsers. Keep colors for the default human-readable stream.
        colored=not _json,
    )
    return log_level.upper()


def _resolve_api_key(argv_api_key: str | None = None) -> str | None:
    global _api_key
    import os

    if argv_api_key:
        return argv_api_key
    if _api_key:
        return _api_key
    return os.environ.get("FUSION_MLX_API_KEY")


def _resolve_effective_api_key(
    argv_key: str | None = None,
    settings_key: str | None = None,
) -> tuple[str | None, str]:
    # Single source of truth for the startup key priority:
    #   CLI --api-key  >  FUSION_MLX_API_KEY env  >  settings.json auth.api_key
    # ``_resolve_api_key`` folds CLI+env into the module global before
    # ``Server.__init__`` runs, so callers may pass that global as
    # ``argv_key`` (the operator-provided key). Returns ``(key, source)``
    # where source is one of ``"cli"`` / ``"env"`` / ``"settings"`` /
    # ``"none"`` for the startup log line.
    import os

    if argv_key:
        if settings_key and settings_key != argv_key:
            logger.warning(
                "api_key from cli overrides settings.json auth.api_key "
                "(they differ); clients sending the settings.json key "
                "will get 401"
            )
        return argv_key, "cli"
    env_key = os.environ.get("FUSION_MLX_API_KEY")
    if env_key:
        if settings_key and settings_key != env_key:
            logger.warning(
                "api_key from env (FUSION_MLX_API_KEY) overrides "
                "settings.json auth.api_key (they differ); clients "
                "sending the settings.json key will get 401"
            )
        return env_key, "env"
    if settings_key:
        return settings_key, "settings"
    return None, "none"


_cors_origins: list[str] | None = None
_cors_mounted: bool = False


class _SpecAlignedCORSMiddleware(CORSMiddleware):
    """L-02 spec-aligned preflight rejection envelope.

    Overrides stock Starlette ``400 Disallowed CORS origin`` with
    ``200 OK`` + no ``Access-Control-Allow-Origin`` + ``Vary: Origin``.
    Browsers block either way (ACAO absent is the only browser-observable
    signal), but the 200+missing-header shape is cleaner in devtools and
    avoids reverse-proxy operators reading a 4xx as "upstream unhealthy".
    Same 200-not-400 shape applies to disallowed-method preflights.

    Wide-open ``*`` default stance (#52) is untouched — this only
    changes the rejection branch when an explicit allowlist is set.
    """

    def preflight_response(self, request_headers):  # type: ignore[override]
        from starlette.responses import PlainTextResponse

        requested_origin = request_headers["origin"]
        requested_method = request_headers["access-control-request-method"]
        requested_headers = request_headers.get("access-control-request-headers")

        headers = dict(self.preflight_headers)
        headers["Vary"] = "Origin"
        failures = []

        if self.is_allowed_origin(origin=requested_origin):
            if self.preflight_explicit_allow_origin:
                headers["Access-Control-Allow-Origin"] = requested_origin
        else:
            failures.append("origin")
            headers.pop("Access-Control-Allow-Origin", None)

        if requested_method not in self.allow_methods:
            failures.append("method")

        if self.allow_all_headers and requested_headers is not None:
            headers["Access-Control-Allow-Headers"] = requested_headers
        elif requested_headers is not None:
            for header in [h.lower() for h in requested_headers.split(",")]:
                if header.strip() not in self.allow_headers:
                    failures.append("headers")
                    break

        if failures:
            logger.debug(
                "CORS preflight rejected (spec-aligned 200): failures=%s origin=%s method=%s",
                failures,
                requested_origin,
                requested_method,
            )
            return PlainTextResponse("OK", status_code=200, headers=headers)

        return PlainTextResponse("OK", status_code=200, headers=headers)


def _resolve_cors_origins(cors_origins) -> list[str] | None:
    # Returns the resolved origin allowlist. Distinguishes three states:
    #   None  → unset (caller falls back to wildcard ``*``)
    #   []    → fail-closed: env was set to whitespace-only CSV (operator
    #           templating bug, e.g. ``" , ,, "``); mount NO middleware so
    #           preflight 405s — the operator-visible signal (#758 3da8230)
    #   [...]  → explicit allowlist
    import os

    if cors_origins:
        origins = [o.strip() for o in cors_origins if o and o.strip()]
        if origins:
            return origins
        return []  # CLI passed whitespace-only → fail-closed
    env_raw = os.environ.get("FUSION_MLX_CORS_ALLOW_ORIGINS", "")
    if env_raw.strip():
        origins = [o.strip() for o in env_raw.split(",") if o.strip()]
        if origins:
            return origins
        return []  # env whitespace-only CSV → fail-closed
    return None


def _resolve_cors_methods() -> list[str]:
    # Narrowed default (POST, GET, OPTIONS) replaces the prior wide-open
    # ``["*"]`` so disallowed-method preflights reject with a 200 + no echo
    # (L-02). Override via FUSION_MLX_CORS_ALLOW_METHODS env CSV. ``*``
    # in the env value expands to all methods.
    import os

    env_raw = os.environ.get("FUSION_MLX_CORS_ALLOW_METHODS", "").strip()
    if env_raw:
        methods = [m.strip().upper() for m in env_raw.split(",") if m.strip()]
        if methods:
            return methods
        # env present but parsed to empty list (e.g. " , ,, ") — operator
        # templating bug. Warn and fall back rather than silently broadening
        # back to the default (#675).
        logger.warning(
            "FUSION_MLX_CORS_ALLOW_METHODS parsed to an empty list — "
            "falling back to default POST,GET,OPTIONS. Check the env "
            "value for a templating bug."
        )
    return ["POST", "GET", "OPTIONS"]


# F-091 narrowed default header allowlist for the env-driven CORS path.
# The legacy ``--cors-origins`` CLI path keeps ``["*"]`` (back-compat).
_DEFAULT_CORS_HEADERS_ENV = ["content-type", "authorization", "x-rapid-mlx-internal"]


def _resolve_cors_headers(env_path: bool) -> list[str]:
    # ``FUSION_MLX_CORS_ALLOW_HEADERS`` env CSV overrides the default.
    # Env unset:
    #   - env-driven path (#675 F-091) → narrowed default header allowlist
    #   - legacy CLI path → wide-open ``["*"]`` (back-compat contract)
    # Env present + non-empty → parsed list. Env present + empty → warn +
    # fall back to the path-appropriate default (#675).
    import os

    default = _DEFAULT_CORS_HEADERS_ENV if env_path else ["*"]
    env_raw = os.environ.get("FUSION_MLX_CORS_ALLOW_HEADERS", "").strip()
    if env_raw:
        headers = [h.strip().lower() for h in env_raw.split(",") if h.strip()]
        if headers:
            return headers
        logger.warning(
            "FUSION_MLX_CORS_ALLOW_HEADERS parsed to an empty list — "
            "falling back to default. Check the env value for a "
            "templating bug."
        )
    return default


def _resolve_cors_max_age() -> int:
    # ``FUSION_MLX_CORS_MAX_AGE`` env (seconds). Bad/empty → warn + default
    # 3600. Default 3600 replaces Starlette's silent 600 so preflight results
    # cache longer and reduce OPTIONS traffic (#675).
    import os

    env_raw = os.environ.get("FUSION_MLX_CORS_MAX_AGE", "").strip()
    if env_raw:
        try:
            value = int(env_raw)
            if value < 0:
                raise ValueError("negative max-age")
            return value
        except ValueError:
            logger.warning(
                "FUSION_MLX_CORS_MAX_AGE=%r is not a non-negative integer — "
                "falling back to default 3600.",
                env_raw,
            )
    return 3600


def _resolve_cors_credentials(origins: list[str] | None) -> bool:
    # ``FUSION_MLX_CORS_ALLOW_CREDENTIALS`` env opts credentials in. Default
    # False (#675: reverses #641's ``bool(_cors_origins)`` which auto-enabled
    # credentials on any explicit origin). Wildcard ``["*"]`` origins force
    # False per the fetch spec (``ACAO: *`` + credentials is illegal).
    import os

    if origins == ["*"]:
        return False
    env_raw = os.environ.get("FUSION_MLX_CORS_ALLOW_CREDENTIALS", "").strip().lower()
    if env_raw in ("true", "1", "yes", "on"):
        return True
    if env_raw in ("false", "0", "no", "off"):
        return False
    return False


def configure_cors_from_env(cors_origins=None, cli_origins=None):
    # ``cli_origins`` is the legacy Rapid-MLX kwarg; accept it as an alias
    # so callers/tests using the old contract still resolve correctly.
    # ``cli_origins`` non-None marks the legacy CLI path which keeps the
    # wide-open ``["*"]`` header default; the env-driven path (cors_origins
    # None and cli_origins None) gets the F-091 narrowed header default.
    if cors_origins is None and cli_origins is not None:
        cors_origins = cli_origins
    env_path = cli_origins is None and cors_origins is None
    global _cors_origins, _cors_mounted
    _cors_origins = _resolve_cors_origins(cors_origins)
    if _cors_origins:
        logger.info("CORS origins pinned to: %s", ", ".join(_cors_origins))
    elif _cors_origins == []:
        logger.warning(
            "CORS origins set to whitespace-only CSV — fail-closed "
            "(no CORS middleware, preflight will 405). Check "
            "FUSION_MLX_CORS_ALLOW_ORIGINS for a templating bug."
        )
    else:
        logger.debug("CORS origins defaulting to wildcard '*'")
    _mount_cors_middleware(env_path=env_path)
    return _cors_origins


def _mount_cors_middleware(env_path: bool = False):
    # Mount the spec-aligned CORS middleware onto the module-global app.
    # Idempotent: skip if already mounted this process. Relocated here from
    # ``Server._create_app`` so CORS config applied via configure_cors_from_env
    # (the test-harness + cli path) takes effect without re-building the app.
    # ``env_path`` selects the F-091 narrowed header default (env origins)
    # vs the legacy wide-open ``["*"]`` default (CLI origins).
    global _cors_mounted
    if _cors_mounted:
        return
    if app is None:
        logger.debug("CORS mount deferred — module-global app not built yet")
        return
    # Fail-closed: empty-CSV origins → mount nothing (preflight 405s).
    if _cors_origins == []:
        _cors_mounted = True
        return
    # P2-25 (#0909 audit): default CORS origins to localhost only — not
    # ["*"]. fusion-mlx is local-first on 127.0.0.1; wildcard origins allow
    # any website to cross-origin call the API (DNS rebinding, malicious
    # browser tab). Override via FUSION_MLX_CORS_ALLOW_ORIGINS env or
    # --cors-origins CLI when remote access is needed.
    if _cors_origins:
        cors_origins = _cors_origins
        cors_origin_regex = None
    else:
        cors_origins = []
        # Starlette does exact-string matching on allow_origins, so
        # http://localhost:3000 would not match. Use a regex to cover
        # any localhost/loopback port.
        cors_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    cors_methods = _resolve_cors_methods()
    cors_headers = _resolve_cors_headers(env_path=env_path)
    cors_credentials = _resolve_cors_credentials(_cors_origins)
    cors_max_age = _resolve_cors_max_age()
    app.add_middleware(
        _SpecAlignedCORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex,
        allow_methods=cors_methods,
        allow_headers=cors_headers,
        allow_credentials=cors_credentials,
        max_age=cors_max_age,
    )
    _cors_mounted = True
    logger.debug(
        "CORS middleware mounted (spec-aligned): origins=%s methods=%s "
        "headers=%s credentials=%s max_age=%s",
        cors_origins,
        cors_methods,
        cors_headers,
        cors_credentials,
        cors_max_age,
    )


def register_audio_routes_if_enabled(*args, **kwargs):
    pass


def _runtime_base_info(pool) -> dict[str, Any]:
    # Issue #104: base-binding runtime info for Fusion-Model-Hub. Exposes MLX
    # capabilities (Metal, quant formats, GPU) so ecosystem components verify
    # the base before model operations. Honest about mlx limits: gpu_cores and
    # metal_family are NOT reported by mx.device_info, so they stay None rather
    # than fabricated. max_context_length is model-dependent, not a constant.
    try:
        import mlx.core as mx

        metal_available = bool(mx.metal.is_available())
        di = mx.device_info() if metal_available else {}
    except Exception:
        logger.debug("base info: mlx probe failed", exc_info=True)
        metal_available = False
        di = {}
    mem_bytes = di.get("memory_size", 0) if isinstance(di, dict) else 0
    compatible: list[str] = []
    if pool is not None:
        try:
            compatible = list(pool.get_loaded_model_ids())
        except Exception:
            logger.debug("base info: get_loaded_model_ids failed", exc_info=True)
    return {
        "version": __version__,
        "metal_available": metal_available,
        "metal_family": None,
        "kv_cache_supported": True,
        "quantization_formats": [
            "mxfp4",
            "mxfp8",
            "mixed_3_4",
            "quant2",
            "quant2_all",
        ],
        "max_context_length": None,
        "gpu_info": {
            "chip_name": di.get("device_name") if isinstance(di, dict) else None,
            "gpu_cores": None,
            "memory_gb": round(mem_bytes / 1e9, 1) if mem_bytes else None,
        },
        "compatible_models": compatible,
    }


_NODE_PLATFORM_CACHE: str | None = None


def _node_platform() -> str:
    """Cached platform tag for this node (#365).

    Detected once per process via ``cluster.platform.detect_platform`` and
    cached so repeated snapshots don't re-probe torch.cuda.
    """
    global _NODE_PLATFORM_CACHE
    if _NODE_PLATFORM_CACHE is None:
        from .cluster.platform import detect_platform

        _NODE_PLATFORM_CACHE = str(detect_platform())
        logger.info("node platform: %s", _NODE_PLATFORM_CACHE)
    return _NODE_PLATFORM_CACHE


def _node_load_snapshot(pool, config) -> dict[str, Any]:
    # Issue #264: node-level load snapshot for Multi-Node cluster routing.
    # Reuses pool.get_status() + ServerMetrics + psutil; adds system memory
    # and node identity so a Cluster Manager can do load-aware routing in
    # one call. Apple Silicon unified memory => memory.* is the model budget.
    import socket

    metrics = get_server_metrics().to_dict()
    host = getattr(config, "bind_host", None) or getattr(config, "host", "127.0.0.1")
    port = getattr(config, "bind_port", None) or getattr(config, "port", 0)
    try:
        hostname = socket.gethostname()
    except Exception:
        logger.debug("node load: gethostname failed", exc_info=True)
        hostname = host
    node_id = f"{hostname}:{port}"

    mem_total = mem_avail = mem_used = 0
    available_percent = 0.0
    try:
        import psutil

        vm = psutil.virtual_memory()
        mem_total = vm.total
        mem_avail = vm.available
        mem_used = vm.used
        available_percent = round(mem_avail / mem_total * 100, 1) if mem_total else 0.0
    except Exception:
        logger.debug("node load: psutil virtual_memory failed", exc_info=True)

    models: list[dict[str, Any]] = []
    current_model_memory = 0
    final_ceiling = None
    if pool is not None:
        try:
            status = pool.get_status()
            current_model_memory = status.get("current_model_memory", 0)
            final_ceiling = status.get("final_ceiling")
            for m in status.get("models", []):
                models.append(
                    {
                        "id": m.get("id"),
                        "loaded": bool(m.get("loaded")),
                        "is_loading": bool(m.get("is_loading")),
                        "resident_bytes": m.get("estimated_size", 0),
                    }
                )
        except Exception:
            logger.debug("node load: pool.get_status failed", exc_info=True)

    can_load = mem_avail
    if final_ceiling:
        can_load = max(0, final_ceiling - current_model_memory)

    logger.debug(
        "node load snapshot: active=%d models_loaded=%d mem_avail=%d can_load=%d",
        metrics.get("active_requests", 0),
        sum(1 for m in models if m["loaded"]),
        mem_avail,
        can_load,
    )
    return {
        "node_id": node_id,
        "host": host,
        "port": port,
        "platform": _node_platform(),
        "uptime_seconds": round(metrics.get("uptime_seconds", 0.0), 3),
        "active_requests": metrics.get("active_requests", 0),
        "memory": {
            "total_bytes": mem_total,
            "available_bytes": mem_avail,
            "used_bytes": mem_used,
            "available_percent": available_percent,
        },
        "models": models,
        "capacity": {
            "free_memory_bytes": mem_avail,
            "can_load_estimate_bytes": can_load,
        },
        "throughput": {
            "avg_prefill_tps": round(metrics.get("avg_prefill_tps", 0.0), 3),
            "avg_generation_tps": round(metrics.get("avg_generation_tps", 0.0), 3),
        },
    }


def load_embedding_model(*args, **kwargs):
    raise NotImplementedError(
        "Use POST /v1/embeddings with a model already loaded in the pool. "
        "Load an embedding model via POST /v1/chat/completions or the CLI "
        "'fusion load <model>' first. See GET /v1/models for available models."
    )


def get_max_context_window(model_id: str) -> int | None:
    """Return the configured max context window for a model, or None if unset."""
    srv = get_server()
    if srv is None:
        return None
    return getattr(srv.config, "max_context_window", None)


def get_embedding_max_length(model_id: str, max_length: int | None) -> int | None:
    """Resolve per-request embedding token cap.

    Priority: request override > configured context window > None (model resolves).
    """
    if max_length is not None:
        return max_length
    return get_max_context_window(model_id)


def get_app():
    global _server_instance, app
    # AS-11 (#0907 audit): double-checked lock so the first-call Server()
    # construction is serialized — two concurrent callers cannot both
    # build a Server/FastAPI app and orphan one. Warm calls skip the lock.
    if _server_instance is None:
        with _app_init_lock:
            if _server_instance is None:
                _server_instance = Server()
    if app is None:
        app = _server_instance.app
    return app


def _resolve_single_model_path(name: str) -> str:
    # Resolve a model name to a loadable path/id. Reuses the fusion-mlx
    # model-discovery advantage: a bare name like ``Qwen3.6-27B-mxfp8``
    # resolves to a local model directory under the standard model dirs
    # instead of falling through to a HuggingFace lookup that 404s (the
    # released ``serve --model Qwen3-4B-Q4_K_M`` form). Exact aliases,
    # slash-names (HF repos), and existing local paths pass through.
    from .model_aliases import resolve_model

    resolved = resolve_model(name)
    if Path(resolved).exists():
        return resolved
    if "/" in resolved:
        return resolved
    home = Path.home()
    for cand in (
        home / ".fusion-mlx" / "models" / "mlx-community" / resolved,
        home / ".fusion-mlx" / "models" / resolved,
        home / ".fusion-mlx" / "models" / resolved,
    ):
        if cand.exists():
            return str(cand)
    hf_cache = home / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        norm = resolved.replace("/", "--")
        for snap in (hf_cache / f"models--{norm}").glob("snapshots/*"):
            return str(snap)
    return resolved


def load_model(
    model_name: str,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int | None = None,
    gpu_memory_utilization: float = 0.90,
    cloud_model: str | None = None,
    cloud_threshold: int = 20000,
    cloud_api_base: str | None = None,
    cloud_api_key: str | None = None,
    served_model_name: str | None = None,
    mtp: bool = False,
    *,
    max_tokens_is_explicit: bool | None = None,
    force_text: bool = False,
    force_hybrid: bool = False,
    no_hybrid: bool = False,
    force_spec_decode: bool = False,
    no_spec_decode: bool = False,
    force_openai_harmony_streaming: bool = False,
    no_openai_harmony_streaming: bool = False,
    lora_path: str | None = None,
):
    # ``serve --model <X>`` single-model entry. The migration left this as a
    # NotImplementedError stub, which broke even full local paths. We stage the
    # resolved model + scheduler config on a module global; ``Server._startup``
    # loads + registers the engine into the pool once the pool exists (it is
    # created in the lifespan, after this call). Routes then resolve the engine
    # through the pool like the multi-model ``--model-dir`` path.
    global _pending_single_model

    resolved = _resolve_single_model_path(model_name)
    from .config import get_config

    cfg = get_config()
    cfg.model_path = resolved
    cfg.model_name = served_model_name or resolved
    if not cfg.model_alias:
        cfg.model_alias = model_name
    _pending_single_model = {
        "model_path": resolved,
        "original_name": model_name,
        "scheduler_config": scheduler_config,
        "stream_interval": stream_interval,
        "served_model_name": served_model_name,
        "mtp": mtp,
        "force_text": force_text,
        "force_hybrid": force_hybrid,
        "no_hybrid": no_hybrid,
        "force_spec_decode": force_spec_decode,
        "no_spec_decode": no_spec_decode,
        "gpu_memory_utilization": gpu_memory_utilization,
        "cloud_model": cloud_model,
        "cloud_threshold": cloud_threshold,
        "cloud_api_base": cloud_api_base,
        "cloud_api_key": cloud_api_key,
        "max_tokens": max_tokens,
        "max_tokens_is_explicit": max_tokens_is_explicit,
        "lora_path": lora_path,
    }
    # Ensure the singleton Server + app exist so _startup will pick up the
    # staged model when uvicorn starts the lifespan.
    get_app()
    _sync_config()
    logger.info(
        "load_model: staged single model %s (resolved=%s, served=%s)",
        model_name,
        resolved,
        cfg.model_name,
    )


def resolve_model_id(model_id: str) -> str:
    """Resolve a model alias to its real ID."""
    from .config import DEFAULT_ALIASES

    resolved = DEFAULT_ALIASES.get(model_id)
    if resolved:
        return resolved
    # Only strip known provider prefixes — preserve HF paths
    for prefix in ["fusion-mlx/", "fusion/"]:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


def resolve_model_with_profile(model_id: str) -> tuple[str, dict[str, Any]]:
    """Resolve model:profile syntax into (resolved_model_id, profile_overrides).

    If model_id contains ':' and the suffix matches an exposed profile,
    returns the base model ID plus a dict of sampling overrides from the
    profile.  Otherwise returns (resolve_model_id(model_id), {}).

    This enables zero-extra-memory profile selection via API calls like:
        POST /v1/chat/completions  {"model": "qwen3:creative", ...}
    """
    if ":" not in model_id:
        return resolve_model_id(model_id), {}

    sm = _server_state.get("settings_manager")
    if sm is None:
        logger.debug(
            "resolve_model_with_profile: no settings_manager, stripping profile"
        )
        base = model_id.split(":", 1)[0]
        return resolve_model_id(base), {}

    result = sm.get_exposed_profile_runtime_settings_for_request(model_id)
    if result is not None:
        base_model_id, profile_settings = result
        overrides = {}
        for fname in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "repetition_penalty",
            "presence_penalty",
        ):
            val = getattr(profile_settings, fname, None)
            if val is not None:
                overrides[fname] = val
        logger.info(
            "resolve_model_with_profile: %s -> base=%s, overrides=%s",
            model_id,
            base_model_id,
            overrides,
        )
        return resolve_model_id(base_model_id), overrides

    # No profile match — treat whole string as model name (colon may be in model ID)
    return resolve_model_id(model_id), {}


def get_settings() -> Any:
    from .settings import Settings

    global _server_instance
    if _server_instance is not None:
        return _server_instance.settings
    return Settings()


def get_server() -> "Server | None":
    return _server_instance


async def init_mcp(config_path: str | None = None):
    """Initialize MCP manager from config path (standalone, test-friendly).

    Loads config, creates MCPClientManager, starts it, and wires the
    getter into mcp_routes. Safe to call multiple times — replaces the
    previous manager if any.
    """
    from .api.mcp_routes import set_mcp_manager_getter
    from .mcp import MCPClientManager, load_mcp_config

    try:
        mcp_config = load_mcp_config(config_path)
        if not mcp_config.servers:
            logger.info("init_mcp: no servers configured")
            return
        manager = MCPClientManager(mcp_config)
        await manager.start()
        set_mcp_manager_getter(lambda: manager)
        _server_state["mcp_manager"] = manager
        logger.info("init_mcp: %d servers started", len(mcp_config.servers))
    except FileNotFoundError:
        logger.info("init_mcp: config not found at %s", config_path)
    except ImportError as e:
        logger.info("init_mcp: MCP SDK not installed: %s", e)
    except Exception as e:
        logger.warning("init_mcp: failed: %s", e)


class Server:
    """Main fusion-mlx server with engine pool, routing, and API endpoints."""

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self.pool: EnginePool | None = None
        self.request_router: RequestRouter | None = None
        self.cloud_router: CloudRouter | None = None
        self.engine_cores: dict[str, AsyncEngineCore] = {}
        self._load_lock = asyncio.Lock()
        self._mdns = None
        self._cluster_lb_monitor = None  # #811 multi-instance LB health monitor
        self._training_services: list = []  # ENG-02: cleanup on shutdown
        self._startup_failures: list[str] = []  # ENG-01: track broken subsystems

        # R-7: resolve profile early — route registration in __init__ needs it.
        # Sync from global config singleton (set by _stage_server_config via
        # --profile flag) into self.config so profile_from_config sees it.
        from .config import get_config as _get_global_config
        from .profile import profile_from_config

        _gc = _get_global_config()
        if getattr(_gc, "profile", None) and not self.config.profile:
            self.config.profile = _gc.profile
        self._profile = profile_from_config(self.config)

        warnings.filterwarnings(
            "ignore",
            message="You are using a model of type .* to instantiate",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="resource_tracker: There appear to be .* leaked semaphore",
            category=UserWarning,
        )
        self.settings = Settings.load(Path(self.config.settings_dir) / "settings.json")

        # Daily-rotated file logging — writes {settings_dir}/logs/server.log so
        # the admin /admin/api/logs endpoint has content to serve. Appends a
        # file handler to the root logger; console logging is configured
        # separately by ``configure_logging`` from cli_serve. Best-effort: a
        # filesystem failure here must not block server startup.
        try:
            import os

            from .logging_config import configure_file_logging

            log_dir = Path(self.config.settings_dir) / "logs"
            _json = os.environ.get("FUSION_LOG_JSON", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            # OPS-P4-8 (#0907 audit): retention was hard-coded to the 7-day
            # default in configure_file_logging. Read it from env, then the
            # settings.json logging.retention_days field, so a long-lived
            # deployment can grow/shrink the rotation window without editing
            # code. Invalid values fall back to 7 loudly.
            _retention = 7
            _raw_retention = os.environ.get("FUSION_LOG_RETENTION_DAYS", "").strip()
            if not _raw_retention and isinstance(self.settings, object):
                try:
                    _cfg = getattr(self.settings, "as_dict", lambda: {})()
                    _retention_cfg = (_cfg.get("logging", {}) or {}).get(
                        "retention_days"
                    )
                    if _retention_cfg is not None:
                        _raw_retention = str(_retention_cfg)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "logging retention read from settings failed", exc_info=True
                    )
            if _raw_retention:
                try:
                    _retention = int(_raw_retention)
                    if _retention < 1:
                        raise ValueError
                except (ValueError, TypeError):
                    logger.warning(
                        "FUSION_LOG_RETENTION_DAYS/logging.retention_days=%r is "
                        "not a positive int; falling back to 7-day retention",
                        _raw_retention,
                    )
                    _retention = 7
            # OP-1: mirror the console JSON knob onto the file handler so the
            # rotated server.log stays consistent with the stream format.
            # OPS-FIX: level was hardcoded "INFO", ignoring --log-level DEBUG
            # set via cli_serve/settings.json. Read the root logger's effective
            # level so the file handler matches the console handler.
            import logging as _stdlib_logging

            _root_level = _stdlib_logging.getLogger().getEffectiveLevel()
            _file_level = (
                _stdlib_logging.getLevelName(_root_level) if _root_level > 0 else "INFO"
            )
            configure_file_logging(
                log_dir=log_dir,
                level=_file_level,
                retention_days=_retention,
                format_style="json" if _json else "standard",
            )
            logger.info("File logging enabled: %s", log_dir / "server.log")
        except Exception:
            logger.debug("configure_file_logging failed (non-fatal)", exc_info=True)

        from .admin.auth import set_api_key

        # Resolve the effective API key ONCE and sync to every read path
        # so the middleware and admin auth agree. Pre-fix this block ran
        # ``if self.settings.api_key: set_api_key(self.settings.api_key)``
        # which OVERWROTE the CLI/env key with the settings.json key AND
        # left ``self.settings.api_key`` (read by _get_configured_api_key
        # via global_settings_getter) as the settings.json value, so an
        # operator passing ``--api-key <X>`` still got 401 "Invalid API
        # key" because the /v1 middleware enforced the settings.json key.
        # Priority: CLI --api-key > FUSION_MLX_API_KEY env > settings.json.
        effective_key, key_source = _resolve_effective_api_key(
            argv_key=_api_key,
            settings_key=self.settings.api_key,
        )
        if effective_key:
            self.settings.api_key = effective_key
            set_api_key(effective_key)
            try:
                from .config import get_config

                get_config().api_key = effective_key
            except Exception:
                logger.debug("effective api_key sync to config failed", exc_info=True)
        logger.info(
            "auth: effective api_key source=%s configured=%s",
            key_source,
            bool(effective_key),
        )

        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            gen = self._lifespan()
            await gen.__anext__()
            try:
                yield
            finally:
                try:
                    await gen.__anext__()
                except StopAsyncIteration:
                    pass

        app = FastAPI(
            title="fusion-mlx",
            description="Unified local model management for Apple Silicon",
            version=__version__,
            lifespan=lifespan,
        )

        # CORS — wildcard by default for friendly single-machine UX.
        # ``configure_cors_from_env`` (called before Server init in the
        # serve flow) may pin this to specific origins via --cors-origins
        # or FUSION_MLX_CORS_ALLOW_ORIGINS; None falls back to ``*``. When
        # origins are wildcard, do NOT set allow_credentials=True (browser
        # spec forbids credentials + wildcard origins). Uses the L-02
        # spec-aligned subclass so preflight rejections are 200 + no ACAO
        # + Vary: Origin (not stock 400). The module-global ``app`` is not
        # bound yet here, so mount on the local instance directly.
        global _cors_mounted
        if not _cors_mounted:
            if _cors_origins == []:
                # Fail-closed: empty-CSV origins → mount nothing.
                _cors_mounted = True
            else:
                if _cors_origins:
                    cors_origins = _cors_origins
                    cors_origin_regex = None
                else:
                    # P2-25: localhost-only via regex (Starlette exact-match
                    # can't handle port wildcards).
                    cors_origins = []
                    cors_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
                cors_methods = _resolve_cors_methods()
                # P2-6: route credentials through _resolve_cors_credentials
                # so wildcard ["*"] origins force credentials=False per the
                # fetch spec. bool(_cors_origins) auto-enabled credentials on
                # any explicit origin and never guarded the wildcard case.
                app.add_middleware(
                    _SpecAlignedCORSMiddleware,
                    allow_origins=cors_origins,
                    allow_origin_regex=cors_origin_regex,
                    allow_methods=cors_methods,
                    allow_headers=_resolve_cors_headers(env_path=True),
                    allow_credentials=_resolve_cors_credentials(cors_origins),
                )
                _cors_mounted = True
                logger.debug(
                    "CORS middleware mounted in _create_app (spec-aligned): origins=%s methods=%s",
                    cors_origins,
                    cors_methods,
                )

        # Body-size and depth guards (ASGI-level, run before FastAPI routing).
        # Order matters: Starlette runs the LAST-added middleware outermost.
        # body_depth buffers the body into memory, so body_size MUST be
        # outermost (added last) to reject an oversized body with 413 BEFORE
        # body_depth receives a single chunk — otherwise a 10GiB flat JSON
        # body (depth 2, under the 64 cap) is buffered entirely and OOMs the
        # server before the 8MiB size cap fires (#807 P1-11).
        install_request_body_depth_middleware(app)
        install_request_body_limit_middleware(app)

        # ENG-08 (#0909 audit): auth pre-check runs BEFORE body_limit
        # reads the body. Installed after body_limit (outer to it) but
        # before request_id (inner to it) — so execution order is
        # request_id → auth_precheck → body_limit → body_depth. This
        # means a 401 carries a request-id, and the body is never
        # buffered for unauthenticated requests (closes the memory DoS
        # vector where a bad key + 7.9 MiB body is buffered before the
        # Depends() auth fires).
        install_auth_precheck_middleware(app)

        # Request-ID correlation — stamps the logging ContextVar per request
        # and echoes X-Request-Id on the response. Pure ASGI so the ContextVar
        # propagates into the handler's task.
        install_request_id_middleware(app)

        # #343: X-Fusion-Route source validation. Warn-only by default;
        # rejects 403 when FUSION_ROUTE_ENFORCE=true. Health probes and
        # CORS preflight stay exempt (handled inside the middleware).
        install_route_guard_middleware(app)

        # Probe fast-path (OUTERMOST — installed last so it runs first)
        install_probe_fastpath_middleware(app)

        # Unified exception handlers (OpenAI/Anthropic envelope shapes)
        install_exception_handlers(app)

        # R-7: profile-gated route registration. Each route maps to the
        # modality it requires; disabled modalities' routes are NOT mounted
        # (unreachable = no attack surface). Routes with modality=None always
        # mount (health, metrics, admin infra).
        _profile = self._profile
        _FORBIDDEN_UNTIL_FIXED: set[str] = set()
        # H2 stub: flywheel routes a fake runner. Until fixed, hard-ban even
        # in full profile — code-level prohibition, stronger than profile gate.
        _FORBIDDEN_UNTIL_FIXED.add("flywheel")

        _ROUTE_REGISTRY: list[tuple[str, Any, str | None]] = [
            ("ollama", ollama_router, "llm"),
            ("openai", openai_router, "llm"),
            ("anthropic", anthropic_router, "llm"),
            ("responses", responses_router, "llm"),
            ("audio", audio_router, "audio"),
            ("images", images_router, "image"),
            ("images_sr", images_sr_router, "image"),
            ("videos", videos_router, "video"),
            ("mcp", mcp_router, "mcp"),
            ("openclaw", openclaw_router, "agent"),
            ("agent", agent_router, "agent"),
            ("convert", convert_router, "tools"),
            ("watermark", watermark_router, "tools"),
            ("layered_quantize", layered_quantize_router, "llm"),
            ("distributed", distributed_router, "multitenant"),
            ("recommend", recommend_router, "llm"),
            ("bench", bench_router, "bench"),
            ("recommend_batch", recommend_batch_router, "llm"),
            ("flywheel", flywheel_router, "bench"),
            ("spec", spec_router, "llm"),
            ("embeddings", embeddings_router, "embedding"),
            ("rerank", rerank_router, "reranker"),
            ("ner", ner_router, "ner"),
            ("ocr", ocr_router, "ocr"),
            ("reasoning", reasoning_router, "llm"),
            ("sessions", sessions_router, "llm"),
            ("health_probe", health_probe_router, None),
            ("health", health_router, None),
            ("health_admin", health_admin_router, None),
            ("metrics", metrics_router, None),
            ("cache", cache_router, None),
            ("gc", gc_router, None),
            ("config_reload", config_reload_router, None),
            ("admin", admin_router, None),
            ("cluster", cluster_router, None),
        ]
        _mounted: list[str] = []
        _skipped: list[str] = []
        for _name, _router, _mod in _ROUTE_REGISTRY:
            if _name in _FORBIDDEN_UNTIL_FIXED:
                _skipped.append(f"{_name}(forbidden-stub)")
                continue
            if _mod is not None and not _profile.engine_allowed(_mod):
                _skipped.append(f"{_name}(modality={_mod})")
                continue
            app.include_router(_router)
            _mounted.append(_name)
        logger.info(
            "R-7 profile '%s': mounted %d routes [%s], skipped %d [%s]",
            _profile.name,
            len(_mounted),
            ",".join(_mounted),
            len(_skipped),
            ",".join(_skipped),
        )
        self._profile_mounted_routes = _mounted
        self._profile_skipped_routes = _skipped

        # #357: /v1/models/status MUST be registered before the gui_compat
        # router's /v1/models/{model_name} catch-all. Starlette matches routes
        # in registration order; a specific path shadowed by a parameter route
        # is unreachable (status was captured as model_name -> 404). Keep
        # specific /v1/models/* routes above the gui_compat include.
        @app.get("/v1/models/status")
        async def models_status(is_admin: bool = Depends(require_admin)):
            if self.pool is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            status = self.pool.get_status()
            n = len(status.get("models", [])) if isinstance(status, dict) else 0
            logger.debug("GET /v1/models/status -> %d models", n)
            return status

        # Register GUI compatibility router (discovery, settings, manager, admin UI)
        if get_gui_compat_router:
            app.include_router(get_gui_compat_router())

        # Stats endpoint (combined pool + metrics)
        @app.get("/stats")
        async def stats(is_admin: bool = Depends(require_admin)):
            pool_status = self.pool.get_status() if self.pool else {}
            metrics = get_server_metrics().to_dict()
            return {**pool_status, **metrics}

        # Issue #104: base-binding runtime info (version/Metal/quant/GPU) for
        # Fusion-Model-Hub. Separate from /stats so existing consumers are
        # unaffected; /stats stays the pool+metrics shape.
        @app.get("/v1/base")
        async def base_info(is_admin: bool = Depends(require_admin)):
            return _runtime_base_info(self.pool)

        @app.get("/api/status")
        async def api_status(is_admin: bool = Depends(require_admin)):
            from .pool.model_discovery import format_size

            metrics = get_server_metrics().to_dict()
            models_discovered = 0
            models_loaded = 0
            models_loading = 0
            loaded_models = []
            model_memory_used = 0
            model_memory_max = None
            if self.pool:
                models_discovered = self.pool.model_count
                models_loaded = self.pool.loaded_model_count
                loaded_models = self.pool.get_loaded_model_ids()
                model_memory_used = self.pool.current_model_memory
                enforcer = self.pool.process_memory_enforcer
                if enforcer:
                    try:
                        model_memory_max = enforcer.get_final_ceiling()
                    except Exception:
                        # #82: was a silent pass; log so a broken enforcer
                        # surfaces in debug instead of hiding wrong stats.
                        logger.debug("stats: get_final_ceiling failed", exc_info=True)
                # ENG-03 (#0909 audit): use public loading_count instead of
                # iterating pool._entries directly.
                models_loading = self.pool.loading_count
            return {
                "status": "ok",
                "version": __version__,
                "uptime_seconds": metrics.get("total_requests", 0),
                "models_discovered": models_discovered,
                "models_loaded": models_loaded,
                "models_loading": models_loading,
                "default_model": _server_state.get("default_model"),
                "loaded_models": loaded_models,
                "total_requests": metrics.get("total_requests", 0),
                "total_prompt_tokens": metrics.get("total_prompt_tokens", 0),
                "total_completion_tokens": metrics.get("total_tokens_generated", 0),
                "model_memory_used": model_memory_used,
                "model_memory_max": model_memory_max,
                "model_memory_used_formatted": (
                    format_size(model_memory_used) if model_memory_used else "0B"
                ),
                "model_memory_max_formatted": (
                    format_size(model_memory_max) if model_memory_max else "unlimited"
                ),
            }

        @app.get("/api/stats/alltime")
        async def api_stats_alltime(is_admin: bool = Depends(require_admin)):
            # E-30 (#811): /api/stats/alltime exposed aggregate request
            # counts, token throughput, model-load counts, and uptime
            # with no auth — reconnaissance goldmine for an attacker
            # profiling the instance. Sibling /stats (line 1086) is
            # already require_admin-gated; this was the outlier.
            return get_server_metrics().to_alltime_dict()

        @app.post("/v1/models/{model_id:path}/load")
        async def load_model_public(
            model_id: str,
            is_admin: bool = Depends(require_admin),
            _source: bool = Depends(require_model_hub_source),
        ):
            if self.pool is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            resolved = resolve_model_id(model_id)
            entry = self.pool.get_entry(resolved)
            if entry is None and "/" in resolved:
                hyphen = resolved.replace("/", "-")
                hyphen_entry = self.pool.get_entry(hyphen)
                if hyphen_entry is not None:
                    logger.debug(
                        "load: slash->hyphen resolve %s -> %s", resolved, hyphen
                    )
                    resolved = hyphen
                    entry = hyphen_entry
            if entry is None:
                raise HTTPException(
                    status_code=404, detail=f"Model not found: {model_id}"
                )
            if getattr(entry, "engine", None) is not None:
                return {
                    "status": "ok",
                    "model_id": model_id,
                    "message": f"Already loaded: {model_id}",
                }
            try:
                await self.pool.get_engine(resolved)
            except HTTPException:
                raise
            except (ModelLoadingError, ModelBusyError) as e:
                raise HTTPException(
                    status_code=503,
                    detail=str(e),
                    headers={"Retry-After": "5"},
                ) from e
            except (InsufficientMemoryError, ModelTooLargeError) as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except Exception as e:
                logger.exception("Load model failed: %s(%s)", type(e).__name__, e)
                raise HTTPException(status_code=500, detail="Internal server error")
            return {
                "status": "ok",
                "model_id": model_id,
                "message": f"Loaded {model_id}",
            }

        @app.post("/v1/models/{model_id:path}/unload")
        async def unload_model_public(
            model_id: str,
            is_admin: bool = Depends(require_admin),
            _source: bool = Depends(require_model_hub_source),
        ):
            if self.pool is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            resolved = resolve_model_id(model_id)
            entry = self.pool.get_entry(resolved)
            if entry is None and "/" in resolved:
                hyphen = resolved.replace("/", "-")
                hyphen_entry = self.pool.get_entry(hyphen)
                if hyphen_entry is not None:
                    logger.debug(
                        "unload: slash->hyphen resolve %s -> %s", resolved, hyphen
                    )
                    resolved = hyphen
                    entry = hyphen_entry
            if entry is None:
                raise HTTPException(
                    status_code=404, detail=f"Model not found: {model_id}"
                )
            if getattr(entry, "engine", None) is None:
                raise HTTPException(
                    status_code=400, detail=f"Model not loaded: {model_id}"
                )
            await self.pool.unload_engine_async(resolved)
            return {"status": "ok", "model_id": model_id}

        @app.post("/v1/set_default_model")
        async def set_default_model(
            request: dict, is_admin: bool = Depends(require_admin)
        ):
            # Issue #277: JSON-RPC-compatible mlx.set_model endpoint.
            # Accepts {"model": "<model_id>"} and sets the default model.
            model_id = request.get("model")
            if not model_id:
                raise HTTPException(status_code=400, detail="Missing 'model' field")
            resolved = resolve_model_id(model_id)
            entry = self.pool.get_entry(resolved) if self.pool else None
            if entry is None:
                raise HTTPException(
                    status_code=404, detail=f"Model not found: {model_id}"
                )
            _server_state["default_model"] = resolved
            logger.info("Default model set to: %s (requested: %s)", resolved, model_id)
            return {"status": "ok", "model": resolved}

        @app.post("/rpc")
        async def json_rpc_dispatch(
            request: dict, is_admin: bool = Depends(require_admin)
        ):
            # Issue #277: JSON-RPC 2.0 dispatch endpoint.
            # Supports: mlx.set_model, mlx.start, mlx.stop, mlx.status
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id")

            def rpc_result(result):
                resp = {"jsonrpc": "2.0", "result": result}
                if req_id is not None:
                    resp["id"] = req_id
                return resp

            def rpc_error(code, message):
                resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": code, "message": message},
                }
                if req_id is not None:
                    resp["id"] = req_id
                return resp

            if method == "mlx.set_model":
                model_id = params.get("model") if isinstance(params, dict) else None
                if not model_id:
                    return rpc_error(-32602, "Missing 'model' parameter")
                resolved = resolve_model_id(model_id)
                entry = self.pool.get_entry(resolved) if self.pool else None
                if entry is None:
                    return rpc_error(-32602, f"Model not found: {model_id}")
                _server_state["default_model"] = resolved
                logger.info(
                    "JSON-RPC mlx.set_model: %s (requested: %s)",
                    resolved,
                    model_id,
                )
                return rpc_result({"status": "ok", "model": resolved})

            elif method == "mlx.status":
                if self.pool is None:
                    return rpc_error(-32000, "Server not initialized")
                metrics = self.pool.get_metrics()
                return rpc_result(
                    {
                        "status": "ok",
                        "default_model": _server_state.get("default_model"),
                        "models_loaded": self.pool.loaded_model_count,
                        "models_discovered": self.pool.model_count,
                        "uptime_seconds": metrics.get("total_requests", 0),
                    }
                )

            elif method == "mlx.start":
                return rpc_error(
                    -32601,
                    "mlx.start not supported via JSON-RPC; use HTTP /v1/models/{id}/load",
                )

            elif method == "mlx.stop":
                return rpc_error(
                    -32601,
                    "mlx.stop not supported via JSON-RPC; use HTTP /v1/models/{id}/unload",
                )

            else:
                return rpc_error(-32601, f"Method not found: {method}")

        @app.get("/v1/node/load")
        async def node_load(is_admin: bool = Depends(require_admin)):
            # Issue #264: node-level load snapshot for Multi-Node cluster routing.
            return _node_load_snapshot(self.pool, self.config)

        return app

    def _convert_scheduler_config(self):
        """Convert ServerConfig.scheduler to scheduler SchedulerConfig.

        Carries every spec-decode field (dflash2/dspark/suffix) through to
        the engine pool. Previously this dropped dflash2_drafter_path etc,
        so --enable-dflash2 was silently ignored in --model-dir mode.
        """
        from .scheduler.config import SchedulerConfig as SchedConfig

        src = self.config.scheduler
        kw = dict(
            max_num_seqs=src.max_num_seqs,
            max_num_batched_tokens=src.max_num_batched_tokens,
            completion_batch_size=src.completion_batch_size,
            prefill_step_size=src.prefill_step_size,
            chunked_prefill=src.chunked_prefill_tokens > 0,
            model_name="",
        )
        for _f in (
            "spec_decode",
            "dflash_drafter_path",
            "dflash2_drafter_path",
            "dflash2_block_size",
            "dflash2_draft_bits",
            "dspark_drafter_path",
            "dspark_draft_quant_bits",
            "enable_suffix_decoding",
            "suffix_max_draft",
            "suffix_max_suffix_len",
            "suffix_min_confidence",
            "suffix_min_draft_len",
        ):
            if hasattr(src, _f):
                kw[_f] = getattr(src, _f)
        return SchedConfig(**kw)

    def run(self):
        """Start the server using uvicorn."""
        from ._cli_base import (
            _cleanup_uds_socket,
            _prepare_uds_socket,
            _uds_path_from_host,
        )

        uds_path = _uds_path_from_host(self.config.host)
        if uds_path is not None:
            uds_fd = _prepare_uds_socket(uds_path)
            logger.info("UDS listen mode: %s", uds_path)
            try:
                uvicorn.run(
                    self.app,
                    fd=uds_fd,
                    log_level="info",
                    timeout_graceful_shutdown=15,
                )
            finally:
                _cleanup_uds_socket(uds_path, fd=uds_fd)
        else:
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info",
                timeout_graceful_shutdown=15,
            )

    async def _lifespan(self):
        """Startup/shutdown lifecycle."""
        from ._parent_watchdog import (
            clear_crash_counter,
            install_signal_handlers,
            record_crash,
            remove_pid_file,
            stop_watchdog,
            write_exit_status,
            write_pid_file,
            write_status,
        )

        install_signal_handlers()
        # ENG-04: pass the port so the PID file is port-suffixed
        # (server.{port}.pid) for multi-instance safety.
        write_pid_file(port=self.config.port)
        write_status("starting")
        logger.info("fusion-mlx starting up...")
        _startup_ok = False
        try:
            await self._startup()
            write_status("running")
            clear_crash_counter()
            _install_sighup_reload()
            yield
            _startup_ok = True
        except Exception as exc:
            record_crash()
            try:
                from .telemetry import emit

                emit.error(
                    category="lifespan_failure",
                    phase="startup",
                    exc=exc,
                )
            except Exception:
                logger.debug("telemetry error emit failed", exc_info=True)
            write_status("crashed")
            write_exit_status("crash")
            raise
        finally:
            from .server_metrics import get_server_metrics

            # ENG-12 (#0909 audit): wrap flush_alltime in try/except so a
            # flush failure (disk full, serialization error) does NOT skip
            # the entire _shutdown() chain — MCP manager, EnginePool, mDNS,
            # prefix cache save, temp file cleanup would all be skipped.
            try:
                get_server_metrics().flush_alltime()
            except Exception:
                logger.error(
                    "flush_alltime() failed — continuing shutdown anyway "
                    "(ENG-12 fail-visible)",
                    exc_info=True,
                )
            # ENG-05 (#0909 audit): stop the watchdog before shutdown so it
            # does not fire SIGKILL during graceful request draining.
            stop_watchdog()
            # P1-10: run _shutdown even when _startup raised partway, so MCP
            # manager / engines / mDNS partly initialized get torn down. The
            # method internally None-checks every subsystem it touches.
            # OP-8 (#0907 audit): hard timeout so a slow teardown (engine
            # stop() hanging, in-flight requests not draining) cannot block
            # shutdown indefinitely. A rolling update / SIGTERM must let the
            # old instance exit within a bounded window. Env-tunable; 0
            # disables the guard (legacy behavior) for single-machine dev.
            import asyncio as _asyncio
            import os as _os

            _shutdown_timeout = float(_os.environ.get("FUSION_SHUTDOWN_TIMEOUT", "30"))
            try:
                if _shutdown_timeout > 0:
                    await _asyncio.wait_for(self._shutdown(), timeout=_shutdown_timeout)
                else:
                    await self._shutdown()
            except TimeoutError:
                logger.error(
                    "OP-8: graceful shutdown exceeded hard timeout %.1fs — "
                    "force-canceling remaining teardown (in-flight requests "
                    "may be dropped). Raise FUSION_SHUTDOWN_TIMEOUT if a "
                    "longer drain window is needed.",
                    _shutdown_timeout,
                )
            except Exception:
                logger.debug("shutdown after partial startup failed", exc_info=True)
            # P2-9: remove pid file AFTER shutdown so start.sh status still
            # reports running while the port is held during teardown.
            remove_pid_file()
        if _startup_ok:
            write_exit_status("clean")
            write_status("stopped")

    async def _startup(self):
        """Initialize engine pool, routers, and load models."""
        # Telemetry: check consent state at server startup so we can log
        # the current status for operators auditing their install.
        try:
            from fusion_mlx.telemetry import consent_source, is_enabled

            src = consent_source()
            enabled = is_enabled()
            logger.info("telemetry consent: enabled=%s source=%s", enabled, src)
        except Exception:
            logger.debug("telemetry consent check failed (non-fatal)", exc_info=True)

        # Set memory limit
        mem_cfg = self.config.memory
        if mem_cfg.ssd_cache_enabled:
            avail_mb = _available_ram_mb()
            limit_mb = (
                mem_cfg.cache_memory_mb
                if mem_cfg.cache_memory_mb
                else int(mem_cfg.cache_memory_percent * avail_mb)
            )
            if limit_mb > 0:
                mx.set_memory_limit(limit_mb)
                logger.info(
                    "MLX memory limit set to %d MB (available: %d MB)",
                    limit_mb,
                    avail_mb,
                )

        # Create engine pool with scheduler config from ServerConfig.
        # R-7: profile gate controls which modalities can load (resolved
        # in __init__ before route registration).
        self.pool = EnginePool(
            scheduler_config=self._convert_scheduler_config(),
            profile=self._profile,
        )

        # Create and wire memory enforcer
        tier_str = getattr(mem_cfg, "tier", "balanced")
        if hasattr(tier_str, "name"):
            tier_str = tier_str.name.lower()
        # E-9 (#811): the config-file custom tier (MemoryConfig.custom_limit_mb)
        # was silently ignored at construction — only the runtime admin route set
        # the custom ceiling. Pass it here so a custom tier in settings.json takes
        # effect at boot. custom_limit_mb is None when unset; leave 0 (disabled).
        custom_ceiling_gb = 0.0
        if tier_str == "custom" and getattr(mem_cfg, "custom_limit_mb", None):
            custom_ceiling_gb = float(mem_cfg.custom_limit_mb) / 1024.0
        self.pool.set_process_memory_enforcer(
            ProcessMemoryEnforcer(
                engine_pool=self.pool,
                memory_guard_tier=tier_str,
                soft_threshold=mem_cfg.soft_threshold,
                hard_threshold=mem_cfg.hard_threshold,
                memory_guard_custom_ceiling_gb=custom_ceiling_gb,
            )
        )
        self.pool.process_memory_enforcer.start()
        self.pool.set_final_ceiling_callback(
            self.pool.process_memory_enforcer.get_final_ceiling
        )
        # RC-3 (#811 audit 0906): every unload path funnels through
        # EnginePool._detach_engine, but only Server.unload_model popped
        # engine_cores — LRU eviction / admin route / TTL unloaded via
        # pool.unload_engine_async without popping, leaking a stale
        # AsyncEngineCore (holding MLX weights). Register a callback so the
        # pool keeps engine_cores in sync regardless of the trigger.
        self.pool.set_engine_detached_callback(self._drop_engine_core)

        # Populate _server_state so admin helpers that import it directly
        # (instead of using getter functions) can find engine_pool etc.
        _server_state["engine_pool"] = self.pool
        _server_state["process_memory_enforcer"] = self.pool.process_memory_enforcer
        # Initialize ModelSettingsManager for per-model settings + profiles
        settings_manager = None
        try:
            from .model_settings import ModelSettingsManager

            settings_path = Path(self.config.settings_dir)
            settings_manager = ModelSettingsManager(settings_path)
            logger.info("ModelSettingsManager initialized at %s", settings_path)
        except Exception as e:
            # ENG-01 (#0909 audit): surface as ERROR, not just warning —
            # a broken ModelSettingsManager silently degrades per-model
            # overrides and profiles.
            logger.error("Failed to initialize ModelSettingsManager: %s", e)
            self._startup_failures.append("ModelSettingsManager")

        _server_state["settings_manager"] = settings_manager
        self.pool.set_settings_manager(settings_manager)
        _server_state["default_model"] = None  # set when a model is marked default
        # Simple namespace for sampling defaults (read by admin helpers)
        import types

        _server_state["sampling"] = types.SimpleNamespace(
            max_context_window=getattr(self.config, "max_context_window", 4096),
            max_tokens=getattr(self.config, "max_tokens", 4096),
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
        )

        # Create request router
        # RT-12 (#0909 audit): pass cloud_fallback_consent so the
        # RequestRouter consent gate is consistent with SmartRouter.
        self.request_router = RequestRouter(
            cloud_fallback_consent=self.config.cloud_fallback_consent,
        )

        # Create cloud router if enabled
        if self.config.cloud_router_enabled:
            import os

            cloud_model = self.config.cloud_router_model or os.environ.get(
                "FUSION_CLOUD_MODEL"
            )
            if not cloud_model:
                # Fail visibly — CloudRouter requires a target model string.
                # Booting with --cloud-router but no --cloud-model used to crash
                # with a cryptic TypeError (#809 P0).
                raise RuntimeError(
                    "Cloud router enabled but no cloud model configured. "
                    "Pass --cloud-model <litellm-string> or set "
                    "FUSION_CLOUD_MODEL in the environment."
                )
            self.cloud_router = CloudRouter(
                cloud_model=cloud_model,
                api_key=self.config.cloud_router_api_key,
                threshold=self.config.cloud_router_threshold,
            )
            logger.info(
                "CloudRouter enabled: target=%s threshold=%d",
                cloud_model,
                self.config.cloud_router_threshold,
            )
            # Inject cloud_router into request_router so route_chat can
            # use it (was previously created but never connected).
            self.request_router.cloud_router = self.cloud_router

        # Inject context into route modules
        global _server_instance
        _server_instance = self
        # ARCH-02 (#0909 audit): removed duplicate set_ollama_context call
        # (was called at L1721 and L1723 — copy-paste bug exposing implicit
        # call-order dependency).
        set_ollama_context(self.pool)
        set_openai_context(self.pool, self.request_router)
        set_anthropic_context(self.pool)
        set_responses_context(self.pool)
        set_images_context(self.pool)
        set_images_sr_context(self.pool)
        set_videos_context(self.pool)
        set_audio_context(self.pool)
        set_openclaw_agent_pool(self.pool)
        set_mcp_manager_getter(lambda: None)  # placeholder, replaced below

        # Wire MCP client manager
        _mcp_manager = None
        try:
            from .mcp import MCPClientManager, load_mcp_config

            mcp_config = load_mcp_config()
            if mcp_config.servers:
                _mcp_manager = MCPClientManager(mcp_config)
                await _mcp_manager.start()
                set_mcp_manager_getter(lambda: _mcp_manager)
                logger.info(
                    "MCP manager started: %d servers configured",
                    len(mcp_config.servers),
                )
            else:
                logger.info("MCP: no servers configured, MCP disabled")
        except FileNotFoundError:
            logger.info("MCP: no config found, MCP disabled")
        except ImportError as e:
            logger.info("MCP SDK not installed, MCP disabled: %s", e)
        except Exception as e:
            # ENG-01 (#0909 audit): surface MCP init failure at ERROR —
            # a broken MCP manager means tool-calling routes silently fail.
            logger.error("MCP init failed: %s", e)
            self._startup_failures.append("MCP")
        _server_state["mcp_manager"] = _mcp_manager
        set_embeddings_context(self.pool, _server_state)
        set_rerank_context(self.pool, _server_state)
        set_ner_context(self.pool, _server_state)
        set_ocr_context(self.pool)
        set_reasoning_context(self.pool)
        set_sessions_context(self.pool, _server_state)
        set_models_context(self.pool)

        # Wire fine-tune service
        from .admin.fine_tune_route import set_fine_tune_context
        from .training.service import FineTuneService

        _fine_tune_svc = FineTuneService()
        _fine_tune_svc.set_engine_pool(self.pool)
        _fine_tune_svc.set_loop(asyncio.get_running_loop())
        set_fine_tune_context(self.pool, _fine_tune_svc)
        self._training_services.append(_fine_tune_svc)

        # Wire GRPO service (#363)
        from .admin.fine_tune_route import set_grpo_context
        from .training.grpo_service import GRPOService

        _grpo_svc = GRPOService()
        _grpo_svc.set_engine_pool(self.pool)
        _grpo_svc.set_loop(asyncio.get_running_loop())
        set_grpo_context(self.pool, _grpo_svc)
        self._training_services.append(_grpo_svc)

        # Wire RFT (rejection-sampling fine-tuning) service (#9)
        from .admin.fine_tune_route import set_rft_context
        from .training.rft_service import RFTService

        _rft_svc = RFTService()
        _rft_svc.set_engine_pool(self.pool)
        _rft_svc.set_loop(asyncio.get_running_loop())
        set_rft_context(self.pool, _rft_svc)
        self._training_services.append(_rft_svc)

        # Wire VLM (vision-language) fine-tune service (#797)
        from .admin.fine_tune_route import set_vlm_context
        from .training.vlm_service import VLMFineTuneService

        _vlm_svc = VLMFineTuneService()
        _vlm_svc.set_engine_pool(self.pool)
        _vlm_svc.set_loop(asyncio.get_running_loop())
        set_vlm_context(self.pool, _vlm_svc)
        self._training_services.append(_vlm_svc)

        # Wire DPO/ORPO service (#399)
        from .admin.fine_tune_route import set_dpo_context
        from .training.dpo_service import DPOService

        _dpo_svc = DPOService()
        _dpo_svc.set_engine_pool(self.pool)
        _dpo_svc.set_loop(asyncio.get_running_loop())
        set_dpo_context(self.pool, _dpo_svc)
        self._training_services.append(_dpo_svc)

        # Wire reward-model training service (#424)
        from .admin.fine_tune_route import set_reward_context
        from .training.reward_service import RewardService

        _reward_svc = RewardService()
        _reward_svc.set_engine_pool(self.pool)
        _reward_svc.set_loop(asyncio.get_running_loop())
        set_reward_context(self.pool, _reward_svc)
        self._training_services.append(_reward_svc)

        # Auto-add adapters dir to FUSION_LORA_ALLOWED_DIRS so trained
        # adapters can be served via EnginePool hot-swap without manual env config
        import os
        from pathlib import Path as _P

        _adapters_dir = str(_P.home() / ".fusion-mlx" / "adapters")
        _allowed = os.environ.get("FUSION_LORA_ALLOWED_DIRS", "")
        if _allowed:
            _dirs = [d.strip() for d in _allowed.split(":") if d.strip()]
        else:
            _dirs = []
        if _adapters_dir not in _dirs:
            _dirs.append(_adapters_dir)
            os.environ["FUSION_LORA_ALLOWED_DIRS"] = ":".join(_dirs)
            logger.info("Added %s to FUSION_LORA_ALLOWED_DIRS", _adapters_dir)

        # Wire admin getters so require_admin can access global settings/auth
        set_admin_getters(
            state_getter=lambda: _server_state,
            pool_getter=lambda: self.pool,
            settings_manager_getter=lambda: _server_state.get("settings_manager"),
            global_settings_getter=lambda: self.settings,
        )

        # Initialize HFDownloader so admin download routes work
        if self.config.model_dir:
            try:
                from .admin.hf_downloader import HFDownloader

                hf_dl = HFDownloader(model_dir=self.config.model_dir)
                set_hf_downloader(hf_dl)
                logger.info(
                    "HFDownloader initialized with model_dir=%s", self.config.model_dir
                )
            except Exception as e:
                # ENG-01 (#0909 audit): surface at ERROR — broken HFDownloader
                # means admin download routes silently fail.
                logger.error("Failed to initialize HFDownloader: %s", e)
                self._startup_failures.append("HFDownloader")

        # Initialize the oQ quantizer, ModelScope downloader, and HF uploader.
        # All three share a refresh callback that re-discovers models in the
        # pool after a download/quantization completes (mirrors fusion-mlx wiring).
        if self.config.model_dir:
            model_dirs = [self.config.model_dir]

            async def _refresh_models_after_task():
                if self.pool is None:
                    return
                await self.pool.discover_models_async(self.config.model_dir)
                logger.info("Model pool refreshed after admin task completion")

            # oQ Quantizer (always available — only needs mlx)
            try:
                from .admin.oq_manager import OQManager

                set_oq_manager(
                    OQManager(
                        model_dirs=model_dirs,
                        on_complete=_refresh_models_after_task,
                    )
                )
                logger.info("oQ Quantizer initialized")
            except Exception as e:
                # ENG-01 (#0909 audit): surface at ERROR — broken oQManager
                # means quantization routes silently fail.
                logger.error("Failed to initialize oQManager: %s", e)
                self._startup_failures.append("oQManager")

            # ModelScope downloader (requires modelscope SDK)
            try:
                from .admin.ms_downloader import MS_SDK_AVAILABLE, MSDownloader

                if MS_SDK_AVAILABLE:
                    set_ms_downloader(
                        MSDownloader(
                            model_dir=self.config.model_dir,
                            on_complete=_refresh_models_after_task,
                        )
                    )
                    logger.info("ModelScope Downloader initialized")
                else:
                    logger.info("ModelScope SDK not installed, MS downloader disabled")
            except Exception as e:
                logger.error("Failed to initialize MSDownloader: %s", e)
                self._startup_failures.append("MSDownloader")

            # HuggingFace uploader (requires huggingface_hub, lazy per-call)
            try:
                from .admin.hf_uploader import HFUploader

                set_hf_uploader(HFUploader(model_dirs=model_dirs))
                logger.info("HF Uploader initialized")
            except Exception as e:
                logger.error("Failed to initialize HFUploader: %s", e)
                self._startup_failures.append("HFUploader")

        # Apply model aliases
        aliases = {**self.config.model_aliases}
        if aliases:
            logger.info("Applied %d model aliases", len(aliases))

        # Auto-discover and register models in pool
        if self.config.model_dir:
            await self.pool.discover_models_async(self.config.model_dir)
            logger.info(
                "Discovered %d models in %s",
                self.pool.model_count,
                self.config.model_dir,
            )

        # Auto-pin locked embedding model so memory enforcer never evicts it
        locked_embed = self.config.embedding_model_locked
        if locked_embed:
            entry = self.pool.get_entry(locked_embed)
            if entry is not None:
                self.pool.set_pinned(locked_embed, True)
                logger.info(
                    "Embedding model %s pinned (embedding_model_locked)", locked_embed
                )
            else:
                logger.warning(
                    "embedding_model_locked=%s not found in pool", locked_embed
                )

        # Single-model ``serve --model <X>`` path: load_model() staged the
        # resolved model on ``_pending_single_model`` before uvicorn started.
        # The pool now exists, so load + register the engine via the same
        # AsyncEngineCore single-engine path the benchmark uses (preserves the
        # rich scheduler config: kv quant, prefix cache, spec-decode knobs).
        if _pending_single_model:
            await self._load_single_model(_pending_single_model)

        # Preload models from PRELOAD_MODELS env var or settings.json
        await self._preload_models()

        # Load prefix cache from disk (best-effort)
        try:
            from .runtime.cache import load_prefix_cache_from_disk

            await load_prefix_cache_from_disk()
        except Exception as e:
            # ENG-01 (#0909 audit): surface at WARNING (not debug) — a failed
            # prefix cache load means cold starts; operator should know.
            logger.warning("prefix cache load failed (non-fatal): %s", e)
            self._startup_failures.append("prefix_cache_load")

        # Initialize GUI database (for compat layer)
        if get_database_manager:
            try:
                get_database_manager()
                logger.info("GUI database initialized")
            except Exception as e:
                logger.warning(f"GUI database init failed (non-fatal): {e}")

        logger.info("fusion-mlx startup complete")

        # Security: warn if running without API key authentication
        try:
            from .middleware.auth import _get_configured_api_key

            if _get_configured_api_key() is None:
                logger.warning(
                    "SECURITY: No API key configured — all endpoints allow "
                    "anonymous access. Set FUSION_MLX_API_KEY env var or "
                    "api_key in config for production deployments."
                )
        except Exception:
            # ENG-01 (#0909 audit): was bare `pass` — a broken auth config
            # check should at least log, not silently disappear.
            logger.error(
                "Failed to check API key configuration for security warning",
                exc_info=True,
            )
            self._startup_failures.append("security_check")

        # mDNS/Bonjour cluster advertising (#264 part 2)
        if getattr(self.config, "cluster_advertise", False):
            # CL-5 (#811 audit 0906): this node serves HTTP plaintext on the
            # advertised port. Node-to-node prompt/completion traffic is
            # unencrypted — an on-subnet sniffer reads prompts. fusion-mlx
            # does not terminate TLS in-process (that is the reverse
            # proxy / fusion-gateway's job). Fail visibly: warn loudly so an
            # operator deploying cluster advertising on a shared subnet
            # knows to put the node behind a TLS-terminating gateway, or set
            # FUSION_CLUSTER_TLS_ACK to acknowledge the plaintext risk.
            if not os.environ.get("FUSION_CLUSTER_TLS_ACK", "").strip():
                logger.warning(
                    "CL-5 (#811 audit 0906): cluster advertising is ON but "
                    "this node serves plaintext HTTP — node-to-node prompt "
                    "traffic is unencrypted and sniffable on the subnet. "
                    "Deploy behind a TLS-terminating gateway (fusion-gateway) "
                    "or set FUSION_CLUSTER_TLS_ACK=1 to acknowledge the risk."
                )
            try:
                from .cluster.mdns import MdnsAdvertiser, build_txt_records

                snapshot = _node_load_snapshot(self.pool, self.config)
                txt = build_txt_records(snapshot)
                self._mdns = MdnsAdvertiser(
                    node_id=snapshot["node_id"],
                    host=snapshot["host"],
                    port=snapshot["port"],
                    txt_records=txt,
                )
                await self._mdns.start(
                    refresh_fn=lambda: _node_load_snapshot(self.pool, self.config)
                )
            except Exception:
                logger.warning(
                    "mDNS: advertising failed to start (non-fatal)", exc_info=True
                )

        # Multi-instance load balancing (#811) — OPT-IN. Bootstrap peers
        # into the NodeRegistry and start a health monitor, activating the
        # dormant cluster self-heal layer for single-host multi-port
        # deployments. With cluster_lb_enabled off (default), this is a no-op
        # and single-instance behavior is unchanged.
        if getattr(self.config, "cluster_lb_enabled", False):
            try:
                from .cluster.peer_lb import bootstrap_peers, start_health_monitor

                count = await bootstrap_peers(self.config)
                if count > 0:
                    self._cluster_lb_monitor = await start_health_monitor(
                        interval=float(
                            getattr(self.config, "cluster_lb_health_interval", 5.0)
                        ),
                        max_missed=int(
                            getattr(self.config, "cluster_lb_health_max_missed", 3)
                        ),
                    )
                    logger.info("cluster_lb (#811): activated with %d peer(s)", count)
                else:
                    logger.warning(
                        "cluster_lb (#811): enabled but no valid peers "
                        "configured — running in single-instance mode"
                    )
            except Exception:
                # ENG-01 (#0909 audit): surface at ERROR — a failed cluster
                # LB activation means multi-instance routing silently broken.
                logger.error(
                    "cluster_lb (#811): activation failed (non-fatal)",
                    exc_info=True,
                )
                self._startup_failures.append("cluster_lb")

        # ENG-01 (#0909 audit): if any subsystems failed during startup,
        # log a prominent summary so operators know what is broken before
        # the "startup complete" message. Previously these were all silent.
        if self._startup_failures:
            logger.warning(
                "startup completed with %d failed subsystem(s): %s — "
                "affected features may be unavailable",
                len(self._startup_failures),
                ", ".join(self._startup_failures),
            )

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("fusion-mlx shutting down...")

        # Stop MCP manager
        _mcp_mgr = _server_state.get("mcp_manager")
        if _mcp_mgr:
            try:
                await _mcp_mgr.stop()
                logger.info("MCP manager stopped")
            except Exception as e:
                logger.debug("MCP manager stop failed (non-fatal): %s", e)

        # mDNS: unregister service before teardown
        if self._mdns is not None:
            try:
                await self._mdns.stop()
                logger.info("mDNS: advertising stopped")
            except Exception:
                logger.debug("mDNS: stop failed (non-fatal)", exc_info=True)
            self._mdns = None

        # Multi-instance LB (#811): stop the health monitor
        if self._cluster_lb_monitor is not None:
            try:
                from .cluster.peer_lb import stop_health_monitor

                await stop_health_monitor(self._cluster_lb_monitor)
                logger.info("cluster_lb (#811): health monitor stopped")
            except Exception:
                logger.debug(
                    "cluster_lb: monitor stop failed (non-fatal)", exc_info=True
                )
            self._cluster_lb_monitor = None

        # Telemetry: fire the session_end hook registered by cli.py.
        # SIGTERM from systemd/Docker/K8s triggers FastAPI lifespan
        # shutdown, NOT atexit, so without this the session_end event
        # would be lost. The latch inside fire_session_end_hook makes
        # the second invocation (atexit fallback) a no-op.
        try:
            from fusion_mlx.telemetry.emit import fire_session_end_hook

            fire_session_end_hook()
        except Exception:
            logger.debug("telemetry session_end hook failed (non-fatal)", exc_info=True)

        # Cleanup GUI resources
        if close_database:
            try:
                from fusion_mlx.gui_compat.inference_queue_manager import (
                    shutdown_inference_manager,
                )
                from fusion_mlx.gui_compat.model_manager import shutdown_model_manager

                shutdown_inference_manager()
                shutdown_model_manager()
                close_database()
                logger.info("GUI resources cleaned up")
            except Exception as e:
                logger.warning(f"GUI cleanup warning: {e}")
        # ENG-02 (#0909 audit): cancel pending training tasks BEFORE
        # pool.shutdown so in-flight training jobs don't outlive the
        # event loop. Previously training services had no cleanup path.
        for svc in self._training_services:
            try:
                await svc.shutdown()
            except Exception:
                logger.debug(
                    "training service shutdown failed (non-fatal)",
                    exc_info=True,
                )
        self._training_services.clear()

        # ENG-02 (#0909 audit): wrap pool.shutdown() in try/except so a
        # failure here doesn't skip prefix cache save, temp cleanup, and
        # mx.clear_cache() — the entire shutdown chain must complete.
        if self.pool:
            try:
                await self.pool.shutdown()
            except Exception as e:
                logger.error(
                    "pool.shutdown() failed: %s — continuing shutdown "
                    "to save prefix cache and clean temp files (ENG-02)",
                    e,
                )

        # A-P1-5 (#0908 audit): save prefix cache AFTER pool.shutdown Phase 1
        # (abort+drain) completes — no in-flight requests mutating the cache
        # during save → consistent snapshot. The prefix cache is process-level
        # (survives engine unload), so saving post-shutdown is safe.
        try:
            from .runtime.cache import save_prefix_cache_to_disk

            await save_prefix_cache_to_disk()
        except Exception as e:
            logger.debug("prefix cache save failed (non-fatal): %s", e)
        try:
            from .utils.video import cleanup_all_temp_files

            cleaned = cleanup_all_temp_files()
            if cleaned:
                logger.info("Cleaned up %d temp video files on shutdown", cleaned)
        except Exception:
            logger.debug("temp video file cleanup failed (non-fatal)", exc_info=True)
        try:
            from ._tempfile_safe import _atexit_reap_all

            _atexit_reap_all()
        except Exception:
            logger.debug("tempfile_safe reap failed (non-fatal)", exc_info=True)
        mx.clear_cache()
        logger.info("fusion-mlx shutdown complete")

    async def load_model(self, model_id: str, **kwargs):
        """Dynamically load a model via the engine pool."""
        if self.pool is None:
            raise RuntimeError("Server not started")
        async with self._load_lock:
            resolved = resolve_model_id(model_id)
            engine = await self.pool.get_engine(resolved)
            logger.info(
                "Loaded model %s into pool (engine=%s)", model_id, type(engine).__name__
            )

    async def unload_model(self, model_id: str):
        """Unload a model from the pool."""
        core = self.engine_cores.pop(model_id, None)
        if core:
            await core.stop()
        if self.pool:
            self.pool.unload_engine(model_id)
        logger.info("Unloaded model %s from pool", model_id)

    def _drop_engine_core(self, model_id: str) -> None:
        # RC-3 (#811 audit 0906): pop is idempotent — unload_model already
        # pops and fires this on the same path, so a missing key is normal.
        self.engine_cores.pop(model_id, None)
        logger.debug("RC-3: engine_cores dropped detached model %s", model_id)

    async def _load_single_model(self, pending: dict) -> None:
        # Load the staged single model (``serve --model <X>``) into the pool.
        # Runs in _startup after the pool exists. Dispatches to the correct
        # engine type: DiffusionEngine for diffusion_gemma models,
        # BatchedEngine for everything else. Then registers it under the
        # served name + original name so routes resolve it via the pool
        # exactly like a discovered model.
        model_path = pending["model_path"]
        served = pending.get("served_model_name") or model_path
        scheduler_config = pending.get("scheduler_config")
        stream_interval = pending.get("stream_interval", 1)

        # Issue #256: detect diffusion_gemma from config and route to
        # DiffusionEngine instead of BatchedEngine.
        _is_diffusion = False
        try:
            import json
            from pathlib import Path

            cfg_path = Path(model_path) / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                _is_diffusion = (
                    cfg.get("model_type", "").lower().replace("-", "_")
                    == "diffusion_gemma"
                )
        except Exception:  # noqa: BLE001
            # FT-P4-4 (#0907 audit): silent pass here defaulted _is_diffusion
            # to False on any config read/parse error, so a diffusion model
            # whose config.json was corrupt/unreadable loaded via the wrong
            # engine (BatchedEngine) and failed later with a cryptic shape
            # error instead of a clear "could not classify" message. Log the
            # real cause so the operator can fix the config or move the file.
            logger.warning(
                "Failed to read/parse %s/config.json for diffusion "
                "classification — defaulting to non-diffusion (BatchedEngine). "
                "If this model is diffusion_gemma, it will fail to load; "
                "fix the config file or report the path.",
                model_path,
                exc_info=True,
            )

        logger.info(
            "Loading single model: %s (diffusion=%s)", model_path, _is_diffusion
        )
        if _is_diffusion:
            from .runtime.diffusion_lane import DiffusionEngine

            engine = DiffusionEngine(
                model_name=model_path,
                scheduler_config=scheduler_config,
            )
        else:
            from .engines.batched import BatchedEngine

            # Resolve per-model settings (dflash2_disabled, ttl, etc.) so
            # the single-model serve path gets the same settings as the pool
            # load_model path. Without this, model_settings is None and gates
            # like dflash2_disabled never fire for CLI-served models.
            _ms = None
            _sm = getattr(self.pool, "_settings_manager", None)
            if _sm is not None:
                _ms = _sm.get_settings(model_path)
            engine = BatchedEngine(
                model_name=model_path,
                scheduler_config=scheduler_config,
                stream_interval=stream_interval,
                model_settings=_ms,
                lora_path=pending.get("lora_path"),
            )
        await engine.start()
        self.pool.register_engine(served, engine)
        orig = pending.get("original_name")
        if orig and orig != served:
            self.pool.register_engine(orig, engine)
        # Track for unload/shutdown (unload_model calls core.stop()).
        self.engine_cores[served] = engine
        logger.info(
            "Single model registered: %s (engine=%s)", served, type(engine).__name__
        )

    async def _preload_models(self) -> None:
        import json as _json
        import os

        # Resolve preload list: PRELOAD_MODELS env > settings.json model.preload
        preload_str = os.environ.get("PRELOAD_MODELS", "").strip()
        if not preload_str:
            # Read model.preload from settings.json directly (same key layout
            # start.sh uses for model.model_dir)
            settings_path = Path(self.config.settings_dir) / "settings.json"
            if settings_path.exists():
                try:
                    raw = _json.loads(settings_path.read_text())
                    model_cfg = raw.get("model", {})
                    if isinstance(model_cfg, dict):
                        preload_val = model_cfg.get("preload")
                        if isinstance(preload_val, list):
                            preload_str = ",".join(preload_val)
                        elif isinstance(preload_val, str):
                            preload_str = preload_val
                except Exception as e:
                    logger.debug("Failed to read model.preload from settings: %s", e)
        if not preload_str:
            return

        model_ids = [m.strip() for m in preload_str.split(",") if m.strip()]
        if not model_ids:
            return

        _server_state["preloading"] = True
        logger.info("Preloading %d model(s): %s", len(model_ids), model_ids)

        loaded = []
        failed = []
        for model_id in model_ids:
            try:
                resolved = resolve_model_id(model_id)
                engine = await self.pool.get_engine(resolved)
                loaded.append(model_id)
                logger.info(
                    "Preloaded model %s (resolved=%s, engine=%s)",
                    model_id,
                    resolved,
                    type(engine).__name__,
                )
            except Exception as e:
                failed.append(model_id)
                logger.error(
                    "Failed to preload model %s: %s: %s",
                    model_id,
                    type(e).__name__,
                    e,
                )

        _server_state["preloading"] = False
        if loaded:
            logger.info(
                "Preload complete: %d loaded, %d failed", len(loaded), len(failed)
            )
        if failed:
            logger.warning("Preload failures: %s", failed)


def _available_ram_mb() -> int:
    """Get truly available system RAM in MB, using psutil."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        # Reserve 4 GB for OS + other processes as a safety margin
        return max(0, int(vm.available // (1024 * 1024)) - 4096)
    except Exception:
        return 16 * 1024  # fallback: 12 GB effective (16 - 4 GB reserve)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create the FastAPI app (convenience function for external use)."""
    server = Server(config)
    return server.app


def main():
    """CLI entry point for `fusion-mlx serve`."""
    import argparse

    parser = argparse.ArgumentParser(description="fusion-mlx server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=11434, help="Port")
    parser.add_argument("--model-dir", default=None, help="Model directory")
    parser.add_argument(
        "--memory-tier",
        choices=["safe", "balanced", "aggressive", "custom"],
        default="balanced",
        help="Memory enforcement tier",
    )
    parser.add_argument(
        "--ssd-cache", action="store_true", help="Enable SSD cold layer"
    )
    parser.add_argument(
        "--cloud-router", action="store_true", help="Enable cloud fallback"
    )
    parser.add_argument("--cloud-api-key", default=None, help="Cloud router API key")
    parser.add_argument(
        "--cloud-model",
        default=None,
        help="Cloud router target model (litellm string, e.g. anthropic/claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for authentication (if not set, falls back to the "
            "FUSION_MLX_API_KEY env var; if neither, no auth required)"
        ),
    )
    args = parser.parse_args()

    # Route both the standalone entry and the shared serve command through
    # _resolve_api_key so env-only sidecars keep the bearer out of argv/ps.
    global _api_key
    _api_key = _resolve_api_key(args.api_key)
    logger.info(
        "server entry: api_key resolved (source=%s)",
        "argv" if args.api_key else "env" if _api_key else "none",
    )

    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
    )
    config.memory.tier = getattr(
        config.memory.tier.__class__, args.memory_tier, config.memory.tier
    )
    config.memory.ssd_cache_enabled = args.ssd_cache
    config.cloud_router_enabled = args.cloud_router
    if args.cloud_api_key:
        config.cloud_router_api_key = args.cloud_api_key
    if args.cloud_model:
        config.cloud_router_model = args.cloud_model

    server = Server(config)
    server.run()


if __name__ == "__main__":
    main()
