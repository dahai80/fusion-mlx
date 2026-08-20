"""FastAPI server for fusion-mlx.

Wires together all API routes:
- OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models
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
from .api.images import router as images_router
from .api.images import set_images_context
from .api.mcp_routes import router as mcp_router
from .api.mcp_routes import set_mcp_manager_getter
from .exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelTooLargeError,
)
from .middleware import (
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
from .api.embeddings_routes import router as embeddings_router
from .api.embeddings_routes import set_embeddings_context
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
from .api.recommend_routes import router as recommend_router
from .api.rerank_routes import router as rerank_router
from .api.rerank_routes import set_rerank_context
from .api.session_routes import router as sessions_router
from .api.session_routes import set_sessions_context
from .api.spec_routes import router as spec_router
from .api.videos_routes import router as videos_router
from .api.videos_routes import set_videos_context
from .config import ServerConfig
from .dispatch import CloudRouter, RequestRouter
from .engine_core import AsyncEngineCore
from .pool import EnginePool, ProcessMemoryEnforcer
from .routes_internal.cache import router as cache_router
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


class _ServerState(dict):
    """Dict subclass that also supports attribute access for admin helpers."""

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
    """
    from .logging_config import configure_logging as _configure_logging

    _configure_logging(level=log_level)
    return log_level.upper()


def _resolve_api_key(argv_api_key: str | None = None) -> str | None:
    global _api_key
    import os

    if argv_api_key:
        return argv_api_key
    if _api_key:
        return _api_key
    return os.environ.get("FUSION_MLX_API_KEY")


_cors_origins: list[str] | None = None


def _resolve_cors_origins(cors_origins) -> list[str] | None:
    import os

    if cors_origins:
        origins = [o.strip() for o in cors_origins if o and o.strip()]
        if origins:
            return origins
    env_raw = os.environ.get("FUSION_MLX_CORS_ALLOW_ORIGINS", "").strip()
    if env_raw:
        origins = [o.strip() for o in env_raw.split(",") if o.strip()]
        if origins:
            return origins
    return None


def configure_cors_from_env(cors_origins=None):
    global _cors_origins
    _cors_origins = _resolve_cors_origins(cors_origins)
    if _cors_origins:
        logger.info("CORS origins pinned to: %s", ", ".join(_cors_origins))
    else:
        logger.debug("CORS origins defaulting to wildcard '*'")
    return _cors_origins


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
            from .logging_config import configure_file_logging

            log_dir = Path(self.config.settings_dir) / "logs"
            configure_file_logging(log_dir=log_dir, level="INFO")
            logger.info("File logging enabled: %s", log_dir / "server.log")
        except Exception:
            logger.debug("configure_file_logging failed (non-fatal)", exc_info=True)

        from .admin.auth import set_api_key

        if self.settings.api_key:
            set_api_key(self.settings.api_key)

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
        # or FUSION_MLX_CORS_ALLOW_ORIGINS; None falls back to ``*``.
        # When origins are wildcard, do NOT set allow_credentials=True
        # (browser spec forbids credentials + wildcard origins).
        cors_origins = _cors_origins if _cors_origins else ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=bool(_cors_origins),
        )

        # Body-size and depth guards (ASGI-level, run before FastAPI routing)
        install_request_body_limit_middleware(app)
        install_request_body_depth_middleware(app)

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

        # Register all route modules
        app.include_router(ollama_router)
        app.include_router(openai_router)
        app.include_router(anthropic_router)
        app.include_router(audio_router)
        app.include_router(images_router)
        app.include_router(videos_router)
        app.include_router(mcp_router)
        app.include_router(openclaw_router)
        app.include_router(agent_router)
        app.include_router(convert_router)
        app.include_router(recommend_router)
        app.include_router(spec_router)
        app.include_router(embeddings_router)
        app.include_router(rerank_router)
        app.include_router(ner_router)
        app.include_router(ocr_router)
        app.include_router(reasoning_router)
        app.include_router(sessions_router)
        app.include_router(responses_router)
        app.include_router(health_probe_router)
        app.include_router(health_router)
        app.include_router(health_admin_router)
        app.include_router(metrics_router)
        app.include_router(cache_router)
        app.include_router(gc_router)
        app.include_router(admin_router)

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
                enforcer = self.pool._process_memory_enforcer
                if enforcer:
                    try:
                        model_memory_max = enforcer.get_final_ceiling()
                    except Exception:
                        # #82: was a silent pass; log so a broken enforcer
                        # surfaces in debug instead of hiding wrong stats.
                        logger.debug("stats: get_final_ceiling failed", exc_info=True)
                for entry in self.pool._entries.values():
                    if getattr(entry, "is_loading", False):
                        models_loading += 1
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
        async def api_stats_alltime():
            return get_server_metrics().to_alltime_dict()

        @app.post("/v1/models/{model_id}/load")
        async def load_model_public(
            model_id: str,
            is_admin: bool = Depends(require_admin),
            _source: bool = Depends(require_model_hub_source),
        ):
            if self.pool is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            resolved = resolve_model_id(model_id)
            entry = self.pool.get_entry(resolved)
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

        @app.post("/v1/models/{model_id}/unload")
        async def unload_model_public(
            model_id: str,
            is_admin: bool = Depends(require_admin),
            _source: bool = Depends(require_model_hub_source),
        ):
            if self.pool is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            resolved = resolve_model_id(model_id)
            entry = self.pool.get_entry(resolved)
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
        """Convert ServerConfig.scheduler to scheduler SchedulerConfig."""
        from .scheduler.config import SchedulerConfig as SchedConfig

        src = self.config.scheduler
        return SchedConfig(
            max_num_seqs=src.max_num_seqs,
            max_num_batched_tokens=src.max_num_batched_tokens,
            completion_batch_size=src.completion_batch_size,
            prefill_step_size=src.prefill_step_size,
            chunked_prefill=src.chunked_prefill_tokens > 0,
            model_name="",
        )

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
            write_exit_status,
            write_pid_file,
            write_status,
        )

        install_signal_handlers()
        write_pid_file()
        write_status("starting")
        logger.info("fusion-mlx starting up...")
        try:
            await self._startup()
            write_status("running")
            clear_crash_counter()
            yield
        except Exception:
            record_crash()
            write_status("crashed")
            write_exit_status("crash")
            raise
        finally:
            from .server_metrics import get_server_metrics

            get_server_metrics().flush_alltime()
            remove_pid_file()
        await self._shutdown()
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

        # Create engine pool with scheduler config from ServerConfig
        self.pool = EnginePool(scheduler_config=self._convert_scheduler_config())

        # Create and wire memory enforcer
        tier_str = getattr(mem_cfg, "tier", "balanced")
        if hasattr(tier_str, "name"):
            tier_str = tier_str.name.lower()
        self.pool._process_memory_enforcer = ProcessMemoryEnforcer(
            engine_pool=self.pool,
            memory_guard_tier=tier_str,
            soft_threshold=mem_cfg.soft_threshold,
            hard_threshold=mem_cfg.hard_threshold,
        )
        self.pool._process_memory_enforcer.start()
        self.pool._get_final_ceiling = (
            self.pool._process_memory_enforcer.get_final_ceiling
        )

        # Populate _server_state so admin helpers that import it directly
        # (instead of using getter functions) can find engine_pool etc.
        _server_state["engine_pool"] = self.pool
        _server_state["process_memory_enforcer"] = self.pool._process_memory_enforcer
        # Initialize ModelSettingsManager for per-model settings + profiles
        settings_manager = None
        try:
            from .model_settings import ModelSettingsManager

            settings_path = Path(self.config.settings_dir)
            settings_manager = ModelSettingsManager(settings_path)
            logger.info("ModelSettingsManager initialized at %s", settings_path)
        except Exception as e:
            logger.warning("Failed to initialize ModelSettingsManager: %s", e)

        _server_state["settings_manager"] = settings_manager
        self.pool._settings_manager = settings_manager
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
        self.request_router = RequestRouter()

        # Create cloud router if enabled
        if self.config.cloud_router_enabled:
            self.cloud_router = CloudRouter(
                api_key=self.config.cloud_router_api_key,
                threshold=self.config.cloud_router_threshold,
            )

        # Inject context into route modules
        global _server_instance
        _server_instance = self
        set_ollama_context(self.pool)
        set_openai_context(self.pool, self.request_router)
        set_ollama_context(self.pool)
        set_anthropic_context(self.pool)
        set_responses_context(self.pool)
        set_images_context(self.pool)
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
            logger.warning("MCP init failed: %s", e)
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

        # Wire GRPO service (#363)
        from .admin.fine_tune_route import set_grpo_context
        from .training.grpo_service import GRPOService

        _grpo_svc = GRPOService()
        _grpo_svc.set_engine_pool(self.pool)
        _grpo_svc.set_loop(asyncio.get_running_loop())
        set_grpo_context(self.pool, _grpo_svc)

        # Wire DPO/ORPO service (#399)
        from .admin.fine_tune_route import set_dpo_context
        from .training.dpo_service import DPOService

        _dpo_svc = DPOService()
        _dpo_svc.set_engine_pool(self.pool)
        _dpo_svc.set_loop(asyncio.get_running_loop())
        set_dpo_context(self.pool, _dpo_svc)

        # Wire reward-model training service (#424)
        from .admin.fine_tune_route import set_reward_context
        from .training.reward_service import RewardService

        _reward_svc = RewardService()
        _reward_svc.set_engine_pool(self.pool)
        _reward_svc.set_loop(asyncio.get_running_loop())
        set_reward_context(self.pool, _reward_svc)

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
                logger.warning("Failed to initialize HFDownloader: %s", e)

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
                logger.warning("Failed to initialize oQManager: %s", e)

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
                logger.warning("Failed to initialize MSDownloader: %s", e)

            # HuggingFace uploader (requires huggingface_hub, lazy per-call)
            try:
                from .admin.hf_uploader import HFUploader

                set_hf_uploader(HFUploader(model_dirs=model_dirs))
                logger.info("HF Uploader initialized")
            except Exception as e:
                logger.warning("Failed to initialize HFUploader: %s", e)

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

            load_prefix_cache_from_disk()
        except Exception as e:
            logger.debug("prefix cache load failed (non-fatal): %s", e)

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
            pass

        # mDNS/Bonjour cluster advertising (#264 part 2)
        if getattr(self.config, "cluster_advertise", False):
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

        # Save prefix cache to disk (best-effort, budget-aware)
        try:
            from .runtime.cache import save_prefix_cache_to_disk

            save_prefix_cache_to_disk()
        except Exception as e:
            logger.debug("prefix cache save failed (non-fatal): %s", e)

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
        if self.pool:
            await self.pool.shutdown()
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
        except Exception:
            pass

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

            engine = BatchedEngine(
                model_name=model_path,
                scheduler_config=scheduler_config,
                stream_interval=stream_interval,
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

    server = Server(config)
    server.run()


if __name__ == "__main__":
    main()
