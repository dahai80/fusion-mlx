"""FastAPI server for fusion-mlx.

Wires together all API routes:
- OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models
- Anthropic-compatible: /v1/messages, /v1/count_tokens
- Audio: /v1/audio/transcriptions, /v1/audio/speech, /v1/audio/process
- Images: /v1/images/generate
- MCP: /v1/mcp/tools, /v1/mcp/servers, /v1/mcp/execute
- OpenClaw Agent: /v1/openclaw/agent/*
- Admin: /admin/*
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
from .api.anthropic_routes import router as anthropic_router
from .api.anthropic_routes import set_anthropic_context
from .api.audio_routes import router as audio_router
from .api.audio_routes import set_audio_context
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
)

# GUI compatibility layer
try:
    from fusion_gui.database import close_database, get_database_manager
    from fusion_gui.server import get_gui_compat_router
except ImportError:
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
from .api.openai_routes import router as openai_router
from .api.openai_routes import set_openai_context
from .api.openclaw_routes import router as openclaw_router
from .api.openclaw_routes import set_openclaw_agent_pool
from .api.recommend_routes import router as recommend_router
from .api.rerank_routes import router as rerank_router
from .api.rerank_routes import set_rerank_context
from .api.videos_routes import router as videos_router
from .api.videos_routes import set_videos_context
from .config import ServerConfig
from .engine_core import AsyncEngineCore
from .pool import EnginePool, ProcessMemoryEnforcer
from .router import CloudRouter, RequestRouter
from .routes.cache import router as cache_router
from .routes.health import probe_router as health_probe_router
from .routes.health import router as health_router
from .routes.metrics import router as metrics_router
from .routes.responses import router as responses_router
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
_model_alias: str | None = None
_model_name: str | None = None
_model_path: str | None = None
_default_timeout: float = 1800.0
_max_request_bytes: int | None = None
_rate_limiter = None
_gc_control: bool = True
_no_thinking: bool = False
_pin_system_prompt: bool = False
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None
_enable_tool_logits_bias: bool = False
_default_temperature: float | None = None
_default_top_p: float | None = None
_default_top_k: int | None = None
_default_min_p: float | None = None
_default_repetition_penalty: float | None = None
_default_presence_penalty: float | None = None
_default_frequency_penalty: float | None = None
_reasoning_parser = None
_reasoning_parser_name: str | None = None
_enable_audio_lane: bool = False
_sse_keepalive_seconds: float = 0.0
# Staged single-model request from ``serve --model <X>``. ``load_model``
# populates this before uvicorn starts; ``Server._startup`` loads + registers
# the engine into the pool once the pool exists. None on the multi-model
# ``--model-dir`` path (which discovers into the pool directly).
_pending_single_model: dict | None = None


def _sync_config() -> None:
    # Copy staged server globals into the config singleton + auth. Bridges the
    # global-variable staging pattern (cli_serve sets ``server._api_key`` etc.
    # before uvicorn starts) with the config object middleware reads. Idempotent
    # — every assignment is a straight overwrite. Best-effort: ServerConfig
    # only carries a subset of these fields, so guard each with hasattr.
    try:
        from .config import get_config

        cfg = get_config()
    except Exception:
        cfg = None
    if cfg is not None:
        for _attr, _val in (
            ("model_name", _model_name),
            ("model_path", _model_path),
            ("model_alias", _model_alias),
            ("api_key", _api_key),
            ("max_request_bytes", _max_request_bytes),
            ("default_timeout", _default_timeout),
            ("gc_control", _gc_control),
            ("no_thinking", _no_thinking),
            ("pin_system_prompt", _pin_system_prompt),
            ("enable_auto_tool_choice", _enable_auto_tool_choice),
            ("tool_call_parser", _tool_call_parser),
            ("enable_tool_logits_bias", _enable_tool_logits_bias),
            ("reasoning_parser", _reasoning_parser),
            ("reasoning_parser_name", _reasoning_parser_name),
            ("enable_audio_lane", _enable_audio_lane),
            ("sse_keepalive_seconds", _sse_keepalive_seconds),
            ("default_temperature", _default_temperature),
            ("default_top_p", _default_top_p),
            ("default_top_k", _default_top_k),
            ("default_min_p", _default_min_p),
            ("default_repetition_penalty", _default_repetition_penalty),
            ("default_presence_penalty", _default_presence_penalty),
            ("default_frequency_penalty", _default_frequency_penalty),
            # rate_limiter is intentionally excluded: it is a module-level
            # singleton in middleware/auth.py (configure_rate_limiter mutates
            # it in place and returns it), NOT a ServerConfig field. The
            # previous ("rate_limiter", _rate_limiter) entry was dead code --
            # hasattr(cfg, "rate_limiter") was always False so it never synced.
        ):
            if hasattr(cfg, _attr):
                try:
                    setattr(cfg, _attr, _val)
                except Exception:
                    # #69: setattr failures mean config silently drifts out of
                    # sync (frozen dataclass / property setter). Warn so the
                    # drift is visible, not buried at debug.
                    logger.warning(
                        "_sync_config: setattr %s failed (config may drift)",
                        _attr,
                        exc_info=True,
                    )
    # Auth: propagate ``--api-key`` to the admin.auth module global the
    # middleware checks. Settings.json api_key is wired in Server.__init__;
    # this covers the CLI override for the single-model ``serve --model`` path.
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


def load_embedding_model(*args, **kwargs):
    raise NotImplementedError("Embedding models not available in this build")


def get_app():
    global _server_instance, app
    if _server_instance is None:
        _server_instance = Server()
    if app is None:
        app = _server_instance.app
    return app


def _resolve_single_model_path(name: str) -> str:
    # Resolve a model name to a loadable path/id. Reuses the omlx
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
        home / ".omlx" / "models" / "mlx-community" / resolved,
        home / ".omlx" / "models" / resolved,
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
    global _model_name, _model_path, _model_alias, _pending_single_model

    resolved = _resolve_single_model_path(model_name)
    _model_path = resolved
    _model_name = served_model_name or resolved
    if not _model_alias:
        _model_alias = model_name
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
        _model_name,
    )


def resolve_model_id(model_id: str) -> str:
    """Resolve a model alias to its real ID."""
    from .config import DEFAULT_ALIASES

    resolved = DEFAULT_ALIASES.get(model_id)
    if resolved:
        return resolved
    # Only strip known provider prefixes — preserve HF paths
    for prefix in ["omlx/", "fusion/"]:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


def get_settings() -> Any:
    from .settings import Settings

    global _server_instance
    if _server_instance is not None:
        return _server_instance.settings
    return Settings()


def get_server() -> "Server | None":
    return _server_instance


class Server:
    """Main fusion-mlx server with engine pool, routing, and API endpoints."""

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self.pool: EnginePool | None = None
        self.request_router: RequestRouter | None = None
        self.cloud_router: CloudRouter | None = None
        self.engine_cores: dict[str, AsyncEngineCore] = {}
        self._load_lock = asyncio.Lock()

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
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins if _cors_origins else ["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Body-size and depth guards (ASGI-level, run before FastAPI routing)
        install_request_body_limit_middleware(app)
        install_request_body_depth_middleware(app)

        # Request-ID correlation — stamps the logging ContextVar per request
        # and echoes X-Request-Id on the response. Pure ASGI so the ContextVar
        # propagates into the handler's task.
        install_request_id_middleware(app)

        # Probe fast-path (OUTERMOST — installed last so it runs first)
        install_probe_fastpath_middleware(app)

        # Unified exception handlers (OpenAI/Anthropic envelope shapes)
        install_exception_handlers(app)

        # Register all route modules
        app.include_router(openai_router)
        app.include_router(anthropic_router)
        app.include_router(audio_router)
        app.include_router(images_router)
        app.include_router(videos_router)
        app.include_router(mcp_router)
        app.include_router(openclaw_router)
        app.include_router(recommend_router)
        app.include_router(embeddings_router)
        app.include_router(rerank_router)
        app.include_router(responses_router)
        app.include_router(health_probe_router)
        app.include_router(health_router)
        app.include_router(metrics_router)
        app.include_router(cache_router)
        app.include_router(admin_router)

        # Register GUI compatibility router (discovery, settings, manager, admin UI)
        if get_gui_compat_router:
            app.include_router(get_gui_compat_router())

        # Stats endpoint (combined pool + metrics)
        @app.get("/stats")
        async def stats():
            pool_status = self.pool.get_status() if self.pool else {}
            metrics = get_server_metrics().to_dict()
            return {**pool_status, **metrics}

        @app.get("/api/status")
        async def api_status():
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

        @app.get("/v1/models/status")
        async def models_status():
            if self.pool is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            status = self.pool.get_status()
            return status

        @app.post("/v1/models/{model_id}/load")
        async def load_model_public(
            model_id: str, is_admin: bool = Depends(require_admin)
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
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
            return {
                "status": "ok",
                "model_id": model_id,
                "message": f"Loaded {model_id}",
            }

        @app.post("/v1/models/{model_id}/unload")
        async def unload_model_public(
            model_id: str, is_admin: bool = Depends(require_admin)
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
            await self.pool._unload_engine(resolved)
            return {"status": "ok", "model_id": model_id}

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

    @asynccontextmanager
    def run(self):
        """Start the server using uvicorn."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )

    async def _lifespan(self):
        """Startup/shutdown lifecycle."""
        logger.info("fusion-mlx starting up...")
        await self._startup()
        yield
        await self._shutdown()

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
        set_openai_context(self.pool, self.request_router)
        set_anthropic_context(self.pool)
        set_images_context(self.pool)
        set_videos_context(self.pool)
        set_audio_context(self.pool)
        set_openclaw_agent_pool(self.pool)
        set_mcp_manager_getter(lambda: None)  # TODO: wire MCP manager
        set_embeddings_context(self.pool, _server_state)
        set_rerank_context(self.pool, _server_state)

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
        # pool after a download/quantization completes (mirrors omlx wiring).
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

        # Single-model ``serve --model <X>`` path: load_model() staged the
        # resolved model on ``_pending_single_model`` before uvicorn started.
        # The pool now exists, so load + register the engine via the same
        # AsyncEngineCore single-engine path the benchmark uses (preserves the
        # rich scheduler config: kv quant, prefix cache, spec-decode knobs).
        if _pending_single_model:
            await self._load_single_model(_pending_single_model)

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

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("fusion-mlx shutting down...")

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
                from fusion_gui.inference_queue_manager import (
                    shutdown_inference_manager,
                )
                from fusion_gui.model_manager import shutdown_model_manager

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
        # Runs in _startup after the pool exists. Uses BatchedEngine — the
        # same wrapper the multi-model serve path uses — so the engine
        # exposes .chat()/.stream_chat() (which /v1/* routes call) and
        # inherits the rich scheduler config, TurboQuant KV, and spec-decode
        # wiring. Then registers it under the served name + original name so
        # routes resolve it via the pool exactly like a discovered model.
        from .engines.batched import BatchedEngine

        model_path = pending["model_path"]
        served = pending.get("served_model_name") or model_path
        scheduler_config = pending.get("scheduler_config")
        stream_interval = pending.get("stream_interval", 1)
        logger.info("Loading single model: %s", model_path)
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
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
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
    args = parser.parse_args()

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
