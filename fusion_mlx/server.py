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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .admin.routes import router as admin_router
from .api.anthropic_routes import router as anthropic_router
from .api.anthropic_routes import set_anthropic_context
from .api.audio_routes import router as audio_router
from .api.images import router as images_router
from .api.images import set_images_context
from .api.mcp_routes import router as mcp_router
from .api.mcp_routes import set_mcp_manager_getter
# GUI compatibility layer
try:
    from fusion_gui.server import get_gui_compat_router
    from fusion_gui.database import get_database_manager, close_database
except ImportError:
    get_gui_compat_router = None
    get_database_manager = None
    close_database = None

# Import route modules
from .api.openai_routes import router as openai_router
from .api.openai_routes import set_openai_context
from .api.openclaw_routes import router as openclaw_router
from .api.openclaw_routes import set_openclaw_agent_pool
from .api.recommend_routes import router as recommend_router
from .config import SchedulerConfig as FusionSchedulerConfig
from .config import ServerConfig
from .engine_core import AsyncEngineCore, EngineConfig
from .pool import EnginePool, ProcessMemoryEnforcer
from .router import CloudRouter, RequestRouter
from .server_metrics import get_server_metrics
from .settings import Settings

logger = logging.getLogger(__name__)

_server_state: dict[str, Any] = {}


def resolve_model_id(model_id: str) -> str:
    """Resolve a model alias to its real ID."""
    from .config import DEFAULT_ALIASES
    resolved = DEFAULT_ALIASES.get(model_id)
    if resolved:
        return resolved
     # Only strip known provider prefixes — preserve HF paths
    for prefix in ["omlx/", "fusion/"]:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


class Server:
    """Main fusion-mlx server with engine pool, routing, and API endpoints."""

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self.pool: EnginePool | None = None
        self.request_router: RequestRouter | None = None
        self.cloud_router: CloudRouter | None = None
        self.engine_cores: dict[str, AsyncEngineCore] = {}
        self._load_lock = asyncio.Lock()
        self.settings = Settings.load(Path(self.config.settings_dir) / "settings.json")
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with self._lifespan():
                yield

        app = FastAPI(
            title="fusion-mlx",
            description="Unified local model management for Apple Silicon",
            version="0.1.0",
            lifespan=lifespan,
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register all route modules
        app.include_router(openai_router)


        @app.exception_handler(HTTPException)
        async def _http_exception_handler(_req, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"message": str(exc.detail)}},
             )

        app.include_router(anthropic_router)
        app.include_router(audio_router)
        app.include_router(images_router)
        app.include_router(mcp_router)
        app.include_router(openclaw_router)
        app.include_router(recommend_router)
        app.include_router(admin_router)

         # Register GUI compatibility router (discovery, settings, manager, admin UI)
        if get_gui_compat_router:
            app.include_router(get_gui_compat_router())

        # Root endpoint (health check for clients)
        @app.get("/")
        @app.head("/")
        async def root():
            return {"status": "ok", "service": "fusion-mlx"}

        # Health check
        @app.get("/health")
        async def health():
            engines_list = []
            if self.pool:
                status = self.pool.get_status()
                engines_list = status.get("models", [])
            return {
                "status": "ok",
                "version": "0.1.0",
                "engines": engines_list,
                "mx_memory": {
                     "active": f"{mx.get_active_memory() / 1e9:.2f} GB",
                     "cached": f"{mx.get_cache_memory() / 1e9:.2f} GB",
                      "peak": f"{mx.get_peak_memory() / 1e9:.2f} GB",
                  },
                 "model_dir": self.config.model_dir,
              }
        # Stats endpoint (combined pool + metrics)
        @app.get("/stats")
        async def stats():
            pool_status = self.pool.get_status() if self.pool else {}
            metrics = get_server_metrics().to_dict()
            return {**pool_status, **metrics}

        # Metrics endpoint
        @app.get("/metrics")
        async def metrics():
            return JSONResponse(get_server_metrics().to_dict())

        return app

    @asynccontextmanager
    async def _lifespan(self):
        """Startup/shutdown lifecycle."""
        logger.info("fusion-mlx starting up...")
        await self._startup()
        yield
        await self._shutdown()

    async def _startup(self):
        """Initialize engine pool, routers, and load models."""
        # Set memory limit
        mem_cfg = self.config.memory
        if mem_cfg.ssd_cache_enabled:
            avail_mb = _available_ram_mb()
            limit_mb = (mem_cfg.cache_memory_mb if mem_cfg.cache_memory_mb
                         else int(mem_cfg.cache_memory_percent * avail_mb))
            if limit_mb > 0:
                mx.set_memory_limit(limit_mb)
                logger.info("MLX memory limit set to %d MB (available: %d MB)", limit_mb, avail_mb)

        # Create engine pool
        self.pool = EnginePool()

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
        await self.pool._process_memory_enforcer.start()
        self.pool._get_final_ceiling = self.pool._process_memory_enforcer.get_final_ceiling

        # Create request router
        self.request_router = RequestRouter()

        # Create cloud router if enabled
        if self.config.cloud_router_enabled:
            self.cloud_router = CloudRouter(
                api_key=self.config.cloud_router_api_key,
                threshold=self.config.cloud_router_threshold,
            )

        # Inject context into route modules
        set_openai_context(self.pool, self.request_router)
        set_anthropic_context(self.pool)
        set_images_context(self.pool)
        set_openclaw_agent_pool(self.pool)
        set_mcp_manager_getter(lambda: None)  # TODO: wire MCP manager

        # Apply model aliases
        aliases = {**self.config.model_aliases}
        if aliases:
            logger.info("Applied %d model aliases", len(aliases))

          # Auto-discover and register models in pool
        if self.config.model_dir:
            self.pool.discover_models(self.config.model_dir)
            logger.info("Discovered %d models in %s", self.pool.model_count, self.config.model_dir)

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

         # Cleanup GUI resources
        if close_database:
            try:
                from fusion_gui.model_manager import shutdown_model_manager
                from fusion_gui.inference_queue_manager import shutdown_inference_manager
                shutdown_inference_manager()
                shutdown_model_manager()
                close_database()
                logger.info("GUI resources cleaned up")
            except Exception as e:
                logger.warning(f"GUI cleanup warning: {e}")
        if self.pool:
            await self.pool.shutdown()
        mx.clear_cache()
        logger.info("fusion-mlx shutdown complete")

    async def load_model(self, model_id: str, **kwargs):
        """Dynamically load a model via the engine pool."""
        if self.pool is None:
            raise RuntimeError("Server not started")
        async with self._load_lock:
            resolved = resolve_model_id(model_id)
            engine = await self.pool.get_engine(resolved)
            logger.info("Loaded model %s into pool (engine=%s)", model_id, type(engine).__name__)

    async def unload_model(self, model_id: str):
        """Unload a model from the pool."""
        core = self.engine_cores.pop(model_id, None)
        if core:
            await core.stop()
        if self.pool:
            self.pool.unload_engine(model_id)
        logger.info("Unloaded model %s from pool", model_id)




def _available_ram_mb() -> int:
    """Get truly available system RAM in MB, using psutil."""
    try:
        import psutil
        vm = psutil.virtual_memory()
          # Reserve 4 GB for OS + other processes as a safety margin
        return max(0, int(vm.available // (1024 * 1024)) - 4096)
    except Exception:
        return 16 * 1024    # fallback: 12 GB effective (16 - 4 GB reserve)


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
    parser.add_argument("--memory-tier", choices=["safe", "balanced", "aggressive", "custom"],
                        default="balanced", help="Memory enforcement tier")
    parser.add_argument("--ssd-cache", action="store_true", help="Enable SSD cold layer")
    parser.add_argument("--cloud-router", action="store_true", help="Enable cloud fallback")
    parser.add_argument("--cloud-api-key", default=None, help="Cloud router API key")
    args = parser.parse_args()

    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
    )
    config.memory.tier = getattr(config.memory.tier.__class__, args.memory_tier, config.memory.tier)
    config.memory.ssd_cache_enabled = args.ssd_cache
    config.cloud_router_enabled = args.cloud_router
    if args.cloud_api_key:
        config.cloud_router_api_key = args.cloud_api_key

    server = Server(config)
    server.run()


if __name__ == "__main__":
    main()
