"""FastAPI server for fusion-mlx.

Stubs the server skeleton. Full implementation comes in Phase 5 (API + routing).
"""

import logging
from typing import Optional

import mlx.core as mx
from fastapi import FastAPI

from .config import ServerConfig

logger = logging.getLogger(__name__)


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    if config is None:
        config = ServerConfig()

    app = FastAPI(
        title="fusion-mlx",
        description="Unified local model management for Apple Silicon",
        version="0.1.0",
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "engines": [],
            "mx_memory": {
                "active": f"{mx.get_memory_usage()[0] / 1e9:.2f} GB",
                "cached": f"{mx.get_memory_usage()[1] / 1e9:.2f} GB",
                "memory_limit": f"{mx.get_memory_limit() / 1e9:.2f} GB",
            },
        }

    @app.get("/v1/models")
    async def list_models():
        return {"data": []}

    return app
