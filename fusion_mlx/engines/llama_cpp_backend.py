# SPDX-License-Identifier: Apache-2.0
"""llama.cpp subprocess bridge — routes ultra-low-bit (Q3 and below) GGUF
models to a llama-server child process over HTTP.

Design (from PR-D6 spike):
  - llama-server is HTTP-only (httplib), no JSONL inference protocol.
  - Its /v1/chat/completions is OpenAI-shaped — thin HTTP forwarder, no
    protocol translation needed.
  - Subprocess lifecycle mirrors media/job_manager.py: spawn, health-gate,
    forward requests, kill on idle/shutdown.

Routing: quant level Q3/IQ3/Q2/IQ2/IQ1 → llama.cpp; everything else → MLX.
The bridge is opt-in via env FUSION_MLX_LLAMA_CPP=1 or auto-detected when a
GGUF model with low-bit quant is loaded.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_LLAMA_SERVER_BIN = os.environ.get("FUSION_MLX_LLAMA_SERVER", "llama-server")
_DEFAULT_TIMEOUT = 30.0
_IDLE_TIMEOUT = 300.0  # 5min idle → kill server
_HEALTH_RETRIES = 60
_HEALTH_INTERVAL = 0.5


@dataclass
class LlamaCppConfig:
    """Configuration for a llama-server subprocess."""

    model_path: str
    n_gpu_layers: int = 0  # 0=CPU, -1=all-GPU
    ctx_size: int = 4096
    n_threads: int = 0  # 0=auto
    port: int = 0  # 0=auto-allocate
    extra_args: list[str] = field(default_factory=list)

    def to_args(self) -> list[str]:
        args = [
            _LLAMA_SERVER_BIN,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--model",
            self.model_path,
            "--ctx-size",
            str(self.ctx_size),
            "--no-ui",
            "--jinja",
        ]
        if self.n_gpu_layers != 0:
            ngl = "all" if self.n_gpu_layers == -1 else str(self.n_gpu_layers)
            args.extend(["--n-gpu-layers", ngl])
        if self.n_threads > 0:
            args.extend(["--threads", str(self.n_threads)])
        args.extend(self.extra_args)
        return args


def _allocate_port() -> int:
    """Bind ephemeral port, close, return port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCppBackend:
    """Manages a llama-server subprocess for one GGUF model.

    One backend instance per model. The server stays alive across requests;
    killed on idle timeout or explicit shutdown.
    """

    def __init__(self, config: LlamaCppConfig):
        self.config = config
        if config.port == 0:
            config.port = _allocate_port()
        self._proc: asyncio.subprocess.Process | None = None
        self._base_url = f"http://127.0.0.1:{config.port}"
        self._last_active: float = 0.0
        self._started: bool = False
        logger.info(
            "LlamaCppBackend init: model=%s port=%d ngl=%d ctx=%d",
            Path(config.model_path).name,
            config.port,
            config.n_gpu_layers,
            config.ctx_size,
        )

    async def start(self) -> None:
        """Spawn llama-server, wait for health."""
        if self._started:
            return
        args = self.config.to_args()
        logger.info("Starting llama-server: %s", " ".join(args))
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._wait_health()
        self._started = True
        self._last_active = time.monotonic()
        logger.info("llama-server ready on %s", self._base_url)

    async def _wait_health(self) -> None:
        """Poll GET /health until 200."""
        for attempt in range(_HEALTH_RETRIES):
            if self._proc and self._proc.returncode is not None:
                stderr = await self._proc.stderr.read() if self._proc.stderr else b""
                raise RuntimeError(
                    f"llama-server exited code={self._proc.returncode}: "
                    f"{stderr.decode()[:500]}"
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    r = await c.get(f"{self._base_url}/health")
                    if r.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(_HEALTH_INTERVAL)
        raise TimeoutError(
            f"llama-server health check failed after {_HEALTH_RETRIES} attempts"
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Forward a chat completion request to llama-server."""
        if not self._started:
            await self.start()
        self._last_active = time.monotonic()
        payload = {
            "model": Path(self.config.model_path).stem,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as c:
            r = await c.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            return r.json()

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Stream chat completion as SSE chunks."""
        if not self._started:
            await self.start()
        self._last_active = time.monotonic()
        payload = {
            "model": Path(self.config.model_path).stem,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as c:
            async with c.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        yield line

    async def stop(self) -> None:
        """Kill the llama-server subprocess."""
        if self._proc and self._proc.returncode is None:
            logger.info("Stopping llama-server (pid=%s)", self._proc.pid)
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                logger.warning("llama-server SIGTERM timeout, SIGKILL")
                self._proc.kill()
                await self._proc.wait()
        self._started = False
        self._proc = None

    @property
    def is_idle(self) -> bool:
        return self._started and (time.monotonic() - self._last_active) > _IDLE_TIMEOUT

    @property
    def is_running(self) -> bool:
        return (
            self._started and self._proc is not None and self._proc.returncode is None
        )


class LlamaCppBackendPool:
    """Manages multiple LlamaCppBackend instances (one per GGUF model).

    LRU eviction when pool is full. Idle backends killed after timeout.
    """

    def __init__(self, max_backends: int = 2):
        self._backends: dict[str, LlamaCppBackend] = {}
        self._max = max_backends

    async def get(self, model_path: str, **kwargs) -> LlamaCppBackend:
        """Get or create a backend for the given GGUF model."""
        key = str(model_path)
        if key in self._backends:
            b = self._backends[key]
            if not b.is_running:
                await b.start()
            return b

        if len(self._backends) >= self._max:
            await self._evict_oldest()

        config = LlamaCppConfig(model_path=model_path, **kwargs)
        backend = LlamaCppBackend(config)
        await backend.start()
        self._backends[key] = backend
        return backend

    async def _evict_oldest(self) -> None:
        """Evict the least-recently-active backend."""
        if not self._backends:
            return
        oldest_key = min(self._backends, key=lambda k: self._backends[k]._last_active)
        logger.info("Evicting llama-server backend: %s", oldest_key)
        await self._backends[oldest_key].stop()
        del self._backends[oldest_key]

    async def stop_all(self) -> None:
        """Kill all backends (shutdown)."""
        for b in self._backends.values():
            await b.stop()
        self._backends.clear()

    async def reap_idle(self) -> int:
        """Kill idle backends, return count reaped."""
        idle = [k for k, b in self._backends.items() if b.is_idle]
        for k in idle:
            logger.info("Reaping idle llama-server backend: %s", k)
            await self._backends[k].stop()
            del self._backends[k]
        return len(idle)
