"""Windows CUDA node implementation (#365).

A vLLM-backed OpenAI-compatible HTTP server for heavy LLM inference on Windows
CUDA nodes. The node advertises ``platform=windows-cuda`` over mDNS so the
fusion-gateway can route heavy-model intents to it.

Design:
- vLLM is imported lazily inside ``_build_engine``. Importing this module on a
  Mac never touches vLLM (Rule 12: fail visibly, not silently — the loud
  failure happens only when the node is started without vLLM/CUDA).
- The FastAPI app mirrors fusion-mlx's OpenAI routes so the gateway treats the
  node uniformly: ``/v1/models``, ``/v1/chat/completions``, ``/v1/completions``.
- mDNS advertising reuses ``fusion_mlx.cluster.mdns`` so the gateway's existing
  discovery picks the node up by the ``platform`` TXT record.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _cuda_available_percent() -> float:
    # P3 (#811): the CUDA node always advertised available_percent: 0.0,
    # so the cluster load balancer (lower load_score is better) saw this
    # node as maximally loaded and never routed to it. vLLM requires torch
    # with CUDA, so query the real free/total ratio here. Falls back to
    # 100.0 (not 0.0) when torch.cuda is unavailable so the node is not
    # blackholed by a stale-zero snapshot.
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            logger.warning(
                "cuda-node: torch.cuda unavailable — advertising available_percent=100.0"
            )
            return 100.0
        free, total = torch.cuda.mem_get_info()
        if total <= 0:
            return 100.0
        return round((free / total) * 100.0, 1)
    except Exception as e:
        logger.warning(
            "cuda-node: GPU memory probe failed — advertising available_percent=100.0: %s",
            e,
        )
        return 100.0


@dataclass
class CudaNodeConfig:
    """Configuration for the vLLM CUDA backend node (#365)."""

    model: str
    # CL-2 (#811 audit 0906): bind loopback by default. A cluster-exposed
    # CUDA node on 0.0.0.0 with no auth is an open inference relay — any
    # peer (or attacker on the subnet) can drive the GPU unauthenticated.
    # Operators who genuinely need 0.0.0.0 must pass it explicitly AND
    # configure an api_key (see the startup guard in create_cuda_app).
    host: str = "127.0.0.1"
    port: int = 8000
    # CL-2 (#811 audit 0906): API key gating the node's inference routes.
    # Sourced from CudaNodeConfig.api_key (CLI --api-key / FUSION_MLX_API_KEY).
    # When unset AND the bind is loopback, anonymous access is permitted
    # (dev/single-host); when unset AND the bind is non-loopback, the node
    # REFUSES to start (fail visible — no silent open relay).
    api_key: str | None = None
    # vLLM engine kwargs
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    quantization: str | None = None  # e.g. "fp8", "awq", "gptq"
    max_model_len: int | None = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    # Cluster discovery
    cluster_advertise: bool = True
    node_id: str | None = None
    # Extra vLLM engine kwargs passthrough
    extra: dict = field(default_factory=dict)


def _import_vllm():
    """Lazily import vLLM; raise a clear error when unavailable (Rule 12)."""
    try:
        from vllm import AsyncLLMEngine  # type: ignore
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
        from vllm.sampling_params import SamplingParams  # type: ignore

        return AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "CUDA node requires vLLM (pip install vllm) on a Windows CUDA host. "
            "Importing this node on a non-CUDA machine is not supported. "
            f"Original error: {exc}"
        ) from exc


def _build_engine(config: CudaNodeConfig):
    AsyncLLMEngine, AsyncEngineArgs, _ = _import_vllm()
    engine_args = AsyncEngineArgs(
        model=config.model,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        quantization=config.quantization,
        max_model_len=config.max_model_len,
        trust_remote_code=config.trust_remote_code,
        dtype=config.dtype,
        **config.extra,
    )
    logger.info(
        "cuda-node: building vLLM engine model=%s tp=%d util=%.2f quant=%s",
        config.model,
        config.tensor_parallel_size,
        config.gpu_memory_utilization,
        config.quantization,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


def _engine_to_engine_args(engine):
    return getattr(engine, "engine_args", None)


def _resolve_node_id(config: CudaNodeConfig) -> str:
    if config.node_id:
        return config.node_id
    import socket

    return f"{socket.gethostname()}:{config.port}"


def create_cuda_app(config: CudaNodeConfig):
    """Build the FastAPI app for the CUDA node.

    vLLM is imported here (not at module load). On a Mac this raises loudly
    when called — callers (CLI, tests with monkeypatched vllm) handle it.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    engine = _build_engine(config)
    AsyncLLMEngine, AsyncEngineArgs, SamplingParams = _import_vllm()
    served_model = config.model

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("cuda-node: startup platform=windows-cuda model=%s", served_model)
        advertiser = None
        if config.cluster_advertise:
            try:
                from fusion_mlx.cluster.mdns import MdnsAdvertiser, build_txt_records

                snapshot = {
                    "node_id": _resolve_node_id(config),
                    "host": config.host,
                    "port": config.port,
                    "platform": "windows-cuda",
                    "models": [{"id": served_model, "loaded": True}],
                    "memory": {"available_percent": _cuda_available_percent()},
                }
                txt = build_txt_records(snapshot)
                advertiser = MdnsAdvertiser(
                    node_id=snapshot["node_id"],
                    host=snapshot["host"],
                    port=snapshot["port"],
                    txt_records=txt,
                )
                await advertiser.start()
                logger.info(
                    "cuda-node: mDNS advertising started (platform=windows-cuda)"
                )
            except Exception:
                logger.warning(
                    "cuda-node: mDNS advertising failed (non-fatal)", exc_info=True
                )
        app.state._mdns = advertiser
        try:
            yield
        finally:
            if advertiser is not None:
                await advertiser.stop()

    from fusion_mlx._version import __version__

    # CL-2 (#811 audit 0906): fail-visible startup guard. A non-loopback
    # bind with no api_key is an open inference relay — refuse to serve
    # rather than silently expose an unauthenticated GPU to the network.
    _configured_key = config.api_key
    import os

    if not _configured_key:
        env_key = os.environ.get("FUSION_MLX_API_KEY")
        if env_key:
            _configured_key = env_key
    _bind_loopback = config.host in {"127.0.0.1", "::1", "localhost"}
    if not _bind_loopback and not _configured_key:
        raise RuntimeError(
            "CL-2 (#811 audit 0906): CUDA node bound to non-loopback host "
            f"'{config.host}' with no api_key. An unauthenticated node on a "
            "cluster network is an open inference relay. Set --api-key "
            "(or FUSION_MLX_API_KEY) or bind 127.0.0.1. Refusing to start."
        )
    if _configured_key:
        logger.info(
            "cuda-node: API-key auth ENABLED (bind=%s)", config.host
        )
    else:
        logger.warning(
            "cuda-node: no api_key configured on loopback bind — anonymous "
            "access permitted (dev/single-host only). Do NOT bind 0.0.0.0."
        )

    app = FastAPI(
        title="fusion-mlx CUDA node",
        description="vLLM-powered OpenAI-compatible heavy LLM node (#365)",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _cuda_node_auth(request: Request, call_next):
        # CL-2 (#811 audit 0906): gate every route behind the configured
        # api_key (Bearer or x-api-key), mirroring the main server's
        # verify_api_key. /health is the only unauthenticated probe — a
        # load balancer needs a liveness check without credentials.
        path = request.url.path
        if path == "/health" or _configured_key is None:
            return await call_next(request)
        import secrets

        bearer = None
        authz = request.headers.get("authorization", "")
        if authz.lower().startswith("bearer "):
            bearer = authz[7:].strip()
        x_api_key = request.headers.get("x-api-key")
        provided = [k for k in (bearer, x_api_key) if k]
        if not provided:
            logger.warning(
                "cuda-node: 401 no api_key host=%s path=%s",
                request.client.host if request.client else "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required"},
            )
        if not all(
            secrets.compare_digest(k, _configured_key) for k in provided
        ):
            logger.warning(
                "cuda-node: 401 invalid api_key host=%s path=%s",
                request.client.host if request.client else "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "platform": "windows-cuda", "model": served_model}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fusion-mlx-cuda-node",
                }
            ],
        }

    async def _generate(prompt: str, request: dict, is_chat: bool):
        sampling = SamplingParams(
            temperature=float(request.get("temperature", 0.7)),
            top_p=float(request.get("top_p", 1.0)),
            max_tokens=int(
                request.get("max_tokens", request.get("max_completion_tokens", 512))
            ),
            stop=request.get("stop"),
            presence_penalty=float(request.get("presence_penalty", 0.0)),
            frequency_penalty=float(request.get("frequency_penalty", 0.0)),
        )
        # P3 (#811): int(time.time()*1000) collides under concurrent requests.
        # Append a uuid4 short suffix so each request_id is unique.
        request_id = f"cuda-node-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        final_text = ""
        final_finish = None
        try:
            async for output in engine.generate(prompt, sampling, request_id):
                final_text = output.outputs[0].text
                final_finish = output.outputs[0].finish_reason
        except Exception as exc:
            logger.error("cuda-node: generation failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return final_text, final_finish

    @app.post("/v1/completions")
    async def completions(request: dict):
        prompt = request.get("prompt")
        if prompt is None:
            raise HTTPException(status_code=400, detail="prompt is required")
        if isinstance(prompt, list):
            prompt = "".join(str(p) for p in prompt)
        text, finish = await _generate(str(prompt), request, is_chat=False)
        return JSONResponse(
            {
                "id": f"cmpl-cuda-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": served_model,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "finish_reason": finish,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        messages = request.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")
        # Render messages to a prompt string. vLLM's chat path uses its own
        # tokenizer chat template; fall back to plain concatenation if the
        # tokenizer helper is unavailable.
        try:
            from vllm.transformers_utils.tokenizer import get_tokenizer  # type: ignore

            tokenizer = get_tokenizer(
                served_model, trust_remote_code=config.trust_remote_code
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            logger.warning("cuda-node: chat template unavailable, using plain concat")
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            )
        text, finish = await _generate(prompt, request, is_chat=True)
        return JSONResponse(
            {
                "id": f"chatcmpl-cuda-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    app.state.engine = engine
    app.state.config = config
    return app


def run_cuda_node(config: CudaNodeConfig) -> None:
    """Start the CUDA node uvicorn server (blocking)."""
    import uvicorn

    app = create_cuda_app(config)
    logger.info("cuda-node: serving on %s:%d", config.host, config.port)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
