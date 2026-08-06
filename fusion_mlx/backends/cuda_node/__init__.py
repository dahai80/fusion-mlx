"""Windows CUDA backend node (#365) — vLLM-powered OpenAI-compatible server.

Heavy LLM inference (DeepSeek 70B / Qwen 72B FP8) runs on a Windows CUDA node
via vLLM. This package exposes the node as a FastAPI app with the same
OpenAI routes fusion-mlx serves on Mac, and self-registers with the cluster
mDNS under ``platform=windows-cuda`` so fusion-gateway can route heavy-model
intents here.

vLLM is a heavy, CUDA-only dependency. It is imported lazily inside
``create_cuda_app``/``run_cuda_node`` so that merely importing this package on
a Mac (no CUDA, no vLLM) never fails — the failure is surfaced loudly only
when the node is actually started (Rule 12).
"""

from .node import CudaNodeConfig, create_cuda_app, run_cuda_node

__all__ = ["CudaNodeConfig", "create_cuda_app", "run_cuda_node"]
