"""fusion-mlx optional backends (#365).

The native engine is Apple-Silicon MLX. Optional backends target other
platforms and are imported lazily so the package stays importable everywhere.
``cuda_node`` runs a vLLM-powered OpenAI-compatible server on Windows CUDA.
"""
