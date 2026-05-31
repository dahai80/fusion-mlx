"""fusion-mlx — unified local model management for Apple Silicon.

Merges the best of omlx (long-context, memory control, multi-model
concurrency) with Rapid-MLX (speculative decoding, multi-modal,
cloud routing) into a single codebase.

Key features:
- EnginePool with LRU eviction, pinning, TTL auto-unload
- 4-tier ProcessMemoryEnforcer (safe/balanced/aggressive/custom)
- Paged KV cache with SSD cold layer
- Block-aware prefix cache with copy-on-write
- Speculative decoding (SuffixDecoding, DFlash, MTP, VLM-MTP)
- OpenAI/Anthropic/Responses API compatibility
- Claude Code, OpenClaw, ComfyUI integrations
"""

from ._version import __version__

__all__ = ["__version__"]
