"""Runtime module — macOS-specific helpers (UBC eviction, prefix cache, etc.)."""

# Submodules are importable as fusion_mlx.runtime.ubc_evict, etc.
# Do NOT re-export names that shadow submodule names here —
# Python's dotted import uses attribute lookup on the package,
# so `from .ubc_evict import ubc_evict` would make
# `import fusion_mlx.runtime.ubc_evict as m` return the function.

__all__ = ["ubc_evict", "ubc_evict_paths"]
