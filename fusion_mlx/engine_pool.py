# SPDX-License-Identifier: Apache-2.0
# A-P1-2 (#0908 audit): this was a no-op stub `class EnginePool: def
# __init__(self,**kwargs): pass` left from a test migration. The real class
# lives at fusion_mlx/pool/engine_pool.py. `from fusion_mlx.engine_pool
# import EnginePool` silently returned the stub (no-op) instead of the real
# pool — dangerous if any code path used the short import. Re-export the
# real class so both import paths resolve to the same object.
from .pool.engine_pool import EnginePool, EngineEntry

__all__ = ["EnginePool", "EngineEntry"]
