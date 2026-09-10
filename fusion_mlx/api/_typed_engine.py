# SPDX-License-Identifier: Apache-2.0
"""Shared typed-engine resolver for modality route modules.

R-P1-11 (#0908 audit): get_ner_engine / get_reranker_engine were
near-identical copies — one shared factory eliminates the divergence risk.
"""

from typing import Any

from fastapi import HTTPException

_pool: Any = None


def set_pool(pool) -> None:
    global _pool
    _pool = pool


async def get_typed_engine(
    model_id: str,
    engine_class_path: str,
    label: str,
) -> Any:
    if _pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    engine = await _pool.get_engine(model_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    import importlib

    module_path, class_name = engine_class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    if not isinstance(engine, cls):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not a {label} model",
        )
    return engine
