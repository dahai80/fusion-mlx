# SPDX-License-Identifier: Apache-2.0
# /api/ps must list one row per engine instance, not one per alias entry:
# boot registers the served model under repo-id and flattened cache-key
# spellings sharing a single engine, and ps showed a phantom duplicate.

from __future__ import annotations

import json
from unittest.mock import MagicMock

import fusion_mlx.api.ollama_routes as ollama_routes
from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _entry(model_id: str, engine) -> EngineEntry:
    return EngineEntry(
        model_id=model_id,
        model_path="/m",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
        engine=engine,
    )


def _pool_two_spellings_one_engine():
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {}
    eng = MagicMock()
    pool._entries["mlx-community--Qwen3.5-4B-MLX-4bit"] = _entry(
        "mlx-community--Qwen3.5-4B-MLX-4bit", eng
    )
    pool._entries["mlx-community/Qwen3.5-4B-MLX-4bit"] = _entry(
        "mlx-community/Qwen3.5-4B-MLX-4bit", eng
    )
    return pool


async def test_ps_lists_alias_spellings_once(monkeypatch):
    monkeypatch.setattr(
        ollama_routes, "_pool", _pool_two_spellings_one_engine(), raising=False
    )
    resp = await ollama_routes.api_ps(_auth=True)
    names = [m["name"] for m in json.loads(resp.body)["models"]]
    assert len(names) == 1
    assert names[0] == "mlx-community--Qwen3.5-4B-MLX-4bit"


async def test_ps_lists_distinct_engines_separately(monkeypatch):
    pool = _pool_two_spellings_one_engine()
    pool._entries["BAAI--bge-m3"] = _entry("BAAI--bge-m3", MagicMock())
    monkeypatch.setattr(ollama_routes, "_pool", pool, raising=False)
    resp = await ollama_routes.api_ps(_auth=True)
    assert len(json.loads(resp.body)["models"]) == 2
