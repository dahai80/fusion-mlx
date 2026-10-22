# SPDX-License-Identifier: Apache-2.0
# #1316: a model written after startup (merged adapter) must be servable by
# its directory path and by its id, without a restart or an admin reload.

import json
import os

import pytest

from fusion_mlx.pool.engine_pool import EnginePool


def _make_pool(ceiling: int = 10 * 1024**3) -> EnginePool:
    pool = EnginePool()
    pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


def _write_model(root, name):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (path / "model.safetensors").write_bytes(b"0" * 1024)
    return path


def test_resolve_model_path_to_entry_id(tmp_path):
    root = tmp_path / "models"
    merged = _write_model(root, "sft_merged_001")
    pool = _make_pool()
    pool.discover_models(str(root))

    resolved = pool._resolve_hf_repo_id(str(merged))
    assert resolved == "sft_merged_001"
    logger_ok = pool._match_entry_by_model_path(str(merged)) == "sft_merged_001"
    assert logger_ok


def test_resolve_trailing_slash_path(tmp_path):
    root = tmp_path / "models"
    merged = _write_model(root, "sft_merged_002")
    pool = _make_pool()
    pool.discover_models(str(root))

    assert pool._resolve_hf_repo_id(f"{merged}/") == "sft_merged_002"


@pytest.mark.asyncio
async def test_late_written_model_registers_on_first_request(tmp_path):
    root = tmp_path / "models"
    _write_model(root, "model-a")
    pool = _make_pool()
    pool.discover_models(str(root))
    assert "merged-later" not in pool._entries

    merged = _write_model(root, "merged-later")
    resolved = await pool._ensure_local_path_registered(str(merged))
    assert resolved == "merged-later"
    assert "merged-later" in pool._entries
    assert pool._entries["merged-later"].model_path == str(merged)


@pytest.mark.asyncio
async def test_late_registration_keeps_entries_from_other_roots(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    _write_model(root_a, "model-a")
    _write_model(root_b, "model-b")
    pool = _make_pool()
    pool.discover_models([str(root_a), str(root_b)])
    assert {"model-a", "model-b"} <= set(pool._entries)

    merged = _write_model(root_a, "merged-later")
    await pool._ensure_local_path_registered(str(merged))
    assert "model-a" in pool._entries
    assert "model-b" in pool._entries
    assert "merged-later" in pool._entries


@pytest.mark.asyncio
async def test_non_path_ids_untouched(tmp_path):
    root = tmp_path / "models"
    _write_model(root, "model-a")
    pool = _make_pool()
    pool.discover_models(str(root))

    assert await pool._ensure_local_path_registered("model-a") == "model-a"
    assert await pool._ensure_local_path_registered("qwen3.5-9b-4bit") == (
        "qwen3.5-9b-4bit"
    )
    assert pool.model_count == 1


@pytest.mark.asyncio
async def test_missing_path_returns_input(tmp_path):
    pool = _make_pool()
    missing = os.path.join(str(tmp_path), "nope", "model")
    assert await pool._ensure_local_path_registered(missing) == missing
    assert pool.model_count == 0
