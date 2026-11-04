# SPDX-License-Identifier: Apache-2.0
# Regression: one model registered under two spellings (repo-id alias vs
# flattened cache key) loaded two live engines. Unloading one spelling left
# the stopped engine attached to the alias entry, so get_engine reloaded
# under the alias key and both spellings kept separate engines resident.

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _live_engine():
    eng = MagicMock(spec=["start", "stop", "safe_evict", "is_running"])
    eng.start = AsyncMock()
    eng.stop = AsyncMock()
    eng.safe_evict = AsyncMock()
    eng.is_running = MagicMock(return_value=True)
    return eng


def _dead_engine():
    eng = _live_engine()
    eng.is_running = MagicMock(return_value=False)
    return eng


def _entry(model_id: str, model_path: str) -> EngineEntry:
    return EngineEntry(
        model_id=model_id,
        model_path=model_path,
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
    )


def _two_spelling_pool(tmpdir: str):
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {}
    pool._on_engine_detached = None
    pool._active_swap = {}
    pool._adapter_swap_locks = {}
    slug = "mlx-community--Qwen3.5-4B-MLX-4bit"
    slash = "mlx-community/Qwen3.5-4B-MLX-4bit"
    path = os.path.join(tmpdir, "snapshots", "abc123")
    os.makedirs(path, exist_ok=True)
    pool._entries[slug] = _entry(slug, path)
    pool._entries[slash] = _entry(slash, path)
    return pool, slug, slash


class TestLiveSiblingRedirect:
    def test_dead_entry_redirects_to_live_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool, slug, slash = _two_spelling_pool(tmpdir)
            pool._entries[slug].engine = _live_engine()
            pool._entries[slash].engine = _dead_engine()
            assert pool._resolve_hf_repo_id(slash) == slug

    def test_both_dead_no_redirect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool, slug, slash = _two_spelling_pool(tmpdir)
            pool._entries[slug].engine = _dead_engine()
            pool._entries[slash].engine = _dead_engine()
            assert pool._resolve_hf_repo_id(slash) == slash

    def test_live_entry_not_redirected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool, slug, slash = _two_spelling_pool(tmpdir)
            pool._entries[slash].engine = _live_engine()
            pool._entries[slug].engine = _dead_engine()
            assert pool._resolve_hf_repo_id(slash) == slash

    def test_different_model_path_no_redirect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool, slug, slash = _two_spelling_pool(tmpdir)
            other = _entry("other-model", os.path.join(tmpdir, "elsewhere"))
            os.makedirs(other.model_path, exist_ok=True)
            pool._entries["other-model"] = other
            pool._entries["other-model"].engine = _live_engine()
            pool._entries[slash].engine = _dead_engine()
            assert pool._resolve_hf_repo_id(slash) == slash


class TestAliasRefClearOnUnload:
    @pytest.mark.asyncio
    async def test_detach_clears_sibling_refs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool, slug, slash = _two_spelling_pool(tmpdir)
            eng = _live_engine()
            pool._entries[slug].engine = eng
            pool._entries[slash].engine = eng
            await pool._detach_engine(slug)
            assert pool._entries[slug].engine is None
            assert pool._entries[slash].engine is None
            eng.safe_evict.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_detach_keeps_unrelated_engines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool, slug, slash = _two_spelling_pool(tmpdir)
            eng = _live_engine()
            other_eng = _live_engine()
            pool._entries[slug].engine = eng
            pool._entries[slash].engine = other_eng
            await pool._detach_engine(slug)
            assert pool._entries[slash].engine is other_eng
