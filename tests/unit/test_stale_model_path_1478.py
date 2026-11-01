# SPDX-License-Identifier: Apache-2.0
"""#1478: stale discovery entry — model dir removed after boot must fail
with a clear 404 (ModelNotFoundError), not a confusing HFValidationError
from mlx_lm treating the dead absolute path as an HF repo id.
"""

from __future__ import annotations

import pytest

from fusion_mlx.exceptions import ModelNotFoundError
from fusion_mlx.pool.engine_pool import EnginePool


class _FakeEntry:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_id = "stale-model"
        self.priority_tier = None
        self.actual_size = 0
        self.estimated_size = 0
        self.is_loading = True
        self.loading_started_at = 0.0
        self.abort_loading = False
        self.last_load_failed = False
        self.adapter_path = None
        self.engine = None
        self.last_observed_size = None
        self.loading_event = None
        self.runtime_settings_signature = None


def _make_pool(model_path: str) -> tuple[EnginePool, _FakeEntry]:
    pool = EnginePool.__new__(EnginePool)
    entry = _FakeEntry(model_path)
    pool._entries = {"stale-model": entry}
    pool._tier_budgets_bytes = {}
    pool._load_seconds_per_gb_ema = None
    pool._load_time_observations = 0
    pool._wake_process_memory_enforcer = lambda *a, **k: None
    return pool, entry


def test_stale_absolute_path_raises_model_not_found(tmp_path):
    # Dir discovered at boot, removed since: must 404 with the path in
    # the message instead of reaching mlx_lm's repo-id validator.
    gone = tmp_path / "models--org--repo" / "snapshots" / "deadbeef"
    pool, entry = _make_pool(str(gone))
    with pytest.raises(ModelNotFoundError) as exc_info:
        import asyncio

        asyncio.run(pool._load_engine("stale-model"))
    assert "no longer exists on disk" in str(exc_info.value)
    assert str(gone) in str(exc_info.value)
    # finally block: loading slot released, failure marker set.
    assert entry.is_loading is False
    assert entry.last_load_failed is True


def test_existing_path_passes_guard(tmp_path):
    # A real dir must get past the guard (the load then fails later on
    # the stub entry — any non-ModelNotFoundError proves the guard let
    # it through).
    live = tmp_path / "models--org--repo" / "snapshots" / "live"
    live.mkdir(parents=True)
    pool, _ = _make_pool(str(live))
    with pytest.raises(Exception) as exc_info:
        import asyncio

        asyncio.run(pool._load_engine("stale-model"))
    assert not isinstance(exc_info.value, ModelNotFoundError)


def test_relative_repo_id_not_guarded():
    # HF repo ids ("org/repo") are not absolute paths — the guard must
    # not reject them (they go to snapshot_download by design).
    pool, _ = _make_pool("mlx-community/Qwen3-0.6B-4bit")
    with pytest.raises(Exception) as exc_info:
        import asyncio

        asyncio.run(pool._load_engine("stale-model"))
    assert not isinstance(exc_info.value, ModelNotFoundError)
