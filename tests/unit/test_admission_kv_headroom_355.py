# SPDX-License-Identifier: Apache-2.0
# #355: admission memory accounting - actual_size feedback + tunable KV headroom.
#
# estimated_size (weights-only) underestimates real load footprint and ignores
# runtime KV-cache growth, so the pre-load admission projection under-reserves
# memory and models break through the ceiling after loading. The fix:
#   - last_observed_size (persists across unload) replaces estimated_size in the
#     projection once a model has been loaded once.
#   - _kv_admission_headroom() adds a tunable KV reservation
#     (FUSION_MLX_ADMISSION_KV_HEADROOM_GB; default min(max_kv_cache_memory,
#     2 GiB); 0 disables).
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from fusion_mlx.exceptions import ModelTooLargeError
from fusion_mlx.pool.engine_pool import EnginePool

ENV = "FUSION_MLX_ADMISSION_KV_HEADROOM_GB"
_DEFAULT_CAP = 2 * 1024**3


def _make_pool(ceiling=None):
    pool = EnginePool()
    if ceiling is None or ceiling <= 0:
        pool._get_final_ceiling = lambda: 0
    else:
        pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


@pytest.fixture
def small_model_dir(tmp_path):
    d = tmp_path / "model-a"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path


class TestKvAdmissionHeadroom:
    # _kv_admission_headroom() env parsing + default.

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        pool = _make_pool()
        hr = pool._kv_admission_headroom()
        # default = min(scheduler max_kv_cache_memory or 4 GiB, 2 GiB)
        assert 0 < hr <= _DEFAULT_CAP

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(ENV, "1.5")
        pool = _make_pool()
        assert pool._kv_admission_headroom() == int(1.5 * 1024**3)

    def test_env_zero_disables(self, monkeypatch):
        monkeypatch.setenv(ENV, "0")
        pool = _make_pool()
        assert pool._kv_admission_headroom() == 0

    def test_invalid_env_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv(ENV, "not-a-number")
        caplog.set_level(logging.WARNING, logger="fusion_mlx.pool.engine_pool")
        pool = _make_pool()
        hr = pool._kv_admission_headroom()
        assert 0 < hr <= _DEFAULT_CAP
        assert any(ENV in r.getMessage() for r in caplog.records)


class TestAdmissionProjection:
    # #355: projection = current + effective_size + kv_headroom, where
    # effective_size = last_observed_size or estimated_size.

    def test_uses_last_observed_size_plus_headroom(self, monkeypatch, small_model_dir):
        # 1 GiB headroom -> clean arithmetic
        monkeypatch.setenv(ENV, "1")
        pool = _make_pool(ceiling=100)
        pool.discover_models(str(small_model_dir))
        entry = pool.get_entry("model-a")
        assert entry is not None
        # simulate a prior observed footprint larger than the weights estimate
        observed = 5 * 1024**3
        entry.last_observed_size = observed

        with pytest.raises(ModelTooLargeError) as exc_info:
            asyncio.run(pool.get_engine("model-a"))

        # model_size = effective_size + kv_headroom = 5 GiB + 1 GiB
        assert exc_info.value.model_size == observed + 1 * 1024**3
        assert exc_info.value.ceiling == 100

    def test_falls_back_to_estimated_when_no_observed(
        self, monkeypatch, small_model_dir
    ):
        # disable headroom to isolate the estimated_size fallback
        monkeypatch.setenv(ENV, "0")
        pool = _make_pool(ceiling=100)
        pool.discover_models(str(small_model_dir))
        entry = pool.get_entry("model-a")
        assert entry is not None
        assert entry.last_observed_size is None

        with pytest.raises(ModelTooLargeError) as exc_info:
            asyncio.run(pool.get_engine("model-a"))

        # model_size = estimated_size + 0 headroom
        assert exc_info.value.model_size == entry.estimated_size
