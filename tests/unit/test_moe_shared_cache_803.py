# #803 headless tests for the DSA shared-expert activation cache.
# Deterministic, no model weights. Verifies: env gating, exact-key dedup with
# no output drift, hit/miss accounting, size guard, single-row pass-through,
# metrics flat snapshot, and that wiring is present (import-time, no crash).

import mlx.core as mx
import pytest

from fusion_mlx.patches._moe_shared_cache import (
    SharedExpertActivationCache,
    get_stats,
    get_stats_flat,
    make_layer_cache,
    reset_stats,
    shared_cache_enabled,
)


def _setup_env(value, monkeypatch):
    monkeypatch.setenv("FUSION_MOE_SHARED_CACHE", value)
    import fusion_mlx.patches._moe_shared_cache as mod

    monkeypatch.setattr(mod, "_enabled", None)
    return mod


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_stats()
    # Default to enabled for the dedup tests; individual tests override.
    _setup_env("1", monkeypatch)
    reset_stats()
    yield
    reset_stats()


def test_env_gating_disabled(monkeypatch):
    _setup_env("0", monkeypatch)
    assert shared_cache_enabled() is False
    assert make_layer_cache(0) is None


def test_env_gating_enabled(monkeypatch):
    _setup_env("1", monkeypatch)
    assert shared_cache_enabled() is True
    c = make_layer_cache(0)
    assert c is not None
    assert c.layer_id == 0


def test_disabled_returns_none_by_default(monkeypatch):
    _setup_env("", monkeypatch)
    assert make_layer_cache(0) is None


def test_dedup_all_identical_rows():
    # 4 identical rows -> 1 miss, 3 hits; result equals fresh compute.
    c = SharedExpertActivationCache(0)
    x = mx.broadcast_to(mx.array([1.0, 2.0, 3.0]), (4, 3)).astype(mx.float32)
    calls = {"n": 0}

    def compute_fn(batch):
        calls["n"] += 1
        return batch * 10.0

    out = c.get_or_compute(x, compute_fn)
    assert calls["n"] == 1  # computed once for the unique row
    expected = x * 10.0
    assert mx.array_equal(out, expected)
    flat = get_stats_flat()
    assert flat["moe_shared_cache_requests"] == 4
    assert flat["moe_shared_cache_hits"] == 3
    assert flat["moe_shared_cache_misses"] == 1
    assert flat["moe_shared_cache_hit_rate"] == 0.75


def test_no_dedup_all_unique_rows():
    # 4 distinct rows -> 4 misses, 0 hits; compute called 4x via the batch
    # path (one batched call for the misses).
    c = SharedExpertActivationCache(1)
    x = mx.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=mx.float32)
    calls = {"n": 0}

    def compute_fn(batch):
        calls["n"] += 1
        return batch * 5.0

    out = c.get_or_compute(x, compute_fn)
    expected = x * 5.0
    assert mx.array_equal(out, expected)
    flat = get_stats_flat()
    assert flat["moe_shared_cache_hits"] == 0
    assert flat["moe_shared_cache_misses"] == 4
    assert flat["moe_shared_cache_hit_rate"] == 0.0


def test_no_output_drift_mixed():
    # Mixed batch: rows 0,2 identical; 1,3 unique. Verify exact equality vs a
    # plain compute on the full batch — a collision-key bug would diverge.
    c = SharedExpertActivationCache(2)
    row = mx.array([0.5, -1.5, 2.25, 9.0], dtype=mx.float32)
    x = mx.stack([row, row * 2.0, row, row + 1.0])

    def compute_fn(batch):
        return mx.sin(batch) + 0.5

    cached_out = c.get_or_compute(x, compute_fn)
    direct_out = (mx.sin(x) + 0.5).astype(mx.float32)
    assert mx.array_equal(cached_out, direct_out)
    # rows 0,2 identical -> 1 hit; rows 1,3 unique -> 2 misses
    flat = get_stats_flat()
    assert flat["moe_shared_cache_hits"] == 1
    assert flat["moe_shared_cache_misses"] == 3


def test_single_row_passthrough_no_stats():
    # B==1 is the decode step: no dedup possible, must skip instrumentation.
    c = SharedExpertActivationCache(3)
    x = mx.array([[1.0, 2.0]], dtype=mx.float32)
    calls = {"n": 0}

    def compute_fn(batch):
        calls["n"] += 1
        return batch * 2.0

    out = c.get_or_compute(x, compute_fn)
    assert mx.array_equal(out, x * 2.0)
    assert calls["n"] == 1
    flat = get_stats_flat()
    assert flat["moe_shared_cache_requests"] == 0  # not instrumented


def test_size_guard_skips_huge_batch():
    # Above _SIZE_GUARD elements -> plain compute, no stats, no key building.
    import fusion_mlx.patches._moe_shared_cache as mod

    c = SharedExpertActivationCache(4)
    big = mx.zeros((8, mod._SIZE_GUARD // 4 + 1), dtype=mx.float32)
    calls = {"n": 0}

    def compute_fn(batch):
        calls["n"] += 1
        return batch

    out = c.get_or_compute(big, compute_fn)
    assert calls["n"] == 1
    assert mx.array_equal(out, big)
    flat = get_stats_flat()
    assert flat["moe_shared_cache_requests"] == 0


def test_per_forward_state_reset():
    # Two consecutive calls with different inputs must not bleed state: the
    # second call's hits reflect only its own duplicates, not the first call's.
    c = SharedExpertActivationCache(5)
    x1 = mx.zeros((3, 2), dtype=mx.float32)  # 3 identical
    x2 = mx.ones((2, 2), dtype=mx.float32)  # 2 identical

    def compute_fn(batch):
        return batch + 1.0

    c.get_or_compute(x1, compute_fn)
    c.get_or_compute(x2, compute_fn)
    flat = get_stats_flat()
    # call1: 2 hits 1 miss; call2: 1 hit 1 miss
    assert flat["moe_shared_cache_hits"] == 3
    assert flat["moe_shared_cache_misses"] == 2


def test_get_stats_per_layer():
    c0 = SharedExpertActivationCache(0)
    c1 = SharedExpertActivationCache(1)
    x = mx.zeros((2, 2), dtype=mx.float32)
    c0.get_or_compute(x, lambda b: b)
    c1.get_or_compute(x, lambda b: b)
    stats = get_stats()
    assert set(stats.keys()) == {0, 1}
    assert stats[0]["hits"] == 1
    assert stats[1]["hits"] == 1


def test_v4_moe_wiring_present():
    # The V4 patch module is loaded by the takeover patcher, not by a direct
    # package import (it has in-package imports with no sibling base/cache
    # modules on disk). So we assert the wiring at the source level: the
    # shared cache is constructed in __init__ and consulted in __call__.
    from pathlib import Path

    src = Path("fusion_mlx/patches/deepseek_v4/deepseek_v4_model.py").read_text()
    assert "make_layer_cache" in src
    assert "_shared_cache.get_or_compute(x, self.shared_experts)" in src


def test_v32_moe_wiring_present():
    from pathlib import Path

    src = Path("fusion_mlx/patches/glm_moe_dsa/deepseek_v32.py").read_text()
    assert "make_layer_cache" in src
    assert "_shared_cache.get_or_compute(x, self.shared_experts)" in src


def test_metrics_route_renders_shared_cache(monkeypatch):
    # The /metrics renderer must include shared-cache series when stats exist
    # and omit them when zero (sparse-counter convention).
    _setup_env("1", monkeypatch)
    reset_stats()
    c = SharedExpertActivationCache(0)
    x = mx.zeros((3, 2), dtype=mx.float32)
    c.get_or_compute(x, lambda b: b)
    from fusion_mlx.routes_internal.metrics import (
        _render_moe_shared_cache_metrics,
    )

    lines = _render_moe_shared_cache_metrics()
    body = "\n".join(lines)
    assert "fusion_mlx_moe_shared_cache_requests_total" in body
    assert "fusion_mlx_moe_shared_cache_hit_rate" in body

    # Zero-stats case: no series.
    reset_stats()
    assert _render_moe_shared_cache_metrics() == []
