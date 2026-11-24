# SPDX-License-Identifier: Apache-2.0
"""Tests for the LSA decode-time repair quota (#1707 / fusion-lsa #12).

Covers the serving-side plumbing: the ContextVar carrier, the scheduler-step
publisher (max quota of running requests, restored after the step), the
selector's page expansion (_append_repair), and the API request field.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("fusion_lsa")

from fusion_mlx.patches.qwen35_lsa.repair import (  # noqa: E402
    current_repair_pages,
    reset_repair_pages,
    set_repair_pages,
)


def test_contextvar_default_zero():
    assert current_repair_pages() == 0


def test_contextvar_set_reset_roundtrip():
    token = set_repair_pages(3)
    try:
        assert current_repair_pages() == 3
    finally:
        reset_repair_pages(token)
    assert current_repair_pages() == 0


def test_contextvar_negative_clamped():
    token = set_repair_pages(-5)
    try:
        assert current_repair_pages() == 0
    finally:
        reset_repair_pages(token)


def _config(**overrides):
    from fusion_mlx.patches.qwen35_lsa.selection import LsaSelectionConfig

    defaults = dict(
        pool_size=2,
        bucket_size=4,
        bucket_cap=8,
        top_p=16,
        top_k=64,
        sink_size=2,
        window_size=32,
    )
    defaults.update(overrides)
    return LsaSelectionConfig(**defaults)


def _make_selector(config, kv_bytes_per_token=256):
    from fusion_lsa.core.index_space import LearnedProjection

    from fusion_mlx.patches.qwen35_lsa.selection import LsaLayerSelector

    def _unit(rows):
        return rows / mx.linalg.norm(rows, axis=-1, keepdims=True)

    # index input dim = kv_heads * (dh - rotary_dim) = 1 * (32 - 8)
    projection = LearnedProjection(
        query_projection=_unit(mx.random.normal((24, 8), key=mx.random.key(42))),
        key_projection=_unit(mx.random.normal((24, 8), key=mx.random.key(7))),
    )
    return LsaLayerSelector(
        projection,
        config,
        rotary_dim=8,
        kv_heads=1,
        kv_bytes_per_token=kv_bytes_per_token,
    )


def _keys(tokens, dh=32, seed=3):
    key = mx.random.key(seed)
    rows = mx.random.normal((1, 1, tokens, dh), key=key)
    return rows / mx.linalg.norm(rows, axis=-1, keepdims=True)


def _query(position, dh=32, seed=9):
    key = mx.random.key(seed)
    rows = mx.random.normal((1, 2, 1, dh), key=key)
    return rows / mx.linalg.norm(rows, axis=-1, keepdims=True)


def test_repair_expands_whole_pages():
    # 4 closed pages (window_size=32, 128 keys appended); a repair quota of 2
    # must add every row of the 2 best-scoring closed pages to the picked set.
    config = _config(window_size=32)
    selector = _make_selector(config)
    k = _keys(128)
    selector.append_keys(k)
    assert selector.index.key_count == 128
    q = _query(140)
    base = selector.select(q, 140)
    picked = selector.select(q, 140, repair_count=2)
    base_set = set(int(x) for x in np.array(base) if x >= 0)
    picked_set = set(int(x) for x in np.array(picked) if x >= 0)
    assert base_set <= picked_set
    # Compute the expected repair pages exactly like _append_repair does:
    # static = pages already covered by base picked rows, repair = the top
    # scoring pages outside that static set.
    buckets = selector.index.buckets()
    per_page = buckets.block_bucket_count
    query = selector.index_queries(q)[:, 0, :]
    scores = query.astype(mx.float32) @ buckets.bucket_centroids.T
    page_scores = np.asarray(scores)[0][: 4 * per_page].reshape(4, per_page).max(axis=1)
    base_np = np.array(base)
    in_index = (base_np >= 0) & (base_np < 128)
    static = set(int(p) for p in np.unique(base_np[in_index] // 32))
    order = np.argsort(-page_scores, kind="stable")
    repair_ids = [int(p) for p in order if int(p) not in static][:2]
    assert not (set(repair_ids) & static), "repair pages must be outside static"
    for page in repair_ids:
        page_rows = set(range(page * 32, (page + 1) * 32))
        assert page_rows <= picked_set, f"page {page} rows missing from picked"


def test_repair_zero_bit_identical():
    # Regression guard: repair_count=0 must return the exact static-tier
    # selection (same picked rows, no extra page expansion).
    config = _config(window_size=32)
    selector = _make_selector(config)
    selector.append_keys(_keys(128))
    q = _query(100)
    without = np.array(selector.select(q, 100))
    with_zero = np.array(selector.select(q, 100, repair_count=0))
    assert np.array_equal(without, with_zero)


def test_repair_no_visible_pages_returns_picked():
    # position 0 with an empty index takes the window-only path; a repair
    # quota with no closed page visible must be a no-op, not a crash.
    config = _config(window_size=32)
    selector = _make_selector(config)
    q = _query(0)
    picked = selector.select(q, 0, repair_count=4)
    assert set(range(0, 1)) <= set(int(x) for x in np.array(picked) if x >= 0)


def test_scheduler_step_publishes_max_quota():
    # The scheduler step publishes max(_lsa_repair_pages) of running requests
    # into the ContextVar for the attention layer, and restores 0 afterwards.
    from fusion_mlx.scheduler.core import Scheduler

    class _Req:
        def __init__(self, rid, pages):
            self.request_id = rid
            self._lsa_repair_pages = pages

    sched = object.__new__(Scheduler)
    sched.running = {
        "a": _Req("a", 0),
        "b": _Req("b", 2),
        "c": _Req("c", 5),
    }
    seen = []
    sched._step_impl = lambda: seen.append(current_repair_pages()) or "out"
    assert sched.step() == "out"
    assert seen == [5]
    assert current_repair_pages() == 0


def test_scheduler_step_restores_zero_without_quota():
    from fusion_mlx.scheduler.core import Scheduler

    sched = object.__new__(Scheduler)
    sched.running = {}
    seen = []
    sched._step_impl = lambda: seen.append(current_repair_pages()) or "out"
    sched.step()
    assert seen == [0]
    assert current_repair_pages() == 0


def test_chat_request_field_validation():
    from pydantic import ValidationError

    from fusion_mlx.api.openai_models import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], lsa_repair_pages=4
    )
    assert req.lsa_repair_pages == 4
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert req.lsa_repair_pages is None
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            lsa_repair_pages=-1,
        )
