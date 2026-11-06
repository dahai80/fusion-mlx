# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen3_5 LSA selection glue: incremental index + decode selection."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

# fusion-lsa is a sibling monorepo package (pip install -e ~/fusion/fusion-lsa),
# not a fusion-mlx dependency — CI runners don't have it. Skip instead of
# erroring at collection; these tests run on dev machines with it installed.
pytest.importorskip("fusion_lsa")

from fusion_lsa.core.index_space import LearnedProjection  # noqa: E402
from fusion_lsa.core.reference import build_buckets, read  # noqa: E402


def _unit(rows: mx.array) -> mx.array:
    return rows / mx.linalg.norm(rows, axis=-1, keepdims=True)


def _make_selector(
    T_total_free: int, index_dim: int, rotary_dim: int, kv_heads: int, config
):
    from fusion_mlx.patches.qwen35_lsa.selection import LsaLayerSelector

    key = mx.random.key(42)
    wq = _unit(mx.random.normal((T_total_free, index_dim), key=key))
    wk = _unit(mx.random.normal((T_total_free, index_dim), key=mx.random.key(7)))
    projection = LearnedProjection(query_projection=wq, key_projection=wk)
    return LsaLayerSelector(
        projection, config, rotary_dim=rotary_dim, kv_heads=kv_heads
    )


def _synthetic_inputs(kv_heads: int, q_heads: int, tokens: int, dh: int, seed: int = 3):
    key = mx.random.key(seed)
    k = _unit(mx.random.normal((1, kv_heads, tokens, dh), key=key))
    q = _unit(mx.random.normal((1, q_heads, tokens, dh), key=mx.random.key(seed + 1)))
    return k, q


def _config(**overrides):
    from fusion_mlx.patches.qwen35_lsa.selection import LsaSelectionConfig

    defaults = dict(
        pool_size=2,
        bucket_size=4,
        bucket_cap=8,
        top_p=16,
        top_k=64,
        sink_size=2,
        window_size=64,
    )
    defaults.update(overrides)
    return LsaSelectionConfig(**defaults)


def test_selection_matches_batch_oracle():
    kv_heads, q_heads, tokens, dh, rotary = 2, 6, 2048, 64, 16
    free = dh - rotary
    config = _config()
    k, q = _synthetic_inputs(kv_heads, q_heads, tokens, dh)
    selector = _make_selector(kv_heads * free, 8, rotary, kv_heads, config)

    # oracle: whole-document index space, batch buckets, batch read
    positions = mx.arange(256, tokens, 97)
    index_q = selector.index_queries(q[:, :, positions])[0]
    buckets = build_buckets(
        selector.index_keys(k)[0],
        config.pool_size,
        config.bucket_size,
        config.bucket_cap,
        config.window_size,
    )
    want, _ = read(
        index_q,
        positions,
        buckets,
        config.top_p,
        config.top_k,
        config.sink_size,
        config.window_size,
    )

    selector.append_keys(k)
    got_rows = []
    for position in positions.tolist():
        got_rows.append(
            np.array(selector.select(q[:, :, position : position + 1], position))
        )
    want_rows = np.array(want)
    for row in range(want_rows.shape[0]):
        want_set = set(int(x) for x in want_rows[row] if x >= 0)
        got_set = set(int(x) for x in got_rows[row] if x >= 0)
        assert (
            want_set == got_set
        ), f"position {positions[row].item()}: {len(want_set)} vs {len(got_set)}"


def test_midpage_tail_always_selected():
    kv_heads, q_heads, tokens, dh, rotary = 2, 6, 1000, 64, 16
    free = dh - rotary
    config = _config(window_size=512, top_k=64, top_p=16)
    k, q = _synthetic_inputs(kv_heads, q_heads, tokens, dh)
    selector = _make_selector(kv_heads * free, 8, rotary, kv_heads, config)
    selector.append_keys(k)
    assert selector.index.key_count == 512, "one closed page of 512"
    for position in (600, 800, 999):
        picked = set(
            int(x)
            for x in np.array(
                selector.select(q[:, :, position : position + 1], position)
            )
            if x >= 0
        )
        tail = set(range(512, position + 1))
        assert (
            tail <= picked
        ), f"position {position}: {len(tail - picked)} tail keys missing"


def test_small_config_page_semantics():
    kv_heads, q_heads, tokens, dh, rotary = 1, 2, 200, 32, 8
    free = dh - rotary
    config = _config(pool_size=2, bucket_size=4, window_size=32, top_k=32, top_p=8)
    k, q = _synthetic_inputs(kv_heads, q_heads, tokens, dh)
    selector = _make_selector(kv_heads * free, 8, rotary, kv_heads, config)
    selector.append_keys(k[:, :, :128])
    assert selector.index.key_count == 128
    selector.append_keys(k[:, :, 128:])
    assert (
        selector.index.key_count == 192
    ), "64 more keys close a page, 8 stay in the tail"
    assert selector._tail_rows == 8
    picked = set(
        int(x) for x in np.array(selector.select(q[:, :, 199:200], 199)) if x >= 0
    )
    assert len(picked) > 0


def test_rejects_wrong_head_count():
    kv_heads, q_heads, tokens, dh, rotary = 2, 6, 512, 64, 16
    free = dh - rotary
    config = _config()
    k, q = _synthetic_inputs(kv_heads, q_heads, tokens, dh)
    selector = _make_selector(kv_heads * free, 8, rotary, kv_heads, config)
    with __import__("pytest").raises(ValueError):
        selector.index_queries(q[:, : q_heads - 1])
