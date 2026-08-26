# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /distributed/decode_step + /distributed/reset_cache (#630).

Fast validation/route tests — no model load. Real-model bit-exact coverage
lives in test_distributed_decode_step_e2e.py (this file's sibling convention
follows test_distributed_pipeline.py, but e2e is split out for clarity)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")


def test_shard_info_exposes_kv_offset_zero_on_fresh_shard():
    """A freshly loaded shard has kv_cache=None → list_shards reports
    kv_offset=0."""
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    # Use a dummy shard entry to avoid a real model load in this unit test.
    mgr._shards["shard-fresh"] = {
        "shard_id": "shard-fresh",
        "model_id": "dummy",
        "shard_index": 0,
        "layer_range": [0, 4],
        "dtype": None,
        "num_layers": 16,
        "kv_cache": None,
    }
    from fusion_mlx.api.distributed_routes import ShardInfo

    info = ShardInfo(**mgr.list_shards()[0])
    assert info.kv_offset == 0


def _dummy_shard(mgr, shard_id="shard-x", start=0, end=4, total=16):
    """Register a dummy shard without loading a model (validation tests only)."""
    mgr._shards[shard_id] = {
        "shard_id": shard_id,
        "model_id": "dummy",
        "shard_index": 0,
        "layer_range": [start, end],
        "dtype": None,
        "num_layers": total,
        "kv_cache": None,
    }
    mgr._models["dummy"] = object()  # placeholder; validation fails before use
    return shard_id


def test_decode_step_rejects_both_input_modes():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError):
        mgr.decode_step(
            sid, hidden_states_b64="AAAA", input_ids=[1], is_last_shard=False
        )


def test_decode_step_rejects_neither_input_mode():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError):
        mgr.decode_step(
            sid, hidden_states_b64=None, input_ids=None, is_last_shard=False
        )


def test_decode_step_rejects_single_token_on_empty_cache():
    """Single-token input_ids with kv_cache=None is a decode call with no
    prefill — fail visibly (400), do not silently produce garbage attention."""
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError, match="prefill"):
        mgr.decode_step(sid, hidden_states_b64=None, input_ids=[42], is_last_shard=True)


def test_decode_step_unknown_shard_404():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.decode_step(
            "shard-nope", hidden_states_b64=None, input_ids=[1, 2], is_last_shard=False
        )
