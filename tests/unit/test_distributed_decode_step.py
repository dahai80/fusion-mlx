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


def test_reset_cache_idempotent_on_empty():
    """reset_cache on kv_cache=None is a no-op: prev_offset=0, no error."""
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    out = mgr.reset_cache(sid)
    assert out == {"shard_id": sid, "kv_cleared": True, "prev_offset": 0}
    # idempotent
    out2 = mgr.reset_cache(sid)
    assert out2 == {"shard_id": sid, "kv_cleared": True, "prev_offset": 0}


def test_reset_cache_unknown_shard_404():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.reset_cache("shard-nope")


def test_sync_weights_clears_kv_cache():
    """A weight swap invalidates cached K/V. sync_weights sets kv_cache=None
    (logged) even though its response is unchanged."""
    import base64

    from fusion_mlx.distributed import shard as shard_mod

    mgr = shard_mod.ShardManager()
    sid = _dummy_shard(mgr)
    # Simulate a populated cache (don't need a real model; just set the field).
    fake_cache = [type("C", (), {"offset": 5})() for _ in range(4)]
    mgr._shards[sid]["kv_cache"] = fake_cache
    # Build a minimal valid weights payload so sync_weights succeeds.
    import os
    import tempfile

    import mlx.core as mx

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    mx.savez(path, **{"model.layers.0.weight": mx.array([1.0])})
    try:
        with open(path, "rb") as fh:
            payload = base64.b64encode(fh.read()).decode("ascii")
    finally:
        os.unlink(path)

    # Need a real-ish model object whose load_weights won't crash on dummy.
    class _DummyModel:
        args = type("A", (), {"tie_word_embeddings": False})()

        def load_weights(self, items, strict=False):
            return None

    mgr._models["dummy"] = _DummyModel()
    out = mgr.sync_weights(sid, payload, None)
    assert out["params_updated"] == 1
    assert mgr._shards[sid]["kv_cache"] is None, "sync_weights must clear KV"
