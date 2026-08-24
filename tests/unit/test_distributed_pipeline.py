# SPDX-License-Identifier: Apache-2.0
"""Round-trip test for the distributed pipeline-parallelism surface (#621).

Verifies the core invariant Pipeline Parallelism depends on: splitting a
transformer forward at a layer boundary, serializing the intermediate hidden
states through the HTTP activation format (base64 .npy), and continuing on
the next shard reproduces the un-split forward bit-exactly.

Runs against a REAL small LM (mlx-community/Llama-3.2-1B-Instruct-4bit, ~0.7GB)
so the layer-slicing forward exercises actual quantized weights. Skipped when
the model is absent or mlx is unavailable.

This is the single-machine round-trip the issue acceptance asks for:
"激活张量能跨节点 round-trip（数值一致）" — here "cross-node" is simulated by
the serialize→deserialize hop between two shard forward calls.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mlx.core")

# Resolve the small LM from the configured model dir.
_MODEL_CANDIDATES = [
    "models--mlx-community--Llama-3.2-1B-Instruct-4bit",
    "models--mlx-community--Qwen3-0.6B-8bit",
]


def _find_small_lm() -> str | None:
    base = os.path.expanduser(
        os.environ.get("FUSION_MLX_MODEL_DIR", "~/.fusion-mlx/models")
    )
    for name in _MODEL_CANDIDATES:
        snap_root = os.path.join(base, name, "snapshots")
        if not os.path.isdir(snap_root):
            continue
        for snap in os.listdir(snap_root):
            snap_dir = os.path.join(snap_root, snap)
            if any(f.endswith(".safetensors") for f in os.listdir(snap_dir)):
                return snap_dir
    return None


_LM_PATH = _find_small_lm()

skip_no_model = pytest.mark.skipif(
    _LM_PATH is None, reason="no small LM with safetensors found in model dir"
)


@skip_no_model
def test_activation_roundtrip_preserves_dtype_and_values():
    """serialize→deserialize is bit-exact across float32 / bfloat16 / int32 —
    the property that makes cross-node activation transfer safe."""
    import mlx.core as mx

    from fusion_mlx.distributed.shard import (
        deserialize_activation,
        serialize_activation,
    )

    cases = [
        mx.array([[1.5, 2.0], [3.0, 4.0]], dtype=mx.float32),
        mx.array([[1.5, 2.0], [3.0, 4.0]], dtype=mx.bfloat16),
        mx.array([0, 1, 2, 3, 100], dtype=mx.int32),
        mx.zeros((1, 7, 2048), dtype=mx.float32),
    ]
    for arr in cases:
        got = deserialize_activation(serialize_activation(arr))
        assert mx.array_equal(arr, got), f"{arr.dtype} round-trip mismatch"
        assert str(got.dtype) == str(arr.dtype)
        assert list(got.shape) == list(arr.shape)


@skip_no_model
def test_deserialize_rejects_garbage():
    from fusion_mlx.distributed.shard import ShardError, deserialize_activation

    with pytest.raises(ShardError):
        deserialize_activation("!!!not-base64!!!")
    with pytest.raises(ShardError):
        deserialize_activation("")


@skip_no_model
def test_pipeline_split_matches_unsplit_forward():
    """The acceptance test: forward split at a layer boundary, with a
    serialize→deserialize hop at the boundary, equals the un-split forward."""
    import mlx.core as mx
    import mlx_lm

    from fusion_mlx.distributed.shard import (
        deserialize_activation,
        serialize_activation,
    )

    model, _tok = mlx_lm.load(_LM_PATH)
    inner = model.model
    total = len(inner.layers)
    split = total // 2  # e.g. 8 of 16 for Llama-3.2-1B

    input_ids = [1, 2, 3, 4, 5, 6, 7]
    ids = mx.array(input_ids, dtype=mx.int32)

    # Reference: un-split forward over ALL layers.
    ref = inner.embed_tokens(ids[None, :])
    for i in range(total):
        ref = inner.layers[i](ref)
    mx.eval(ref)

    # Split: shard A = layers [0, split), shard B = layers [split, total),
    # with a serialize→deserialize hop (simulating the cross-node transfer).
    h_a = inner.embed_tokens(ids[None, :])
    for i in range(0, split):
        h_a = inner.layers[i](h_a)
    mx.eval(h_a)
    transferred = deserialize_activation(serialize_activation(h_a))
    h_b = transferred
    for i in range(split, total):
        h_b = inner.layers[i](h_b)
    mx.eval(h_b)

    assert list(h_b.shape) == list(
        ref.shape
    ), f"shape mismatch: split {h_b.shape} vs ref {ref.shape}"
    assert str(h_b.dtype) == str(ref.dtype)
    # Quantized weights are deterministic; the split must reproduce the
    # un-split forward bit-exactly (same op graph, just split at a boundary
    # where the intermediate is fully materialized).
    assert mx.array_equal(h_b, ref), "split forward diverged from un-split reference"


@skip_no_model
def test_shard_manager_load_step_round_trip():
    """End-to-end through the ShardManager: load two shards of the same model,
    run shard A (first, embeds) → serialize → shard B (continues) → compare
    to a direct full forward. This is the single-machine round-trip the
    endpoints compose."""
    import mlx.core as mx
    import mlx_lm

    from fusion_mlx.distributed.shard import (
        ShardManager,
        deserialize_activation,
    )

    mgr = ShardManager()
    model_id = _LM_PATH
    model, _tok = mlx_lm.load(model_id)
    total = len(model.model.layers)
    split = total // 2

    info_a = mgr.load_shard(model_id, shard_index=0, layer_range=[0, split])
    info_b = mgr.load_shard(model_id, shard_index=1, layer_range=[split, total])
    assert info_a["layer_range"] == [0, split]
    assert info_b["layer_range"] == [split, total]
    assert info_a["num_layers"] == total

    input_ids = [10, 20, 30, 40]

    # shard A: first shard, embeds + runs [0, split)
    out_a = mgr.pipeline_step(info_a["shard_id"], None, input_ids, None)
    assert out_a["shape"][0] == 1  # batch
    assert out_a["shape"][1] == len(input_ids)  # seq
    assert out_a["dtype"].startswith("mlx")

    # shard B: receives A's hidden states, runs [split, total)
    out_b = mgr.pipeline_step(info_b["shard_id"], out_a["hidden_states"], None, None)
    assert len(out_b["shape"]) == 3

    # Reference full forward.
    ids = mx.array(input_ids, dtype=mx.int32)
    ref = model.model.embed_tokens(ids[None, :])
    for i in range(total):
        ref = model.model.layers[i](ref)
    mx.eval(ref)

    final = deserialize_activation(out_b["hidden_states"])
    assert list(final.shape) == list(ref.shape)
    assert str(final.dtype) == str(ref.dtype)
    assert mx.array_equal(
        final, ref
    ), "shard-manager round-trip diverged from reference"

    # idempotent load_shard reuse
    info_a2 = mgr.load_shard(model_id, shard_index=0, layer_range=[0, split])
    assert info_a2["shard_id"] == info_a["shard_id"]

    # list + drop
    assert len(mgr.list_shards()) == 2
    mgr.drop_shard(info_a["shard_id"])
    assert len(mgr.list_shards()) == 1


@skip_no_model
def test_shard_manager_rejects_bad_ranges():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.load_shard(_LM_PATH, 0, [0])  # not a pair
    with pytest.raises(ShardError):
        mgr.load_shard(_LM_PATH, 0, [5, 2])  # end <= start
    with pytest.raises(ShardError):
        mgr.load_shard(_LM_PATH, 0, [-1, 2])  # negative


@skip_no_model
def test_pipeline_step_unknown_shard_errors():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.pipeline_step("shard-does-not-exist", None, [1, 2, 3], None)


@skip_no_model
def test_first_shard_requires_input_ids():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    info = mgr.load_shard(_LM_PATH, 0, [0, 4])
    with pytest.raises(ShardError):
        mgr.pipeline_step(info["shard_id"], None, None, None)  # no ids, no hidden


# --- security: path-traversal confinement (#621 hardening) ---


def test_resolve_model_path_rejects_traversal():
    """model_id with '..' escaping the allowed roots must be rejected —
    never reaches mlx_lm.load."""
    from fusion_mlx.distributed.shard import ShardError, _resolve_model_path

    with pytest.raises(ShardError):
        _resolve_model_path("../../../etc/passwd")
    with pytest.raises(ShardError):
        _resolve_model_path("/etc/passwd")
    with pytest.raises(ShardError):
        _resolve_model_path("")


def test_resolve_model_path_accepts_allowed_root():
    """A snapshot path under ~/.fusion-mlx/models is confined and accepted."""
    from fusion_mlx.distributed.shard import _resolve_model_path

    resolved = _resolve_model_path(_LM_PATH)
    assert os.path.realpath(_LM_PATH) == resolved


def test_load_shard_rejects_traversal_before_load():
    """load_shard with a traversal model_id raises ShardError without
    attempting a model load (no spurious cache entry)."""
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.load_shard("../../../../etc/passwd", 0, [0, 4])
    assert "../../../../etc/passwd" not in mgr._models


# --- security: payload size caps (#621 hardening) ---


def test_deserialize_rejects_oversized_activation(monkeypatch):
    import base64

    from fusion_mlx.distributed import shard as shard_mod

    monkeypatch.setattr(shard_mod, "_MAX_ACTIVATION_BYTES", 8)
    # 16 bytes of valid base64 -> decodes to 12 bytes > cap 8.
    payload = base64.b64encode(b"x" * 12).decode("ascii")
    with pytest.raises(shard_mod.ShardError):
        shard_mod.deserialize_activation(payload)


def test_sync_weights_rejects_oversized(monkeypatch):
    import base64

    from fusion_mlx.distributed import shard as shard_mod

    monkeypatch.setattr(shard_mod, "_MAX_WEIGHTS_BYTES", 8)
    mgr = shard_mod.ShardManager()
    # Register a dummy shard entry so sync_weights gets past the lookup.
    mgr._shards["shard-x"] = {
        "shard_id": "shard-x",
        "model_id": "dummy",
        "shard_index": 0,
        "layer_range": [0, 1],
        "dtype": None,
        "num_layers": 1,
    }
    mgr._models["dummy"] = object()  # placeholder; never reaches load_weights
    payload = base64.b64encode(b"x" * 64).decode("ascii")
    with pytest.raises(shard_mod.ShardError):
        mgr.sync_weights("shard-x", payload, None)
