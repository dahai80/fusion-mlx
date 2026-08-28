# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /distributed/kv_cache/export + /import (#650).

Fast validation/route/round-trip tests — no model load (mock cache objects).
Real-model bit-exact round-trip coverage lives in the *_e2e sibling functions
at the bottom, gated to FUSION_MLX_REAL_MODEL_TESTS per the #630 convention."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mlx.core")


def _dummy_shard(mgr, shard_id="shard-kv", start=0, end=4, total=16):
    mgr._shards[shard_id] = {
        "shard_id": shard_id,
        "model_id": "dummy",
        "shard_index": 0,
        "layer_range": [start, end],
        "dtype": None,
        "num_layers": total,
        "kv_cache": None,
    }
    mgr._models["dummy"] = object()
    return shard_id


class _FakeCache:
    """Minimal stand-in for KVCache: .state returns (k, v); .offset tracks
    seq_len. state setter writes keys/values/offset like the real class."""

    def __init__(self, seq=0):
        self.keys = None
        self.values = None
        self.offset = seq

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]


def _populated_shard(mgr, sid, seq_len=5, n_layers=16, slice_start=0):
    """Register a shard with a fake populated KV cache list."""
    cache = [_FakeCache(seq=seq_len) for _ in range(n_layers)]
    import mlx.core as mx

    for i in range(n_layers):
        k = mx.zeros((1, 4, seq_len, 16), dtype=mx.float32)
        v = mx.zeros((1, 4, seq_len, 16), dtype=mx.float32)
        k += i  # distinct per layer so round-trip can detect ordering
        mx.eval(k, v)  # materialize in main-thread stream; route handlers run
        # serialize (mx.save) in a worker thread that has no MLX Stream(gpu,0),
        # so the arrays must be eval'd here first (#630 stream GOTCHA).
        cache[i].keys = k
        cache[i].values = v
    mgr._shards[sid]["kv_cache"] = cache
    mgr._shards[sid]["layer_range"] = [slice_start, slice_start + 4]
    mgr._shards[sid]["num_layers"] = n_layers
    return cache


# ---- ShardManager.export_kv_cache ----


def test_export_rejects_empty_cache():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError, match="no active KV cache"):
        mgr.export_kv_cache(sid)


def test_export_unknown_shard_404():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.export_kv_cache("shard-nope")


def test_export_rejects_bad_layer_range():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr, start=4, end=8)
    _populated_shard(mgr, sid, seq_len=5, slice_start=4)
    # 3-element range
    with pytest.raises(ShardError, match="must be"):
        mgr.export_kv_cache(sid, layer_range=[4, 6, 8])
    # outside slice
    with pytest.raises(ShardError, match="outside shard slice"):
        mgr.export_kv_cache(sid, layer_range=[0, 6])
    # inverted
    with pytest.raises(ShardError, match="outside shard slice"):
        mgr.export_kv_cache(sid, layer_range=[6, 4])


def test_export_returns_serialized_layers_and_seq_len():
    import mlx.core as mx

    from fusion_mlx.distributed.shard import ShardManager, deserialize_activation

    mgr = ShardManager()
    sid = _dummy_shard(mgr, start=0, end=4, total=16)
    cache = _populated_shard(mgr, sid, seq_len=5, n_layers=16, slice_start=0)
    out = mgr.export_kv_cache(sid)
    assert out["shard_id"] == sid
    assert out["seq_len"] == 5
    assert len(out["layers"]) == 4
    # layer indices are the shard's slice [0,4)
    assert [l["layer"] for l in out["layers"]] == [0, 1, 2, 3]
    # round-trip a key tensor: deserialize == original
    k0 = deserialize_activation(out["layers"][0]["keys"])
    assert mx.array_equal(k0, cache[0].keys)
    assert out["layers"][0]["shape"] == [1, 4, 5, 16]
    assert out["layers"][0]["dtype"] == "mlx.core.float32"


def test_export_layer_range_subset():
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr, start=0, end=4, total=16)
    _populated_shard(mgr, sid, seq_len=5, n_layers=16, slice_start=0)
    out = mgr.export_kv_cache(sid, layer_range=[1, 3])
    assert [l["layer"] for l in out["layers"]] == [1, 2]
    assert out["seq_len"] == 5


# ---- ShardManager.import_kv_cache ----


def test_import_unknown_shard_404():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    with pytest.raises(ShardError):
        mgr.import_kv_cache("shard-nope", [], 5)


def test_import_rejects_empty_layers():
    from fusion_mlx.distributed.shard import ShardError, ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    with pytest.raises(ShardError, match="empty layers"):
        mgr.import_kv_cache(sid, [], 5)


def test_import_rejects_layer_outside_slice():
    import mlx.core as mx

    from fusion_mlx.distributed.shard import (
        ShardError,
        ShardManager,
        serialize_activation,
    )

    mgr = ShardManager()
    sid = _dummy_shard(mgr, start=4, end=8, total=16)
    k = mx.zeros((1, 4, 5, 16), dtype=mx.float32)
    v = mx.zeros((1, 4, 5, 16), dtype=mx.float32)
    entry = {
        "layer": 0,  # outside [4,8)
        "keys": serialize_activation(k),
        "values": serialize_activation(v),
        "shape": [1, 4, 5, 16],
        "dtype": "float32",
    }
    with pytest.raises(ShardError, match="outside shard slice"):
        mgr.import_kv_cache(sid, [entry], 5)


def test_import_rejects_seq_len_mismatch():
    import mlx.core as mx

    from fusion_mlx.distributed.shard import (
        ShardError,
        ShardManager,
        serialize_activation,
    )

    mgr = ShardManager()
    sid = _dummy_shard(mgr, start=0, end=4, total=16)
    k = mx.zeros((1, 4, 3, 16), dtype=mx.float32)  # len 3
    v = mx.zeros((1, 4, 3, 16), dtype=mx.float32)
    entry = {
        "layer": 0,
        "keys": serialize_activation(k),
        "values": serialize_activation(v),
        "shape": [1, 4, 3, 16],
        "dtype": "float32",
    }
    with pytest.raises(ShardError, match="!= seq_len"):
        mgr.import_kv_cache(sid, [entry], 5)  # seq_len=5


def test_import_lazy_inits_cache_and_restores_tensors():
    import mlx.core as mx

    from fusion_mlx.distributed.shard import (
        ShardManager,
        serialize_activation,
    )

    mgr = ShardManager()
    sid = _dummy_shard(mgr, start=0, end=4, total=16)
    assert mgr._shards[sid]["kv_cache"] is None

    seq = 7
    layers = []
    for li in range(0, 4):
        k = mx.ones((1, 4, seq, 16), dtype=mx.float32) * li
        v = mx.ones((1, 4, seq, 16), dtype=mx.float32) * (li + 100)
        layers.append(
            {
                "layer": li,
                "keys": serialize_activation(k),
                "values": serialize_activation(v),
                "shape": [1, 4, seq, 16],
                "dtype": "float32",
            }
        )
    out = mgr.import_kv_cache(sid, layers, seq)
    assert out["imported_layers"] == 4
    assert out["seq_len"] == seq
    cache = mgr._shards[sid]["kv_cache"]
    assert cache is not None
    assert len(cache) == 16
    for li in range(4):
        assert int(cache[li].offset) == seq
        k_back = cache[li].state[0]
        assert k_back.shape == (1, 4, seq, 16)
        assert float(k_back.max()) == float(li)


def test_export_import_round_trip_tensor_equal():
    """Full round-trip on mock caches: export A, import into B's empty cache,
    assert tensor-equal per layer."""
    import mlx.core as mx

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid_a = _dummy_shard(mgr, shard_id="shard-a", start=0, end=4, total=16)
    sid_b = _dummy_shard(mgr, shard_id="shard-b", start=0, end=4, total=16)
    cache_a = _populated_shard(mgr, sid_a, seq_len=5, n_layers=16, slice_start=0)

    exported = mgr.export_kv_cache(sid_a)
    mgr.import_kv_cache(sid_b, exported["layers"], exported["seq_len"])
    cache_b = mgr._shards[sid_b]["kv_cache"]
    for li in range(4):
        ka, va = cache_a[li].state
        kb, vb = cache_b[li].state
        assert mx.array_equal(ka, kb), f"layer {li} keys differ"
        assert mx.array_equal(va, vb), f"layer {li} values differ"


# ---- routes (TestClient) ----


def _client_with_manager(mgr, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import fusion_mlx.distributed.shard as shard_mod
    from fusion_mlx.api import distributed_routes as dr
    from fusion_mlx.middleware.auth import verify_api_key

    shard_mod._manager = mgr
    app = FastAPI()
    app.include_router(dr.router)
    app.dependency_overrides[verify_api_key] = lambda: None
    return TestClient(app)


def test_export_route_rejects_empty_cache_400(monkeypatch):
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid = _dummy_shard(mgr)
    client = _client_with_manager(mgr, monkeypatch)
    r = client.post("/distributed/kv_cache/export", json={"shard_id": sid})
    assert r.status_code == 400
    assert "no active KV cache" in r.json()["detail"]


def test_export_route_unknown_shard_404(monkeypatch):
    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    client = _client_with_manager(mgr, monkeypatch)
    r = client.post("/distributed/kv_cache/export", json={"shard_id": "shard-nope"})
    assert r.status_code == 404


def test_export_import_route_round_trip_200(monkeypatch):

    from fusion_mlx.distributed.shard import ShardManager

    mgr = ShardManager()
    sid_a = _dummy_shard(mgr, shard_id="shard-a", start=0, end=4, total=16)
    sid_b = _dummy_shard(mgr, shard_id="shard-b", start=0, end=4, total=16)
    _populated_shard(mgr, sid_a, seq_len=5, n_layers=16, slice_start=0)
    client = _client_with_manager(mgr, monkeypatch)

    r = client.post("/distributed/kv_cache/export", json={"shard_id": sid_a})
    assert r.status_code == 200
    body = r.json()
    assert body["shard_id"] == sid_a
    assert body["seq_len"] == 5
    assert len(body["layers"]) == 4

    r2 = client.post(
        "/distributed/kv_cache/import",
        json={
            "shard_id": sid_b,
            "layers": body["layers"],
            "seq_len": body["seq_len"],
        },
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["imported_layers"] == 4
    assert body2["seq_len"] == 5
    cache_b = mgr._shards[sid_b]["kv_cache"]
    assert cache_b is not None
    for li in range(4):
        assert int(cache_b[li].offset) == 5


# ---- real-model round-trip (gated) ----


_MODEL_CANDIDATES = ["models--mlx-community--Llama-3.2-1B-Instruct-4bit"]


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


def _skip_unless_real_model():
    if not os.environ.get("FUSION_MLX_REAL_MODEL_TESTS"):
        pytest.skip(
            "set FUSION_MLX_REAL_MODEL_TESTS=1 to run real-model KV export/import e2e"
        )
    if _find_small_lm() is None:
        pytest.skip("no small LM with safetensors found in model dir")


@pytest.mark.real_model
def test_kv_cache_export_import_resume_decode_matches_no_reset():
    """Prefill A, export KV, reset A, import KV back, decode one more token.
    The resumed token must equal decoding WITHOUT the reset+import (KV state
    restored bit-exactly). This is the #650 acceptance: import restores the
    prefix so decode continues as if computed locally."""
    _skip_unless_real_model()
    import mlx_lm

    from fusion_mlx.distributed.shard import ShardManager

    lm_path = _find_small_lm()
    mgr = ShardManager()
    _model, tok = mlx_lm.load(lm_path)
    total = len(_model.model.layers)
    info = mgr.load_shard(lm_path, 0, [0, total])

    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt)

    # prefill -> token 1
    out = mgr.decode_step(
        info["shard_id"], None, prompt_ids, is_last_shard=True, temperature=0.0
    )
    tok_id = out["token_ids"][0]
    assert out["kv_offset"] == len(prompt_ids)

    # baseline: continue decode WITHOUT export/reset (the reference)
    base_out = mgr.decode_step(
        info["shard_id"], None, [tok_id], is_last_shard=True, temperature=0.0
    )
    base_next = base_out["token_ids"][0]

    # now: export the KV (has prompt cached), reset, import back, decode same
    # single token. Must reproduce base_next.
    mgr.reset_cache(info["shard_id"])
    # after reset cache is None -> export must fail visibly
    from fusion_mlx.distributed.shard import ShardError

    with pytest.raises(ShardError, match="no active KV cache"):
        mgr.export_kv_cache(info["shard_id"])

    # re-prefill to rebuild a cache, then export BEFORE decoding the extra token
    mgr.decode_step(
        info["shard_id"], None, prompt_ids, is_last_shard=True, temperature=0.0
    )
    exported = mgr.export_kv_cache(info["shard_id"])
    assert exported["seq_len"] == len(prompt_ids)
    assert len(exported["layers"]) == total

    mgr.reset_cache(info["shard_id"])
    imported = mgr.import_kv_cache(
        info["shard_id"], exported["layers"], exported["seq_len"]
    )
    assert imported["imported_layers"] == total
    assert imported["seq_len"] == len(prompt_ids)

    # decode the SAME single token on the imported cache
    imp_out = mgr.decode_step(
        info["shard_id"], None, [tok_id], is_last_shard=True, temperature=0.0
    )
    imp_next = imp_out["token_ids"][0]
    assert (
        imp_next == base_next
    ), f"KV export/import resume decode {imp_next} != baseline {base_next}"
    assert imp_out["kv_offset"] == len(prompt_ids) + 1
