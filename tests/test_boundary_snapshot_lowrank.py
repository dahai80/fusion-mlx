import json

import numpy as np
import pytest

from fusion_mlx.cache.boundary_snapshot_store import (
    _LOWRANK_MIN_RANK,
    BoundarySnapshotSSDStore,
)

pytest.importorskip("mlx.core", reason="mlx required for snapshot store")


def _make_gdn_state(heads=4, dim=128, rank=3, seed=0):
    # Low-rank plus small noise: models a real GDN state slot
    # (1, heads, dim, dim) in bf16.
    rng = np.random.default_rng(seed)
    mat = np.zeros((1, heads, dim, dim), dtype=np.float32)
    for h in range(heads):
        a = rng.standard_normal((dim, rank))
        b = rng.standard_normal((rank, dim))
        mat[0, h] = a @ b + 0.01 * rng.standard_normal((dim, dim))
    import mlx.core as mx

    return mx.array(mat).astype(mx.bfloat16)


def _make_extracted(state):
    return [
        {
            "state": (state,),
            "meta_state": (),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
        }
    ]


def test_offset_roundtrip_through_disk(tmp_path):
    # Persisted write cursor must survive save -> disk -> load; rebuild
    # uses it to avoid continuing at over-allocated array length.

    store = BoundarySnapshotSSDStore(base_dir=tmp_path, lowrank_gdn_state=True)
    state = _make_gdn_state(heads=2, dim=64, seed=5)
    extracted = [
        {
            "state": (state,),
            "meta_state": (),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
            "offset": 1209,
        }
    ]
    assert store.save(
        "req-offset",
        1209,
        ["c"],
        lambda raw: (extracted, None),
    )
    import time

    deadline = time.time() + 10
    while time.time() < deadline and not store._file_registry.get("req-offset"):
        time.sleep(0.05)
    time.sleep(0.5)
    with store._pending_lock:
        store._pending_writes.pop(("req-offset", 1209), None)
    loaded = store.load("req-offset", 1209)
    assert loaded is not None
    assert loaded[0].get("offset") == 1209
    assert loaded[0]["state"][0].shape == state.shape
    store.shutdown()


def _serialize(store, extracted):
    return store._serialize_extracted(extracted, "req-test", 128)


def _factorized_metadata_info(metadata):
    return json.loads(metadata["layer_info"])[0]


def test_factorize_replaces_dense_with_smaller_factors():
    store = BoundarySnapshotSSDStore(
        base_dir=__import__("pathlib").Path("/tmp/fusion-mlx-test-lowrank"),
        lowrank_gdn_state=True,
    )
    state = _make_gdn_state()
    tensors_raw, metadata = _serialize(store, _make_extracted(state))
    dense_bytes = sum(len(v[0]) for v in tensors_raw.values())

    tensors_raw, metadata = store._factorize_gdn_state_tensors(
        dict(tensors_raw), dict(metadata)
    )

    assert "layer_0_state_0" not in tensors_raw
    assert {"layer_0_state_0_u", "layer_0_state_0_s", "layer_0_state_0_v"} <= set(
        tensors_raw
    )
    factored_bytes = sum(len(v[0]) for v in tensors_raw.values())
    assert factored_bytes < dense_bytes
    info = _factorized_metadata_info(metadata)
    assert "lowrank_0" in info
    spec = json.loads(info["lowrank_0"])
    assert spec["orig_shape"] == [1, 4, 128, 128]


def test_roundtrip_reconstruction_error_below_tail_energy():
    store = BoundarySnapshotSSDStore(
        base_dir=__import__("pathlib").Path("/tmp/fusion-mlx-test-lowrank"),
        lowrank_gdn_state=True,
    )
    import mlx.core as mx

    state = _make_gdn_state(seed=7)
    tensors_raw, metadata = _serialize(store, _make_extracted(state))
    tensors_raw, metadata = store._factorize_gdn_state_tensors(
        dict(tensors_raw), dict(metadata)
    )
    rebuilt = store._deserialize(tensors_raw, metadata)
    assert rebuilt is not None
    out = rebuilt[0]["state"][0]
    assert out.shape == state.shape
    diff = np.abs(
        np.array(out.astype(mx.float32).tolist(), dtype=np.float64)
        - np.array(state.astype(mx.float32).tolist(), dtype=np.float64)
    )
    scale = np.abs(np.array(state.astype(mx.float32).tolist())).max()
    # Truncation keeps 99.9% energy per head; bf16 factor rounding adds
    # a little more. Anything beyond 1% relative means the path is wrong.
    assert diff.max() / scale < 0.01


def test_rank_floor_respected():
    store = BoundarySnapshotSSDStore(
        base_dir=__import__("pathlib").Path("/tmp/fusion-mlx-test-lowrank"),
        lowrank_gdn_state=True,
    )
    state = _make_gdn_state(rank=1, seed=3)
    tensors_raw, metadata = _serialize(store, _make_extracted(state))
    tensors_raw, metadata = store._factorize_gdn_state_tensors(
        dict(tensors_raw), dict(metadata)
    )
    info = _factorized_metadata_info(metadata)
    spec = json.loads(info["lowrank_0"])
    # Energy rank is 1 here, but the probe-validated floor is 8.
    assert spec["rmax"] >= _LOWRANK_MIN_RANK


def test_disabled_flag_leaves_tensors_dense():
    store = BoundarySnapshotSSDStore(
        base_dir=__import__("pathlib").Path("/tmp/fusion-mlx-test-lowrank"),
        lowrank_gdn_state=False,
    )
    state = _make_gdn_state()
    tensors_raw, metadata = _serialize(store, _make_extracted(state))
    tensors_raw, metadata = store._factorize_gdn_state_tensors(
        dict(tensors_raw), dict(metadata)
    )
    assert "layer_0_state_0" in tensors_raw
    assert "lowrank_0" not in _factorized_metadata_info(metadata)


def test_non_candidate_tensors_untouched():
    store = BoundarySnapshotSSDStore(
        base_dir=__import__("pathlib").Path("/tmp/fusion-mlx-test-lowrank"),
        lowrank_gdn_state=True,
    )
    import mlx.core as mx

    kv = mx.zeros((1, 8, 64, 128), dtype=mx.bfloat16)  # non-square last dims
    gdn = _make_gdn_state(heads=2, dim=64)
    extracted = [
        {
            "state": (kv, gdn),
            "meta_state": (),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
        }
    ]
    tensors_raw, metadata = _serialize(store, extracted)
    tensors_raw, metadata = store._factorize_gdn_state_tensors(
        dict(tensors_raw), dict(metadata)
    )
    # KV-shaped element stays dense; the square GDN slot is factorized.
    assert "layer_0_state_0" in tensors_raw
    assert "layer_0_state_1" not in tensors_raw
    assert "layer_0_state_1_u" in tensors_raw


def test_full_disk_roundtrip_through_writer_thread(tmp_path):
    import mlx.core as mx

    store = BoundarySnapshotSSDStore(base_dir=tmp_path, lowrank_gdn_state=True)
    state = _make_gdn_state(heads=2, dim=128, seed=11)
    extracted = _make_extracted(state)
    assert store.save(
        "req-e2e",
        64,
        ["sentinel-raw-cache"],
        lambda raw_cache: (extracted, None),
    )
    # Wait for the writer to flush to disk, then drop the in-memory
    # pending copy so load() must read the compressed file back.
    import time

    deadline = time.time() + 10
    while time.time() < deadline and not store._file_registry.get("req-e2e"):
        time.sleep(0.05)
    time.sleep(0.5)
    file_path = store._file_registry["req-e2e"][64]
    disk_bytes = file_path.stat().st_size
    with store._pending_lock:
        store._pending_writes.pop(("req-e2e", 64), None)
    loaded = store.load("req-e2e", 64)
    assert loaded is not None
    out = loaded[0]["state"][0]
    assert out.shape == state.shape
    diff = np.abs(
        np.array(out.astype(mx.float32).tolist(), dtype=np.float64)
        - np.array(state.astype(mx.float32).tolist(), dtype=np.float64)
    )
    scale = np.abs(np.array(state.astype(mx.float32).tolist(), dtype=np.float64)).max()
    # bf16 factor rounding + 0.1% tail truncation lands around 1-2%
    # relative state error; probe9 showed generation robust to state
    # perturbations orders of magnitude larger (83% on the worst head).
    assert diff.max() / scale < 0.02
    dense_bytes = 2 * 2 * 128 * 128  # bf16, 2 heads, 128x128
    assert disk_bytes < dense_bytes
    store.shutdown()
