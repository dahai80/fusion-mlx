# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from fusion_mlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from fusion_mlx.scheduler.sched_boundary import _try_prefix_snapshot_warm_start

try:
    import mlx.core as mx

    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

pytestmark = pytest.mark.skipif(not _HAS_MLX, reason="MLX not available")

_BLOCK_SIZE = 32
_MODEL_NAME = "test-model"


def _make_extracted(token_count: int, num_layers: int = 2) -> list[dict]:
    states = []
    for i in range(num_layers):
        k = mx.array([float(token_count), float(i), 1.0, 2.0])
        v = mx.array([float(token_count + i), 3.0, 4.0, 5.0])
        states.append(
            {
                "state": (k, v),
                "meta_state": (token_count, num_layers),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            }
        )
    return states


def _extract_fn(snapshot_cache):
    return snapshot_cache, None


def _prefix_hash(token_count: int) -> bytes:
    token_ids = list(range(token_count))
    hashes = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
        token_ids, _BLOCK_SIZE, _MODEL_NAME
    )
    assert hashes, "expected at least one full-block boundary"
    return hashes[-1]


def _wait_for(store, prefix_hash, token_count, timeout=5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.has_prefix(prefix_hash, token_count):
            return True
        time.sleep(0.02)
    return False


def test_chain_hashes_deterministic_and_model_scoped() -> None:
    ids = list(range(128))
    h1 = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
        ids, _BLOCK_SIZE, _MODEL_NAME
    )
    h2 = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
        ids, _BLOCK_SIZE, _MODEL_NAME
    )
    assert h1 == h2
    assert len(h1) == 128 // _BLOCK_SIZE
    assert all(len(h) == 32 for h in h1)
    h_other = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
        ids, _BLOCK_SIZE, "other-model"
    )
    assert h1 != h_other
    assert BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
        [], _BLOCK_SIZE, _MODEL_NAME
    ) == []


def test_prefix_persist_disabled_is_noop(tmp_path) -> None:
    store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=False)
    try:
        h = _prefix_hash(64)
        assert store.save_prefix(h, 64, _make_extracted(64), _extract_fn, _MODEL_NAME) is False
        assert store.load_prefix(h, 64) is None
        assert store.has_prefix(h, 64) is False
        assert store.find_prefix_snapshot([h]) is None
        assert store.get_prefix_stats() == {}
    finally:
        store.shutdown()


def test_prefix_roundtrip_preserves_states(tmp_path) -> None:
    store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
    try:
        extracted = _make_extracted(64)
        h = _prefix_hash(64)
        assert store.save_prefix(h, 64, extracted, _extract_fn, _MODEL_NAME) is True
        assert _wait_for(store, h, 64)
        loaded = store.load_prefix(h, 64)
        assert loaded is not None
        assert len(loaded) == len(extracted)
        for orig, got in zip(extracted, loaded):
            assert got["class_name"] == orig["class_name"]
            assert got["cache_type"] == orig["cache_type"]
            assert tuple(got["meta_state"]) == tuple(orig["meta_state"])
            assert len(got["state"]) == len(orig["state"])
            for a, b in zip(got["state"], orig["state"]):
                assert mx.array_equal(a, b)
        stats = store.get_prefix_stats()
        assert stats["writes"] == 1
        assert stats["entries"] == 1
        assert stats["total_bytes"] > 0
    finally:
        store.shutdown()


def test_find_prefix_snapshot_returns_longest(tmp_path) -> None:
    store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
    try:
        for tc in (64, 128, 192):
            h = _prefix_hash(tc)
            assert store.save_prefix(h, tc, _make_extracted(tc), _extract_fn, _MODEL_NAME)
            assert _wait_for(store, h, tc)
        full_chain = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
            list(range(192)), _BLOCK_SIZE, _MODEL_NAME
        )
        match = store.find_prefix_snapshot(full_chain)
        assert match is not None
        assert match[1] == 192
        partial = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
            list(range(96)), _BLOCK_SIZE, _MODEL_NAME
        )
        assert store.find_prefix_snapshot(partial)[1] == 64
        miss = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
            list(range(160)), _BLOCK_SIZE, "nope-model"
        )
        assert store.find_prefix_snapshot(miss) is None
    finally:
        store.shutdown()


def test_prefix_cross_restart_recovery(tmp_path) -> None:
    store_a = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
    h = _prefix_hash(64)
    try:
        assert store_a.save_prefix(h, 64, _make_extracted(64), _extract_fn, _MODEL_NAME)
        assert _wait_for(store_a, h, 64)
    finally:
        store_a.shutdown()
    assert (tmp_path / "_prefix_snapshots").exists()
    store_b = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
    try:
        assert store_b.has_prefix(h, 64) is True
        loaded = store_b.load_prefix(h, 64)
        assert loaded is not None
        assert len(loaded) == 2
        assert mx.array_equal(
            loaded[0]["state"][0], mx.array([64.0, 0.0, 1.0, 2.0])
        )
        stats = store_b.get_prefix_stats()
        assert stats["entries"] == 1
        assert stats["writes"] == 0
    finally:
        store_b.shutdown()


def test_prefix_lru_eviction_drops_oldest(tmp_path) -> None:
    probe = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
    h_probe = _prefix_hash(64)
    try:
        assert probe.save_prefix(
            h_probe, 64, _make_extracted(64), _extract_fn, _MODEL_NAME
        )
        assert _wait_for(probe, h_probe, 64)
        snap_size = probe.get_prefix_stats()["total_bytes"]
    finally:
        probe.shutdown()
        probe.cleanup_prefix_all()
    assert snap_size > 0
    cap = int(snap_size * 2.5)
    store = BoundarySnapshotSSDStore(
        base_dir=tmp_path, prefix_persist=True, prefix_max_bytes=cap
    )
    try:
        hashes = [_prefix_hash(tc) for tc in (64, 128, 192)]
        for tc, h in zip((64, 128, 192), hashes):
            assert store.save_prefix(h, tc, _make_extracted(tc), _extract_fn, _MODEL_NAME)
            assert _wait_for(store, h, tc)
            time.sleep(0.02)
        stats = store.get_prefix_stats()
        assert stats["evictions"] >= 1
        assert store.has_prefix(hashes[0], 64) is False
        assert store.has_prefix(hashes[1], 128) is True
        assert store.has_prefix(hashes[2], 192) is True
        assert stats["total_bytes"] <= cap + snap_size
    finally:
        store.shutdown()


def test_cleanup_prefix_all_clears_disk_and_index(tmp_path) -> None:
    store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
    try:
        h = _prefix_hash(64)
        assert store.save_prefix(h, 64, _make_extracted(64), _extract_fn, _MODEL_NAME)
        assert _wait_for(store, h, 64)
        store.cleanup_prefix_all()
        assert store.has_prefix(h, 64) is False
        assert store.get_prefix_stats()["entries"] == 0
        leftover = list((tmp_path / "_prefix_snapshots").rglob("*.safetensors"))
        assert leftover == []
    finally:
        store.shutdown()


# ---------------------------------------------------------------------------
# Read-hook (_try_prefix_snapshot_warm_start) wiring tests.
# Uses the real BoundarySnapshotSSDStore for hash/find/load; block_aware_cache
# and paged_cache_manager are fakes so the control flow (gating,
# full-materialization guard, fallback, request field assignment) is tested
# without a live inference stack.
# ---------------------------------------------------------------------------


class _FakeBlockTable:
    def __init__(self, num_tokens: int, num_blocks: int):
        self.num_tokens = num_tokens
        self.block_ids = list(range(num_blocks))


class _FakeBlockAwareCache:
    def __init__(self, block_table=None, reconstructed=None):
        self._block_table = block_table
        self._reconstructed = reconstructed
        self.store_calls = []

    def store_cache(self, request_id, tokens, cache_data, model_cache_config=None,
                    boundary_snapshots=None, **kwargs):
        self.store_calls.append(
            (request_id, len(tokens), model_cache_config, boundary_snapshots)
        )
        return self._block_table

    def reconstruct_cache(self, block_table, promote_to_hot_cache=False):
        return self._reconstructed


class _FakePagedCacheManager:
    def __init__(self):
        self.deleted = []

    def delete_block_table(self, request_id):
        self.deleted.append(request_id)


def _fake_self(tmp_path, *, persist=True, block_size=_BLOCK_SIZE,
               block_table=None, reconstructed="recon"):
    cfg = SimpleNamespace(
        boundary_prefix_persist=persist,
        paged_cache_block_size=block_size,
        model_name=_MODEL_NAME,
    )
    store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=persist)
    bac = _FakeBlockAwareCache(block_table=block_table, reconstructed=reconstructed)
    pcm = _FakePagedCacheManager()
    fake = SimpleNamespace(
        config=cfg,
        _boundary_snapshot_store=store,
        block_aware_cache=bac,
        paged_cache_manager=pcm,
    )
    return fake, store, bac, pcm


def _request(token_count: int, vlm: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(token_count)),
        vlm_extra_keys_for_cache=("img",) if vlm else None,
        prompt_cache=None,
        block_table=None,
        cached_tokens=0,
        shared_prefix_blocks=0,
        remaining_tokens=None,
    )


def test_warm_start_disabled_returns_false(tmp_path) -> None:
    fake, store, _, _ = _fake_self(tmp_path, persist=False)
    try:
        assert _try_prefix_snapshot_warm_start(fake, _request(128)) is False
    finally:
        store.shutdown()


def test_warm_start_vlm_request_skipped(tmp_path) -> None:
    fake, store, bac, _ = _fake_self(
        tmp_path, block_table=_FakeBlockTable(64, 2)
    )
    try:
        h = _prefix_hash(64)
        assert store.save_prefix(h, 64, _make_extracted(64), _extract_fn, _MODEL_NAME)
        assert _wait_for(store, h, 64)
        assert _try_prefix_snapshot_warm_start(fake, _request(128, vlm=True)) is False
        assert bac.store_calls == []
    finally:
        store.shutdown()


def test_warm_start_no_match_returns_false(tmp_path) -> None:
    fake, store, bac, _ = _fake_self(
        tmp_path, block_table=_FakeBlockTable(64, 2)
    )
    try:
        # Nothing persisted -> find_prefix_snapshot returns None.
        assert _try_prefix_snapshot_warm_start(fake, _request(128)) is False
        assert bac.store_calls == []
    finally:
        store.shutdown()


def test_warm_start_partial_materialization_falls_back(tmp_path) -> None:
    # store_cache returns a block_table covering fewer tokens than matched_tc:
    # the hook must release it and fall back to a full prefill (return False).
    fake, store, bac, pcm = _fake_self(
        tmp_path, block_table=_FakeBlockTable(32, 1)
    )
    try:
        h = _prefix_hash(64)
        assert store.save_prefix(h, 64, _make_extracted(64), _extract_fn, _MODEL_NAME)
        assert _wait_for(store, h, 64)
        req = _request(128)
        assert _try_prefix_snapshot_warm_start(fake, req) is False
        assert req.prompt_cache is None
        assert req.remaining_tokens is None
        assert pcm.deleted == ["req-1"]
    finally:
        store.shutdown()


def test_warm_start_reconstruct_none_falls_back(tmp_path) -> None:
    fake, store, bac, pcm = _fake_self(
        tmp_path, block_table=_FakeBlockTable(64, 2), reconstructed=None
    )
    try:
        h = _prefix_hash(64)
        assert store.save_prefix(h, 64, _make_extracted(64), _extract_fn, _MODEL_NAME)
        assert _wait_for(store, h, 64)
        req = _request(128)
        assert _try_prefix_snapshot_warm_start(fake, req) is False
        assert pcm.deleted == ["req-1"]
    finally:
        store.shutdown()


def test_warm_start_success_sets_request_fields(tmp_path) -> None:
    bt = _FakeBlockTable(64, 2)
    recon = object()
    fake, store, bac, pcm = _fake_self(
        tmp_path, block_table=bt, reconstructed=recon
    )
    try:
        h = _prefix_hash(64)
        assert store.save_prefix(h, 64, _make_extracted(64), _extract_fn, _MODEL_NAME)
        assert _wait_for(store, h, 64)
        req = _request(128)
        assert _try_prefix_snapshot_warm_start(fake, req) is True
        # store_cache called with prefix tokens, model_cache_config=None,
        # and a boundary_snapshots dict keyed by the matched token count.
        assert len(bac.store_calls) == 1
        rid, ntokens, mcc, snaps = bac.store_calls[0]
        assert rid == "req-1"
        assert ntokens == 64
        assert mcc is None
        assert snaps is not None and 64 in snaps
        # Request warm-start state mirrors the paged-cache HIT path.
        assert req.prompt_cache is recon
        assert req.block_table is bt
        assert req.cached_tokens == 64
        assert req.shared_prefix_blocks == 2
        assert req.remaining_tokens == list(range(64, 128))
        assert pcm.deleted == []
    finally:
        store.shutdown()
