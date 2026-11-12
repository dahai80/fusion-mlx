# SPDX-License-Identifier: Apache-2.0
"""#1532: tail snapshot capture + warm-start for non-block-aligned prompts.

Hybrid models force block_size=2048, so the paged block store discards up
to 2047 trailing tokens per prompt and every repeat request re-prefills
them. The tail snapshot persists the full kickoff state (prompt_len - 1,
the shape both single-request prefill paths leave after tokens[:-1]) keyed
by a domain-separated tail hash.

Covers: hash determinism + domain separation, capture gating, warm-start
request field assignment, and the direct states-to-cache-objects rebuild.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from fusion_mlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from fusion_mlx.scheduler.sched_boundary import (
    _build_cache_from_full_states,
    _maybe_capture_tail_snapshot,
    _try_tail_snapshot_warm_start,
)

try:
    import mlx.core as mx

    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

pytestmark = pytest.mark.skipif(not _HAS_MLX, reason="MLX not available")

_BLOCK_SIZE = 32
_MODEL_NAME = "test-model"
_PROMPT_LEN = 100  # 100 % 32 = 4 -> non-aligned; tc = 99


def _make_extracted(token_count: int, num_layers: int = 2) -> list[dict]:
    states = []
    for i in range(num_layers):
        k = mx.zeros((1, 2, token_count, 4))
        v = mx.zeros((1, 2, token_count, 4))
        states.append(
            {
                "state": (k, v),
                "meta_state": (token_count, num_layers),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            }
        )
    return states


def _full_cache(token_count: int, num_layers: int = 2) -> SimpleNamespace:
    # Live-cache stand-in for capture: _extract_cache_states is faked to
    # return _make_extracted, so only the token-count marker matters.
    return SimpleNamespace(layers=[i for i in range(num_layers)], _tail_tc=token_count)


def _extract_fn(snapshot_cache):
    tc = getattr(snapshot_cache, "_tail_tc", _PROMPT_LEN - 1)
    return _make_extracted(tc), None


def _tail_keyed(token_count: int, block_size: int = _BLOCK_SIZE) -> tuple:
    return BoundarySnapshotSSDStore.compute_tail_prefix_hash(
        list(range(token_count)), block_size, _MODEL_NAME
    )


def _wait_for(store, prefix_hash, token_count, timeout=5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.has_prefix(prefix_hash, token_count):
            return True
        time.sleep(0.02)
    return False


def _fake_self(tmp_path, *, persist=True, block_size=_BLOCK_SIZE, stateful=True):
    cfg = SimpleNamespace(
        boundary_prefix_persist=persist,
        paged_cache_block_size=block_size,
        model_name=_MODEL_NAME,
    )
    store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=persist)
    fake = SimpleNamespace(
        config=cfg,
        _boundary_snapshot_store=store,
        block_aware_cache=object(),
        _cache_list_needs_boundary_snapshot=lambda cache_list: stateful,
        _extract_cache_states=_extract_fn,
        _model_cache_sig="",
    )
    return fake, store


def _request(token_count: int, vlm: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-tail",
        prompt_token_ids=list(range(token_count)),
        vlm_extra_keys_for_cache=("img",) if vlm else None,
        prompt_cache=None,
        block_table=None,
        cached_tokens=0,
        shared_prefix_blocks=0,
        remaining_tokens=None,
    )


class TestTailHash:
    def test_deterministic_and_tc(self):
        a = _tail_keyed(_PROMPT_LEN - 1)
        b = _tail_keyed(_PROMPT_LEN - 1)
        assert a is not None
        assert a == b
        assert a[1] == _PROMPT_LEN - 1

    def test_differs_when_tail_tokens_differ(self):
        h1 = BoundarySnapshotSSDStore.compute_tail_prefix_hash(
            list(range(99)), _BLOCK_SIZE, _MODEL_NAME
        )
        other = list(range(64)) + list(range(100, 135))
        h2 = BoundarySnapshotSSDStore.compute_tail_prefix_hash(
            other, _BLOCK_SIZE, _MODEL_NAME
        )
        assert h1[0] != h2[0]

    def test_domain_separated_from_chain_hash(self):
        chain = BoundarySnapshotSSDStore.compute_prefix_chain_hashes(
            list(range(64)), _BLOCK_SIZE, _MODEL_NAME
        )
        tail = BoundarySnapshotSSDStore.compute_tail_prefix_hash(
            list(range(64)), _BLOCK_SIZE, _MODEL_NAME
        )
        # Tail of a block-aligned prefix must not equal the block chain hash
        # (64 % 32 == 0: no tail tokens, but b"tail" separator still applies).
        assert tail[0] != chain[-1]

    def test_model_scoped(self):
        h1 = BoundarySnapshotSSDStore.compute_tail_prefix_hash(
            list(range(99)), _BLOCK_SIZE, _MODEL_NAME
        )
        h2 = BoundarySnapshotSSDStore.compute_tail_prefix_hash(
            list(range(99)), _BLOCK_SIZE, "other-model"
        )
        assert h1[0] != h2[0]

    def test_empty_returns_none(self):
        assert (
            BoundarySnapshotSSDStore.compute_tail_prefix_hash([], _BLOCK_SIZE, "m")
            is None
        )


class TestCaptureGating:
    def test_success_persists_tail(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            cache = _full_cache(_PROMPT_LEN - 1)
            cache._tail_tc = _PROMPT_LEN - 1
            req = _request(_PROMPT_LEN)
            _maybe_capture_tail_snapshot(fake, req, cache, _PROMPT_LEN - 1)
            tail_hash, tc = _tail_keyed(_PROMPT_LEN - 1)
            assert _wait_for(store, tail_hash, tc)
        finally:
            store.shutdown()

    def test_persist_disabled_noop(self, tmp_path):
        fake, store = _fake_self(tmp_path, persist=False)
        try:
            req = _request(_PROMPT_LEN)
            _maybe_capture_tail_snapshot(
                fake, req, _full_cache(_PROMPT_LEN - 1), _PROMPT_LEN - 1
            )
            assert store.get_prefix_stats() == {}
        finally:
            store.shutdown()

    def test_aligned_prompt_skipped(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            req = _request(96)  # 96 % 32 == 0
            _maybe_capture_tail_snapshot(fake, req, _full_cache(95), 95)
            assert store.get_prefix_stats().get("entries", 0) == 0
        finally:
            store.shutdown()

    def test_vlm_request_skipped(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            req = _request(_PROMPT_LEN, vlm=True)
            _maybe_capture_tail_snapshot(
                fake, req, _full_cache(_PROMPT_LEN - 1), _PROMPT_LEN - 1
            )
            assert store.get_prefix_stats().get("entries", 0) == 0
        finally:
            store.shutdown()

    def test_wrong_token_count_skipped(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            req = _request(_PROMPT_LEN)
            _maybe_capture_tail_snapshot(
                fake, req, _full_cache(_PROMPT_LEN - 1), _PROMPT_LEN - 2
            )
            _maybe_capture_tail_snapshot(
                fake, req, _full_cache(_PROMPT_LEN), _PROMPT_LEN
            )
            assert store.get_prefix_stats().get("entries", 0) == 0
        finally:
            store.shutdown()

    def test_sliceable_only_model_skipped(self, tmp_path):
        fake, store = _fake_self(tmp_path, stateful=False)
        try:
            req = _request(_PROMPT_LEN)
            _maybe_capture_tail_snapshot(
                fake, req, _full_cache(_PROMPT_LEN - 1), _PROMPT_LEN - 1
            )
            assert store.get_prefix_stats().get("entries", 0) == 0
        finally:
            store.shutdown()


class TestBuildFromStates:
    def test_kvcache_layers_rebuilt(self):
        extracted = _make_extracted(99)
        caches = _build_cache_from_full_states(None, extracted, 99)
        assert caches is not None
        assert len(caches) == 2
        assert int(caches[0].keys.shape[2]) == 99

    def test_empty_state_fails_visibly(self):
        extracted = _make_extracted(99)
        extracted[0]["state"] = ()
        assert _build_cache_from_full_states(None, extracted, 99) is None

    def test_empty_states_returns_none(self):
        assert _build_cache_from_full_states(None, [], 99) is None


class TestWarmStart:
    def test_success_sets_kickoff_fields(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            tail_hash, tc = _tail_keyed(_PROMPT_LEN - 1)
            assert store.save_prefix(
                tail_hash, tc, _make_extracted(tc), _extract_fn, _MODEL_NAME
            )
            assert _wait_for(store, tail_hash, tc)
            req = _request(_PROMPT_LEN)
            assert _try_tail_snapshot_warm_start(fake, req) is True
            assert req.cached_tokens == _PROMPT_LEN - 1
            assert req.remaining_tokens == [_PROMPT_LEN - 1]
            assert req.block_table is None
            assert req.prompt_cache is not None
            assert len(req.prompt_cache) == 2
            assert req.shared_prefix_blocks == 0
        finally:
            store.shutdown()

    def test_aligned_prompt_returns_false(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            req = _request(96)
            assert _try_tail_snapshot_warm_start(fake, req) is False
            assert req.prompt_cache is None
        finally:
            store.shutdown()

    def test_no_entry_returns_false(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            req = _request(_PROMPT_LEN)
            assert _try_tail_snapshot_warm_start(fake, req) is False
        finally:
            store.shutdown()

    def test_persist_disabled_returns_false(self, tmp_path):
        fake, store = _fake_self(tmp_path, persist=False)
        try:
            assert _try_tail_snapshot_warm_start(fake, _request(_PROMPT_LEN)) is False
        finally:
            store.shutdown()

    def test_vlm_request_returns_false(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            assert (
                _try_tail_snapshot_warm_start(fake, _request(_PROMPT_LEN, vlm=True))
                is False
            )
        finally:
            store.shutdown()

    def test_short_prompt_returns_false(self, tmp_path):
        fake, store = _fake_self(tmp_path)
        try:
            assert _try_tail_snapshot_warm_start(fake, _request(1)) is False
        finally:
            store.shutdown()


class TestPendingFlushWindow:
    """#1543: has_prefix must cover the async write flush window."""

    def test_has_prefix_accounts_for_pending_write(self, tmp_path):
        store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
        try:
            tail_hash, tail_tc = _tail_keyed(_PROMPT_LEN - 1)
            assert store.has_prefix(tail_hash, tail_tc) is False
            assert store.save_prefix(
                tail_hash,
                tail_tc,
                _full_cache(tail_tc),
                _extract_fn,
                _MODEL_NAME,
            )
            # Immediately after enqueue (writer may not have flushed yet):
            # the bounded wait must report the snapshot as present.
            assert store.has_prefix(tail_hash, tail_tc) is True
            # A different token_count for the same hash is still a miss.
            assert store.has_prefix(tail_hash, tail_tc - 1) is False
            # Unknown hash never blocks the wait window.
            t0 = time.monotonic()
            assert store.has_prefix(b"\x00" * 16, tail_tc) is False
            assert time.monotonic() - t0 < 0.5
        finally:
            store.shutdown()

    def test_failed_write_clears_pending(self, tmp_path):
        store = BoundarySnapshotSSDStore(base_dir=tmp_path, prefix_persist=True)
        try:
            tail_hash, tail_tc = _tail_keyed(_PROMPT_LEN - 1)

            def _boom(_snapshot_cache):
                raise RuntimeError("extract failed")

            assert (
                store.save_prefix(tail_hash, tail_tc, _full_cache(tail_tc), _boom,
                                  _MODEL_NAME)
                is False
            )
            assert (tail_hash, tail_tc) not in store._prefix_pending
            assert store.has_prefix(tail_hash, tail_tc) is False
        finally:
            store.shutdown()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
