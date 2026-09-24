# SPDX-License-Identifier: Apache-2.0
"""Tests for PR-H: CoW PagedKVCache + two-level addressing + FA-tile.

Validates CoW refcount semantics, FA-tile alignment, prefix page donation,
and KL alignment vs stock FusionPagedKVCache (golden reference < 1e-6).
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
from fusion_mlx.custom_kernels.paged_kv_cow import (
    CoWPagedKVCache,
    CoWPagedRequestCache,
    PoolPrefixPageBinder,
    PrefixPageBinder,
    align_block_size_to_fa_tile,
    is_two_level_kv_enabled,
)
from fusion_mlx.eval.golden_reference import assert_logits_aligned


def _enable(monkeypatch):
    monkeypatch.setenv("FUSION_SHIM_TWO_LEVEL_KV", "1")


def _kv(batch=1, heads=2, steps=4, kdim=8, vdim=8, dtype=mx.float32):
    k = mx.array(np.random.randn(batch, heads, steps, kdim).astype(np.float32)).astype(
        dtype
    )
    v = mx.array(np.random.randn(batch, heads, steps, vdim).astype(np.float32)).astype(
        dtype
    )
    return k, v


class TestFATileAlignment:
    def test_already_aligned(self):
        assert align_block_size_to_fa_tile(64) == 64
        assert align_block_size_to_fa_tile(128) == 128

    def test_rounds_up(self):
        assert align_block_size_to_fa_tile(16) == 64
        assert align_block_size_to_fa_tile(50) == 64
        assert align_block_size_to_fa_tile(100) == 128

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            align_block_size_to_fa_tile(0)


class TestDisabledMatchesStock:
    def test_disabled_flag(self):
        assert is_two_level_kv_enabled() is False

    def test_disabled_fetch_matches_stock(self):
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        stock = FusionPagedKVCache(block_size=4, num_blocks=8)
        k, v = _kv(steps=6)
        out_cow = cow.update_and_fetch(k, v)
        out_stock = stock.update_and_fetch(k, v)
        mx.eval([out_cow[0], out_cow[1], out_stock[0], out_stock[1]])
        assert_logits_aligned(
            out_stock[0], out_cow[0], tol=1e-6, label="cow disabled K matches stock"
        )
        assert_logits_aligned(
            out_stock[1], out_cow[1], tol=1e-6, label="cow disabled V matches stock"
        )


class TestCoWRefcount:
    def test_single_writer_no_copy(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        k, v = _kv(steps=4)
        out = cow.update_and_fetch(k, v)
        mx.eval(out)
        assert cow._cow_copies == 0
        for phys in cow.block_table:
            assert cow._refcount[phys] == 1

    def test_shared_block_copies_on_write(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=16)
        k1, v1 = _kv(steps=4)
        cow.update_and_fetch(k1, v1)
        mx.eval([cow.keys_pool, cow.values_pool])
        phys0 = cow.block_table[0]
        cow._refcount[phys0] = 2
        new_phys = cow.ensure_writable(phys0)
        mx.eval(cow.keys_pool)
        assert new_phys != phys0
        assert cow._cow_copies == 1
        assert cow._refcount[phys0] == 1
        assert cow._refcount[new_phys] == 1

    def test_ensure_writable_noop_when_sole_owner(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        k, v = _kv(steps=4)
        cow.update_and_fetch(k, v)
        mx.eval([cow.keys_pool, cow.values_pool])
        phys0 = cow.block_table[0]
        assert cow.ensure_writable(phys0) == phys0

    def test_free_decrements_refcount(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        k, v = _kv(steps=4)
        cow.update_and_fetch(k, v)
        mx.eval([cow.keys_pool, cow.values_pool])
        phys0 = cow.block_table[0]
        cow._refcount[phys0] = 3
        cow._free_block(phys0)
        assert cow._refcount[phys0] == 2
        assert phys0 not in cow.free_list
        cow._free_block(phys0)
        cow._free_block(phys0)
        assert phys0 in cow.free_list


class TestEnabledMatchesStock:
    def test_enabled_fetch_matches_stock(self, monkeypatch):
        _enable(monkeypatch)
        assert is_two_level_kv_enabled() is True
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        stock = FusionPagedKVCache(block_size=4, num_blocks=8)
        k, v = _kv(steps=10)
        out_cow = cow.update_and_fetch(k, v)
        out_stock = stock.update_and_fetch(k, v)
        mx.eval([out_cow[0], out_cow[1], out_stock[0], out_stock[1]])
        assert_logits_aligned(
            out_stock[0], out_cow[0], tol=1e-6, label="cow enabled K matches stock"
        )
        assert_logits_aligned(
            out_stock[1], out_cow[1], tol=1e-6, label="cow enabled V matches stock"
        )

    def test_multi_step_appends_match_stock(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=16)
        stock = FusionPagedKVCache(block_size=4, num_blocks=16)
        for step in range(3):
            k, v = _kv(steps=3)
            out_cow = cow.update_and_fetch(k, v)
            out_stock = stock.update_and_fetch(k, v)
            mx.eval([out_cow[0], out_cow[1], out_stock[0], out_stock[1]])
        assert_logits_aligned(
            out_stock[0], out_cow[0], tol=1e-6, label="cow multi-step K matches stock"
        )
        assert_logits_aligned(
            out_stock[1], out_cow[1], tol=1e-6, label="cow multi-step V matches stock"
        )
        assert cow.offset == stock.offset

    def test_trim_matches_stock(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=16)
        stock = FusionPagedKVCache(block_size=4, num_blocks=16)
        k, v = _kv(steps=10)
        cow.update_and_fetch(k, v)
        stock.update_and_fetch(k, v)
        mx.eval([cow.keys_pool, cow.values_pool])
        n_cow = cow.trim(3)
        n_stock = stock.trim(3)
        assert n_cow == n_stock
        assert cow.offset == stock.offset

    def test_block_boundary_span(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=16)
        stock = FusionPagedKVCache(block_size=4, num_blocks=16)
        k, v = _kv(steps=9)
        out_cow = cow.update_and_fetch(k, v)
        out_stock = stock.update_and_fetch(k, v)
        mx.eval([out_cow[0], out_cow[1], out_stock[0], out_stock[1]])
        assert out_cow[0].shape == out_stock[0].shape
        assert_logits_aligned(
            out_stock[0], out_cow[0], tol=1e-6, label="cow boundary span K"
        )


class TestPrefixPageBinder:
    def _bhash(self, s: str) -> bytes:
        return hashlib.sha256(s.encode()).digest()

    def test_register_and_lookup(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=16)
        k, v = _kv(steps=8)
        cow.update_and_fetch(k, v)
        mx.eval([cow.keys_pool, cow.values_pool])
        binder = PrefixPageBinder(cow)
        h = self._bhash("prefix-1")
        pages = list(cow.block_table)
        binder.register_prefix(h, pages)
        looked = binder.lookup(h)
        assert looked == pages

    def test_lookup_miss_returns_none(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        binder = PrefixPageBinder(cow)
        assert binder.lookup(self._bhash("nope")) is None

    def test_donate_shares_pages(self, monkeypatch):
        _enable(monkeypatch)
        donor = CoWPagedKVCache(block_size=4, num_blocks=32)
        k, v = _kv(steps=8)
        donor.update_and_fetch(k, v)
        mx.eval([donor.keys_pool, donor.values_pool])
        donor_pages = list(donor.block_table)

        receiver = CoWPagedKVCache(block_size=4, num_blocks=32, slab_size=1)
        receiver._ensure_pool(1, 2, 8, 8, mx.float32)
        receiver.keys_pool = donor.keys_pool
        receiver.values_pool = donor.values_pool
        receiver.free_list = [i for i in range(32) if i not in donor_pages]
        for p in donor_pages:
            receiver._refcount[p] = 1

        binder = PrefixPageBinder(receiver)
        h = self._bhash("shared-prefix")
        binder.register_prefix(h, donor_pages)
        adopted = binder.donate(h)
        assert len(adopted) == len(donor_pages)
        assert receiver.offset == len(donor_pages) * 4
        for p in donor_pages:
            assert receiver._refcount[p] == 2
        assert receiver._shared_pages == len(donor_pages)

    def test_donate_disabled_returns_empty(self):
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        binder = PrefixPageBinder(cow)
        assert binder.donate(self._bhash("x")) == []


class TestStats:
    def test_stats_has_cow_fields(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        k, v = _kv(steps=4)
        cow.update_and_fetch(k, v)
        mx.eval([cow.keys_pool, cow.values_pool])
        s = cow.stats()
        assert "cow_copies" in s
        assert "shared_pages" in s
        assert "refcount_total" in s
        assert "two_level_enabled" in s
        assert s["two_level_enabled"] is True
        assert s["cow_copies"] == 0


class TestAuditSafetyFixes:
    """D1-D8 audit fixes: cross-pool raise, rollback, cap, sweep."""

    def test_d5_share_pages_raises_on_foreign_block(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        # phys 999 not in this cache's _refcount -> cross-pool/freed -> raise
        with pytest.raises(RuntimeError, match="not owned by this cache"):
            cow.share_pages([999])

    def test_d5_share_pages_raises_on_any_foreign_in_list(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=8)
        k, v = _kv(steps=4)
        cow.update_and_fetch(k, v)
        mx.eval([cow.keys_pool, cow.values_pool])
        owned = list(cow._refcount.keys())
        # one owned + one foreign -> raise (no partial adoption)
        with pytest.raises(RuntimeError, match="not owned"):
            cow.share_pages([owned[0], 12345])

    def test_d3_rollback_on_alloc_exhaustion_standalone(self):
        # standalone cache: exhaust pool mid-update -> rollback tail blocks
        cache = FusionPagedKVCache(block_size=4, num_blocks=2)
        k1, v1 = _kv(steps=4)
        cache.update_and_fetch(k1, v1)  # fills 1 block
        mx.eval([cache.keys_pool, cache.values_pool])
        offset_before = cache.offset
        # second 4-step update needs another block but pool has only 2;
        # use 8 steps to force >2 blocks -> exhaust on 3rd
        k2, v2 = _kv(steps=8)
        with pytest.raises(RuntimeError, match="pool exhausted"):
            cache.update_and_fetch(k2, v2)
        # offset unchanged after rollback
        assert cache.offset == offset_before

    def test_d7_prefix_binder_cap_evicts_oldest(self, monkeypatch):
        _enable(monkeypatch)
        cow = CoWPagedKVCache(block_size=4, num_blocks=64)
        binder = PrefixPageBinder(cow, max_prefixes=3)
        for i in range(5):
            binder.register_prefix(hashlib.sha256(str(i).encode()).digest(), [i])
        # cap enforced
        assert len(binder._hash_to_pages) == 3
        # oldest (i=0,1) evicted
        assert binder.lookup(hashlib.sha256(b"0").digest()) is None
        # newest (i=4) present
        assert binder.lookup(hashlib.sha256(b"4").digest()) is not None

    def test_d1_standalone_lock_is_rlock(self):
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        # RLock: reentrant — acquire twice without deadlock
        cache._lock.acquire()
        cache._lock.acquire()
        cache._lock.release()
        cache._lock.release()

    def test_d1_pool_lock_is_rlock(self):
        from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedKVPool

        pool = FusionPagedKVPool(block_size=4, num_blocks=8, n_kv_heads=2, head_dim=8)
        pool._lock.acquire()
        pool._lock.acquire()
        pool._lock.release()
        pool._lock.release()

    def test_d2_sweep_registry_reclaims_stale(self, monkeypatch):
        from fusion_mlx.custom_kernels import fusion_paged_kv as mod

        monkeypatch.setattr(mod, "_GLOBAL_CACHE_REGISTRY", {})
        # simulate two requests, one stale
        fake_active = MagicMock()
        fake_active.free_all = lambda: 4
        fake_stale = MagicMock()
        fake_stale.free_all = lambda: 2
        with mod._REGISTRY_LOCK:
            mod._GLOBAL_CACHE_REGISTRY["active"] = [fake_active]
            mod._GLOBAL_CACHE_REGISTRY["stale"] = [fake_stale]
        reclaimed = mod.sweep_registry({"active"})
        assert reclaimed == 1
        assert "stale" not in mod._GLOBAL_CACHE_REGISTRY
        assert "active" in mod._GLOBAL_CACHE_REGISTRY

    def test_d8_state_setter_frees_old_blocks(self):
        cache = FusionPagedKVCache(block_size=4, num_blocks=8)
        k1, v1 = _kv(steps=4)
        cache.update_and_fetch(k1, v1)
        mx.eval([cache.keys_pool, cache.values_pool])
        used_before = len(cache.block_table)
        free_before = len(cache.free_list)
        # set state with new content -> old blocks freed, not lost
        k2, v2 = _kv(steps=8)
        cache.state = (k2, v2)
        # total blocks conserved: used + free == num_blocks
        assert len(cache.block_table) + len(cache.free_list) == cache.num_blocks
        assert len(cache.block_table) >= 2  # 8 steps / 4 block_size = 2 blocks


class TestPoolCoWRefcount:
    """Pool-level CoW: refcount, share_block, ensure_writable, decrement-free."""

    def _pool(self):
        from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedKVPool

        return FusionPagedKVPool(block_size=4, num_blocks=8, n_kv_heads=2, head_dim=8)

    def test_alloc_init_refcount_one(self):
        pool = self._pool()
        pb = pool.alloc_block("r1")
        assert pool._refcount[pb] == 1
        assert pool._owners[pb] == {"r1"}

    def test_share_block_increments_refcount(self):
        pool = self._pool()
        pb = pool.alloc_block("r1")
        pool.share_block("r2", pb)
        assert pool._refcount[pb] == 2
        assert pool._owners[pb] == {"r1", "r2"}

    def test_share_block_rejects_foreign(self):
        pool = self._pool()
        with pytest.raises(RuntimeError, match="not allocated"):
            pool.share_block("r2", 999)

    def test_ensure_writable_copies_when_shared(self):
        pool = self._pool()
        pb = pool.alloc_block("r1")
        pool.share_block("r2", pb)
        new = pool.ensure_writable("r2", pb)
        assert new != pb
        assert pool._refcount[pb] == 1  # donor kept sole after copy
        assert pool._refcount[new] == 1
        # data copied
        mx.eval([pool.keys_pool, pool.values_pool])

    def test_ensure_writable_noop_when_sole(self):
        pool = self._pool()
        pb = pool.alloc_block("r1")
        assert pool.ensure_writable("r1", pb) == pb

    def test_free_block_decrements_when_shared(self):
        pool = self._pool()
        pb = pool.alloc_block("r1")
        pool.share_block("r2", pb)
        pool.free_block(pb, "r1")
        # still owned by r2 -> not freed
        assert pb not in pool.free_list
        assert pool._refcount[pb] == 1
        pool.free_block(pb, "r2")
        assert pb in pool.free_list

    def test_free_request_decrements_shared(self):
        pool = self._pool()
        pb = pool.alloc_block("r1")
        pool.share_block("r2", pb)
        freed = pool.free_request("r1")
        # r1 freed but block still held by r2 -> not in free_list
        assert pb not in pool.free_list
        freed2 = pool.free_request("r2")
        assert pb in pool.free_list

    def test_eviction_skips_shared_block(self):
        # shared block (refcount>1) must not be freed when an idle co-owner
        # is evicted — the active peer still reads it.
        pool = self._pool()
        pb = pool.alloc_block("r1")
        pool.share_block("r2", pb)
        pool.set_active_ids({"r1"})  # r2 idle/evictable
        # exhaust all other blocks so eviction must consider r2
        for i in range(pool.num_blocks - 1):
            pool.alloc_block("filler")
        # next alloc evicts r2 (idle) -> pb must survive (shared with active r1)
        new_pb = pool.alloc_block("r3", active_ids={"r1", "r3"})
        assert pool._refcount.get(pb, 0) >= 1  # pb still live for r1
        assert pb != new_pb


class TestCoWPagedRequestCacheParity:
    """CoWPagedRequestCache output matches stock FusionPagedRequestCache."""

    def _pool(self):
        from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedKVPool

        return FusionPagedKVPool(block_size=4, num_blocks=16, n_kv_heads=2, head_dim=8)

    def test_single_request_matches_stock(self):
        pool = self._pool()
        cow = CoWPagedRequestCache(pool, "r1")
        stock_pool = self._pool()
        from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedRequestCache

        stock = FusionPagedRequestCache(stock_pool, "r1s")
        k, v = _kv(steps=10)
        oc = cow.update_and_fetch(k, v)
        os_ = stock.update_and_fetch(k, v)
        mx.eval([oc[0], oc[1], os_[0], os_[1]])
        assert_logits_aligned(os_[0], oc[0], tol=1e-6, label="cow pool K matches stock")
        assert_logits_aligned(os_[1], oc[1], tol=1e-6, label="cow pool V matches stock")
        assert cow.offset == stock.offset
        assert cow._cow_copies == 0

    def test_multi_step_appends_match(self):
        pool = self._pool()
        cow = CoWPagedRequestCache(pool, "r1")
        from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedRequestCache

        stock = FusionPagedRequestCache(self._pool(), "r1s")
        for _ in range(3):
            k, v = _kv(steps=3)
            oc = cow.update_and_fetch(k, v)
            os_ = stock.update_and_fetch(k, v)
            mx.eval([oc[0], oc[1], os_[0], os_[1]])
        assert_logits_aligned(os_[0], oc[0], tol=1e-6, label="cow pool multi-step K")
        assert cow.offset == stock.offset

    def test_d9_instance_scoped_free_survives_peer(self):
        # D9 regression: two layer caches of one request share request_id.
        # clear()/free_all() must free ONLY the caller's own block_table
        # blocks, NOT pool.free_request(request_id) which would free every
        # peer's blocks mid-generation and corrupt live KV (phys collision
        # from stale block_table referencing freed-then-reallocated slabs).
        from fusion_mlx.custom_kernels.paged_kv_pool import (
            FusionPagedKVPool,
            FusionPagedRequestCache,
        )

        pool = FusionPagedKVPool(
            block_size=4, num_blocks=16, n_kv_heads=2, head_dim=8
        )
        layer_a = FusionPagedRequestCache(pool, "req_1")
        layer_b = FusionPagedRequestCache(pool, "req_1")
        ka, va = _kv(steps=6)
        kb, vb = _kv(steps=6)
        layer_a.update_and_fetch(ka, va)
        layer_b.update_and_fetch(kb, vb)
        mx.eval([pool.keys_pool, pool.values_pool])
        phys_b = list(layer_b.block_table)
        fetched_b_before = layer_b.state[0]
        mx.eval(fetched_b_before)
        before_vals = fetched_b_before[0, 0, :2, :4].tolist()
        # Free layer A mid-generation — must NOT touch layer B's blocks.
        freed = layer_a.free_all()
        assert freed == len(phys_b)
        assert layer_a.block_table == []
        assert layer_a.offset == 0
        # layer B's blocks must still be live in the pool (refcount intact).
        for pb in phys_b:
            assert pool._refcount.get(pb, 0) >= 1
            assert pb in pool.in_use
        # layer B's KV must be unchanged (no retroactive mutation).
        fetched_b_after = layer_b.state[0]
        mx.eval(fetched_b_after)
        after_vals = fetched_b_after[0, 0, :2, :4].tolist()
        assert before_vals == after_vals
        # layer B can still append (block 3 alloc must not collide with freed).
        kc, vc = _kv(steps=4)
        layer_b.update_and_fetch(kc, vc)
        mx.eval(layer_b.state[0])
        assert layer_b.offset == 10
        # Same for clear().
        layer_c = FusionPagedRequestCache(pool, "req_1")
        kc2, vc2 = _kv(steps=6)
        layer_c.update_and_fetch(kc2, vc2)
        mx.eval([pool.keys_pool, pool.values_pool])
        phys_b2 = list(layer_b.block_table)
        before2 = layer_b.state[0]
        mx.eval(before2)
        before2_vals = before2[0, 0, :2, :4].tolist()
        layer_c.clear()
        for pb in phys_b2:
            assert pool._refcount.get(pb, 0) >= 1
        after2 = layer_b.state[0]
        mx.eval(after2)
        assert before2_vals == after2[0, 0, :2, :4].tolist()


class TestPoolDonation:
    """Concurrent donation: donor -> receiver refcount-shared, CoW on write."""

    def _pool(self):
        from fusion_mlx.custom_kernels.paged_kv_pool import FusionPagedKVPool

        return FusionPagedKVPool(block_size=4, num_blocks=32, n_kv_heads=2, head_dim=8)

    def test_adopt_donated_shares_blocks(self):
        pool = self._pool()
        donor = CoWPagedRequestCache(pool, "donor")
        k, v = _kv(steps=8)
        donor.update_and_fetch(k, v)
        mx.eval([pool.keys_pool, pool.values_pool])
        donor_pages = list(donor.block_table)
        receiver = CoWPagedRequestCache(pool, "receiver")
        n = receiver.adopt_donated(donor_pages)
        assert n == len(donor_pages)
        assert receiver.offset == len(donor_pages) * 4
        for p in donor_pages:
            assert pool._refcount[p] == 2
        assert receiver._shared_pages == len(donor_pages)

    def test_donor_finish_receiver_survives(self):
        pool = self._pool()
        donor = CoWPagedRequestCache(pool, "donor")
        k, v = _kv(steps=8)
        donor.update_and_fetch(k, v)
        mx.eval([pool.keys_pool, pool.values_pool])
        donor_pages = list(donor.block_table)
        receiver = CoWPagedRequestCache(pool, "receiver")
        receiver.adopt_donated(donor_pages)
        # donor finishes -> blocks decremented but not freed (receiver holds)
        donor.free_all()
        for p in donor_pages:
            assert pool._refcount.get(p, 0) == 1
            assert p not in pool.free_list
        # receiver can still fetch its state
        rk, rv = receiver.state
        mx.eval([rk, rv])

    def test_receiver_append_does_not_corrupt_donor(self):
        # donated prefix blocks are full/read-only; receiver appends to NEW
        # blocks — no CoW needed for full-block donation. The invariant under
        # test: receiver's appended tokens do not overwrite donor's shared KV.
        pool = self._pool()
        donor = CoWPagedRequestCache(pool, "donor")
        k, v = _kv(steps=4)
        donor.update_and_fetch(k, v)
        mx.eval([pool.keys_pool, pool.values_pool])
        donor_pages = list(donor.block_table)
        donor_k_before = donor.state[0]
        mx.eval(donor_k_before)
        receiver = CoWPagedRequestCache(pool, "receiver")
        receiver.adopt_donated(donor_pages)
        # receiver appends a divergent continuation (new block, not shared)
        k2, v2 = _kv(steps=4)
        receiver.update_and_fetch(k2, v2)
        mx.eval([pool.keys_pool, pool.values_pool])
        # donor's shared block still holds its original KV (not overwritten)
        donor_k_after = donor.state[0]
        mx.eval(donor_k_after)
        assert_logits_aligned(
            donor_k_before,
            donor_k_after,
            tol=1e-6,
            label="donor KV intact after receiver append",
        )
        # receiver offset = donated 4 + appended 4 = 8
        assert receiver.offset == 8

    def test_receiver_write_into_shared_partial_block_copies(self):
        # CoW-on-write path: force receiver to write into a shared block by
        # adopting a block then writing into its logical position directly via
        # pool.ensure_writable (the safety net for misaligned shared writes).
        pool = self._pool()
        donor = CoWPagedRequestCache(pool, "donor")
        k, v = _kv(steps=4)
        donor.update_and_fetch(k, v)
        mx.eval([pool.keys_pool, pool.values_pool])
        donor_pages = list(donor.block_table)
        receiver = CoWPagedRequestCache(pool, "receiver")
        receiver.adopt_donated(donor_pages)
        # simulate a shared-block write via ensure_writable
        phys = donor_pages[0]
        new_phys = pool.ensure_writable("receiver", phys)
        assert new_phys != phys
        assert (
            receiver._cow_copies == 0
        )  # ensure_writable is pool-level; cache counter only on update_and_fetch path
        assert pool._refcount[phys] == 1

    def test_pool_binder_donate_stale_returns_empty(self):
        pool = self._pool()
        binder = PoolPrefixPageBinder(pool)
        donor = CoWPagedRequestCache(pool, "donor")
        k, v = _kv(steps=4)
        donor.update_and_fetch(k, v)
        mx.eval([pool.keys_pool, pool.values_pool])
        pages = list(donor.block_table)
        binder.register_prefix(b"hash1", pages)
        # donor evicted -> blocks freed -> donate must detect stale + return []
        donor.free_all()
        assert binder.donate(b"hash1", "receiver") == []

    def test_pool_binder_donate_live_returns_pages(self):
        pool = self._pool()
        binder = PoolPrefixPageBinder(pool)
        donor = CoWPagedRequestCache(pool, "donor")
        k, v = _kv(steps=4)
        donor.update_and_fetch(k, v)
        mx.eval([pool.keys_pool, pool.values_pool])
        pages = list(donor.block_table)
        binder.register_prefix(b"hash1", pages)
        got = binder.donate(b"hash1", "receiver")
        # per-layer: single layer wrapped
        assert got == [pages]
        assert len(got) == 1

    def test_pool_binder_cap_evicts_oldest(self):
        pool = self._pool()
        binder = PoolPrefixPageBinder(pool, max_prefixes=3)
        for i in range(5):
            binder.register_prefix(bytes([i]), [i])
        assert len(binder._hash_to_pages) == 3
        assert binder.lookup(bytes([0])) is None
        assert binder.lookup(bytes([4])) is not None
