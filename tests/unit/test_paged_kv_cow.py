# SPDX-License-Identifier: Apache-2.0
"""Tests for PR-H: CoW PagedKVCache + two-level addressing + FA-tile.

Validates CoW refcount semantics, FA-tile alignment, prefix page donation,
and KL alignment vs stock FusionPagedKVCache (golden reference < 1e-6).
"""

from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache
from fusion_mlx.custom_kernels.paged_kv_cow import (
    CoWPagedKVCache,
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
