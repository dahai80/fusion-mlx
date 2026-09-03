# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FusionPagedKVCache (custom_kernels.paged_kv_cache).

Verifies bit-exact storage/fetch vs a naive contiguous reference cache,
covering: single-step decode, multi-step prefill, block-boundary spans,
trim round-trip, state round-trip, and pool-exhaustion fail-visible.
"""

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.custom_kernels.paged_kv_cache import FusionPagedKVCache


def _random_kv(batch, heads, steps, kdim, vdim, dtype=mx.float16):
    k = mx.array(np.random.randn(batch, heads, steps, kdim).astype(np.float32)).astype(
        dtype
    )
    v = mx.array(np.random.randn(batch, heads, steps, vdim).astype(np.float32)).astype(
        dtype
    )
    return k, v


class _NaiveRef:
    def __init__(self):
        self.offset = 0
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def _assert_kv_equal(ref, paged):
    mx.eval(ref[0], ref[1], paged[0], paged[1])
    assert ref[0].shape == paged[0].shape, "keys shape mismatch"
    assert ref[1].shape == paged[1].shape, "values shape mismatch"
    assert mx.allclose(ref[0], paged[0]).item(), "keys not bit-exact"
    assert mx.allclose(ref[1], paged[1]).item(), "values not bit-exact"


def test_single_step_decode_matches_naive():
    ref = _NaiveRef()
    paged = FusionPagedKVCache(block_size=16, num_blocks=64)
    kdim, vdim = 8, 10
    for _ in range(5):
        k, v = _random_kv(1, 2, 1, kdim, vdim)
        rk, rv = ref.update_and_fetch(k, v)
        pk, pv = paged.update_and_fetch(k, v)
        _assert_kv_equal((rk, rv), (pk, pv))
    assert paged.offset == 5


def test_multistep_prefill_matches_naive():
    ref = _NaiveRef()
    paged = FusionPagedKVCache(block_size=16, num_blocks=64)
    k, v = _random_kv(1, 2, 62, 8, 10)
    rk, rv = ref.update_and_fetch(k, v)
    pk, pv = paged.update_and_fetch(k, v)
    _assert_kv_equal((rk, rv), (pk, pv))
    assert paged.offset == 62


def test_block_boundary_span_matches_naive():
    ref = _NaiveRef()
    paged = FusionPagedKVCache(block_size=8, num_blocks=64)
    for steps in [3, 6, 8, 9, 15, 17]:
        k, v = _random_kv(2, 3, steps, 6, 7)
        rk, rv = ref.update_and_fetch(k, v)
        pk, pv = paged.update_and_fetch(k, v)
        _assert_kv_equal((rk, rv), (pk, pv))
    assert paged.offset == sum([3, 6, 8, 9, 15, 17])


def test_interleaved_prefill_and_decode():
    ref = _NaiveRef()
    paged = FusionPagedKVCache(block_size=16, num_blocks=128)
    sequence = [40, 1, 1, 1, 20, 1, 1]
    for steps in sequence:
        k, v = _random_kv(1, 4, steps, 5, 5)
        rk, rv = ref.update_and_fetch(k, v)
        pk, pv = paged.update_and_fetch(k, v)
        _assert_kv_equal((rk, rv), (pk, pv))
    assert paged.offset == sum(sequence)


def test_state_getter_matches_fetch():
    paged = FusionPagedKVCache(block_size=16, num_blocks=64)
    k, v = _random_kv(1, 2, 30, 8, 10)
    fk, fv = paged.update_and_fetch(k, v)
    sk, sv = paged.state
    _assert_kv_equal((fk, fv), (sk, sv))


def test_state_setter_round_trip():
    paged = FusionPagedKVCache(block_size=16, num_blocks=64)
    k, v = _random_kv(1, 2, 30, 8, 10)
    paged.update_and_fetch(k, v)
    sk, sv = paged.state
    mx.eval(sk, sv)
    paged2 = FusionPagedKVCache(block_size=16, num_blocks=64)
    paged2.state = (sk, sv)
    assert paged2.offset == 30
    ok, ov = paged2.state
    _assert_kv_equal((sk, sv), (ok, ov))


def test_trim_recycles_and_keeps_prefix():
    paged = FusionPagedKVCache(block_size=16, num_blocks=64)
    k, v = _random_kv(1, 2, 50, 8, 10)
    pk, pv = paged.update_and_fetch(k, v)
    mx.eval(pk, pv)
    blocks_before = len(paged.block_table)
    trimmed = paged.trim(10)
    assert trimmed == 10
    assert paged.offset == 40
    assert len(paged.block_table) <= blocks_before
    new_k, new_v = _random_kv(1, 2, 5, 8, 10)
    ref_k = mx.concatenate([pk[:, :, :40, :], new_k], axis=2)
    ref_v = mx.concatenate([pv[:, :, :40, :], new_v], axis=2)
    fk, fv = paged.update_and_fetch(new_k, new_v)
    _assert_kv_equal((ref_k, ref_v), (fk, fv))
    assert paged.offset == 45


def test_pool_exhaustion_raises():
    paged = FusionPagedKVCache(block_size=4, num_blocks=2)
    k, v = _random_kv(1, 1, 8, 4, 4)
    paged.update_and_fetch(k, v)
    more_k, more_v = _random_kv(1, 1, 4, 4, 4)
    with pytest.raises(RuntimeError, match="pool exhausted"):
        paged.update_and_fetch(more_k, more_v)


def test_free_all_resets():
    paged = FusionPagedKVCache(block_size=16, num_blocks=64)
    k, v = _random_kv(1, 2, 30, 8, 10)
    paged.update_and_fetch(k, v)
    freed = paged.free_all()
    assert freed == 2
    assert paged.offset == 0
    assert paged.block_table == []
    assert paged.empty()


def test_flat_pool_storage_shape():
    cache = FusionPagedKVCache(block_size=4, num_blocks=8)
    keys = mx.random.uniform(shape=(1, 2, 3, 8))
    values = mx.random.uniform(shape=(1, 2, 3, 8))
    cache.update_and_fetch(keys, values)
    assert hasattr(cache, "keys_pool")
    assert cache.keys_pool.shape == (8, 1, 2, 4, 8)
    assert cache.values_pool.shape == (8, 1, 2, 4, 8)
    assert not hasattr(cache, "keys_slabs")


def test_make_mask_offset_propagates():
    paged = FusionPagedKVCache(block_size=16, num_blocks=64)
    k, v = _random_kv(1, 2, 10, 8, 10)
    paged.update_and_fetch(k, v)
    mask = paged.make_mask(1, return_array=False, window_size=None)
    assert mask is None
    mask2 = paged.make_mask(5, return_array=True, window_size=None)
    assert mask2 is not None
