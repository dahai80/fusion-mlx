import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache

import fusion_mlx.scheduler.monkeypatches  # noqa: F401  (installs the patch)


@pytest.fixture
def cache_zero_pad():
    return BatchKVCache([0])


@pytest.fixture
def cache_left_pad():
    return BatchKVCache([2])


def test_patch_installed():
    assert getattr(BatchKVCache, "_fusion_decode_mask_fastpath", False)


def test_decode_n1_zero_padding_returns_none(cache_zero_pad):
    assert cache_zero_pad.make_mask(1) is None


def test_decode_n1_window_size_kwarg_returns_none(cache_zero_pad):
    assert cache_zero_pad.make_mask(1, window_size=None) is None


def test_decode_n1_return_array_still_builds_mask(cache_zero_pad):
    m = cache_zero_pad.make_mask(1, return_array=True)
    assert isinstance(m, mx.array)


def test_decode_n1_nonzero_left_padding_builds_mask(cache_left_pad):
    m = cache_left_pad.make_mask(1)
    assert isinstance(m, mx.array)


def test_prefill_n_gt_1_builds_mask(cache_zero_pad):
    m = cache_zero_pad.make_mask(5)
    assert isinstance(m, mx.array)


def test_window_size_builds_mask(cache_zero_pad):
    m = cache_zero_pad.make_mask(1, window_size=4)
    assert isinstance(m, mx.array)


def test_mirror_tracks_prepare(cache_zero_pad):
    cache_zero_pad.prepare(left_padding=[0], lengths=[3])
    assert cache_zero_pad._fusion_left_padding_py == [0]
    assert cache_zero_pad.make_mask(1) is None


def test_mirror_tracks_state_restore():
    src = BatchKVCache([0])
    src.keys = mx.zeros((1, 2, 8, 4))
    src.values = mx.zeros((1, 2, 8, 4))
    dst = BatchKVCache([1])
    assert dst.make_mask(1) is not None
    dst.state = src.state
    assert dst._fusion_left_padding_py == [0]
    assert dst.make_mask(1) is None
