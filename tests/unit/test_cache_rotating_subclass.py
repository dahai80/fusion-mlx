# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.cache._rotating_subclass — PrefillReadyRotatingKVCache.

Covers size() clamping (keys None, buffer_len 0, normal), physical_resident_size.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fusion_mlx.cache._rotating_subclass import PrefillReadyRotatingKVCache


class TestPrefillReadyRotatingKVCache:
    def _cache(self, keys_shape2=10, super_size=10):
        cache = PrefillReadyRotatingKVCache.__new__(PrefillReadyRotatingKVCache)
        if keys_shape2 is None:
            cache.keys = None
        else:
            cache.keys = MagicMock()
            cache.keys.shape = [1, 8, keys_shape2]
        return cache

    def test_size_keys_none_returns_0(self):
        cache = self._cache(keys_shape2=None)
        assert cache.size() == 0

    def test_size_buffer_zero_returns_0(self):
        cache = self._cache(keys_shape2=0)
        assert cache.size() == 0

    def test_size_clamps_to_buffer(self):
        cache = self._cache(keys_shape2=5, super_size=10)
        with patch(
            "fusion_mlx.cache._rotating_subclass.RotatingKVCache.size", return_value=10
        ):
            assert cache.size() == 5

    def test_size_returns_super_when_smaller(self):
        cache = self._cache(keys_shape2=20, super_size=8)
        with patch(
            "fusion_mlx.cache._rotating_subclass.RotatingKVCache.size", return_value=8
        ):
            assert cache.size() == 8

    def test_physical_resident_size_keys_none(self):
        cache = self._cache(keys_shape2=None)
        assert cache.physical_resident_size == 0

    def test_physical_resident_size_buffer_zero(self):
        cache = self._cache(keys_shape2=0)
        assert cache.physical_resident_size == 0

    def test_physical_resident_size_clamps(self):
        cache = self._cache(keys_shape2=3, super_size=10)
        with patch(
            "fusion_mlx.cache._rotating_subclass.RotatingKVCache.size", return_value=10
        ):
            assert cache.physical_resident_size == 3
