# SPDX-License-Identifier: Apache-2.0
"""Tests for the N-tuple state interface on CacheTypeHandler.

The legacy interface in `extract_state` / `reconstruct_cache` modeled
state as a 2-tuple `(keys, values)` dict. fusion-mlx core had hard-coded
`state[0], state[1]` unpacking sprinkled across `prefix_cache.py`,
`paged_ssd_cache.py`, and `boundary_snapshot_store.py`, which silently
dropped the third+ element of N-tuple state caches like DeepSeek V4's
`PoolingCache` (`(buf_kv, buf_gate, pooled)`).

This test module pins the handler-driven interface: per-element axis
metadata, generic serialize/deserialize, and seq-len recovery from a
raw state tuple. Subsequent commits wire fusion_mlx core to use this
interface; this test establishes the contract those changes must keep
stable.

Tests that require real MLX tensors (tensor value round-trips, disk I/O
with safetensors) are skipped in the unit-test environment where MLX is
mocked. They should be run in an integration test suite with real MLX.
"""

from __future__ import annotations

import pytest


class _MockArray:
    """Lightweight mock tensor with configurable shape for handler tests.

    Supports .shape, .copy(), and __getitem__ (slicing returns self) so
    the N-tuple handler logic can be exercised without real MLX.
    """

    def __init__(self, shape):
        self._shape = tuple(shape)

    @property
    def shape(self):
        return self._shape

    def copy(self):
        return _MockArray(self._shape)

    def __getitem__(self, key):
        return self

    def __repr__(self):
        return f"_MockArray(shape={self._shape})"


class TestCacheStateAxisInfoDefault:
    """Default axis_info matches the legacy 2-tuple (keys, values) contract."""

    def test_default_axis_info_two_elements(self):
        from fusion_mlx.cache.type_handlers import KVCacheHandler

        info = KVCacheHandler().get_state_axis_info()
        assert len(info) == 2
        assert info[0].name == "keys"
        assert info[1].name == "values"
        assert info[0].sequence_axis == 2
        assert info[1].sequence_axis == 2
        assert info[0].sliceable is True
        assert info[1].sliceable is True

    def test_rotating_axis_info_marks_non_sliceable(self):
        """RotatingKVCache uses circular buffer, must not be per-block sliced."""
        from fusion_mlx.cache.type_handlers import RotatingKVCacheHandler

        info = RotatingKVCacheHandler().get_state_axis_info()
        assert len(info) == 2
        assert info[0].sliceable is False
        assert info[1].sliceable is False
        # Sequence axis is still axis 2 (the circular buffer dim) even
        # though slicing along it is unsafe.
        assert info[0].sequence_axis == 2

    def test_arrays_cache_marked_variable_length(self):
        from fusion_mlx.cache.type_handlers import ArraysCacheHandler

        h = ArraysCacheHandler()
        assert h.is_variable_length_state() is True
        # Variable-length caches return empty axis info — fusion_mlx core
        # consults the `is_variable_length_state` flag instead.
        assert h.get_state_axis_info() == ()

    def test_cache_list_marked_composite(self):
        from fusion_mlx.cache.type_handlers import CacheListHandler

        h = CacheListHandler()
        assert h.is_composite_cache() is True
        assert h.get_state_axis_info() == ()


class TestSerializeStatePassthrough:
    """Default serialize_state passes through cache_obj.state as a tuple."""

    def test_kvcache_state_serialized_as_2tuple(self):
        from fusion_mlx.cache.type_handlers import KVCacheHandler

        class _FakeKVCache:
            @property
            def state(self):
                return (_MockArray((1, 4, 8, 16)), _MockArray((1, 4, 8, 16)))

        cache = _FakeKVCache()
        elements = KVCacheHandler().serialize_state(cache)
        assert isinstance(elements, tuple)
        assert len(elements) == 2

    def test_serialize_state_handles_missing_state_attr(self):
        from fusion_mlx.cache.type_handlers import KVCacheHandler

        class _Empty:
            pass

        elements = KVCacheHandler().serialize_state(_Empty())
        assert elements == ()


class TestDeserializeStateLegacyContract:
    """Default deserialize_state maps tuple elements to legacy keys/values dict."""

    def test_kvcache_round_trip_via_new_interface(self):
        from fusion_mlx.cache.type_handlers import KVCacheHandler

        mock_keys = _MockArray((1, 4, 8, 16))
        mock_values = _MockArray((1, 4, 8, 16))

        class _FakeKVCache:
            @property
            def state(self):
                return (mock_keys, mock_values)

            @property
            def meta_state(self):
                return ()

        original = _FakeKVCache()
        h = KVCacheHandler()
        elements = h.serialize_state(original)
        assert isinstance(elements, tuple)
        assert len(elements) == 2
        assert elements[0] is mock_keys
        assert elements[1] is mock_values

        # deserialize_state maps elements to keys/values dict and calls
        # reconstruct_cache. In the mock env, KVCache() returns a
        # MagicMock, but keys/values/offset are set correctly.
        restored = h.deserialize_state(elements, meta_state=())
        assert restored is not None
        # Verify the reconstructed cache received the right tensors.
        assert restored.keys is mock_keys
        assert restored.values is mock_values


class TestSeqLenFromTuple:
    """get_state_seq_len_from_tuple recovers length from first sliceable elem."""

    def test_kvcache_seq_len_from_tuple(self):
        from fusion_mlx.cache.type_handlers import KVCacheHandler

        keys = _MockArray((1, 4, 13, 16))  # seq_len = 13 on axis 2
        values = _MockArray((1, 4, 13, 16))
        seq_len = KVCacheHandler().get_state_seq_len_from_tuple((keys, values))
        assert seq_len == 13

    def test_rotating_returns_full_length_even_when_non_sliceable(self):
        """Non-sliceable elements still report seq length on the seq axis;
        the *sliceable* flag controls per-block slicing, not length lookup.
        Default impl skips non-sliceable, so RotatingKVCache reports 0
        until a handler explicitly overrides this method."""
        from fusion_mlx.cache.type_handlers import RotatingKVCacheHandler

        keys = _MockArray((1, 4, 128, 16))
        values = _MockArray((1, 4, 128, 16))
        # Default impl walks for first sliceable element. Rotating has no
        # sliceable elements -> returns 0. This is the expected contract.
        assert (
            RotatingKVCacheHandler().get_state_seq_len_from_tuple((keys, values)) == 0
        )

    def test_seq_len_returns_zero_for_empty_tuple(self):
        from fusion_mlx.cache.type_handlers import KVCacheHandler

        assert KVCacheHandler().get_state_seq_len_from_tuple(()) == 0

    def test_seq_len_returns_zero_for_none_element(self):
        from fusion_mlx.cache.type_handlers import KVCacheHandler

        assert KVCacheHandler().get_state_seq_len_from_tuple((None, None)) == 0


class TestPagedSSDV3Format:
    """V3 safetensors format — N-tuple state keys, V2 polyfill on read.

    PagedSSDCacheManager now supports V3 format with __nstate__ markers
    for N-tuple states (3+ elements) like PoolingCache and MiniMaxM3KVCache.
    """

    def test_v3_format_version_constants(self):
        from fusion_mlx.cache.paged_ssd_cache import (
            _CACHE_FORMAT_VERSION,
            _READABLE_CACHE_FORMAT_VERSIONS,
        )

        assert _CACHE_FORMAT_VERSION == "3"
        assert "2" in _READABLE_CACHE_FORMAT_VERSIONS
        assert "3" in _READABLE_CACHE_FORMAT_VERSIONS

    def test_v3_legacy_2tuple_round_trip_via_unwrap(self):
        from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mgr = PagedSSDCacheManager.__new__(PagedSSDCacheManager)
        arrays = {
            "layer_0_state_0": _MockArray((1, 4, 16, 64)),
            "layer_1_state_0": _MockArray((1, 4, 16, 64)),
            "layer_1_state_1": _MockArray((1, 4, 16, 64)),
        }
        metadata = {
            "num_layers": "2",
            "layer_0_state_count": "1",
            "layer_1_state_count": "2",
            "format_version": "3",
        }
        layers = mgr._reconstruct_layers_from_arrays(arrays, metadata, num_layers=2)
        assert len(layers) == 2
        assert isinstance(layers[0], tuple) and len(layers[0]) == 1
        assert isinstance(layers[1], tuple) and len(layers[1]) == 2

    def test_v3_three_tuple_state_preserved_as_marker(self):
        from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mgr = PagedSSDCacheManager.__new__(PagedSSDCacheManager)
        arrays = {
            "layer_0_state_0": _MockArray((1, 4, 64)),
            "layer_0_state_1": _MockArray((1, 4, 64)),
            "layer_0_state_2": _MockArray((1, 32, 64)),
        }
        metadata = {
            "num_layers": "1",
            "layer_0_state_count": "3",
            "layer_0_nstate_class": "PoolingCache",
            "format_version": "3",
        }
        layers = mgr._reconstruct_layers_from_arrays(arrays, metadata, num_layers=1)
        assert len(layers) == 1
        layer = layers[0]
        assert isinstance(layer, tuple)
        assert layer[0] == "__nstate__"
        assert layer[1] == "PoolingCache"
        assert len(layer[2]) == 3

    def test_v3_safetensors_keys_use_state_k_naming(self):
        from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mgr = PagedSSDCacheManager.__new__(PagedSSDCacheManager)
        arrays = {
            "layer_0_sub_0_state_0": _MockArray((1, 4, 16, 64)),
            "layer_0_sub_0_state_1": _MockArray((1, 4, 16, 64)),
            "layer_0_sub_1_state_0": _MockArray((1, 4, 64)),
            "layer_0_sub_1_state_1": _MockArray((1, 4, 64)),
            "layer_0_sub_1_state_2": _MockArray((1, 32, 64)),
        }
        metadata = {
            "num_layers": "1",
            "is_cache_list_layer_0": "true",
            "layer_0_sub_count": "2",
            "layer_0_state_count": "5",
            "layer_0_sub_0_state_count": "2",
            "layer_0_sub_1_state_count": "3",
            "layer_0_sub_1_class": "PoolingCache",
            "format_version": "3",
        }
        layers = mgr._reconstruct_layers_from_arrays(
            arrays, metadata, num_layers=1, layer_cache_types=["CacheList"]
        )
        assert len(layers) == 1
        layer = layers[0]
        assert isinstance(layer, list)
        assert len(layer) == 2
        assert isinstance(layer[0], tuple) and len(layer[0]) == 2
        sub1 = layer[1]
        assert sub1[0] == "__nstate__"
        assert sub1[1] == "PoolingCache"
        assert len(sub1[2]) == 3

    def test_unsupported_format_version_rejected(self, monkeypatch):
        from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mgr = PagedSSDCacheManager.__new__(PagedSSDCacheManager)

        def _fake_load_raw(path):
            return None

        monkeypatch.setattr(
            type(mgr), "_load_safetensors_raw", classmethod(lambda cls, p: None)
        )
        result = mgr._load_safetensors_file("/fake/path.safetensors")
        assert result is None


class TestPrefixCacheNTupleSubState:
    """prefix_cache._extract_block_tensor_slice preserves N-tuple sub-state.

    V4's CacheList(RotatingKVCache, PoolingCache) hits the non-sliceable
    branch (PoolingCache's buf_kv is 3D so all_sub_sliceable=False). Before
    the fix, that branch cloned only ``sub_state[0], sub_state[1]`` from
    each sub_state — silently dropping PoolingCache's ``pooled`` (index 2),
    which corrupted the cross-session prefix cache hit.

    The fix here: clone every element of every sub_state, wrap length>=3
    sub_states in an ``__nstate__`` marker so downstream paged_ssd /
    reconstruct paths see the full tuple.
    """

    def test_cache_list_non_sliceable_preserves_third_element(self):
        """Most direct regression guard for V4 cross-session corruption.

        Builds a cache_data with a CacheList layer whose second sub_state
        is a 3-tuple (mimicking PoolingCache.state = (buf_kv, buf_gate,
        pooled)) and verifies the third element survives the slice path.
        """
        from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache

        prefix_cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        prefix_cache._block_size = 64
        # Override _clone_tensor to return input directly so mock arrays
        # pass through without being replaced by MagicMock from mx.copy.
        prefix_cache._clone_tensor = lambda tensor: tensor

        # Build a CacheList layer with two sub_states:
        # - sub 0: 2-tuple (keys, values) — RotatingKVCache style, 4D
        # - sub 1: 3-tuple (buf_kv, buf_gate, pooled) — PoolingCache style
        rot_keys = _MockArray((1, 4, 16, 8))
        rot_values = _MockArray((1, 4, 16, 8))
        buf_kv = _MockArray((1, 4, 8))  # 3D — fails 4D sliceable check
        buf_gate = _MockArray((1, 4, 8))
        pooled = _MockArray((1, 32, 8))

        cache_data = [
            {
                "cache_type": "CacheList",
                "class_name": "CacheList",
                "state": [
                    (rot_keys, rot_values),
                    (buf_kv, buf_gate, pooled),
                ],
                "sub_class_names": ["RotatingKVCache", "PoolingCache"],
            }
        ]

        block_slices = prefix_cache._extract_block_tensor_slice(
            cache_data, start_idx=0, end_idx=16, is_last_block=True
        )
        assert block_slices is not None
        assert len(block_slices) == 1
        cache_list_marker = block_slices[0]
        assert cache_list_marker[0] == "__cache_list__"
        sub_tensors = cache_list_marker[1]
        assert len(sub_tensors) == 2

        # Sub 0 is length-2 -> unwrapped to legacy (keys, values).
        sub0 = sub_tensors[0]
        assert isinstance(sub0, tuple) and len(sub0) == 2
        assert sub0[0] is rot_keys
        assert sub0[1] is rot_values

        # Sub 1 is length-3 -> preserved as __nstate__ marker. The third
        # element (pooled) MUST survive — this is the V4 fix point.
        sub1 = sub_tensors[1]
        assert isinstance(sub1, tuple)
        assert sub1[0] == "__nstate__"
        assert sub1[1] == "PoolingCache"
        elements = sub1[2]
        assert len(elements) == 3
        # Critical regression guard: all three elements preserved.
        assert elements[0] is buf_kv
        assert elements[1] is buf_gate
        assert elements[2] is pooled

    def test_boundary_snapshot_three_tuple_round_trip(self, monkeypatch):
        """BoundarySnapshotSSDStore preserves all elements of a 3-tuple
        state through serialize -> deserialize. PoolingCache regression
        guard at the boundary-snapshot layer."""
        from fusion_mlx.cache.boundary_snapshot_store import (
            BoundarySnapshotSSDStore,
        )

        class _MockArrayWithDtype(_MockArray):
            def __init__(self, shape, dtype_str="float32"):
                super().__init__(shape)
                self.dtype_str = dtype_str

        store = BoundarySnapshotSSDStore.__new__(BoundarySnapshotSSDStore)

        buf_kv = _MockArrayWithDtype((1, 4, 64))
        buf_gate = _MockArrayWithDtype((1, 4, 64))
        pooled = _MockArrayWithDtype((1, 32, 64))

        extracted = [
            {
                "state": (buf_kv, buf_gate, pooled),
                "meta_state": (4,),
                "class_name": "PoolingCache",
                "cache_type": "PoolingCache",
            }
        ]

        import mlx.core as mx

        monkeypatch.setattr(mx, "eval", lambda *a, **kw: None)
        monkeypatch.setattr(mx, "synchronize", lambda: None)
        monkeypatch.setattr(mx, "zeros", lambda shape, **kw: _MockArray(shape))

        from fusion_mlx.cache import boundary_snapshot_store as bss

        def _fake_extract_tensor_bytes(arr):
            nbytes = 1
            for d in arr.shape:
                nbytes *= d
            nbytes *= 4
            return (b"\x00" * nbytes, "F32", list(arr.shape))

        monkeypatch.setattr(bss, "_extract_tensor_bytes", _fake_extract_tensor_bytes)

        tensors_raw, metadata = store._serialize_extracted(
            extracted, request_id="test_req", token_count=64
        )

        assert "layer_0_state_0" in tensors_raw
        assert "layer_0_state_1" in tensors_raw
        assert "layer_0_state_2" in tensors_raw

        import json

        layer_info = json.loads(metadata["layer_info"])
        assert len(layer_info) == 1
        assert layer_info[0]["has_state"] == "true"
        assert layer_info[0]["state_count"] == "3"

        def _fake_restore_tensor_from_bytes(data, dtype_str, shape):
            return _MockArrayWithDtype(shape, dtype_str)

        monkeypatch.setattr(
            bss, "_restore_tensor_from_bytes", _fake_restore_tensor_from_bytes
        )

        result = store._deserialize(tensors_raw, metadata)
        assert result is not None
        assert len(result) == 1
        state = result[0]["state"]
        assert isinstance(state, tuple)
        assert len(state) == 3

    def test_boundary_snapshot_v2_layer_keys_polyfill(self, monkeypatch):
        """V2 boundary snapshots stored with legacy ``layer_{i}_0/1`` keys
        are still readable by the V3 reader, returned as a 2-tuple."""
        from fusion_mlx.cache.boundary_snapshot_store import (
            BoundarySnapshotSSDStore,
        )

        store = BoundarySnapshotSSDStore.__new__(BoundarySnapshotSSDStore)

        tensors_raw = {
            "layer_0_0": (b"\x00" * 512, "F32", [1, 4, 16, 64]),
            "layer_0_1": (b"\x00" * 512, "F32", [1, 4, 16, 64]),
        }

        import json

        metadata = {
            "request_id": "v2_test",
            "token_count": "16",
            "num_layers": "1",
            "layer_info": json.dumps(
                [
                    {
                        "class_name": "KVCache",
                        "cache_type": "KVCache",
                        "meta_state": "[]",
                        "has_state": "true",
                    }
                ]
            ),
        }

        from fusion_mlx.cache import boundary_snapshot_store as bss

        def _fake_restore(data, dtype_str, shape):
            return _MockArray(shape)

        monkeypatch.setattr(bss, "_restore_tensor_from_bytes", _fake_restore)

        import mlx.core as mx

        monkeypatch.setattr(mx, "zeros", lambda shape, **kw: _MockArray(shape))

        result = store._deserialize(tensors_raw, metadata)
        assert result is not None
        assert len(result) == 1
        state = result[0]["state"]
        assert isinstance(state, tuple)
        assert len(state) == 2

    def test_pooling_cache_handler_axis_info(self):
        """PoolingCacheHandler exposes 3-element axis_info, all non-sliceable."""
        from fusion_mlx.patches.deepseek_v4.cache_handlers import PoolingCacheHandler

        info = PoolingCacheHandler().get_state_axis_info()
        assert len(info) == 3
        assert [i.name for i in info] == ["buf_kv", "buf_gate", "pooled"]
        assert all(i.sequence_axis == 1 for i in info)
        assert all(i.sliceable is False for i in info)

    def test_pooling_cache_deserialize_3tuple_round_trip(self, monkeypatch):
        import mlx_lm.models.cache as _cache_mod

        from fusion_mlx.patches.deepseek_v4.cache_handlers import PoolingCacheHandler

        class _FakePoolingCache:
            def __init__(self, ratio=1):
                self.state = None

        _cache_mod.PoolingCache = _FakePoolingCache
        monkeypatch.setattr(_cache_mod, "PoolingCache", _FakePoolingCache)
        h = PoolingCacheHandler()

        class _MockPoolingCache:
            @property
            def state(self):
                return (None, None, "pooled_value")

        elements = h.serialize_state(_MockPoolingCache())
        assert isinstance(elements, tuple)
        assert len(elements) == 3
        assert elements[2] == "pooled_value"

        restored = h.deserialize_state(elements, meta_state=4)
        assert restored is not None
        rest_state = restored.state
        assert isinstance(rest_state, tuple)
        assert len(rest_state) == 3
        assert rest_state[2] == "pooled_value"

    def test_pooling_cache_deserialize_legacy_2tuple_input(self, monkeypatch):
        import mlx_lm.models.cache as _cache_mod

        from fusion_mlx.patches.deepseek_v4.cache_handlers import PoolingCacheHandler

        class _FakePoolingCache:
            def __init__(self, ratio=1):
                self.state = None

        _cache_mod.PoolingCache = _FakePoolingCache
        monkeypatch.setattr(_cache_mod, "PoolingCache", _FakePoolingCache)
        h = PoolingCacheHandler()
        buf_kv = _MockArray((1, 4, 8))
        buf_gate = _MockArray((1, 4, 8))
        restored = h.deserialize_state((buf_kv, buf_gate), meta_state=4)
        assert restored is not None
        rest_state = restored.state
        assert isinstance(rest_state, tuple)
        assert len(rest_state) == 3
        assert rest_state[0] is buf_kv
        assert rest_state[1] is buf_gate
        assert rest_state[2] is None

    def test_batch_pooling_cache_handler_axis_info(self):
        from fusion_mlx.patches.deepseek_v4.cache_handlers import (
            BatchPoolingCacheHandler,
        )

        info = BatchPoolingCacheHandler().get_state_axis_info()
        assert len(info) == 3
        assert [i.name for i in info] == ["buf_kv", "buf_gate", "pooled"]
        assert all(i.sliceable is False for i in info)

    @pytest.mark.skip(
        reason="requires real MLX + Scheduler for _extract_cache_states end-to-end test"
    )
    def test_extract_cache_states_preserves_pooling_cache_3tuple(self):
        """scheduler._extract_cache_states preserves PoolingCache's 3-tuple
        state without dropping the third element. This is the topmost entry
        point on the prefill -> store_cache path; if state[2] survives here
        and the downstream serializers (paged_ssd, boundary_snapshot,
        prefix_cache) preserve it, V4 multi-session corruption is fully
        prevented."""
        pass

    def test_cache_list_legacy_two_tuple_unchanged(self):
        """CacheList with all 2-tuple sub_states (legacy) round-trips
        unchanged — keeps the V2 shape so existing callers see no
        behavioral change."""
        from fusion_mlx.cache.prefix_cache import BlockAwarePrefixCache

        prefix_cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
        prefix_cache._block_size = 64
        prefix_cache._clone_tensor = lambda tensor: tensor

        keys = _MockArray((1, 4, 16, 8))
        values = _MockArray((1, 4, 16, 8))

        cache_data = [
            {
                "cache_type": "CacheList",
                "class_name": "CacheList",
                "state": [
                    (keys, values),
                    (keys, values),
                ],
                "sub_class_names": ["KVCache", "KVCache"],
            }
        ]

        block_slices = prefix_cache._extract_block_tensor_slice(
            cache_data, start_idx=0, end_idx=16, is_last_block=True
        )
        assert block_slices is not None
        marker = block_slices[0]
        assert marker[0] == "__cache_list__"
        # Both sub_states are length 2 -> legacy (keys, values) tuples.
        for sub in marker[1]:
            assert isinstance(sub, tuple)
            assert len(sub) == 2
            assert sub[0] is keys
            assert sub[1] is values
