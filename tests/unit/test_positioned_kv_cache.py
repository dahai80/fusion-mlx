# SPDX-License-Identifier: Apache-2.0
# Unit tests for fusion_mlx.positioned_kv_cache (#360).
#
# Pins the contract documented in disk_kv_checkpoint.py:443 —
# positioned_update_and_fetch writes at an arbitrary position on plain
# MLX-LM KVCache / QuantizedKVCache layers (no subclassing), buffer
# grows step-aligned, offset advances to max(offset, position+num_steps),
# and the layer still round-trips through mlx_lm save/load_prompt_cache.

from __future__ import annotations

import logging

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    KVCache,
    QuantizedKVCache,
    load_prompt_cache,
    save_prompt_cache,
)

from fusion_mlx.positioned_kv_cache import _grow_kv_cache, positioned_update_and_fetch

logger = logging.getLogger(__name__)


def _dense_kv(num_steps=8, head_dim=4):
    c = KVCache()
    k = mx.random.normal((1, 2, num_steps, head_dim))
    v = mx.random.normal((1, 2, num_steps, head_dim))
    c.update_and_fetch(k, v)
    return c


def _quant_kv(num_steps=8, head_dim=64):
    c = QuantizedKVCache(group_size=64, bits=8)
    k = mx.random.normal((1, 2, num_steps, head_dim))
    v = mx.random.normal((1, 2, num_steps, head_dim))
    c.update_and_fetch(k, v)
    return c


class TestPositionedUpdateDense:
    def test_writes_at_position_and_advances_offset(self):
        c = _dense_kv(num_steps=8)
        assert c.offset == 8
        k = mx.random.normal((1, 2, 5, 4))
        v = mx.random.normal((1, 2, 5, 4))
        fk, fv = positioned_update_and_fetch(c, k, v, position=3)
        # offset stays 8 (max(8, 3+5)=8) since we overwrite within range
        assert c.offset == 8
        # fetched slice length == offset
        assert fk.shape == (1, 2, 8, 4)
        assert fv.shape == (1, 2, 8, 4)

    def test_extends_offset_beyond_current(self):
        c = _dense_kv(num_steps=8)
        k = mx.random.normal((1, 2, 5, 4))
        v = mx.random.normal((1, 2, 5, 4))
        positioned_update_and_fetch(c, k, v, position=10)
        assert c.offset == 15
        assert c.keys.shape[2] >= 15

    def test_buffer_grows_step_aligned(self):
        c = _dense_kv(num_steps=8)
        # buffer starts at 256 (one step). Force growth past it.
        big = 300
        k = mx.random.normal((1, 2, big, 4))
        v = mx.random.normal((1, 2, big, 4))
        positioned_update_and_fetch(c, k, v, position=0)
        # grown buffer must be a multiple of step (256)
        assert c.keys.shape[2] % KVCache.step == 0
        assert c.keys.shape[2] >= big

    def test_overwrite_matches_data(self):
        c = _dense_kv(num_steps=8)
        sentinel_k = mx.full((1, 2, 3, 4), 7.0)
        sentinel_v = mx.full((1, 2, 3, 4), 9.0)
        positioned_update_and_fetch(c, sentinel_k, sentinel_v, position=2)
        # positions 2,3,4 now == 7.0 / 9.0
        fetched_k = c.keys[..., : c.offset, :]
        assert mx.all(fetched_k[..., 2:5, :] == 7.0).item()

    def test_negative_position_rejected(self):
        c = _dense_kv(num_steps=4)
        k = mx.random.normal((1, 2, 2, 4))
        v = mx.random.normal((1, 2, 2, 4))
        with pytest.raises(ValueError):
            positioned_update_and_fetch(c, k, v, position=-1)


class TestPositionedUpdateQuantized:
    def test_quantized_write_advances_offset(self):
        c = _quant_kv(num_steps=8)
        assert c.offset == 8
        k = mx.random.normal((1, 2, 5, 64))
        v = mx.random.normal((1, 2, 5, 64))
        fk, fv = positioned_update_and_fetch(c, k, v, position=3)
        assert c.offset == 8
        # quantized fetched slice is a nested tuple; layer0 length == offset
        assert fk[0].shape[2] == 8

    def test_quantized_buffer_grows_step_aligned(self):
        c = _quant_kv(num_steps=8)
        big = 300
        k = mx.random.normal((1, 2, big, 64))
        v = mx.random.normal((1, 2, big, 64))
        positioned_update_and_fetch(c, k, v, position=0)
        assert c.keys[0].shape[2] % QuantizedKVCache.step == 0
        assert c.keys[0].shape[2] >= big


class TestGrowKvCache:
    def test_noop_when_buffer_large_enough(self):
        c = _dense_kv(num_steps=8)
        before = c.keys.shape[2]
        _grow_kv_cache(c, 100)
        assert c.keys.shape[2] == before

    def test_grows_to_step_multiple(self):
        c = _dense_kv(num_steps=8)
        _grow_kv_cache(c, 300)
        assert c.keys.shape[2] % KVCache.step == 0
        assert c.keys.shape[2] >= 300


class TestRoundTripSaveLoad:
    # The documented invariant (disk_kv_checkpoint.py:443-447): a plain
    # KVCache layer written via positioned_update_and_fetch must round-trip
    # through mlx_lm.save_prompt_cache / load_prompt_cache, because we do
    # NOT subclass (load_prompt_cache looks the class name up in upstream
    # globals).
    def test_dense_layer_round_trips(self, tmp_path):
        c = _dense_kv(num_steps=8)
        k = mx.random.normal((1, 2, 5, 4))
        v = mx.random.normal((1, 2, 5, 4))
        positioned_update_and_fetch(c, k, v, position=10)
        path = str(tmp_path / "cp.safetensors")
        save_prompt_cache(path, [c])
        loaded = load_prompt_cache(path)
        assert len(loaded) == 1
        assert isinstance(loaded[0], KVCache)
        assert loaded[0].offset == c.offset
        assert mx.allclose(
            loaded[0].keys[..., : c.offset, :], c.keys[..., : c.offset, :]
        ).item()

    def test_quantized_layer_round_trips(self, tmp_path):
        c = _quant_kv(num_steps=8)
        k = mx.random.normal((1, 2, 5, 64))
        v = mx.random.normal((1, 2, 5, 64))
        positioned_update_and_fetch(c, k, v, position=10)
        path = str(tmp_path / "cpq.safetensors")
        save_prompt_cache(path, [c])
        loaded = load_prompt_cache(path)
        assert len(loaded) == 1
        assert isinstance(loaded[0], QuantizedKVCache)
        assert loaded[0].offset == c.offset
