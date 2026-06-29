# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.memory_monitor module.

NOTE: fusion-mlx MemoryMonitor has a significantly different API from omlx:
- No MemoryInfo dataclass
- No eviction_enabled parameter
- No get_memory_info(), is_under_pressure(), bytes_to_free()
- No estimate_block_memory(), estimate_blocks_to_free()
- No get_stats(), set_baseline_memory(), set_request_stats()
- No check_interval parameter
- set_model_info() has different signature (no kv_quant)
- _predicted_chunk_transient() replaces estimate_chunk_transient_bytes()
- _estimate_sdpa_activation_bytes() exists (not estimate_sdpa_activation_bytes)
- SDPA constants are not module-level exports
"""

import pytest

from fusion_mlx.memory_monitor import MemoryMonitor


class TestMemoryMonitor:
    """Test cases for MemoryMonitor (adapted to fusion-mlx API)."""

    def test_creation(self):
        monitor = MemoryMonitor()
        assert monitor is not None

    def test_creation_with_max_kv_cache(self):
        monitor = MemoryMonitor(max_kv_cache_memory=2 * 1024**3)
        assert monitor.max_kv_cache_memory == 2 * 1024**3

    def test_set_model_info(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, head_dim=128, num_kv_heads=8)
        assert monitor._model_num_layers == 32
        assert monitor._model_head_dim == 128
        assert monitor._model_num_kv_heads == 8

    def test_set_model_info_with_optional(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(
            num_layers=32,
            head_dim=128,
            num_kv_heads=8,
            num_query_heads=64,
            dtype_bytes=4,
        )
        assert monitor._model_num_query_heads == 64
        assert monitor._model_dtype_bytes == 4

    def test_estimate_prefill_peak_bytes(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, head_dim=128, num_kv_heads=8)
        peak = monitor.estimate_prefill_peak_bytes(
            new_tokens=1000,
            prefill_step_size=512,
            cached_tokens=500,
        )
        assert isinstance(peak, int)
        assert peak > 0

    def test_estimate_prompt_kv_bytes(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, head_dim=128, num_kv_heads=8)
        result = monitor.estimate_prompt_kv_bytes(new_tokens=1000, cached_tokens=200)
        assert isinstance(result, tuple)
        assert len(result) == 2
        new_kv, cached_kv = result
        assert isinstance(new_kv, int)
        assert isinstance(cached_kv, int)

    def test_estimate_sdpa_activation_bytes(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, head_dim=128, num_kv_heads=8)
        result = monitor._estimate_sdpa_activation_bytes(query_tokens=100, kv_len=1000)
        assert isinstance(result, int)
        assert result > 0


@pytest.mark.skip(reason="omlx-only: MemoryInfo not in fusion-mlx")
class TestMemoryInfo:
    """MemoryInfo dataclass does not exist in fusion-mlx."""

    def test_memory_info_creation(self):
        pass


@pytest.mark.skip(reason="omlx-only: eviction_enabled not in fusion-mlx")
class TestEvictionEnabled:
    def test_eviction_enabled_default(self):
        pass


@pytest.mark.skip(reason="omlx-only: get_memory_info not in fusion-mlx")
class TestGetMemoryInfo:
    def test_get_memory_info(self):
        pass


@pytest.mark.skip(reason="omlx-only: is_under_pressure not in fusion-mlx")
class TestIsUnderPressure:
    def test_is_under_pressure(self):
        pass


@pytest.mark.skip(reason="omlx-only: bytes_to_free not in fusion-mlx")
class TestBytesToFree:
    def test_bytes_to_free(self):
        pass


@pytest.mark.skip(reason="omlx-only: estimate_block_memory not in fusion-mlx")
class TestEstimateBlockMemory:
    def test_estimate_block_memory(self):
        pass


@pytest.mark.skip(reason="omlx-only: estimate_blocks_to_free not in fusion-mlx")
class TestEstimateBlocksToFree:
    def test_estimate_blocks_to_free(self):
        pass


@pytest.mark.skip(reason="omlx-only: get_stats not in fusion-mlx")
class TestGetStats:
    def test_get_stats(self):
        pass


@pytest.mark.skip(reason="omlx-only: set_baseline_memory not in fusion-mlx")
class TestSetBaselineMemory:
    def test_set_baseline_memory(self):
        pass


@pytest.mark.skip(reason="omlx-only: set_request_stats not in fusion-mlx")
class TestSetRequestStats:
    def test_set_request_stats(self):
        pass


@pytest.mark.skip(reason="omlx-only: SDPA constants not exported from fusion-mlx")
class TestSDPAConstants:
    def test_sdpa_constants(self):
        pass


@pytest.mark.skip(reason="omlx-only: estimate_chunk_transient_bytes not in fusion-mlx")
class TestEstimateChunkTransientBytes:
    def test_estimate_chunk_transient_bytes(self):
        pass
