# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.memory_monitor module."""

import pytest

from fusion_mlx.memory_monitor import (
    _SDPA_FALLBACK_SCORE_DTYPE_SIZE,
    _SDPA_FULL_SUPPORTED_HEAD_DIMS,
    _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD,
    _SDPA_VECTOR_SUPPORTED_HEAD_DIMS,
    MemoryInfo,
    MemoryMonitor,
)


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
        assert monitor._num_layers == 32
        assert monitor._head_dim == 128
        assert monitor._num_kv_heads == 8

    def test_set_model_info_with_optional(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(
            num_layers=32,
            head_dim=128,
            num_kv_heads=8,
            num_query_heads=64,
            dtype_bytes=4,
        )
        assert monitor._num_attention_heads == 64
        assert monitor._dtype_size == 4

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


class TestMemoryInfo:
    """MemoryInfo dataclass tests."""

    def test_memory_info_creation(self):
        info = MemoryInfo(
            total_bytes=16 * 1024**3,
            used_bytes=8 * 1024**3,
            available_bytes=8 * 1024**3,
            utilization=0.5,
        )
        assert info.total_bytes == 16 * 1024**3
        assert info.used_bytes == 8 * 1024**3
        assert info.available_bytes == 8 * 1024**3
        assert info.utilization == 0.5


class TestEvictionEnabled:
    def test_eviction_enabled_default(self):
        monitor = MemoryMonitor()
        assert monitor.eviction_enabled is True

    def test_eviction_enabled_false(self):
        monitor = MemoryMonitor(eviction_enabled=False)
        assert monitor.eviction_enabled is False


class TestGetMemoryInfo:
    def test_get_memory_info(self):
        monitor = MemoryMonitor()
        info = monitor.get_memory_info()
        assert isinstance(info, MemoryInfo)
        assert info.total_bytes >= 0
        assert info.used_bytes >= 0
        assert info.available_bytes >= 0
        assert 0.0 <= info.utilization <= 1.0


class TestIsUnderPressure:
    def test_is_under_pressure(self):
        monitor = MemoryMonitor()
        result = monitor.is_under_pressure()
        assert isinstance(result, bool)


class TestBytesToFree:
    def test_bytes_to_free(self):
        monitor = MemoryMonitor()
        result = monitor.bytes_to_free()
        assert isinstance(result, int)
        assert result >= 0

    def test_bytes_to_free_no_eviction(self):
        monitor = MemoryMonitor(eviction_enabled=False)
        assert monitor.bytes_to_free() == 0


class TestEstimateBlockMemory:
    def test_estimate_block_memory(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, head_dim=128, num_kv_heads=8)
        result = monitor.estimate_block_memory(64)
        assert isinstance(result, (int, float))
        assert result > 0


class TestEstimateBlocksToFree:
    def test_estimate_blocks_to_free(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, head_dim=128, num_kv_heads=8)
        block_mem = monitor.estimate_block_memory(64)
        result = monitor.estimate_blocks_to_free(int(block_mem * 3), 64)
        assert isinstance(result, int)
        assert result >= 1

    def test_estimate_blocks_to_free_disabled_raises(self):
        monitor = MemoryMonitor(eviction_enabled=False)
        with pytest.raises(RuntimeError, match="eviction_enabled=False"):
            monitor.estimate_blocks_to_free(1024, 64)


class TestGetStats:
    def test_get_stats(self):
        monitor = MemoryMonitor()
        stats = monitor.get_stats()
        assert isinstance(stats, dict)
        assert "total_bytes" in stats
        assert "used_bytes" in stats
        assert "available_bytes" in stats
        assert "utilization" in stats
        assert "max_kv_cache_memory" in stats
        assert "baseline_memory" in stats
        assert "has_model_info" in stats


class TestSetBaselineMemory:
    def test_set_baseline_memory(self):
        monitor = MemoryMonitor()
        monitor.set_baseline_memory()
        assert isinstance(monitor._baseline_memory, int)
        assert monitor._baseline_memory >= 0


class TestSetRequestStats:
    def test_set_request_stats(self):
        monitor = MemoryMonitor()
        monitor.set_request_stats(running=3, waiting=5)
        assert monitor._running_requests == 3
        assert monitor._waiting_requests == 5


class TestSDPAConstants:
    def test_sdpa_constants(self):
        assert _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD == 8
        assert 64 in _SDPA_FULL_SUPPORTED_HEAD_DIMS
        assert 80 in _SDPA_FULL_SUPPORTED_HEAD_DIMS
        assert 128 in _SDPA_FULL_SUPPORTED_HEAD_DIMS
        assert 64 in _SDPA_VECTOR_SUPPORTED_HEAD_DIMS
        assert 96 in _SDPA_VECTOR_SUPPORTED_HEAD_DIMS
        assert 128 in _SDPA_VECTOR_SUPPORTED_HEAD_DIMS
        assert 256 in _SDPA_VECTOR_SUPPORTED_HEAD_DIMS
        assert _SDPA_FALLBACK_SCORE_DTYPE_SIZE == 2


class TestEstimateChunkTransientBytes:
    def test_estimate_chunk_transient_bytes(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, head_dim=128, num_kv_heads=8)
        result = monitor.estimate_chunk_transient_bytes(n_tokens=512, kv_len=2000)
        assert isinstance(result, int)
        assert result > 0
