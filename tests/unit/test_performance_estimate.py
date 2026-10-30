import logging

import pytest

from fusion_mlx.hardware.types import GPUInfo
from fusion_mlx.performance import estimate_tok_per_sec

# 2026-10-30: params=1 with a low-bit quant made _estimate_weight_bytes
# truncate to int(1 * 0.1875) = 0, and estimate_tok_per_sec divided by it —
# POST /v1/recommend returned 500. A non-positive model size must return 0.0
# instead of raising ZeroDivisionError.


@pytest.fixture
def apple_gpu():
    return GPUInfo(
        name="Apple M3 Max",
        vendor="apple",
        vram_bytes=64 * 1024**3,
        memory_bandwidth_gbps=400.0,
        shared_memory=True,
    )


def test_estimate_zero_model_size_returns_zero(apple_gpu, caplog):
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.performance"):
        result = estimate_tok_per_sec(1, "TQ1_0", apple_gpu)
    assert result == 0.0
    assert "non-positive model size" in caplog.text


def test_estimate_zero_bandwidth_returns_zero():
    gpu = GPUInfo(name="Apple M3 Max", vendor="apple", vram_bytes=64 * 1024**3)
    assert estimate_tok_per_sec(7_000_000_000, "Q4_K_M", gpu) == 0.0


def test_estimate_positive_params_positive_result(apple_gpu):
    result = estimate_tok_per_sec(7_000_000_000, "Q4_K_M", apple_gpu)
    assert result > 0


def test_estimate_cpu_only_guard():
    assert estimate_tok_per_sec(0, "Q4_K_M", None) == 0.0
    assert estimate_tok_per_sec(7_000_000_000, "Q4_K_M", None, "cpu_only") > 0
