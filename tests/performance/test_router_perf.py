# SPDX-License-Identifier: Apache-2.0
"""Performance tests for SmartRouter — decision latency, throughput under load."""

import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from fusion_mlx.router.smart_router import (
    SmartRouter, RouterConfig, EngineBackend, BenchmarkResult,
)
from fusion_mlx.router.cloud_router import CloudRouter


class TestRouterDecisionLatency:

    def _make_router(self):
        config = RouterConfig()
        return SmartRouter(
            config=config, llm_engine=AsyncMock(), rapid_engine=AsyncMock(),
           )

    def test_decide_latency_under_1ms(self):
        router = self._make_router()
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            router.decide(prompt_length=512, task_tag="claude_code")
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        assert avg_ms < 1.0, f"Avg decide latency {avg_ms:.2f}ms exceeds 1ms budget"

    def test_decide_latency_with_benchmark_under_5ms(self):
        router = self._make_router()
        router.store_benchmark(BenchmarkResult(
            model_id="perf-test", backend=EngineBackend.OMLX, quant_format="4bit",
            tps=100.0, latency_p50=50.0, latency_p99=80.0, memory_peak_bytes=1024,
         ))
        router.store_benchmark(BenchmarkResult(
            model_id="perf-test", backend=EngineBackend.RAPID, quant_format="4bit",
            tps=80.0, latency_p50=30.0, latency_p99=45.0, memory_peak_bytes=512,
         ))
        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            router.decide(prompt_length=512, model_id="perf-test", quant_format="4bit")
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        assert avg_ms < 5.0, f"Avg benchmark decide latency {avg_ms:.2f}ms exceeds 5ms"

    def test_token_estimation_10k_chars_under_1ms(self):
        router = self._make_router()
        long_text = "Hello world " * 500
        messages = [{"role": "user", "content": long_text}]
        start = time.perf_counter()
        for _ in range(100):
            router._estimate_tokens(messages)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 1.0, f"Token estimation {avg_ms:.2f}ms exceeds 1ms for 10k chars"


class TestCircuitBreakerThroughput:

    def test_rapid_fail_recover_cycle(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=100000)
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            cr.report_local_failure()
            cr.report_local_success()
        elapsed = time.perf_counter() - start
        tps = iterations / elapsed
        assert tps > 50000, f"Circuit breaker ops {tps:.0f}/s, expected >50k"


class TestCloudRouterTimeout:

    @pytest.mark.asyncio
    async def test_call_cloud_times_out(self):
        import asyncio
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        litellm_mock = MagicMock()
        async def slow_completion(**kwargs):
            await asyncio.sleep(60)
        litellm_mock.acompletion = slow_completion
        cr._litellm = litellm_mock
        with pytest.raises((asyncio.TimeoutError, Exception)):
            await cr._call_cloud(litellm_mock, {"model": "gpt-4", "messages": [], "stream": False}, False)
