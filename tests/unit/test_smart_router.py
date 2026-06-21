# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SmartRouter — routing decisions, phase split, circuit breaker tracking."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fusion_mlx.router.cloud_router import CloudRouter
from fusion_mlx.router.smart_router import (
    BenchmarkResult,
    EngineBackend,
    PhaseHandoff,
    RouteDecision,
    RouterConfig,
    SmartRouter,
    TaskPriority,
)


class TestSmartRouterDecide:

    def _make_router(self, **config_overrides):
        config = RouterConfig(**config_overrides)
        return SmartRouter(
            config=config,
            llm_engine=AsyncMock(),
            rapid_engine=AsyncMock(),
        )

    def test_explicit_override(self):
        router = self._make_router()
        decision = router.decide(prompt_length=100, backend_override=EngineBackend.OMLX)
        assert decision.prefill_backend == EngineBackend.OMLX
        assert decision.decode_backend == EngineBackend.OMLX
        assert "explicit override" in decision.reason

    def test_cloud_fallback_for_massive_context(self):
        cr = MagicMock()
        cr.cloud_fallback_threshold = 8192
        config = RouterConfig(cloud_fallback_threshold=8192)
        router = SmartRouter(config=config, cloud_router=cr, llm_engine=AsyncMock())
        decision = router.decide(prompt_length=20000, cache_hit_rate=0.0)
        assert decision.prefill_backend == EngineBackend.CLOUD

    def test_phase_split_for_long_prompt_low_cache(self):
        router = self._make_router(phase_split_threshold=1024)
        decision = router.decide(prompt_length=4096, cache_hit_rate=0.2)
        assert decision.split_phases is True
        assert decision.prefill_backend == EngineBackend.OMLX
        assert decision.decode_backend == EngineBackend.RAPID

    def test_no_split_for_high_cache_hit(self):
        router = self._make_router(phase_split_threshold=1024)
        decision = router.decide(prompt_length=4096, cache_hit_rate=0.8)
        assert decision.split_phases is False

    def test_no_split_for_short_prompt(self):
        router = self._make_router(phase_split_threshold=8192)
        decision = router.decide(prompt_length=512)
        assert decision.split_phases is False

    def test_realtime_priority_prefers_rapid(self):
        router = self._make_router()
        decision = router.decide(prompt_length=512, task_tag="claude_code")
        assert decision.prefill_backend == EngineBackend.RAPID

    def test_batch_priority_prefers_omlx(self):
        router = self._make_router()
        decision = router.decide(prompt_length=512, task_tag="openclaw")
        assert decision.prefill_backend == EngineBackend.OMLX

    def test_benchmark_based_routing(self):
        router = self._make_router()
        router.store_benchmark(BenchmarkResult(
            model_id="test-3b", backend=EngineBackend.OMLX, quant_format="4bit",
            tps=100.0, latency_p50=50.0, latency_p99=80.0, memory_peak_bytes=1024,
         ))
        router.store_benchmark(BenchmarkResult(
            model_id="test-3b", backend=EngineBackend.RAPID, quant_format="4bit",
            tps=80.0, latency_p50=30.0, latency_p99=45.0, memory_peak_bytes=512,
         ))
         # Set EMA counts past cold-start threshold so measured values dominate
        router._ema_state["test-3b"] = {
             "omlx": {"tps": 100.0, "latency_p50": 50.0, "count": 25},
             "rapid": {"tps": 80.0, "latency_p50": 30.0, "count": 25},
         }
        decision = router.decide(prompt_length=512, model_id="test-3b", quant_format="4bit")
        assert decision.prefill_backend == EngineBackend.OMLX
        assert decision.decode_backend == EngineBackend.RAPID

    def test_benchmark_filters_auto_and_cloud(self):
        router = self._make_router()
        router.store_benchmark(BenchmarkResult(
            model_id="m2", backend=EngineBackend.AUTO, quant_format="4bit",
            tps=200.0, latency_p50=10.0, latency_p99=20.0, memory_peak_bytes=256,
        ))
        router.store_benchmark(BenchmarkResult(
            model_id="m2", backend=EngineBackend.CLOUD, quant_format="4bit",
            tps=500.0, latency_p50=5.0, latency_p99=10.0, memory_peak_bytes=0,
        ))
        decision = router.decide(prompt_length=512, model_id="m2", quant_format="4bit")
        assert decision.prefill_backend in (EngineBackend.OMLX, EngineBackend.RAPID)

    def test_ema_smoothing(self):
        router = self._make_router(ema_alpha=0.5)
        router.store_benchmark(BenchmarkResult(
            model_id="m3", backend=EngineBackend.OMLX, quant_format="4bit",
            tps=100.0, latency_p50=50.0, latency_p99=80.0, memory_peak_bytes=1024,
        ))
        router.store_benchmark(BenchmarkResult(
            model_id="m3", backend=EngineBackend.RAPID, quant_format="4bit",
            tps=60.0, latency_p50=30.0, latency_p99=45.0, memory_peak_bytes=512,
        ))
        d1 = router.decide(prompt_length=512, model_id="m3", quant_format="4bit")
        router.store_benchmark(BenchmarkResult(
            model_id="m3", backend=EngineBackend.OMLX, quant_format="4bit",
            tps=200.0, latency_p50=20.0, latency_p99=35.0, memory_peak_bytes=2048,
        ))
        router.store_benchmark(BenchmarkResult(
            model_id="m3", backend=EngineBackend.RAPID, quant_format="4bit",
            tps=40.0, latency_p50=60.0, latency_p99=90.0, memory_peak_bytes=256,
        ))
        d2 = router.decide(prompt_length=512, model_id="m3", quant_format="4bit")
        assert d1 is not None
        assert d2 is not None


class TestSmartRouterRouteChat:

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_success(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock())
        cr = CloudRouter(cloud_model="gpt-4", threshold=100000)
        cr.report_local_success = MagicMock()
        cr.report_local_failure = MagicMock()
        router = SmartRouter(
            config=RouterConfig(), cloud_router=cr, llm_engine=llm, rapid_engine=llm,
        )
        await router.route_chat(
            [{"role": "user", "content": "hi"}], {}, prompt_length=10,
        )
        cr.report_local_success.assert_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_failure(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("crash"))
        cr = CloudRouter(cloud_model="gpt-4", threshold=100000)
        cr.report_local_success = MagicMock()
        cr.report_local_failure = MagicMock()
        router = SmartRouter(
            config=RouterConfig(), cloud_router=cr, llm_engine=llm, rapid_engine=llm,
        )
        with pytest.raises(RuntimeError, match="crash"):
            await router.route_chat(
                [{"role": "user", "content": "hi"}], {}, prompt_length=10,
            )
        cr.report_local_failure.assert_called()


class TestSmartRouterTokenEstimation:

    def test_cjk_heavy_text(self):
        router = SmartRouter(llm_engine=AsyncMock())
        messages = [{"role": "user", "content": "这是一个中文测试文本"}]
        tokens = router._estimate_tokens(messages)
        assert tokens >= 15

    def test_ascii_heavy_text(self):
        router = SmartRouter(llm_engine=AsyncMock())
        messages = [{"role": "user", "content": "Hello world this is a test of ASCII token estimation"}]
        tokens = router._estimate_tokens(messages)
        assert tokens >= 15

    def test_mixed_content(self):
        router = SmartRouter(llm_engine=AsyncMock())
        messages = [{"role": "user", "content": "Hello 世界 test 测试"}]
        tokens = router._estimate_tokens(messages)
        assert tokens >= 5

    def test_list_content_with_text_parts(self):
        router = SmartRouter(llm_engine=AsyncMock())
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "url": "http://x.png"},
            ],
        }]
        tokens = router._estimate_tokens(messages)
        assert tokens >= 1


class TestSmartRouterStats:

    def test_reset_stats_with_lock(self):
        router = SmartRouter(llm_engine=AsyncMock())
        router._route_count["omlx"] = 10
        router._split_count = 5
        router._cloud_count = 3
        router.reset_stats()
        assert router._route_count == {}
        assert router._split_count == 0
        assert router._cloud_count == 0

    def test_get_stats(self):
        router = SmartRouter(llm_engine=AsyncMock())
        router._route_count["test"] = 1
        stats = router.get_stats()
        assert "route_count" in stats
        assert "split_count" in stats
        assert "cloud_count" in stats


class TestSmartRouterPriorityResolution:

    def test_claude_code_is_realtime(self):
        router = SmartRouter(llm_engine=AsyncMock())
        p = router._resolve_priority("claude_code")
        assert p == TaskPriority.REALTIME

    def test_openclaw_is_batch(self):
        router = SmartRouter(llm_engine=AsyncMock())
        p = router._resolve_priority("openclaw")
        assert p == TaskPriority.BATCH

    def test_embedding_is_background(self):
        router = SmartRouter(llm_engine=AsyncMock())
        p = router._resolve_priority("embedding")
        assert p == TaskPriority.BACKGROUND

    def test_unknown_tag_uses_default(self):
        config = RouterConfig(default_priority=TaskPriority.REALTIME)
        router = SmartRouter(config=config, llm_engine=AsyncMock())
        p = router._resolve_priority("unknown_tag")
        assert p == TaskPriority.REALTIME


class TestPhaseHandoff:

    def test_handoff_creation(self):
        h = PhaseHandoff(
            request_id="r1", block_table=MagicMock(), kv_buffers={},
            meta_states=[], model_name="test-model", num_tokens=100,
            prefill_backend=EngineBackend.OMLX, decode_backend=EngineBackend.RAPID,
        )
        assert h.request_id == "r1"
        assert h.prefill_backend == EngineBackend.OMLX
        assert h.decode_backend == EngineBackend.RAPID


class TestRouteDecision:

    def test_unified_backend(self):
        d = RouteDecision(
            prefill_backend=EngineBackend.OMLX, decode_backend=EngineBackend.OMLX,
            reason="test", split_phases=False,
        )
        assert d.unified_backend is True

    def test_split_phases(self):
        d = RouteDecision(
            prefill_backend=EngineBackend.OMLX, decode_backend=EngineBackend.RAPID,
            reason="split", split_phases=True,
        )
        assert d.unified_backend is False
