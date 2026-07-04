# SPDX-License-Identifier: Apache-2.0
import threading

import pytest

from fusion_mlx.server_metrics import ServerMetrics, get_server_metrics


class TestServerMetrics:
    def test_initial_state(self):
        metrics = ServerMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_tokens_generated == 0
        assert metrics.total_tokens_prompt == 0
        assert metrics.total_cached_tokens == 0
        assert metrics.active_requests == 0
        assert metrics.model_stats == {}

    def test_inc_tokens(self):
        metrics = ServerMetrics()
        metrics.inc_tokens(generated=50, prompt=100, cached=30)
        assert metrics.total_tokens_generated == 50
        assert metrics.total_tokens_prompt == 100
        assert metrics.total_cached_tokens == 30

    def test_inc_tokens_accumulates(self):
        metrics = ServerMetrics()
        metrics.inc_tokens(generated=10, prompt=20, cached=5)
        metrics.inc_tokens(generated=15, prompt=30, cached=10)
        assert metrics.total_tokens_generated == 25
        assert metrics.total_tokens_prompt == 50
        assert metrics.total_cached_tokens == 15

    def test_update_active_requests(self):
        metrics = ServerMetrics()
        metrics.update_active_requests(1)
        assert metrics.active_requests == 1
        metrics.update_active_requests(2)
        assert metrics.active_requests == 3
        metrics.update_active_requests(-3)
        assert metrics.active_requests == 0

    def test_record_request_complete(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=30,
            prefill_duration=0.5,
        )
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.total_tokens_prompt == 100
        assert metrics.total_tokens_generated == 50
        assert metrics.total_cached_tokens == 30

    def test_multiple_requests(self):
        metrics = ServerMetrics()
        for _ in range(5):
            metrics.record_request_complete(
                prompt_tokens=100,
                completion_tokens=50,
                cached_tokens=20,
                prefill_duration=0.2,
            )
        assert metrics.total_requests == 5
        assert metrics.total_tokens_prompt == 500
        assert metrics.total_tokens_generated == 250
        assert metrics.total_cached_tokens == 100

    def test_per_model_tracking(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="model-a"
        )
        metrics.record_request_complete(
            prompt_tokens=200, completion_tokens=80, model_id="model-b"
        )
        assert "model-a" in metrics.model_stats
        assert "model-b" in metrics.model_stats
        assert metrics.model_stats["model-a"]["prompt_tokens"] == 100
        assert metrics.model_stats["model-a"]["completion_tokens"] == 50
        assert metrics.model_stats["model-b"]["prompt_tokens"] == 200
        assert metrics.model_stats["model-b"]["completion_tokens"] == 80

    def test_per_model_prefill_tps(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=1000,
            completion_tokens=100,
            prefill_duration=2.0,
            model_id="fast-model",
        )
        stats = metrics.model_stats["fast-model"]
        assert stats["avg_prefill_tps"] == pytest.approx(500.0, abs=0.1)
        assert stats["requests"] == 1

    def test_per_model_prefill_tps_no_subtract(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=400,
            prefill_duration=2.0,
            model_id="model-a",
        )
        stats = metrics.model_stats["model-a"]
        assert stats["avg_prefill_tps"] == pytest.approx(500.0, abs=0.1)

    def test_no_model_id_no_per_model_entry(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50
        )
        assert metrics.model_stats == {}

    def test_thread_safety(self):
        metrics = ServerMetrics()
        num_threads = 10
        records_per_thread = 100

        def record_batch():
            for _ in range(records_per_thread):
                metrics.record_request_complete(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cached_tokens=3,
                    prefill_duration=0.01,
                )

        threads = [threading.Thread(target=record_batch) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_expected = num_threads * records_per_thread
        assert metrics.total_requests == total_expected
        assert metrics.total_tokens_prompt == total_expected * 10
        assert metrics.total_tokens_generated == total_expected * 5
        assert metrics.total_cached_tokens == total_expected * 3

    def test_to_dict(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="test"
        )
        d = metrics.to_dict()
        assert d["total_requests"] == 1
        assert d["total_tokens_prompt"] == 100
        assert d["total_tokens_generated"] == 50
        assert "test" in d["model_stats"]
        assert "_lock" not in d

    def test_to_dict_excludes_lock(self):
        metrics = ServerMetrics()
        d = metrics.to_dict()
        assert "_lock" not in d


class TestServerMetricsSingleton:
    def test_get_server_metrics_returns_instance(self):
        metrics = get_server_metrics()
        assert isinstance(metrics, ServerMetrics)

    def test_get_server_metrics_returns_same_instance(self):
        m1 = get_server_metrics()
        m2 = get_server_metrics()
        assert m1 is m2
