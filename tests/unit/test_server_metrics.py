# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from fusion_mlx.server_metrics import ServerMetrics, get_server_metrics


class TestServerMetrics:

    def test_initial_state(self):
        metrics = ServerMetrics()
        d = metrics.to_dict()
        assert d["total_requests"] == 0
        assert d["successful_requests"] == 0
        assert d["failed_requests"] == 0
        assert d["total_tokens_generated"] == 0
        assert d["total_tokens_prompt"] == 0
        assert d["total_cached_tokens"] == 0
        assert d["active_requests"] == 0
        assert d["model_stats"] == {}

    def test_record_request_complete(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=30,
            prefill_duration=0.5,
        )
        d = metrics.to_dict()
        assert d["total_requests"] == 1
        assert d["successful_requests"] == 1
        assert d["total_tokens_prompt"] == 100
        assert d["total_tokens_generated"] == 50
        assert d["total_cached_tokens"] == 30

    def test_multiple_requests(self):
        metrics = ServerMetrics()
        for _ in range(5):
            metrics.record_request_complete(
                prompt_tokens=100,
                completion_tokens=50,
                cached_tokens=20,
                prefill_duration=0.2,
            )
        d = metrics.to_dict()
        assert d["total_requests"] == 5
        assert d["total_tokens_prompt"] == 500
        assert d["total_tokens_generated"] == 250
        assert d["total_cached_tokens"] == 100

    def test_inc_tokens(self):
        metrics = ServerMetrics()
        metrics.inc_tokens(generated=100, prompt=200, cached=50)
        d = metrics.to_dict()
        assert d["total_tokens_generated"] == 100
        assert d["total_tokens_prompt"] == 200
        assert d["total_cached_tokens"] == 50

    def test_update_active_requests(self):
        metrics = ServerMetrics()
        metrics.update_active_requests(1)
        metrics.update_active_requests(1)
        assert metrics.to_dict()["active_requests"] == 2
        metrics.update_active_requests(-1)
        assert metrics.to_dict()["active_requests"] == 1

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

        d = metrics.to_dict()
        total_expected = num_threads * records_per_thread
        assert d["total_requests"] == total_expected
        assert d["total_tokens_prompt"] == total_expected * 10
        assert d["total_tokens_generated"] == total_expected * 5
        assert d["total_cached_tokens"] == total_expected * 3

    def test_per_model_stats(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, prefill_duration=0.2, model_id="model-a"
        )
        metrics.record_request_complete(
            prompt_tokens=200, completion_tokens=80, prefill_duration=0.5, model_id="model-b"
        )
        d = metrics.to_dict()
        assert "model-a" in d["model_stats"]
        assert "model-b" in d["model_stats"]
        assert d["model_stats"]["model-a"]["requests"] == 1
        assert d["model_stats"]["model-a"]["prompt_tokens"] == 100
        assert d["model_stats"]["model-b"]["requests"] == 1
        assert d["model_stats"]["model-b"]["prompt_tokens"] == 200

    def test_per_model_avg_prefill_tps(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=1000, completion_tokens=100, prefill_duration=2.0, model_id="m"
        )
        # 1000 / 2.0 = 500 tok/s
        assert metrics.to_dict()["model_stats"]["m"]["avg_prefill_tps"] == pytest.approx(500.0, abs=0.1)

    def test_per_model_avg_prefill_tps_running_average(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=1000, completion_tokens=100, prefill_duration=2.0, model_id="m"
        )
        metrics.record_request_complete(
            prompt_tokens=1000, completion_tokens=100, prefill_duration=1.0, model_id="m"
        )
        # First: 500 tok/s, Second: 1000 tok/s -> avg = 750
        assert metrics.to_dict()["model_stats"]["m"]["avg_prefill_tps"] == pytest.approx(750.0, abs=0.1)

    def test_singleton(self):
        m1 = get_server_metrics()
        m2 = get_server_metrics()
        assert m1 is m2

    def test_to_dict_keys(self):
        metrics = ServerMetrics()
        d = metrics.to_dict()
        expected_keys = {
            "total_requests", "successful_requests", "failed_requests",
            "total_tokens_generated", "total_tokens_prompt", "total_cached_tokens",
            "active_requests", "model_stats",
        }
        assert set(d.keys()) == expected_keys
