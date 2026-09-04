# SPDX-License-Identifier: Apache-2.0
"""Unit tests for D1 flywheel bench->recommend->apply->rebench loop."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import flywheel_routes
from fusion_mlx.bench import flywheel as fw
from fusion_mlx.bench.flywheel import (
    BenchResult,
    Recommendation,
    recommend,
)
from fusion_mlx.config import reset_config
from fusion_mlx.middleware.auth import verify_api_key


@pytest.fixture
def fresh_config():
    cfg = reset_config()
    yield cfg


class TestRecommend:
    def test_picks_best_throughput(self):
        results = [
            BenchResult("a", 8, 2048, "q4", tok_per_sec=20.0, vram_used_gb=4.0),
            BenchResult("b", 16, 4096, "q4", tok_per_sec=35.0, vram_used_gb=8.0),
            BenchResult("c", 32, 8192, "q4", tok_per_sec=30.0, vram_used_gb=12.0),
        ]
        reco = recommend(results)
        assert reco.batch_size == 16
        assert reco.max_kv_tokens == 4096
        assert reco.expected_tok_per_sec == 35.0

    def test_respects_memory_budget(self):
        results = [
            BenchResult("a", 8, 2048, "q4", tok_per_sec=20.0, vram_used_gb=4.0),
            BenchResult("b", 16, 4096, "q4", tok_per_sec=50.0, vram_used_gb=20.0),
        ]
        reco = recommend(results, memory_budget_gb=10.0)
        assert reco.batch_size == 8
        assert reco.expected_tok_per_sec == 20.0

    def test_falls_back_when_budget_excludes_all(self):
        results = [
            BenchResult("a", 8, 2048, "q4", tok_per_sec=20.0, vram_used_gb=40.0),
        ]
        reco = recommend(results, memory_budget_gb=1.0)
        assert reco.batch_size == 8

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            recommend([])


class TestApply:
    def test_applies_to_config(self, fresh_config):
        reco = Recommendation(
            batch_size=64,
            max_kv_tokens=8192,
            quant_level="q4",
            expected_tok_per_sec=40.0,
            memory_budget_gb=0.0,
        )
        cfg = fw.apply(reco, config=fresh_config)
        assert cfg.scheduler.completion_batch_size == 64
        assert cfg.scheduler.cache_memory_mb == 16384
        assert cfg.scheduler.kv_cache_quantization is True
        assert cfg.scheduler.kv_cache_quantization_bits == 4

    def test_fp16_disables_kv_quant(self, fresh_config):
        reco = Recommendation(
            batch_size=32,
            max_kv_tokens=4096,
            quant_level="fp16",
            expected_tok_per_sec=40.0,
            memory_budget_gb=0.0,
        )
        cfg = fw.apply(reco, config=fresh_config)
        assert cfg.scheduler.kv_cache_quantization is False


class TestFlywheel:
    def test_returns_delta_and_improves(self, fresh_config, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.bench.flywheel._RESULTS_DIR", tmp_path)
        before = BenchResult(
            "before", 8, 2048, "q4", tok_per_sec=20.0, vram_used_gb=4.0
        )

        def fake_runner(config_id, batch_size, max_kv_tokens, quant_level):
            return {
                "tok_per_sec": 45.0,
                "vram_used_gb": 5.0,
                "ttft_ms": 30.0,
            }

        report = fw.flywheel(
            before=before,
            memory_budget_gb=10.0,
            runner=fake_runner,
            config=fresh_config,
        )
        assert report.improved is True
        assert report.tok_per_sec_delta == pytest.approx(25.0)
        assert report.after.tok_per_sec == 45.0
        assert report.recommendation.batch_size == 8

    def test_stores_both_results(self, fresh_config, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.bench.flywheel._RESULTS_DIR", tmp_path)
        before = BenchResult("pre", 4, 1024, "q4", tok_per_sec=10.0, vram_used_gb=2.0)

        def fake_runner(config_id, batch_size, max_kv_tokens, quant_level):
            return {"tok_per_sec": 12.0, "vram_used_gb": 2.5}

        fw.flywheel(
            before=before,
            memory_budget_gb=10.0,
            runner=fake_runner,
            config=fresh_config,
        )
        files = list(Path(tmp_path).glob("*.json"))
        assert len(files) == 2


@pytest.fixture
def app_client():
    app = FastAPI()

    async def _fake_auth():
        return True

    app.dependency_overrides[verify_api_key] = _fake_auth
    app.include_router(flywheel_routes.router)
    return TestClient(app)


class TestRoutes:
    def test_run_stores_result(self, app_client, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.bench.flywheel._RESULTS_DIR", tmp_path)
        r = app_client.post(
            "/v1/bench/flywheel/run",
            json={
                "config_id": "rt-test",
                "batch_size": 16,
                "max_kv_tokens": 4096,
                "quant_level": "q4",
                "tok_per_sec": 33.0,
                "vram_used_gb": 6.0,
            },
        )
        assert r.status_code == 200
        assert r.json()["stored"] is True
        assert (tmp_path / "rt-test.json").exists()

    def test_results_lists_stored(self, app_client, tmp_path, monkeypatch):
        monkeypatch.setattr("fusion_mlx.bench.flywheel._RESULTS_DIR", tmp_path)
        app_client.post(
            "/v1/bench/flywheel/run",
            json={
                "config_id": "lst",
                "batch_size": 8,
                "max_kv_tokens": 2048,
                "quant_level": "q4",
                "tok_per_sec": 20.0,
            },
        )
        r = app_client.get("/v1/bench/flywheel/results")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["results"][0]["config_id"] == "lst"

    def test_recommend_route(self, app_client):
        r = app_client.post(
            "/v1/bench/flywheel/recommend",
            json={
                "memory_budget_gb": 10.0,
                "results": [
                    {
                        "config_id": "a",
                        "batch_size": 8,
                        "max_kv_tokens": 2048,
                        "quant_level": "q4",
                        "tok_per_sec": 20.0,
                        "vram_used_gb": 4.0,
                    },
                    {
                        "config_id": "b",
                        "batch_size": 16,
                        "max_kv_tokens": 4096,
                        "quant_level": "q4",
                        "tok_per_sec": 35.0,
                        "vram_used_gb": 8.0,
                    },
                ],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["batch_size"] == 16
        assert body["expected_tok_per_sec"] == 35.0

    def test_recommend_empty_400(self, app_client):
        r = app_client.post(
            "/v1/bench/flywheel/recommend",
            json={"results": []},
        )
        assert r.status_code == 400

    def test_apply_route(self, app_client, fresh_config):
        r = app_client.post(
            "/v1/bench/flywheel/apply",
            json={
                "recommendation": {
                    "batch_size": 64,
                    "max_kv_tokens": 8192,
                    "quant_level": "q4",
                    "expected_tok_per_sec": 40.0,
                    "memory_budget_gb": 0.0,
                }
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] is True
        assert body["completion_batch_size"] == 64

    def test_flywheel_route(self, app_client, tmp_path, monkeypatch, fresh_config):
        monkeypatch.setattr("fusion_mlx.bench.flywheel._RESULTS_DIR", tmp_path)
        monkeypatch.setattr(
            "fusion_mlx.bench.run_benchmark",
            lambda mid: {"tokens_per_second": 45.0, "vram_used_gb": 5.0},
        )
        r = app_client.post(
            "/v1/bench/flywheel",
            json={
                "before": {
                    "config_id": "fw-pre",
                    "batch_size": 8,
                    "max_kv_tokens": 2048,
                    "quant_level": "q4",
                    "tok_per_sec": 20.0,
                    "vram_used_gb": 4.0,
                },
                "memory_budget_gb": 10.0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["improved"] is True
        assert body["tok_per_sec_delta"] > 0


class TestCommunityBenchDestubbed:
    def test_run_benchmark_returns_local_data(self):
        from fusion_mlx.community_bench.runner import run_benchmark

        result = run_benchmark(model="test-model", num_prompts=2)
        assert result["source"] == "local"
        assert result["model"] == "test-model"
        assert "tokens_per_second" in result

    def test_submit_is_noop(self):
        from fusion_mlx.community_bench.submission import submit_benchmark

        result = submit_benchmark()
        assert result["submitted"] is False
