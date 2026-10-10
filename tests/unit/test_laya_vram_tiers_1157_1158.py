# SPDX-License-Identifier: Apache-2.0
"""Tests for VRAM priority tiers (#1158) and laya decision endpoints (#1157)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.decisions_routes import router as decisions_router
from fusion_mlx.middleware.auth import verify_api_key
from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(decisions_router)
    app.dependency_overrides[verify_api_key] = lambda: True
    return TestClient(app)


class _FakeEngine:
    async def decide(self, state, questions):
        return {
            "answers": {
                qid: {"type": q["type"], "confidence": 0.9}
                for qid, q in questions.items()
            },
            "usage": {"input_tokens": 10},
        }

    @property
    def model_name(self):
        return "test-laya"

    def get_stats(self):
        return {"model_name": "test-laya", "loaded": True}


@pytest.fixture
def laya_engine():
    eng = _FakeEngine()
    with patch("fusion_mlx.api.decisions_routes._engine", eng):
        yield eng


class TestPriorityTiers:
    def test_entry_default_tier(self):
        e = EngineEntry(
            model_id="m",
            model_path="/p",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        assert e.priority_tier == "P0_DYNAMIC"
        assert e.is_pinned is False

    def test_pin_model_never_evict(self):
        pool = EnginePool()
        e = EngineEntry(
            model_id="laya-test",
            model_path="/p",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024 * 1024 * 512,
        )
        e.engine = SimpleNamespace()
        pool._entries["laya-test"] = e
        assert pool.pin_model("laya-test", tier="P0_NEVER_EVICT") is True
        assert e.is_pinned is True
        assert e.priority_tier == "P0_NEVER_EVICT"

    def test_pin_model_invalid_tier(self):
        pool = EnginePool()
        e = EngineEntry(
            model_id="m",
            model_path="/p",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        pool._entries["m"] = e
        assert pool.pin_model("m", tier="INVALID") is False
        assert e.priority_tier == "P0_DYNAMIC"

    def test_pin_model_not_found(self):
        pool = EnginePool()
        assert pool.pin_model("nonexistent") is False

    def test_pin_model_p1_tdm(self):
        pool = EnginePool()
        e = EngineEntry(
            model_id="diffusion",
            model_path="/p",
            model_type="image",
            engine_type="image_gen",
            estimated_size=1024 * 1024 * 20480,
        )
        e.engine = SimpleNamespace()
        pool._entries["diffusion"] = e
        assert pool.pin_model("diffusion", tier="P1_TDM") is True
        assert e.priority_tier == "P1_TDM"
        assert e.is_pinned is False

    def test_get_memory_tiers_loaded_only(self):
        pool = EnginePool()
        loaded = EngineEntry(
            model_id="loaded",
            model_path="/p",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024 * 1024 * 1000,
        )
        loaded.engine = SimpleNamespace()
        loaded.priority_tier = "P0_NEVER_EVICT"
        loaded.is_pinned = True
        unloaded = EngineEntry(
            model_id="unloaded",
            model_path="/p",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        pool._entries["loaded"] = loaded
        pool._entries["unloaded"] = unloaded
        tiers = pool.get_memory_tiers()
        assert len(tiers["P0"]) == 1
        assert tiers["P0"][0]["model"] == "loaded"
        assert tiers["P0"][0]["pinned"] is True
        assert len(tiers["P1"]) == 0

    def test_get_memory_tiers_p1_bucket(self):
        pool = EnginePool()
        e = EngineEntry(
            model_id="wan2",
            model_path="/p",
            model_type="video",
            engine_type="video_gen",
            estimated_size=1024 * 1024 * 25000,
        )
        e.engine = SimpleNamespace()
        e.priority_tier = "P1_TDM"
        pool._entries["wan2"] = e
        tiers = pool.get_memory_tiers()
        assert len(tiers["P1"]) == 1
        assert tiers["P1"][0]["model"] == "wan2"
        assert tiers["P1"][0]["tier"] == "P1_TDM"


class TestLayaDecideEndpoint:
    def test_decide_with_preset(self, laya_engine):
        client = _client()
        resp = client.post(
            "/v1/laya/decide",
            json={"prompt": "def hello():", "preset": "router"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answers" in data
        assert "latency_ms" in data
        assert data["model"] == "test-laya"

    def test_decide_with_custom_questions(self, laya_engine):
        client = _client()
        resp = client.post(
            "/v1/laya/decide",
            json={
                "prompt": "hello world",
                "questions": {
                    "q1": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": ["a", "b"],
                    }
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "q1" in data["answers"]

    def test_decide_preset_and_questions_mutually_exclusive(self, laya_engine):
        client = _client()
        resp = client.post(
            "/v1/laya/decide",
            json={
                "prompt": "hi",
                "preset": "router",
                "questions": {"q1": {"type": "noul"}},
            },
        )
        assert resp.status_code == 400

    def test_decide_neither_preset_nor_questions(self, laya_engine):
        client = _client()
        resp = client.post("/v1/laya/decide", json={"prompt": "hi"})
        assert resp.status_code == 400

    def test_decide_invalid_preset(self, laya_engine):
        client = _client()
        resp = client.post("/v1/laya/decide", json={"prompt": "hi", "preset": "bogus"})
        assert resp.status_code == 400

    def test_decide_too_many_questions(self, laya_engine):
        client = _client()
        qs = {f"q{i}": {"type": "noul"} for i in range(65)}
        resp = client.post("/v1/laya/decide", json={"prompt": "hi", "questions": qs})
        assert resp.status_code == 400


class TestLayaStatusEndpoint:
    def test_status_not_loaded(self):
        with patch("fusion_mlx.api.decisions_routes._engine", None):
            client = _client()
            resp = client.get("/v1/laya/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is False
        assert data["priority_tier"] == "P0_NEVER_EVICT"
        assert data["pinned"] is True

    def test_status_loaded(self, laya_engine):
        client = _client()
        resp = client.get("/v1/laya/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is True
        assert data["model"] == "test-laya"
        assert "stats" in data
