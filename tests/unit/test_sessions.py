# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-session token usage tracking (issue #226)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.session_routes import router as sessions_router
from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key
from fusion_mlx.sessions import (
    SessionTracker,
    get_session_tracker,
    record_chat_session,
    reset_session_tracker_for_tests,
)


@pytest.fixture(autouse=True)
def _isolate_tracker():
    reset_session_tracker_for_tests()
    yield
    reset_session_tracker_for_tests()


def test_record_aggregates_single_session():
    tracker = SessionTracker()
    tracker.record("s1", prompt_tokens=10, completion_tokens=5, cached_tokens=2)
    tracker.record("s1", prompt_tokens=20, completion_tokens=7, cached_tokens=3)
    stats = tracker.get("s1")
    assert stats is not None
    assert stats.prompt_tokens == 30
    assert stats.completion_tokens == 12
    assert stats.cached_tokens == 5
    assert stats.total_tokens == 42
    assert stats.request_count == 2


def test_record_isolates_sessions():
    tracker = SessionTracker()
    tracker.record("s1", prompt_tokens=10, completion_tokens=5)
    tracker.record("s2", prompt_tokens=100, completion_tokens=50)
    assert tracker.get("s1").total_tokens == 15
    assert tracker.get("s2").total_tokens == 150


def test_get_unknown_returns_none():
    tracker = SessionTracker()
    assert tracker.get("nope") is None


def test_record_empty_session_id_noop():
    tracker = SessionTracker()
    tracker.record("", prompt_tokens=10, completion_tokens=5)
    assert tracker.list_sessions() == []


def test_record_chat_session_helper_handles_none():
    record_chat_session(None, prompt_tokens=10, completion_tokens=5)
    assert get_session_tracker().list_sessions() == []


def test_record_chat_session_helper_records():
    record_chat_session(
        "client-1", prompt_tokens=10, completion_tokens=5, cached_tokens=1
    )
    stats = get_session_tracker().get("client-1")
    assert stats is not None
    assert stats.total_tokens == 15
    assert stats.request_count == 1


def test_set_max_context():
    tracker = SessionTracker()
    assert tracker.set_max_context("s1", 8192) is True
    stats = tracker.get("s1")
    assert stats.max_context_tokens == 8192


def test_set_max_context_none_clears():
    tracker = SessionTracker()
    tracker.set_max_context("s1", 4096)
    tracker.set_max_context("s1", None)
    assert tracker.get("s1").max_context_tokens is None


def test_set_max_context_rejects_negative():
    tracker = SessionTracker()
    assert tracker.set_max_context("s1", -1) is False
    assert tracker.set_max_context("", 100) is False


def test_lru_eviction():
    tracker = SessionTracker(max_sessions=3)
    tracker.record("s1", prompt_tokens=1)
    tracker.record("s2", prompt_tokens=1)
    tracker.record("s3", prompt_tokens=1)
    tracker.record("s4", prompt_tokens=1)
    assert tracker.get("s1") is None
    assert tracker.get("s2") is not None
    assert tracker.get("s3") is not None
    assert tracker.get("s4") is not None


def test_lru_touch_prevents_eviction():
    tracker = SessionTracker(max_sessions=2)
    tracker.record("s1", prompt_tokens=1)
    tracker.record("s2", prompt_tokens=1)
    tracker.record("s1", prompt_tokens=1)
    tracker.record("s3", prompt_tokens=1)
    assert tracker.get("s1") is not None
    assert tracker.get("s2") is None
    assert tracker.get("s3") is not None


def test_to_dict_has_all_fields():
    tracker = SessionTracker()
    tracker.record("s1", prompt_tokens=10, completion_tokens=5, cached_tokens=2)
    d = tracker.get("s1").to_dict()
    assert set(d.keys()) == {
        "session_id",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "total_tokens",
        "request_count",
        "max_context_tokens",
        "last_active",
    }
    assert d["session_id"] == "s1"


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(sessions_router)
    app.dependency_overrides[verify_api_key] = lambda: True
    app.dependency_overrides[check_rate_limit] = lambda: None
    return TestClient(app)


def test_route_get_stats_not_found():
    client = _make_client()
    resp = client.get("/v1/sessions/unknown/stats")
    assert resp.status_code == 404


def test_route_get_stats_ok():
    record_chat_session(
        "route-1", prompt_tokens=12, completion_tokens=8, cached_tokens=3
    )
    client = _make_client()
    resp = client.get("/v1/sessions/route-1/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "route-1"
    assert body["prompt_tokens"] == 12
    assert body["completion_tokens"] == 8
    assert body["cached_tokens"] == 3
    assert body["total_tokens"] == 20
    assert body["request_count"] == 1
    assert body["max_context_tokens"] is None


def test_route_put_context_then_stats_reflects_cap():
    record_chat_session("route-2", prompt_tokens=5, completion_tokens=5)
    client = _make_client()
    resp = client.put(
        "/v1/sessions/route-2/context", json={"max_context_tokens": 16384}
    )
    assert resp.status_code == 200
    assert resp.json()["max_context_tokens"] == 16384
    stats = client.get("/v1/sessions/route-2/stats").json()
    assert stats["max_context_tokens"] == 16384


def test_route_put_context_rejects_negative():
    client = _make_client()
    resp = client.put("/v1/sessions/route-3/context", json={"max_context_tokens": -5})
    assert resp.status_code == 400


def test_route_put_context_clears_with_null():
    record_chat_session("route-4", prompt_tokens=1, completion_tokens=1)
    client = _make_client()
    client.put("/v1/sessions/route-4/context", json={"max_context_tokens": 4096})
    resp = client.put("/v1/sessions/route-4/context", json={"max_context_tokens": None})
    assert resp.status_code == 200
    stats = client.get("/v1/sessions/route-4/stats").json()
    assert stats["max_context_tokens"] is None
