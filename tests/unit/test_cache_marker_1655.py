# SPDX-License-Identifier: Apache-2.0
"""#1655: response-cache HIT/MISS marker + benchmark bypass on the
OpenAI chat route (tier-server HTTP surface).

Contract mirrors fusion-bench#47: every /v1/chat/completions response
carries ``X-Cache: HIT`` (served from the response cache) or
``X-Cache: MISS`` (fresh inference), and a request header
``X-Cache: bypass`` forces fresh inference and skips cache storage so
benchmark harnesses get distinct-latency fresh repeats.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.cache.response_cache import reset_response_cache
from fusion_mlx.engine.base import GenerationOutput


class _MarkerEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    async def chat(self, **kwargs):
        return GenerationOutput(
            text="hello",
            raw_text="hello",
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        for i, delta in enumerate(["hel", "lo"]):
            is_last = i == 1
            yield GenerationOutput(
                text="hello"[: (i + 1) * 3],
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
            )


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_response_cache()
    yield
    reset_response_cache()


@pytest.fixture
def client(monkeypatch):
    from fusion_mlx.api.openai import router as chat_router
    from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key

    engine = _MarkerEngine()

    async def _fake_resolve(model_name, adapter_path=None):
        return engine

    async def _fake_release(model_name, adapter_path=None):
        return None

    import fusion_mlx.api.openai.chat as chat_mod
    import fusion_mlx.api.openai.streaming as streaming_mod

    monkeypatch.setattr(chat_mod, "_resolve_engine", _fake_resolve)
    monkeypatch.setattr(chat_mod, "_release_engine", _fake_release)
    monkeypatch.setattr(streaming_mod, "_resolve_engine", _fake_resolve)
    monkeypatch.setattr(streaming_mod, "_release_engine", _fake_release)

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[verify_api_key] = lambda: True
    app.dependency_overrides[check_rate_limit] = lambda: True
    return TestClient(app)


def _body(**over):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    payload.update(over)
    return payload


class TestCacheMarker1655:
    def test_fresh_non_streaming_marked_miss(self, client):
        resp = client.post(
            "/v1/chat/completions", json=_body(), headers={"X-Cache": "bypass"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("X-Cache") == "MISS"

    def test_second_identical_request_hit(self, client):
        # temperature 0.0 → CachePolicy.FORCE (cacheable); default temp
        # resolves to BYPASS which never stores.
        r1 = client.post("/v1/chat/completions", json=_body(temperature=0.0))
        assert r1.status_code == 200, r1.text
        assert r1.headers.get("X-Cache") == "MISS"
        r2 = client.post("/v1/chat/completions", json=_body(temperature=0.0))
        assert r2.status_code == 200, r2.text
        assert r2.headers.get("X-Cache") == "HIT"
        assert r2.json()["choices"][0]["message"]["content"] == (
            r1.json()["choices"][0]["message"]["content"]
        )

    def test_bypass_not_stored(self, client):
        r1 = client.post(
            "/v1/chat/completions", json=_body(), headers={"X-Cache": "bypass"}
        )
        assert r1.status_code == 200
        r2 = client.post("/v1/chat/completions", json=_body())
        assert r2.status_code == 200
        assert r2.headers.get("X-Cache") == "MISS"

    def test_streaming_marked_miss(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json=_body(stream=True),
            headers={"X-Cache": "bypass"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("X-Cache") == "MISS"
