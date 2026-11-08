# SPDX-License-Identifier: Apache-2.0
"""#1518: deterministic replay provider — free zero-GPU model for
pipeline E2E structural runs."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.openai import router as openai_router
from fusion_mlx.api.openai._common import set_openai_context
from fusion_mlx.api.replay import (
    REPLAY_JSON_MARKER,
    REPLAY_MODEL_ID,
    is_replay_model,
    replay_content,
)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(openai_router)
    from fusion_mlx.middleware.auth import verify_api_key

    app.dependency_overrides[verify_api_key] = lambda: True
    from fusion_mlx.api._concurrency import init_request_semaphore

    init_request_semaphore(8)
    set_openai_context(None, SimpleNamespace())
    return TestClient(app)


class TestReplayDetection:
    def test_matches_ids_and_aliases(self):
        assert is_replay_model("replay-dev")
        assert is_replay_model("replay")
        assert is_replay_model("REPLAY-DEV")
        assert not is_replay_model("qwen3-8b")
        assert not is_replay_model(None)


class TestReplayContent:
    def test_echo_last_user_message(self):
        msgs = [
            SimpleNamespace(role="system", content="sys"),
            SimpleNamespace(role="user", content="hello world"),
        ]
        assert replay_content(msgs) == "hello world"

    def test_marker_returns_canned_json(self):
        msgs = [SimpleNamespace(role="user", content=f"{REPLAY_JSON_MARKER} task")]
        out = replay_content(msgs)
        parsed = json.loads(out)
        assert parsed["replay"] is True
        assert parsed["echo"] == "task"
        assert parsed == json.loads(replay_content(msgs))

    def test_deterministic_across_calls(self):
        msgs = [SimpleNamespace(role="user", content="same input")]
        assert replay_content(msgs) == replay_content(list(msgs))


class TestReplayRoute:
    def test_chat_completion_echo(self):
        client = _client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "replay-dev",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "ping"
        assert body["usage"]["total_tokens"] == 0

    def test_chat_completion_json_marker(self):
        client = _client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "replay",
                "messages": [{"role": "user", "content": "[REPLAY_JSON] plan"}],
            },
        )
        assert resp.status_code == 200
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        assert parsed["replay"] is True
        assert parsed["echo"] == "plan"

    def test_streaming_shape(self):
        client = _client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "replay-dev",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        deltas = [json.loads(ln[6:]) for ln in lines[:-1]]
        contents = [
            c["choices"][0]["delta"].get("content")
            for c in deltas
            if c["choices"][0]["delta"].get("content")
        ]
        assert contents == ["hi"]

    def test_models_listing_marks_test_tier(self):
        client = _client()
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        by_id = {m["id"]: m for m in resp.json()["data"]}
        assert REPLAY_MODEL_ID in by_id
        assert by_id[REPLAY_MODEL_ID].get("capabilities") == {"tier": "test"}
        assert by_id[REPLAY_MODEL_ID].get("default_for_chat") is None


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])
