# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /v1/audio/converse — end-to-end voice conversation."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_wav_bytes() -> bytes:
    return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def _make_stt_engine():
    engine = MagicMock()
    engine.__class__ = MagicMock()
    engine.transcribe = AsyncMock(return_value={"text": "hello world"})
    return engine


def _make_llm_engine():
    engine = MagicMock()
    engine.__class__ = MagicMock()
    gen = MagicMock()
    gen.text = "Hi there! How can I help?"
    gen.completion_tokens = 7
    engine.chat = AsyncMock(return_value=gen)
    return engine


def _make_tts_engine():
    engine = MagicMock()
    engine.__class__ = MagicMock()
    engine.synthesize = AsyncMock(return_value=_make_wav_bytes())
    return engine


def _make_app(pool=None):
    from fusion_mlx.api.audio_routes import router
    from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[check_rate_limit] = lambda: None
    return app


@pytest.fixture
def converse_client():
    app = _make_app()
    stt = _make_stt_engine()
    llm = _make_llm_engine()
    tts = _make_tts_engine()
    pool = MagicMock()
    pool.get_engine = AsyncMock(side_effect=[stt, llm, tts])
    with (
        patch("fusion_mlx.api.audio_routes._pool", pool),
        patch(
            "fusion_mlx.api.audio_routes._resolve_model",
            side_effect=lambda m: m,
        ),
        patch("fusion_mlx.server.resolve_model_id", return_value="test-llm"),
        patch(
            "fusion_mlx.routes_internal.audio._resolve_default_voice_literal",
            return_value="af_heart",
        ),
        patch("fusion_mlx.engines.stt.STTEngine", MagicMock),
        patch("fusion_mlx.engines.batched.BatchedEngine", MagicMock),
        patch("fusion_mlx.engines.tts.TTSEngine", MagicMock),
        TestClient(app, raise_server_exceptions=True) as client,
    ):
        yield client, pool, stt, llm, tts


def test_converse_route_registered():
    from fusion_mlx.api.audio_routes import router

    paths = [r.path for r in router.routes]
    assert "/v1/audio/converse" in paths


def test_converse_returns_wav_bytes(converse_client):
    client, pool, stt, llm, tts = converse_client
    resp = client.post(
        "/v1/audio/converse",
        files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
        data={
            "model": "test-llm",
            "stt_model": "whisper-test",
            "tts_model": "kokoro-test",
            "voice": "default",
            "max_tokens": "256",
            "temperature": "0.5",
            "speed": "1.0",
            "include_metadata": "false",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 0
    stt.transcribe.assert_awaited_once()
    llm.chat.assert_awaited_once()
    tts.synthesize.assert_awaited_once()


def test_converse_include_metadata_returns_json(converse_client):
    client, pool, stt, llm, tts = converse_client
    resp = client.post(
        "/v1/audio/converse",
        files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
        data={
            "model": "test-llm",
            "include_metadata": "true",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["transcript"] == "hello world"
    assert body["reply"] == "Hi there! How can I help?"
    assert body["audio"] is not None
    decoded = base64.b64decode(body["audio"])
    assert len(decoded) > 0


def test_converse_system_prompt_passed_to_llm(converse_client):
    client, pool, stt, llm, tts = converse_client
    resp = client.post(
        "/v1/audio/converse",
        files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
        data={
            "model": "test-llm",
            "system_prompt": "You are a helpful assistant.",
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = llm.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    assert any(
        m.get("role") == "system" and "helpful" in m.get("content", "")
        for m in messages
    )


def test_converse_empty_transcript_returns_422():
    app = _make_app()
    stt_engine = MagicMock()
    stt_engine.transcribe = AsyncMock(return_value={"text": ""})
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=stt_engine)
    with (
        patch("fusion_mlx.api.audio_routes._pool", pool),
        patch(
            "fusion_mlx.api.audio_routes._resolve_model",
            side_effect=lambda m: m,
        ),
        patch("fusion_mlx.engines.stt.STTEngine", MagicMock),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        resp = client.post(
            "/v1/audio/converse",
            files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            data={"model": "test-llm"},
        )
    assert resp.status_code == 422


def test_converse_empty_llm_reply_returns_422():
    app = _make_app()
    stt = _make_stt_engine()
    llm = MagicMock()
    gen = MagicMock()
    gen.text = ""
    gen.completion_tokens = 0
    llm.chat = AsyncMock(return_value=gen)
    pool = MagicMock()
    pool.get_engine = AsyncMock(side_effect=[stt, llm])
    with (
        patch("fusion_mlx.api.audio_routes._pool", pool),
        patch(
            "fusion_mlx.api.audio_routes._resolve_model",
            side_effect=lambda m: m,
        ),
        patch("fusion_mlx.server.resolve_model_id", return_value="test-llm"),
        patch("fusion_mlx.engines.stt.STTEngine", MagicMock),
        patch("fusion_mlx.engines.batched.BatchedEngine", MagicMock),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        resp = client.post(
            "/v1/audio/converse",
            files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            data={"model": "test-llm"},
        )
    assert resp.status_code == 422


def test_converse_three_stage_chain():
    import inspect

    from fusion_mlx.api.audio_routes import converse

    src = inspect.getsource(converse)
    assert "stt_engine" in src
    assert "transcribe" in src
    assert "llm_engine" in src
    assert "engine.chat" in src or "llm_engine.chat" in src
    assert "tts_engine" in src
    assert "synthesize" in src
