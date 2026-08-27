# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #660.

The live POST /v1/audio/speech route (fusion_mlx.api.audio_routes.router)
must resolve the ``kokoro`` audio alias via the audio registry
(``mlx-community/Kokoro-82M-bf16``), NOT via the colliding LLM alias table
in model-config.json (which previously mapped ``kokoro`` ->
``Qwen3-TTS-12Hz-1.7B-Base-8bit``). Existing audio tests mount the
``routes_internal.audio`` shim router, so the bug in the live route's
``_resolve_model`` was never exercised. This test mounts the LIVE router and
asserts the resolved id reaching the engine pool is the audio-registry id.
"""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 22050) -> bytes:
    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


DUMMY_WAV = _make_wav_bytes()


def _make_mock_tts_engine() -> MagicMock:
    from fusion_mlx.engines.tts import TTSEngine

    engine = MagicMock(spec=TTSEngine)
    engine.synthesize = AsyncMock(return_value=DUMMY_WAV)
    engine.supports_native_tts_streaming.return_value = False
    return engine


def _make_recording_pool(tts_engine=None, observed: list | None = None):
    # Pool whose get_engine records the resolved model id, so the test can
    # assert WHICH model the live route resolved (the regression core).
    if observed is None:
        observed = []

    async def _get_engine(model_id):
        observed.append(model_id)
        return tts_engine or _make_mock_tts_engine()

    pool = MagicMock()
    pool.get_engine = AsyncMock(side_effect=_get_engine)
    pool.get_entry = MagicMock(
        return_value=MagicMock(model_type="audio_tts", engine_type="tts")
    )
    pool.get_model_ids.return_value = ["mlx-community/Kokoro-82M-bf16"]
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()
    pool.resolve_model_id = MagicMock(side_effect=lambda m, _: m)
    return pool, observed


@pytest.fixture
def speech_client():
    from fusion_mlx.api.audio_routes import router

    app = FastAPI()
    app.include_router(router)

    pool, observed = _make_recording_pool()

    with (
        patch("fusion_mlx.api.audio_routes._get_engine_pool", return_value=pool),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        yield client, observed


class TestKokoroAliasResolution:
    def test_kokoro_resolves_to_audio_registry_hf_id(self, speech_client):
        # #660: live route must consult audio registry FIRST so a colliding
        # LLM alias cannot shadow a TTS alias to the wrong model.
        client, observed = speech_client
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "hello world",
                "voice": "af_heart",
                "response_format": "wav",
            },
        )
        assert response.status_code == 200, response.text
        assert observed == ["mlx-community/Kokoro-82M-bf16"], (
            f"kokoro must resolve via the audio registry to "
            f"mlx-community/Kokoro-82M-bf16, got {observed}"
        )

    def test_kokoro_does_not_resolve_to_qwen3_tts(self, speech_client):
        # Guard against the exact pre-fix regression: the colliding LLM alias
        # mapped kokoro -> Qwen3-TTS-12Hz-1.7B-Base-8bit.
        client, observed = speech_client
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "hello world",
                "voice": "af_heart",
                "response_format": "wav",
            },
        )
        assert response.status_code == 200, response.text
        assert "Qwen3-TTS" not in "".join(
            observed
        ), f"kokoro must NOT resolve to a Qwen3-TTS id, got {observed}"

    def test_full_hf_id_passes_through(self, speech_client):
        # A full HuggingFace repo id has no alias; it must reach the engine
        # unchanged (no shadowing, no structural rejection for a valid id).
        client, observed = speech_client
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "mlx-community/Kokoro-82M-bf16",
                "input": "hello world",
                "voice": "af_heart",
                "response_format": "wav",
            },
        )
        assert response.status_code == 200, response.text
        assert observed == ["mlx-community/Kokoro-82M-bf16"]
