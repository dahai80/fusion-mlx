# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /v1/models listing audio models (INV-02).

Verifies that audio_stt and audio_tts models appear in the /v1/models
response with the current entry contract (id / object / modality /
capabilities / loaded / state), and that they coexist with text (LLM)
models in the same response.

Rescue 2026-08-30: the original file pinned an obsolete OpenAI-strict
shape (``owned_by`` field + a pool-mocked ``_server_state``). The live
``routes_internal.models.list_models`` route is config + registry driven
and emits ``id`` / ``object`` / ``modality`` / ``tool_call_parser`` /
``reasoning_parser`` / ``loaded`` / ``state`` / ``capabilities`` — no
``owned_by``. Audio modality is resolved by the audio alias registry
(``resolve_audio_alias``: whisper-* -> stt, kokoro -> tts), so audio
models surface with ``modality="audio"`` and an audio capability tag
(``audio.transcription`` / ``audio.speech``). Rewritten against the
``_mounted(model_name, pool)`` seam from
``test_v1_models_loaded_state_577.py`` (set_models_context + cfg attrs,
NOT ``_server_state``).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

STT_ID = "whisper-large-v3"
TTS_ID = "kokoro"
LLM_ID = "mlx-community/Qwen3-0.6B-bf16"


@contextmanager
def _mounted(*, model_name=None, pool=None):
    from fusion_mlx.config import get_config
    from fusion_mlx.routes_internal import models as models_route

    app = FastAPI()
    app.include_router(models_route.router)

    cfg = get_config()
    saved = {
        k: getattr(cfg, k, None)
        for k in (
            "model_name",
            "model_alias",
            "model_registry",
            "embedding_model_locked",
            "api_key",
        )
    }
    saved_pool = models_route._pool
    cfg.model_name = model_name
    cfg.model_alias = None
    cfg.model_registry = None
    cfg.embedding_model_locked = None
    cfg.api_key = None
    models_route.set_models_context(pool)
    try:
        yield TestClient(app)
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
        models_route.set_models_context(saved_pool)


def _make_pool_entry(model_id, *, model_type, loaded=True):
    entry = MagicMock()
    entry.model_id = model_id
    entry.engine = MagicMock() if loaded else None
    entry.model_type = model_type
    return entry


def _make_pool(entries):
    pool = MagicMock()
    pool.list_models.return_value = [e.model_id for e in entries]
    pool.get_entry.side_effect = lambda mid: next(
        (e for e in entries if e.model_id == mid), None
    )
    return pool


def _by_id(body, model_id):
    for entry in body["data"]:
        if entry["id"] == model_id:
            return entry
    raise AssertionError(f"id {model_id!r} not in /v1/models: {body}")


def test_served_stt_model_listed_as_audio():
    with _mounted(model_name=STT_ID, pool=None) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, STT_ID)
    assert entry["object"] == "model"
    assert entry["modality"] == "audio"


def test_served_tts_model_listed_as_audio():
    with _mounted(model_name=TTS_ID, pool=None) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, TTS_ID)
    assert entry["object"] == "model"
    assert entry["modality"] == "audio"


def test_served_llm_model_listed_as_text():
    with _mounted(model_name=LLM_ID, pool=None) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, LLM_ID)
    assert entry["modality"] == "text"


def test_pool_stt_model_surfaces_audio_modality():
    stt = _make_pool_entry(STT_ID, model_type="audio_stt", loaded=True)
    pool = _make_pool([stt])
    with _mounted(model_name=LLM_ID, pool=pool) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, STT_ID)
    assert entry["modality"] == "audio"
    assert entry["loaded"] is True
    assert entry["state"] == "loaded"


def test_pool_tts_model_surfaces_audio_modality():
    tts = _make_pool_entry(TTS_ID, model_type="audio_tts", loaded=False)
    pool = _make_pool([tts])
    with _mounted(model_name=LLM_ID, pool=pool) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, TTS_ID)
    assert entry["modality"] == "audio"
    assert entry["loaded"] is False
    assert entry["state"] == "registered"


def test_audio_models_coexist_with_llm():
    stt = _make_pool_entry(STT_ID, model_type="audio_stt", loaded=True)
    tts = _make_pool_entry(TTS_ID, model_type="audio_tts", loaded=True)
    pool = _make_pool([stt, tts])
    with _mounted(model_name=LLM_ID, pool=pool) as client:
        body = client.get("/v1/models").json()
    ids = {e["id"] for e in body["data"]}
    assert STT_ID in ids
    assert TTS_ID in ids
    assert LLM_ID in ids


def test_models_list_response_top_level_fields():
    with _mounted(model_name=STT_ID, pool=None) as client:
        body = client.get("/v1/models").json()
    assert body.get("object") == "list"
    assert isinstance(body.get("data"), list)
    assert len(body["data"]) >= 1


def test_entry_has_no_owned_by_field():
    with _mounted(model_name=STT_ID, pool=None) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, STT_ID)
    assert "owned_by" not in entry, (
        "owned_by was dropped from the /v1/models contract; the route emits "
        "modality/tool_call_parser/reasoning_parser/loaded/state/capabilities"
    )


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
