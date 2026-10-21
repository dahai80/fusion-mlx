# SPDX-License-Identifier: Apache-2.0
"""R-03 + R-04: request-time alias resolution on embeddings + audio.

Bugs caught in 0.8.5 dogfood (Priya F-1/F-2, Karim F-K-MODEL-DEFAULT-NO-RESOLVE
/ F-K-OPENAI-ALIAS-GAP):

* **R-03** — ``model="default"`` (the OpenAI-spec placeholder LangChain /
  LlamaIndex / openai-python emit when the caller hasn't picked a specific
  id) was rejected on ``/v1/embeddings`` (400) and on
  ``/v1/audio/transcriptions`` / ``/v1/audio/speech`` (404), breaking
  drop-in OpenAI-SDK compatibility on multiple routes.
* **R-04** — a short alias (e.g. ``bge-m3-4bit``) was rejected at
  request-time on ``/v1/embeddings`` even though that exact short alias was
  the CLI flag value. PR #805 / D-EMBED-ALIAS only fixed boot-time
  resolution; the route handler did a literal string match against the
  resolved HF path.

Fix shape (#1297): :func:`_resolve_request_alias_or_default` is the single
source of truth — it runs both sides through ``resolve_model`` so the alias
and the HF path compare equal, and maps the ``"default"`` sentinel to the
caller's locked modality id. The embeddings route now calls it, and the
audio resolver maps the sentinel to its own STT/TTS default instead of
falling through to the served chat model.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

# A short alias that really ships in ``aliases.json`` (modality: embedding).
EMBED_ALIAS = "bge-m3-4bit"
EMBED_HF = "mlx-community/bge-m3-mlx-4bit"


@contextmanager
def _fake_server_module(module_name: str, **attrs):
    """Inject a stub ``fusion_mlx.server`` into ``sys.modules``.

    Route handlers lazily import from ``..server`` (``resolve_model_with_profile``
    / ``get_max_context_window``). A real import drags in the MLX text loader,
    which is unnecessary for these contract tests — the stub keeps them fast
    and CI-safe.
    """
    import sys

    prev = sys.modules.get(module_name)
    fake = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(fake, key, value)
    sys.modules[module_name] = fake
    try:
        yield fake
    finally:
        if prev is not None:
            sys.modules[module_name] = prev
        else:
            sys.modules.pop(module_name, None)


# ──────────────────────────────────────────────────────────────────
# Unit tests — the alias-resolution helper itself
# ──────────────────────────────────────────────────────────────────


class TestResolveRequestAliasOrDefault:

    def test_returns_locked_when_request_is_none(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default(None, "mlx-community/foo")
            == "mlx-community/foo"
        )

    def test_returns_locked_when_request_is_empty_string(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default("", "mlx-community/foo")
            == "mlx-community/foo"
        )

    def test_returns_locked_when_request_is_default_sentinel(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default("default", "mlx-community/foo")
            == "mlx-community/foo"
        )

    def test_short_alias_matches_full_hf_path_via_registry(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        result = _resolve_request_alias_or_default(EMBED_ALIAS, EMBED_HF)
        assert result == EMBED_HF

    def test_full_hf_path_matches_self(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        assert _resolve_request_alias_or_default(EMBED_HF, EMBED_HF) == EMBED_HF

    def test_full_hf_path_matches_short_alias_locked(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        result = _resolve_request_alias_or_default(EMBED_HF, EMBED_ALIAS)
        assert result == EMBED_ALIAS

    def test_unknown_alias_returns_none(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default("non-existent-alias-xyz", EMBED_HF)
            is None
        )

    def test_returns_none_when_locked_is_none(self):
        from fusion_mlx.service.helpers import _resolve_request_alias_or_default

        assert _resolve_request_alias_or_default("default", None) is None
        assert _resolve_request_alias_or_default("foo", None) is None

    def test_aliases_match_handles_resolve_model_exception_safely(self, monkeypatch):
        from fusion_mlx.service.helpers import _aliases_match

        def boom(_name: str) -> str:  # noqa: ARG001
            raise RuntimeError("registry corrupt")

        monkeypatch.setattr("fusion_mlx.model_aliases.resolve_model", boom)
        assert _aliases_match("a", "b") is False
        assert _aliases_match("same", "same") is True


# ──────────────────────────────────────────────────────────────────
# Integration — /v1/embeddings (R-03 default + R-04 short alias)
# ──────────────────────────────────────────────────────────────────


class _FakeEntry:
    def __init__(self, engine_type: str):
        self.engine_type = engine_type


class _FakePool:
    # Mirrors the EnginePool surface the embeddings route touches:
    # list_models/get_entry for the "default" lookup, get_engine for loads.
    def __init__(self, entries: dict[str, str], engines: dict[str, object]):
        self._entries = entries
        self._engines = engines
        self.requested: list[str] = []

    def list_models(self) -> list[str]:
        return list(self._entries)

    def get_entry(self, model_id: str):
        engine_type = self._entries.get(model_id)
        return _FakeEntry(engine_type) if engine_type else None

    async def get_engine(self, model_id: str):
        self.requested.append(model_id)
        return self._engines.get(model_id)


def _stub_embedding_engine(model_name: str):
    from fusion_mlx.engines.embedding import EmbeddingEngine, EmbeddingOutput

    class _Stub(EmbeddingEngine):
        def __init__(self):
            self._model_name = model_name

        async def embed(self, texts, max_length=512, padding=True, truncation=True):
            return EmbeddingOutput(
                embeddings=[[0.1, 0.2, 0.3] for _ in texts],
                total_tokens=len(texts),
                dimensions=3,
            )

    return _Stub()


@contextmanager
def _embeddings_client(monkeypatch, pool, cfg_overrides: dict | None = None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fusion_mlx.api import embeddings_routes as emb
    from fusion_mlx.config import get_config
    from fusion_mlx.middleware.exception_handlers import install_exception_handlers

    cfg = get_config()
    saved = {
        "embedding_model_locked": cfg.embedding_model_locked,
        "embedding_model": cfg.embedding_model,
        "api_key": cfg.api_key,
    }
    cfg.embedding_model_locked = None
    cfg.embedding_model = None
    cfg.api_key = None
    for key, value in (cfg_overrides or {}).items():
        setattr(cfg, key, value)

    app = FastAPI()
    app.include_router(emb.router)
    install_exception_handlers(app)

    prev_pool, prev_state = emb._pool, emb._server_state
    emb.set_embeddings_context(pool, None)

    def _noop_context_window(model_id, *_a, **_kw):
        return 512

    try:

        def _alias_resolver(mid):
            # Prod-shaped for the dynamic alias registry: the chat resolver
            # turns the embedding alias into its HF path.
            return (EMBED_HF if mid == EMBED_ALIAS else mid, {})

        with _fake_server_module(
            "fusion_mlx.server",
            resolve_model_with_profile=_alias_resolver,
            get_max_context_window=_noop_context_window,
        ):
            yield TestClient(app)
    finally:
        emb.set_embeddings_context(prev_pool, prev_state)
        for key, value in saved.items():
            setattr(cfg, key, value)


class TestEmbeddingsRouteAliasResolution:

    CHAT_HF = "mlx-community/Qwen3.5-4B-MLX-4bit"

    def _pool(self):
        return _FakePool(
            entries={
                EMBED_HF: "embedding",
                self.CHAT_HF: "llm",
            },
            engines={
                EMBED_HF: _stub_embedding_engine(EMBED_HF),
                self.CHAT_HF: MagicMock(),
            },
        )

    def test_default_sentinel_reaches_embedding_engine(self, monkeypatch):
        """R-03: ``model="default"`` must hit the pool's embedding entry, not
        the served chat model (pre-#1297 it resolved to the LLM and 400'd)."""
        pool = self._pool()
        with _embeddings_client(monkeypatch, pool) as client:
            resp = client.post(
                "/v1/embeddings", json={"model": "default", "input": "hello"}
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert set(pool.requested) == {EMBED_HF}

    def test_short_alias_accepts(self, monkeypatch):
        """R-04: the short alias must reach the same engine as its HF path."""
        pool = self._pool()
        with _embeddings_client(monkeypatch, pool) as client:
            resp = client.post(
                "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "hello"}
            )
        assert resp.status_code == 200, resp.text
        assert set(pool.requested) == {EMBED_HF}

    def test_full_hf_path_accepts(self, monkeypatch):
        pool = self._pool()
        with _embeddings_client(monkeypatch, pool) as client:
            resp = client.post(
                "/v1/embeddings", json={"model": EMBED_HF, "input": "hello"}
            )
        assert resp.status_code == 200, resp.text
        assert set(pool.requested) == {EMBED_HF}

    def test_unknown_alias_404_with_param_field(self, monkeypatch):
        pool = self._pool()
        with _embeddings_client(monkeypatch, pool) as client:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "non-existent-alias", "input": "hello"},
            )
        assert resp.status_code == 404, resp.text
        err = resp.json()["error"]
        assert err["param"] == "model"
        assert err["code"] == "model_not_found"

    def test_configured_embedding_model_wins_over_pool_scan(self, monkeypatch):
        """An explicit configured embed id is the "default" target even when
        the pool holds several embedding entries."""
        cfg_embed = "mlx-community/bge-m3-mlx-8bit"
        pool = _FakePool(
            entries={EMBED_HF: "embedding", cfg_embed: "embedding"},
            engines={cfg_embed: _stub_embedding_engine(cfg_embed)},
        )
        with _embeddings_client(
            monkeypatch, pool, {"embedding_model_locked": cfg_embed}
        ) as client:
            resp = client.post(
                "/v1/embeddings", json={"model": "default", "input": "hello"}
            )
        assert resp.status_code == 200, resp.text
        assert set(pool.requested) == {cfg_embed}

    def test_ambiguous_pool_does_not_pick_a_random_embedding_model(self, monkeypatch):
        """Two embedding entries + no configured id → the sentinel must not
        fall through to the chat-aligned resolver (that would land on the
        served LLM and 400 via the type check); it names the ambiguity."""
        pool = _FakePool(
            entries={
                EMBED_HF: "embedding",
                "mlx-community/bge-m3-mlx-8bit": "embedding",
            },
            engines={},
        )
        with _embeddings_client(monkeypatch, pool) as client:
            resp = client.post(
                "/v1/embeddings", json={"model": "default", "input": "hello"}
            )
        assert resp.status_code == 400, resp.text
        err = resp.json()["error"]
        assert err["code"] == "embedding_model_ambiguous"
        assert err["param"] == "model"
        assert EMBED_HF in err["message"]
        assert pool.requested == []

    def test_named_embedding_model_still_resolves_with_two_entries(self, monkeypatch):
        pool = _FakePool(
            entries={
                EMBED_HF: "embedding",
                "mlx-community/bge-m3-mlx-8bit": "embedding",
            },
            engines={EMBED_HF: _stub_embedding_engine(EMBED_HF)},
        )
        with _embeddings_client(monkeypatch, pool) as client:
            resp = client.post(
                "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "hello"}
            )
        assert resp.status_code == 200, resp.text
        assert set(pool.requested) == {EMBED_HF}


# ──────────────────────────────────────────────────────────────────
# Integration — /v1/audio/* (R-03 default on STT + TTS)
# ──────────────────────────────────────────────────────────────────


class TestAudioRouteWiring:
    async def test_speech_route_passes_the_tts_default(self, monkeypatch):
        """Pins the call site, not just the resolver: reverting
        create_speech's second argument would drop the sentinel back onto
        the chat model."""
        from fastapi import HTTPException

        from fusion_mlx.api import audio_routes
        from fusion_mlx.api.audio_models import AudioSpeechRequest

        calls: list[tuple[str, str | None]] = []

        def _spy(model_id, default_model=None):
            calls.append((model_id, default_model))
            raise HTTPException(status_code=418, detail="stop-after-resolve")

        monkeypatch.setattr(audio_routes, "_resolve_model", _spy)
        req = AudioSpeechRequest(model="default", input="hello", voice="af_heart")
        try:
            await audio_routes.create_speech(req)
        except HTTPException as exc:
            assert exc.status_code == 418
        assert calls == [("default", audio_routes._DEFAULT_TTS_MODEL)]


class TestAudioModelDefaultSentinel:
    """The live audio resolver must map ``"default"`` to the modality's own
    default instead of the chat default (#1297). Same contract the
    registry-only ``_resolve_stt_model`` / ``_resolve_tts_model`` helpers
    pin below, but on the resolver the routes actually call.
    """

    def test_openai_tts_ids_resolve(self):
        """OpenAI SDKs send tts-1 / tts-1-hd explicitly; both must land on the
        Kokoro default instead of 404ing in get_engine."""
        from fusion_mlx.api.audio_routes import _DEFAULT_TTS_MODEL, _resolve_model

        assert _resolve_model("tts-1", _DEFAULT_TTS_MODEL) == _DEFAULT_TTS_MODEL
        assert _resolve_model("tts-1-hd", _DEFAULT_TTS_MODEL) == _DEFAULT_TTS_MODEL

    def test_default_maps_to_stt_default(self):
        from fusion_mlx.api.audio_routes import _DEFAULT_STT_MODEL, _resolve_model

        assert _resolve_model("default", _DEFAULT_STT_MODEL) == _DEFAULT_STT_MODEL
        assert _resolve_model("", _DEFAULT_STT_MODEL) == _DEFAULT_STT_MODEL

    def test_default_maps_to_tts_default(self):
        from fusion_mlx.api.audio_routes import _DEFAULT_TTS_MODEL, _resolve_model

        assert _resolve_model("default", _DEFAULT_TTS_MODEL) == _DEFAULT_TTS_MODEL

    def test_default_without_modality_default_400(self):
        from fastapi import HTTPException

        from fusion_mlx.api.audio_routes import _resolve_model

        with pytest.raises(HTTPException) as exc:
            _resolve_model("default")
        assert exc.value.status_code == 400
        assert exc.value.detail["error"]["param"] == "model"

    def test_explicit_hf_id_unchanged(self):
        from fusion_mlx.api.audio_routes import _resolve_model

        assert (
            _resolve_model("mlx-community/whisper-medium-mlx")
            == "mlx-community/whisper-medium-mlx"
        )

    def test_alias_still_resolves_through_audio_registry(self):
        from fusion_mlx.api.audio_routes import _resolve_model

        assert _resolve_model("kokoro") == "mlx-community/Kokoro-82M-bf16"


class TestAudioRouteAliasResolution:
    """The audio routes have their own alias registry
    (``STT_MODEL_ALIASES`` / TTS ``model_map``) because the engines
    don't share the chat ``aliases.json``.
    """

    def test_stt_resolver_maps_default_to_whisper(self):
        from fusion_mlx.routes_internal.audio import (
            DEFAULT_STT_ALIAS,
            STT_MODEL_ALIASES,
            _resolve_stt_model,
        )

        assert _resolve_stt_model("default") == STT_MODEL_ALIASES[DEFAULT_STT_ALIAS]
        assert _resolve_stt_model("default") == _resolve_stt_model(DEFAULT_STT_ALIAS)

    def test_stt_resolver_still_rejects_bogus_alias(self):
        from fastapi import HTTPException

        from fusion_mlx.routes_internal.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc:
            _resolve_stt_model("non-existent-stt-alias")
        assert exc.value.status_code == 404
        detail = exc.value.detail
        assert isinstance(detail, dict)
        assert detail["error"]["code"] == "model_not_found"
        assert detail["error"]["param"] == "model"

    def test_stt_resolver_passes_through_known_alias(self):
        from fusion_mlx.routes_internal.audio import (
            STT_MODEL_ALIASES,
            _resolve_stt_model,
        )

        for alias, hf in STT_MODEL_ALIASES.items():
            assert _resolve_stt_model(alias) == hf

    def test_stt_resolver_passes_through_hf_id(self):
        from fusion_mlx.routes_internal.audio import _resolve_stt_model

        assert (
            _resolve_stt_model("mlx-community/whisper-medium-mlx")
            == "mlx-community/whisper-medium-mlx"
        )

    def test_stt_resolver_empty_string_400(self):
        from fastapi import HTTPException

        from fusion_mlx.routes_internal.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc:
            _resolve_stt_model("")
        assert exc.value.status_code == 400


# ──────────────────────────────────────────────────────────────────
# Cross-check — /v1/chat/completions
# ──────────────────────────────────────────────────────────────────


class TestChatRouteDefaultNotRegressed:
    """Cross-check: chat already resolves through ``resolve_model_with_profile``
    — the shape embeddings/audio now align with. Pin that a plain HF id passes
    through untouched so a future refactor of the shared helper can't change
    the chat surface.
    """

    def test_chat_resolver_passes_plain_id_and_strips_known_profile(self):
        from fusion_mlx.server import resolve_model_with_profile

        assert resolve_model_with_profile(EMBED_HF) == (EMBED_HF, {})
