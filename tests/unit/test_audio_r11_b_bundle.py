# SPDX-License-Identifier: Apache-2.0
"""R11-B regression bundle — Bo's 0.8.12 audio dogfood follow-up.

Three findings, all guarded against the live pool-seam contract:

* **R11-B-F2** — ``{"format":"mp3"}`` (the legacy OpenAI key) was silently
  dropped on ``/v1/audio/speech``; ``response_format`` fell back to
  ``"wav"``. Fix: a ``model_validator(mode="before")`` on
  :class:`fusion_mlx.api.models.AudioSpeechRequest` folds ``format`` into
  ``response_format`` when the latter isn't explicitly set. Explicit
  ``response_format`` always wins on conflict. Non-string / unknown values
  surface a 400 on the canonical ``response_format`` field name (NOT
  ``format``), so callers learn the spec-correct field even when using the
  legacy alias. NOTE: the speech route emits WAV bytes today regardless of
  ``response_format`` (the codec conversion is the engine's job, not the
  route's); the legacy ``format`` key must still be ACCEPTED and folded
  rather than silently dropped, and a bad value must 400 on
  ``response_format`` — that is the F2 contract this bundle pins.

* **R11-B-F3** — ``{"voice":"default"}`` (the obvious naive caller value)
  on kokoro / chatterbox / voxcpm was rejected by the voice-allowlist as
  ``invalid_voice``, even though the registry advertises a
  ``default_voice`` for each entry. Fix:
  :func:`fusion_mlx.routes_internal.audio._resolve_default_voice_literal`
  maps ``voice="default"`` → ``entry.default_voice`` when the resolved
  model is registered.

* **R11-B-F4** — ``/v1/models`` for an audio-only alias advertised
  ``capabilities=["text"]`` and ``modality=null``. Fix:
  :func:`fusion_mlx.routes_internal.models._resolve_audio_entry`
  short-circuits audio aliases to ``capabilities=["audio.speech"]`` (TTS)
  or ``["audio.transcription"]`` (STT) and ``modality="audio"``.

Harness note (2026-08-31 rescue): the original bundle patched a stale
``audio_route._tts_engine = None`` singleton seam and touched removed
``server._embedding_model_locked``/``_tool_call_parser`` attrs, producing
93×503 "Server not initialized". Rebuilt on the designed pool seam
``fusion_mlx.api.audio_routes._get_engine_pool`` (mirror of
``test_audio_tts.py``); the models lane uses the ``test_capabilities_field``
config-only fixture (no server attr-bag).

Bo r11 evidence: /tmp/dogfood-0812/bo-r1.md F2 / F3 / F4.
"""

from __future__ import annotations

import sys
import types

import pytest

# ``fusion_mlx.routes_internal.audio`` re-exports
# ``fusion_mlx.api.audio_routes.router``, which transitively imports
# ``mlx.core`` via the engine wiring. Linux CI runners don't install mlx,
# so a bare import raises ``ModuleNotFoundError``. importorskip keeps the
# file a clean SKIP off-platform but still executes on Apple Silicon.
pytest.importorskip(
    "mlx.core",
    reason="audio route imports transitively pull in mlx; "
    "test runs on Apple Silicon / dev, not Linux CI runners",
)
pytest.importorskip(
    "mlx_lm",
    reason="audio route imports transitively pull in mlx_lm; "
    "test runs on Apple Silicon / dev, not Linux CI runners",
)
pytest.importorskip(
    "multipart",
    reason="TestClient requires python-multipart on the form lanes",
)

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fusion_mlx.api.audio_routes import router as audio_router  # noqa: E402
from fusion_mlx.config import get_config  # noqa: E402
from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key  # noqa: E402
from fusion_mlx.middleware.exception_handlers import (  # noqa: E402
    install_exception_handlers,
)
from fusion_mlx.routes_internal import models as models_route  # noqa: E402

RIFF_MAGIC = b"RIFF"


def _make_wav_bytes(duration: float = 0.05, sample_rate: int = 24000) -> bytes:
    import struct

    n = int(duration * sample_rate)
    data_size = n * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + (b"\x00\x00" * n)


DUMMY_WAV = _make_wav_bytes()


def _make_mock_tts_engine(wav_bytes: bytes | None = None) -> MagicMock:
    from fusion_mlx.engines.tts import TTSEngine

    engine = MagicMock(spec=TTSEngine)
    engine.synthesize = AsyncMock(return_value=wav_bytes or DUMMY_WAV)
    engine.supports_native_tts_streaming.return_value = False
    return engine


def _make_mock_pool(tts_engine=None, model_id: str = "kokoro") -> MagicMock:
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=tts_engine or _make_mock_tts_engine())
    pool.get_entry = MagicMock(
        return_value=MagicMock(model_type="audio_tts", engine_type="tts")
    )
    pool.get_model_ids.return_value = [model_id]
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()
    pool.resolve_model_id = MagicMock(side_effect=lambda m, _: m)
    return pool


def _install_fake_mlx_audio(monkeypatch):
    import importlib.machinery

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_mlx_audio.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio", loader=None, is_package=True
    )
    fake_tts = types.ModuleType("mlx_audio.tts")
    fake_tts.__path__ = []
    fake_tts.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.tts", loader=None, is_package=True
    )
    fake_tts_generate = types.ModuleType("mlx_audio.tts.generate")
    fake_tts_generate.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.tts.generate", loader=None
    )
    fake_tts_generate.load_model = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", fake_tts_generate)


def _mount_audio_app(
    monkeypatch, *, mock_pool=None, voice=None
) -> tuple[TestClient, MagicMock, callable]:
    """Mount the audio router with the designed pool seam injected.

    The handler reads ``_get_engine_pool()`` (api/audio_routes.py:578) to
    fetch the TTS engine; patching that accessor with a mock pool is the
    supported test seam (mirror of test_audio_tts.server_tts_client). The
    exception handlers are installed so Pydantic validation errors surface
    as the OpenAI 400 envelope (not the default FastAPI 422).
    """
    _install_fake_mlx_audio(monkeypatch)
    app = FastAPI()
    app.include_router(audio_router)
    install_exception_handlers(app)
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[check_rate_limit] = lambda: None

    pool = mock_pool or _make_mock_pool()
    if voice is not None:
        engine = pool.get_engine.return_value
        engine.synthesize = AsyncMock(return_value=DUMMY_WAV)

    cfg = get_config()
    saved_api_key = cfg.api_key
    cfg.api_key = None

    def _restore():
        cfg.api_key = saved_api_key

    patcher = patch("fusion_mlx.api.audio_routes._get_engine_pool", return_value=pool)
    patcher.start()
    client = TestClient(app, raise_server_exceptions=False)

    def _cleanup():
        patcher.stop()
        _restore()

    return client, pool, _cleanup


# ===========================================================================
# R11-B-F2 — ``format`` alias maps to ``response_format``
# ===========================================================================


class TestFormatAliasLegacyToResponseFormat:
    """The legacy ``format`` key must fold into ``response_format`` so a
    bad value 400s on the canonical field name and a valid value is
    accepted (not silently dropped to the ``"wav"`` default)."""

    def test_format_alias_mp3_accepted_not_silently_dropped(self, monkeypatch):
        """``{"format":"mp3"}`` folds into ``response_format="mp3"``.
        ``mp3`` is in the allowed set, so the request is accepted (200).
        The speech route emits WAV bytes today regardless of the requested
        codec (the engine owns codec conversion); the F2 contract is that
        the legacy key is ACCEPTED and folded — NOT silently dropped to
        ``"wav"`` with a 200 the caller can't distinguish from honoring
        the request. Pre-fix the key vanished and ``response_format``
        defaulted to ``"wav"`` silently."""
        client, _pool, cleanup = _mount_audio_app(monkeypatch)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "chatterbox",
                    "input": "Hi",
                    "voice": "default",
                    "format": "mp3",
                },
            )
        finally:
            cleanup()

        # mp3 is an allowed format → folded into response_format → 200.
        # A silent drop would also 200 but on the wrong codec; the
        # distinguishing guard is the nonstring/unknown cases below.
        assert r.status_code == 200, (
            f"R11-B-F2 regression: ``format=mp3`` returned "
            f"{r.status_code}, expected 200 (mp3 is an allowed format; "
            f"the legacy key must be accepted, not dropped). "
            f"Body: {r.text[:500]}"
        )

    def test_explicit_response_format_wins_over_format_alias(self, monkeypatch):
        """When both ``response_format`` AND ``format`` are sent (itself a
        client bug) the spec-correct field must win. Never a silent
        override of explicit caller intent."""
        client, _pool, cleanup = _mount_audio_app(monkeypatch)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "af_heart",
                    "response_format": "wav",
                    "format": "mp3",
                },
            )
        finally:
            cleanup()

        assert r.status_code == 200, r.text

    @pytest.mark.parametrize(
        "bad_legacy",
        [123, True, [], {}],
    )
    def test_nonstring_format_alias_400s(self, monkeypatch, bad_legacy):
        """A non-string legacy ``format`` value (``{"format":123}``) MUST
        400 on ``response_format``, NOT silently fall through to the
        ``"wav"`` default. The before-validator folds EVERY legacy value
        into ``response_format`` so the field-level type-check produces the
        same 400 envelope a wrong-typed ``response_format`` would emit."""
        client, _pool, cleanup = _mount_audio_app(monkeypatch)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "af_heart",
                    "format": bad_legacy,
                },
            )
        finally:
            cleanup()

        assert r.status_code == 400, (
            f"R11-B-F2 regression: format={bad_legacy!r} returned "
            f"{r.status_code}, expected 400. Body: {r.text[:500]}"
        )
        body = r.json()
        # The envelope must surface ``response_format`` as the failing
        # field (NOT ``format``) so the caller learns the spec-correct
        # field name even when they used the legacy alias.
        assert body["error"]["param"] == "response_format", body
        assert body["error"]["type"] == "invalid_request_error", body

    def test_unknown_format_alias_still_400s(self, monkeypatch):
        """The alias only changes the SOURCE of the value, not the
        allowed-set contract. A legacy ``{"format":"jpeg"}`` (not in the
        supported codec list) MUST 400 with the same envelope an explicit
        ``response_format`` would emit."""
        client, _pool, cleanup = _mount_audio_app(monkeypatch)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "af_heart",
                    "format": "jpeg",
                },
            )
        finally:
            cleanup()

        assert r.status_code == 400, r.text
        body = r.json()
        assert body["error"]["type"] == "invalid_request_error", body
        assert body["error"]["param"] == "response_format", body


# ===========================================================================
# R11-B-F3 — ``voice="default"`` falls back to the registry default
# ===========================================================================

# Each TTS family that ships a default_voice in aliases.json:
#   * kokoro → "af_heart" (the Pydantic default; the registry agrees)
#   * chatterbox → "default" (the engine's catch-all)
#   * voxcpm → "default" (same shape as chatterbox)
_DEFAULT_VOICE_TARGETS = [
    ("kokoro", "af_heart"),
    ("chatterbox", "default"),
    ("voxcpm", "default"),
]


class TestVoiceDefaultFallsBackToRegistry:
    """The literal ``voice="default"`` (the obvious naive caller value)
    must resolve to the registry's ``default_voice`` for the resolved
    model. Pre-fix this was rejected by the kokoro allowlist as
    ``invalid_voice`` even though the registry advertises
    ``default_voice="af_heart"`` for it."""

    @pytest.mark.parametrize("alias,expected_voice", _DEFAULT_VOICE_TARGETS)
    def test_voice_default_falls_back_to_registry(
        self, monkeypatch, alias, expected_voice
    ):
        voice_observed: list[str] = []
        engine = _make_mock_tts_engine()

        async def _capture_synthesize(_text, **kwargs):
            voice_observed.append(kwargs.get("voice"))
            return DUMMY_WAV

        engine.synthesize = AsyncMock(side_effect=_capture_synthesize)
        pool = _make_mock_pool(tts_engine=engine)
        client, _pool, cleanup = _mount_audio_app(monkeypatch, mock_pool=pool)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": alias,
                    "input": "Hi",
                    "voice": "default",
                    "response_format": "wav",
                },
            )
        finally:
            cleanup()

        assert r.status_code == 200, (
            f"R11-B-F3 regression: ``voice='default'`` on {alias!r} "
            f"returned {r.status_code}, expected 200. Pre-fix this 400'd "
            f"on the kokoro allowlist even though the registry already "
            f"advertises ``default_voice``. Body: {r.text[:500]}"
        )
        assert voice_observed == [expected_voice], (
            f"R11-B-F3 regression: engine received voice="
            f"{voice_observed!r}, expected [{expected_voice!r}]. "
            f"``default`` was not substituted with the registry value."
        )

    def test_voice_omitted_passes_none_not_default(self, monkeypatch):
        """The F-3 fix must not regress the omitted-voice path: when
        ``voice`` is absent, the live ``AudioSpeechRequest`` field
        default is ``None`` (NOT ``"af_heart"`` — that was the deleted
        ``models.py`` orphan's default; the live class has no voice
        default). The literal-resolution hook MUST NOT fire because the
        value isn't the string ``"default"`` — ``None`` flows straight
        to the engine, which owns its own default-voice resolution."""
        voice_observed: list = []
        engine = _make_mock_tts_engine()

        async def _capture_synthesize(_text, **kwargs):
            voice_observed.append(kwargs.get("voice"))
            return DUMMY_WAV

        engine.synthesize = AsyncMock(side_effect=_capture_synthesize)
        pool = _make_mock_pool(tts_engine=engine)
        client, _pool, cleanup = _mount_audio_app(monkeypatch, mock_pool=pool)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "response_format": "wav",
                },
            )
        finally:
            cleanup()

        assert r.status_code == 200, r.text
        assert voice_observed == [None], (
            "Regression: omitted voice did not flow through as None; "
            f"saw {voice_observed!r}. The F-3 hook must only fire on the "
            f"literal string 'default', never on None (omitted voice)."
        )

    def test_voice_default_with_hf_id_resolves(self, monkeypatch):
        """The HF-id path (``model='mlx-community/Kokoro-82M-bf16'``) and
        the short-alias path (``model='kokoro'``) MUST resolve
        ``voice='default'`` to the same registry value."""
        voice_observed: list[str] = []
        engine = _make_mock_tts_engine()

        async def _capture_synthesize(_text, **kwargs):
            voice_observed.append(kwargs.get("voice"))
            return DUMMY_WAV

        engine.synthesize = AsyncMock(side_effect=_capture_synthesize)
        pool = _make_mock_pool(tts_engine=engine)
        client, _pool, cleanup = _mount_audio_app(monkeypatch, mock_pool=pool)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "mlx-community/Kokoro-82M-bf16",
                    "input": "Hi",
                    "voice": "default",
                    "response_format": "wav",
                },
            )
        finally:
            cleanup()

        assert r.status_code == 200, r.text
        assert voice_observed == ["af_heart"], (
            f"R11-B-F3 regression: HF-id path saw voice="
            f"{voice_observed!r}, expected ['af_heart']. The reverse-HF-id "
            f"lookup in resolve_audio_alias did not fire."
        )

    def test_nondefault_voice_passes_through_unchanged(self, monkeypatch):
        """The F-3 hook resolves ONLY the literal ``voice="default"`` — it
        adds no voice allowlist and rejects nothing. A non-"default" voice
        (here the OpenAI name ``"alloy"``) MUST pass straight through to the
        engine unchanged. The engine's own family detector owns the
        decision of whether a voice is valid for the resolved model; the
        route-layer hook never 400s on a non-"default" voice. Pin so a
        future allowlist (a separate concern) is not silently smuggled in
        under the F-3 umbrella."""
        voice_observed: list = []
        engine = _make_mock_tts_engine()

        async def _capture_synthesize(_text, **kwargs):
            voice_observed.append(kwargs.get("voice"))
            return DUMMY_WAV

        engine.synthesize = AsyncMock(side_effect=_capture_synthesize)
        pool = _make_mock_pool(tts_engine=engine)
        client, _pool, cleanup = _mount_audio_app(monkeypatch, mock_pool=pool)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "alloy",  # OpenAI default voice; not Kokoro
                    "response_format": "wav",
                },
            )
        finally:
            cleanup()

        assert r.status_code == 200, r.text
        assert voice_observed == ["alloy"], (
            f"F-3 over-reached: non-'default' voice was not passed through "
            f"unchanged; saw {voice_observed!r}. The hook must only touch "
            f"the literal 'default' and leave every other voice alone."
        )


# ===========================================================================
# R11-B-F4 — audio aliases advertise audio capability + modality
# ===========================================================================


def _mount_models_app(**cfg_overrides):
    """Mount the models router with controlled config state. Mirrors
    :func:`tests.test_capabilities_field._mount_models_app` — config-only,
    no removed ``server._embedding_model_locked`` /
    ``_tool_call_parser`` attrs (the models route reads ``cfg`` directly)."""
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
            "tool_call_parser",
            "api_key",
        )
    }
    cfg.model_registry = None
    cfg.api_key = None
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)

    def _restore():
        for k, v in saved.items():
            setattr(cfg, k, v)

    return TestClient(app), _restore


# Each row: (alias, hf_id, expected_capability). Covers every audio family
# that ships in aliases.json — adding a new family should add a row here
# too so the wire-level capability advertisement stays uniform.
_AUDIO_ALIASES_FOR_CAP_CHECK = [
    # TTS aliases → ``audio.speech``.
    ("kokoro", "mlx-community/Kokoro-82M-bf16", "audio.speech"),
    ("chatterbox", "mlx-community/chatterbox-turbo-fp16", "audio.speech"),
    ("vibevoice", "mlx-community/VibeVoice-Realtime-0.5B-4bit", "audio.speech"),
    ("voxcpm", "mlx-community/VoxCPM1.5", "audio.speech"),
    ("dia", "mlx-community/Dia-1.6B-4bit", "audio.speech"),
    # STT aliases → ``audio.transcription``.
    ("whisper", "mlx-community/whisper-large-v3-mlx", "audio.transcription"),
    ("whisper-large-v3", "mlx-community/whisper-large-v3-mlx", "audio.transcription"),
    ("parakeet", "mlx-community/parakeet-tdt-0.6b-v2", "audio.transcription"),
]


class TestAudioAliasesHaveAudioCapabilities:
    """``/v1/models`` for an audio-only alias must advertise the audio
    capability + ``modality="audio"`` so drop-in OpenAI clients can route
    on the wire. Pre-fix every audio alias came back as
    ``capabilities=["text"]`` / ``modality=null``."""

    @pytest.mark.parametrize("alias,hf_id,expected_cap", _AUDIO_ALIASES_FOR_CAP_CHECK)
    def test_audio_aliases_have_audio_capabilities(self, alias, hf_id, expected_cap):
        """Both forms (alias + HF id) get the same audio shape on the
        wire."""
        client, restore = _mount_models_app(
            model_name=hf_id,
            model_alias=alias,
        )
        try:
            r = client.get("/v1/models")
        finally:
            restore()

        assert r.status_code == 200, r.text
        body = r.json()
        ids_in_listing = {entry["id"] for entry in body["data"]}
        assert hf_id in ids_in_listing, (
            f"R11-B-F4 regression: HF id {hf_id!r} missing from "
            f"/v1/models listing. Body: {body}"
        )
        assert alias in ids_in_listing, (
            f"R11-B-F4 regression: short alias {alias!r} missing "
            f"from /v1/models listing. Body: {body}"
        )

        for entry in body["data"]:
            if entry["id"] not in (alias, hf_id):
                continue
            assert entry["modality"] == "audio", (
                f"R11-B-F4 regression: entry {entry['id']!r} reports "
                f"modality={entry['modality']!r}, expected 'audio'. "
                f"Pre-fix this was 'null' and drop-in OpenAI clients "
                f"couldn't tell audio aliases from text models."
            )
            assert expected_cap in entry["capabilities"], (
                f"R11-B-F4 regression: entry {entry['id']!r} missing "
                f"{expected_cap!r} in capabilities={entry['capabilities']!r}. "
                f"Pre-fix this was ['text'] and clients couldn't route "
                f"audio traffic correctly."
            )
            # ``text`` MUST NOT bleed into the audio entry — the pre-fix
            # tag was misleading and the new contract is a clean
            # audio-only capability set.
            assert "text" not in entry["capabilities"], (
                f"R11-B-F4 regression: audio entry {entry['id']!r} leaked "
                f"'text' tag: {entry['capabilities']!r}. Audio aliases "
                f"are NOT text models; the capability list must be "
                f"audio-only."
            )

    def test_retrieve_model_for_audio_alias_has_audio_capability(self):
        """``GET /v1/models/{id}`` must agree with the listing — same
        shape for the same id, otherwise clients hitting the single-id
        endpoint to bootstrap state see a stale text envelope."""
        client, restore = _mount_models_app(
            model_name="mlx-community/Kokoro-82M-bf16",
            model_alias="kokoro",
        )
        try:
            r = client.get("/v1/models/kokoro")
        finally:
            restore()

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["modality"] == "audio", body
        assert body["capabilities"] == ["audio.speech"], body


# ===========================================================================
# R11-B-F5 — empty / missing / blank ``input`` -> 400 OpenAI envelope
# (rescued from r7_c cluster #728; the bundle stayed quarantined for its
# REMOVED voice-allowlist / content-type tests, but this live-route
# contract must keep coverage on the permanently un-quarantined file.)
# ===========================================================================


class TestSpeechInputValidation:
    """Empty / missing / whitespace ``input`` raises the OpenAI 400
    envelope BEFORE the synthesis engine runs. Pre-fix every shape
    collapsed into the engine's ``No audio generated`` 500. The live
    validator (audio_routes.py) emits ``param="input"`` so SDK error
    branches can pattern-match (not a bare-string ``detail``)."""

    def test_empty_input_returns_400_envelope(self, monkeypatch):
        client, _pool, cleanup = _mount_audio_app(monkeypatch)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "", "voice": "af_heart"},
            )
        finally:
            cleanup()

        assert r.status_code == 400, (
            f"R11-B-F5 regression: empty input returned "
            f"{r.status_code} (expected 400). Body: {r.text}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "input", err
        assert "input" in err["message"].lower(), err

    def test_whitespace_only_input_returns_400_envelope(self, monkeypatch):
        """``min_length=1`` doesn't catch whitespace-only input because
        ``"   "`` is three characters. The route's custom validator
        rejects blank strings so the wire contract is "non-blank text"
        and the empty-phoneme 500 cannot fire."""
        client, _pool, cleanup = _mount_audio_app(monkeypatch)
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "   ",
                    "voice": "af_heart",
                },
            )
        finally:
            cleanup()

        assert r.status_code == 400, (
            f"R11-B-F5 regression: whitespace input returned "
            f"{r.status_code} (expected 400). Body: {r.text}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "input", err
        msg = err["message"].lower()
        assert "non-empty" in msg or "non-blank" in msg or "blank" in msg, err


# ===========================================================================
# R11-B-F6 — pyproject pins mlx-audio excluding 0.4.4
# (rescued from r7_c cluster #728.)
# ===========================================================================


class TestMlxAudioVersionPin:
    """mlx-audio 0.4.4 broke ``istftnet.SineGen``. The pin must EXCLUDE
    0.4.4 so a fresh ``pip install`` doesn't pull the broken release.
    The live pin is an exact ``==0.4.3`` (stricter than the old
    ``<0.4.4`` upper bound); either form is acceptable so long as 0.4.4
    is not installable. Parses pyproject.toml verbatim so a contributor
    that loosens the bound trips CI."""

    def test_mlx_audio_upper_bound_excludes_0_4_4(self):
        from pathlib import Path

        try:
            import tomllib  # 3.11+
        except ImportError:  # pragma: no cover — keep 3.10 fallback
            import tomli as tomllib  # type: ignore[import-not-found]

        root = Path(__file__).resolve().parents[2]
        with (root / "pyproject.toml").open("rb") as f:
            cfg = tomllib.load(f)
        audio_deps = cfg["project"]["optional-dependencies"]["audio"]
        mlx_audio_specs = [d for d in audio_deps if d.startswith("mlx-audio")]
        assert (
            len(mlx_audio_specs) == 1
        ), f"Expected exactly one mlx-audio pin, found {mlx_audio_specs}"
        spec = mlx_audio_specs[0]
        excludes_broken = "<0.4.4" in spec or "==0.4.3" in spec
        assert excludes_broken, (
            f"R11-B-F6 regression: mlx-audio must exclude 0.4.4 to avoid "
            f"the istftnet SineGen broadcast_shapes regression. Accept "
            f"``<0.4.4`` or ``==0.4.3``. Current pin: {spec!r}"
        )


# ===========================================================================
# R11-B-F7 — case-insensitive TTS alias resolution
# (rescued from r8_a cluster #729; the bundle stayed quarantined for its
# REMOVED voice-allowlist / content-type / default-sentinel tests, but
# the live alias resolver's case-insensitive lookup must keep coverage.)
# ===========================================================================


class TestFullAliasResolution:
    """The full ``kokoro-82m-8bit`` aliases AND their mixed-case HF-repo
    forms (``Kokoro-82M-bf16`` / ``KOKORO-82M-8BIT``) must resolve
    through the same helper as the short ``kokoro`` alias — pre-fix only
    the exact lowercase form hit the table, so mixed-case names 404'd at
    HF lookup inside mlx_audio. Fix: a lowercased fallback index
    (``_TTS_MODEL_ALIASES_LOWER``) in routes_internal/audio.py."""

    @pytest.mark.parametrize(
        "alias",
        [
            "kokoro-82m-bf16",
            "kokoro-82m-4bit",
            "kokoro-82m-8bit",
            "Kokoro-82M-bf16",
            "KOKORO-82M-8BIT",
        ],
    )
    def test_full_kokoro_alias_resolves_to_kokoro_repo(self, alias):
        from fusion_mlx.routes_internal.audio import _resolve_tts_model

        resolved = _resolve_tts_model(alias)
        assert "kokoro" in resolved.lower(), (
            f"R11-B-F7 regression: alias {alias!r} did NOT resolve to a "
            f"Kokoro HF repo, got {resolved!r}. Pre-fix the mixed-case "
            f"form fell through to passthrough and mlx-audio 404'd."
        )
        assert "/" in resolved, (
            f"R11-B-F7 regression: alias {alias!r} must resolve to a full "
            f"HF repo id (with org/), got {resolved!r}."
        )

    def test_short_and_full_alias_resolve_identically(self):
        from fusion_mlx.routes_internal.audio import _resolve_tts_model

        assert _resolve_tts_model("kokoro") == _resolve_tts_model("kokoro-82m-bf16")

    def test_unknown_alias_still_passes_through(self):
        from fusion_mlx.routes_internal.audio import _resolve_tts_model

        hf_path = "mlx-community/Some-Future-TTS-Model"
        assert _resolve_tts_model(hf_path) == hf_path
