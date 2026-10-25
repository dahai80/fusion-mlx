# SPDX-License-Identifier: Apache-2.0
# Tests for server-side reference-voice bundles (#1383): a directory under
# the voice library holding ref.wav + ref.txt that /v1/audio/speech resolves
# from the `voice` field for cloning-capable models.

from __future__ import annotations

import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api.audio_routes import router as audio_router
from fusion_mlx.audio import voice_bundles as vb
from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key


def _wav_bytes(seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return buf.getvalue()


REF_WAV = _wav_bytes()
REF_TEXT = "the quick brown fox jumps over the lazy dog"


@pytest.fixture()
def voice_library(tmp_path, monkeypatch):
    library = tmp_path / "voices"
    library.mkdir()
    from fusion_mlx.training import voice as voice_mod

    monkeypatch.setattr(voice_mod, "VOICE_BASE_DIR", library)
    return library


def _write_bundle(library, name, wav=REF_WAV, text=REF_TEXT):
    directory = library / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ref.wav").write_bytes(wav)
    (directory / "ref.txt").write_text(text, encoding="utf-8")
    return directory


def _engine(cloning: bool) -> MagicMock:
    from fusion_mlx.engines.tts import TTSEngine

    engine = MagicMock(spec=TTSEngine)
    engine.synthesize = AsyncMock(return_value=_wav_bytes(0.1))
    engine.supports_voice_cloning.return_value = cloning
    engine.supports_native_tts_streaming.return_value = False
    return engine


def _client(engine: MagicMock):
    app = FastAPI()
    app.include_router(audio_router)
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[check_rate_limit] = lambda: None
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=engine)
    patcher = patch("fusion_mlx.api.audio_routes._get_engine_pool", return_value=pool)
    patcher.start()
    return TestClient(app), patcher


class TestResolveVoiceBundle:
    def test_resolves_audio_and_transcript(self, voice_library):
        directory = _write_bundle(voice_library, "narrator", text="  hello there  ")
        bundle = vb.resolve_voice_bundle("narrator")
        assert bundle is not None
        assert bundle.audio_path == directory / "ref.wav"
        assert bundle.ref_text == "hello there"

    def test_rejects_path_traversal_and_hidden_names(self, voice_library):
        _write_bundle(voice_library, "ok")
        for bad in ("", "../etc", "a/b", ".hidden", "x" * 65, None):
            assert vb.resolve_voice_bundle(bad) is None

    def test_missing_transcript_is_skipped(self, voice_library):
        directory = voice_library / "no-text"
        directory.mkdir()
        (directory / "ref.wav").write_bytes(REF_WAV)
        assert vb.resolve_voice_bundle("no-text") is None

    def test_short_clip_is_skipped(self, voice_library):
        _write_bundle(voice_library, "tiny", wav=b"RIFF" + b"\x00" * 8)
        assert vb.resolve_voice_bundle("tiny") is None

    def test_fine_tune_pack_is_not_a_bundle(self, voice_library):
        package = voice_library / "trained"
        package.mkdir()
        (package / "voice.safetensors").write_bytes(b"stub")
        (package / "manifest.json").write_text("{}", encoding="utf-8")
        assert vb.resolve_voice_bundle("trained") is None

    def test_list_returns_only_complete_bundles(self, voice_library):
        _write_bundle(voice_library, "alpha")
        _write_bundle(voice_library, "beta")
        (voice_library / "broken").mkdir()
        listed = {entry["name"]: entry for entry in vb.list_voice_bundles()}
        assert set(listed) == {"alpha", "beta"}
        assert listed["alpha"]["ref_audio"] == "ref.wav"
        assert listed["alpha"]["ref_text_chars"] == len(REF_TEXT)


class TestWriteAndDeleteBundle:
    def test_write_then_resolve(self, voice_library):
        bundle = vb.write_voice_bundle("new-voice", REF_WAV, REF_TEXT)
        assert bundle.audio_path.name == "ref.wav"
        assert vb.resolve_voice_bundle("new-voice") is not None

    def test_existing_name_conflicts_without_overwrite(self, voice_library):
        vb.write_voice_bundle("dup", REF_WAV, REF_TEXT)
        with pytest.raises(FileExistsError):
            vb.write_voice_bundle("dup", REF_WAV, REF_TEXT)
        vb.write_voice_bundle("dup", REF_WAV, "new transcript", overwrite=True)
        assert vb.resolve_voice_bundle("dup").ref_text == "new transcript"

    def test_rejects_invalid_input(self, voice_library):
        with pytest.raises(ValueError):
            vb.write_voice_bundle("../escape", REF_WAV, REF_TEXT)
        with pytest.raises(ValueError):
            vb.write_voice_bundle("name", REF_WAV, "   ")
        with pytest.raises(ValueError):
            vb.write_voice_bundle("name", b"\x00" * 4096, REF_TEXT)
        with pytest.raises(ValueError):
            vb.write_voice_bundle("name", b"RIFF\x00\x00WAVE", REF_TEXT)

    def test_delete_removes_bundle_only(self, voice_library):
        vb.write_voice_bundle("gone", REF_WAV, REF_TEXT)
        pack = voice_library / "pack"
        pack.mkdir()
        (pack / "voice.safetensors").write_bytes(b"stub")
        assert vb.delete_voice_bundle("gone") is True
        assert not (voice_library / "gone").exists()
        assert vb.delete_voice_bundle("pack") is False
        assert (pack / "voice.safetensors").exists()
        assert vb.delete_voice_bundle("missing") is False


class TestSpeechUsesBundle:
    def test_voice_name_resolves_to_reference_pair(self, voice_library):
        directory = _write_bundle(voice_library, "narrator")
        engine = _engine(cloning=True)
        client, patcher = _client(engine)
        try:
            resp = client.post(
                "/v1/audio/speech",
                json={"model": "qwen3-tts-base", "input": "hi", "voice": "narrator"},
            )
        finally:
            patcher.stop()
        assert resp.status_code == 200, resp.text
        kwargs = engine.synthesize.await_args.kwargs
        assert kwargs["voice"] is None
        assert kwargs["ref_audio"] == str(directory / "ref.wav")
        assert kwargs["ref_text"] == REF_TEXT
        # The user-owned clip must survive the request cleanup.
        assert (directory / "ref.wav").exists()

    def test_explicit_ref_audio_wins(self, voice_library):
        _write_bundle(voice_library, "narrator")
        engine = _engine(cloning=True)
        client, patcher = _client(engine)
        try:
            resp = client.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts-base",
                    "input": "hi",
                    "voice": "narrator",
                    "ref_audio": base64.b64encode(REF_WAV).decode(),
                    "ref_text": "inline transcript",
                },
            )
        finally:
            patcher.stop()
        assert resp.status_code == 200, resp.text
        kwargs = engine.synthesize.await_args.kwargs
        assert kwargs["voice"] == "narrator"
        assert kwargs["ref_text"] == "inline transcript"

    def test_non_cloning_model_keeps_preset_voice(self, voice_library):
        directory = _write_bundle(voice_library, "af_heart")
        engine = _engine(cloning=False)
        client, patcher = _client(engine)
        try:
            resp = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "hi", "voice": "af_heart"},
            )
        finally:
            patcher.stop()
        assert resp.status_code == 200, resp.text
        kwargs = engine.synthesize.await_args.kwargs
        assert kwargs["voice"] == "af_heart"
        assert kwargs["ref_audio"] is None
        assert (directory / "ref.wav").exists()

    def test_unknown_voice_name_is_untouched(self, voice_library):
        engine = _engine(cloning=True)
        client, patcher = _client(engine)
        try:
            resp = client.post(
                "/v1/audio/speech",
                json={"model": "qwen3-tts-base", "input": "hi", "voice": "Vivian"},
            )
        finally:
            patcher.stop()
        assert resp.status_code == 200, resp.text
        kwargs = engine.synthesize.await_args.kwargs
        assert kwargs["voice"] == "Vivian"
        assert kwargs["ref_audio"] is None


class TestVoicesListing:
    def test_listing_includes_bundles(self, voice_library):
        _write_bundle(voice_library, "narrator")
        client, patcher = _client(_engine(cloning=True))
        try:
            resp = client.get("/v1/audio/voices")
        finally:
            patcher.stop()
        assert resp.status_code == 200, resp.text
        bundles = resp.json()["voice_bundles"]
        assert [entry["name"] for entry in bundles] == ["narrator"]
        assert bundles[0]["ref_audio"] == "ref.wav"


class TestEngineCloningCapability:
    def test_signature_with_ref_audio_supports_cloning(self):
        from fusion_mlx.engines.tts import TTSEngine

        class Cloning:
            def generate(self, text, ref_audio=None, ref_text=None):
                return []

        class Presets:
            def generate(self, text, voice=None, **kwargs):
                return []

        engine = TTSEngine("fake")
        engine._model = Cloning()
        assert engine.supports_voice_cloning() is True
        engine._model = Presets()
        assert engine.supports_voice_cloning() is False
