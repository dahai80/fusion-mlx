# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issues #1060 and #1061.

#1060: scipy wav fallback paths clipped AFTER int16 conversion caused
int16 wraparound for |sample| > 1 (severe clipping artifacts). Fix:
``np.clip(audio, -1.0, 1.0)`` before ``* 32767`` in
``audio/tts.py`` save(), ``audio/processor.py`` save().

#1061: audio-layer ``unload()`` missed ``gc.collect()`` +
``mx.clear_cache()``, leaving Metal cache pinned after unload. Also
STT/STS engine-layer timeouts were hardcoded 60s; now env-configurable
via ``FUSION_STT_TIMEOUT`` / ``FUSION_STS_TIMEOUT``.
"""

import asyncio
import logging
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _read_wav_samples(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2
        n = wf.getnframes()
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype=np.int16)


class TestTtsSaveClipsBeforeInt16:
    # #1060: audio/tts.py TTSEngine.save scipy fallback.
    def test_overrange_clamped_not_wrapped(self, tmp_path):
        from fusion_mlx.audio.tts import AudioOutput, TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        # Samples deliberately outside [-1, 1] — without clip these wrap
        # (2.0 -> -6553, -2.0 -> +6553). With clip they saturate.
        audio = AudioOutput(
            audio=np.array([2.0, -2.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
            sample_rate=24000,
            duration=0.25e-3,
        )
        out = tmp_path / "out.wav"

        # Force the scipy fallback: replace mlx_audio.tts with a spec=[]
        # mock so `from mlx_audio.tts import save_audio` raises ImportError.
        with patch.dict("sys.modules", {"mlx_audio.tts": MagicMock(spec=[])}):
            engine.save(audio, str(out))

        samples = _read_wav_samples(str(out))
        # np.clip(x, -1, 1) * 32767, truncated to int16:
        # 2.0 -> 32767, -2.0 -> -32767, 0.5 -> 16383, -0.5 -> -16383.
        # #1060: WITHOUT the clip, 2.0*32767=65534 wraps to int16 -2.
        assert samples[0] == 32767, samples
        assert samples[1] == -32767, samples
        assert samples[2] == 16383, samples
        assert samples[3] == -16383, samples
        # No wraparound: out-of-range positives stay positive (not wrapped
        # to small negatives), negatives stay negative.
        assert samples[0] > 0 and samples[1] < 0
        assert abs(samples[0]) > 32000 and abs(samples[1]) > 32000


class TestProcessorSaveClipsBeforeInt16:
    # #1060: audio/processor.py AudioProcessor.save scipy fallback.
    def test_overrange_clamped_not_wrapped(self, tmp_path):
        from fusion_mlx.audio.processor import AudioProcessor

        proc = AudioProcessor("mlx-community/sam-audio-large-fp16")
        proc.sample_rate = 44100
        audio = np.array([2.0, -2.0, 0.25, -0.25], dtype=np.float32)
        out = tmp_path / "proc.wav"

        with patch.dict("sys.modules", {"mlx_audio.sts": MagicMock(spec=[])}):
            proc.save(audio, str(out))

        samples = _read_wav_samples(str(out))
        # 2.0 -> 32767, -2.0 -> -32767, 0.25 -> 8191.
        assert samples[0] == 32767, samples
        assert samples[1] == -32767, samples
        assert samples[2] == 8191, samples


class TestAudioUnloadClearsCache:
    # #1061: unload() must gc.collect + mx.clear_cache so Metal cache is
    # actually reclaimed (mirrors engines-layer stop()).
    def _setup_mlx_clear_cache(self):
        # Ensure mx.clear_cache is a spy-able callable. The conftest mocks
        # mlx.core as a MagicMock, so mx.clear_cache already exists; wrap it.
        import mlx.core as mx

        return mx

    def test_tts_unload_clears_cache(self):
        from fusion_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        engine._loaded = True
        engine.model = MagicMock()
        mx = self._setup_mlx_clear_cache()
        with patch.object(mx, "clear_cache") as cc, patch("gc.collect") as gc_cc:
            engine.unload()
        cc.assert_called_once()
        gc_cc.assert_called()
        assert engine._loaded is False
        assert engine.model is None

    def test_stt_unload_clears_cache(self):
        from fusion_mlx.audio.stt import STTEngine

        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        engine._loaded = True
        engine.model = MagicMock()
        mx = self._setup_mlx_clear_cache()
        with patch.object(mx, "clear_cache") as cc, patch("gc.collect") as gc_cc:
            engine.unload()
        cc.assert_called_once()
        gc_cc.assert_called()
        assert engine._loaded is False
        assert engine.model is None

    def test_processor_unload_clears_cache(self):
        from fusion_mlx.audio.processor import AudioProcessor

        proc = AudioProcessor("mlx-community/sam-audio-large-fp16")
        proc._loaded = True
        proc.model = MagicMock()
        proc.processor = MagicMock()
        mx = self._setup_mlx_clear_cache()
        with patch.object(mx, "clear_cache") as cc, patch("gc.collect") as gc_cc:
            proc.unload()
        cc.assert_called_once()
        gc_cc.assert_called()
        assert proc._loaded is False
        assert proc.model is None
        assert proc.processor is None

    def test_tts_unload_safe_without_mx(self):
        # unload must not raise if mx.clear_cache blows up (defensive try/except).
        from fusion_mlx.audio.tts import TTSEngine

        engine = TTSEngine("m")
        engine._loaded = True
        engine.model = MagicMock()
        import mlx.core as mx

        with patch.object(mx, "clear_cache", side_effect=RuntimeError("boom")):
            engine.unload()  # must not raise
        assert engine.model is None


class TestSttEngineTimeoutEnv:
    # #1061: FUSION_STT_TIMEOUT (default 60s) bounds the transcribe executor.
    def test_valid_env_override_honored(self, monkeypatch):
        import time

        from fusion_mlx.engines.stt import STTEngine

        monkeypatch.setenv("FUSION_STT_TIMEOUT", "0.5")
        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        engine._model = MagicMock()

        def _slow_generate(*a, **k):
            time.sleep(5)
            return MagicMock(text="late", language=None, segments=None)

        engine._model.generate = _slow_generate

        async def _run():
            await engine.transcribe("/nonexistent/audio.wav")

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(_run())

    def test_invalid_env_falls_back_with_warning(self, monkeypatch, caplog):
        # Invalid FUSION_STT_TIMEOUT logs a warning and falls back to 60s.
        # With a fast-generating mock the 60s ceiling never fires, so the
        # warning is observable without a slow test.
        from fusion_mlx.engines.stt import STTEngine

        monkeypatch.setenv("FUSION_STT_TIMEOUT", "notanumber")
        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        engine._model = MagicMock()

        def _fast_generate(*a, **k):
            r = MagicMock()
            r.text = "hi"
            r.language = "en"
            r.segments = []
            return r

        engine._model.generate = _fast_generate

        async def _run():
            return await engine.transcribe("/nonexistent/audio.wav")

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.engines.stt"):
            res = asyncio.run(_run())
        assert res["text"] == "hi"
        assert any("Invalid FUSION_STT_TIMEOUT" in r.message for r in caplog.records), [
            r.message for r in caplog.records
        ]


class TestStsEngineTimeoutEnv:
    # #1061: FUSION_STS_TIMEOUT (default 60s) bounds the STS process executor.
    def test_valid_env_override_honored(self, monkeypatch):
        import time

        from fusion_mlx.engines.sts import STSEngine

        monkeypatch.setenv("FUSION_STS_TIMEOUT", "0.5")
        engine = STSEngine("deepfilternet-test", config_model_type="deepfilternet")
        engine._model = MagicMock()

        def _slow_process(*a, **k):
            time.sleep(5)
            return b"x"

        # STS uses _FAMILY_PROCESSORS[family]; patch the processor fn directly.
        from fusion_mlx import engines as engines_mod

        original = engines_mod.sts._FAMILY_PROCESSORS.get("deepfilternet")

        def _proc(model, audio_path, **kwargs):
            return _slow_process()

        engines_mod.sts._FAMILY_PROCESSORS["deepfilternet"] = _proc
        try:

            async def _run():
                await engine.process("/nonexistent/audio.wav")

            with pytest.raises(asyncio.TimeoutError):
                asyncio.run(_run())
        finally:
            if original is not None:
                engines_mod.sts._FAMILY_PROCESSORS["deepfilternet"] = original


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
