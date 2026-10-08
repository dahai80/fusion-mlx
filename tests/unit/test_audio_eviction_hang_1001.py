# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1001.

Symptom: after ``memory_enforcer`` evicts the TTS model, subsequent
``POST /v1/audio/speech`` requests hung indefinitely — the reload path's
settle barrier (``mx.synchronize`` on the shared single-thread mlx
executor) blocked under GPU contention and stranded
``entry.is_unloading=True`` forever, so every ``get_engine`` waited on
``loading_event`` with no engine log.

Fix has two layers:
1. ``EnginePool._bounded_mlx_sync_clear`` wraps the settle's
   ``mx.synchronize()+clear_cache`` executor calls in ``asyncio.wait_for``
   so a blocked Metal device cannot strand ``is_unloading`` indefinitely.
2. ``create_speech`` bounds ``pool.get_engine`` with a timeout and logs
   entry/exit, so a stuck reload fails fast with a retryable 503 instead
   of a silent client hang.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 22050) -> bytes:
    import io
    import wave

    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


DUMMY_WAV = _make_wav_bytes()


class _HangLoop:
    # Stand-in loop whose run_in_executor returns a Future that never
    # resolves, simulating a blocked mx.synchronize() on the mlx executor.
    # The future is created on the currently-running loop so wait_for can
    # cancel it on timeout.
    def run_in_executor(self, executor, fn, *args):
        return asyncio.get_running_loop().create_future()


class TestBoundedMlxSyncClear:
    def test_times_out_does_not_hang(self, caplog):
        pool = EnginePool()
        hang_loop = _HangLoop()
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.pool.engine_pool"):
            asyncio.run(
                pool._bounded_mlx_sync_clear(hang_loop, label="test-hang", timeout=0.2)
            )
        assert any(
            "timed out" in r.message and "test-hang" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_success_path_no_timeout(self):
        pool = EnginePool()

        async def _run():
            loop = asyncio.get_running_loop()
            # Real loop + mocked mx (no-op synchronize/clear_cache) completes
            # instantly; no TimeoutError, no warning.
            await pool._bounded_mlx_sync_clear(loop, label="ok", timeout=5.0)

        asyncio.run(_run())

    def test_env_timeout_override(self, monkeypatch):
        pool = EnginePool()
        hang_loop = _HangLoop()
        monkeypatch.setenv("FUSION_MLX_SYNC_TIMEOUT", "0.15")
        import time

        start = time.monotonic()
        asyncio.run(pool._bounded_mlx_sync_clear(hang_loop, label="env"))
        elapsed = time.monotonic() - start
        # Env override 0.15s honored (bounded well under the 30s default).
        assert elapsed < 2.0

    def test_invalid_env_falls_back(self, monkeypatch, caplog):
        pool = EnginePool()
        monkeypatch.setenv("FUSION_MLX_SYNC_TIMEOUT", "notanumber")

        async def _run():
            # Real loop + mocked mx completes instantly; env IS consulted
            # because timeout param is None (default), exercising the
            # invalid-value fallback warning.
            loop = asyncio.get_running_loop()
            await pool._bounded_mlx_sync_clear(loop, label="badenv")

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.pool.engine_pool"):
            asyncio.run(_run())
        assert any(
            "Invalid FUSION_MLX_SYNC_TIMEOUT" in r.message for r in caplog.records
        )


class TestSettleCompletesOnSyncTimeout:
    # #1001 core guarantee: even when the sync barrier times out, the settle
    # method RETURNS (so unload_engine_async's finally clears is_unloading).
    def test_settle_returns_when_sync_times_out(self, monkeypatch):
        pool = EnginePool()
        entry = EngineEntry(
            model_id="kokoro",
            model_path="/tmp/kokoro",
            model_type="audio",
            engine_type="audio_tts",
            estimated_size=100 * 1024 * 1024,
        )
        pool._entries["kokoro"] = entry

        async def _run():
            loop = asyncio.get_running_loop()
            # _bounded_mlx_sync_clear simulated as a fast no-op (the timeout
            # path inside it already swallows TimeoutError and proceeds).
            pool._bounded_mlx_sync_clear = AsyncMock()
            # active_memory low enough that round 1 settles (freed >= expected).
            with (
                patch(
                    "fusion_mlx.pool.engine_pool.mx.get_active_memory",
                    return_value=10,
                ),
                patch(
                    "fusion_mlx.pool.engine_pool.get_phys_footprint", return_value=10
                ),
            ):
                await pool._settle_unloaded_engine("kokoro", 10)

        asyncio.run(_run())
        # If we got here the settle returned (no hang / no raise).

    def test_settle_unloads_clears_is_unloading(self):
        # End-to-end: unload_engine_async must clear is_unloading even when
        # the sync barrier would hang. Patch _bounded_mlx_sync_clear to a
        # no-op AsyncMock and _detach_engine to set engine=None; verify the
        # finally clears is_unloading.
        pool = EnginePool()
        entry = EngineEntry(
            model_id="kokoro",
            model_path="/tmp/kokoro",
            model_type="audio",
            engine_type="audio_tts",
            estimated_size=100 * 1024 * 1024,
        )
        entry.engine = MagicMock()
        entry.engine.stop = AsyncMock()
        entry.engine.safe_evict = AsyncMock()
        entry.is_unloading = True
        pool._entries["kokoro"] = entry
        pool._bounded_mlx_sync_clear = AsyncMock()

        async def _run():
            await pool.unload_engine_async("kokoro")

        asyncio.run(_run())
        assert entry.is_unloading is False
        assert entry.engine is None
        assert entry.loading_event is None


class _MakeMockTtsEngine:
    pass


def _make_mock_tts_engine() -> MagicMock:
    from fusion_mlx.engines.tts import TTSEngine

    engine = MagicMock(spec=TTSEngine)
    engine.synthesize = AsyncMock(return_value=DUMMY_WAV)
    engine.supports_native_tts_streaming.return_value = False
    return engine


def _make_hanging_pool(delay_s: float = 5.0, observed: list | None = None):
    if observed is None:
        observed = []

    async def _get_engine(model_id):
        observed.append(model_id)
        await asyncio.sleep(delay_s)
        return _make_mock_tts_engine()

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


class TestCreateSpeechLoadTimeout:
    # #1001 route layer: a stuck get_engine must surface as 503 retryable,
    # not a silent hang.
    def test_hanging_get_engine_returns_503(self, monkeypatch):
        from fusion_mlx.api.audio_routes import router

        monkeypatch.setenv("FUSION_AUDIO_LOAD_TIMEOUT", "0.4")
        pool, observed = _make_hanging_pool(delay_s=5.0)

        app = FastAPI()
        app.include_router(router)
        with (
            patch("fusion_mlx.api.audio_routes._get_engine_pool", return_value=pool),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello world",
                    "voice": "af_heart",
                    "response_format": "wav",
                },
            )
        assert response.status_code == 503, response.text
        body = response.json()
        err = body.get("detail", {}).get("error", body.get("detail", {}))
        assert err.get("code") == "engine_load_timeout", err
        # The hanging get_engine WAS invoked (proving the route reached the
        # reload path and timed out, rather than failing earlier).
        assert observed == ["mlx-community/Kokoro-82M-bf16"], observed

    def test_fast_get_engine_succeeds(self, monkeypatch):
        from fusion_mlx.api.audio_routes import router

        monkeypatch.setenv("FUSION_AUDIO_LOAD_TIMEOUT", "30")
        pool, observed = _make_hanging_pool(delay_s=0.0)

        app = FastAPI()
        app.include_router(router)
        with (
            patch("fusion_mlx.api.audio_routes._get_engine_pool", return_value=pool),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
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
        assert observed == ["mlx-community/Kokoro-82M-bf16"], observed


class TestAudioLoadTimeoutHelper:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("FUSION_AUDIO_LOAD_TIMEOUT", raising=False)
        from fusion_mlx.api.audio_routes import _audio_load_timeout

        assert _audio_load_timeout() == 120.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_AUDIO_LOAD_TIMEOUT", "45")
        from fusion_mlx.api.audio_routes import _audio_load_timeout

        assert _audio_load_timeout() == 45.0

    def test_invalid_env_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("FUSION_AUDIO_LOAD_TIMEOUT", "garbage")
        from fusion_mlx.api.audio_routes import _audio_load_timeout

        with caplog.at_level(logging.WARNING):
            val = _audio_load_timeout()
        assert val == 120.0
        assert any(
            "Invalid FUSION_AUDIO_LOAD_TIMEOUT" in r.message for r in caplog.records
        )

    def test_nonpositive_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("FUSION_AUDIO_LOAD_TIMEOUT", "0")
        from fusion_mlx.api.audio_routes import _audio_load_timeout

        assert _audio_load_timeout() == 120.0


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
