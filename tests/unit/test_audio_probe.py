# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.audio.probe — mlx-audio availability probing.

Pure-logic surface: _Verdict, cache, lane status, _probe_lane (mlx_audio
missing/present, unknown lane, submodule import fail), is_audio_model_alias,
require_audio_or_exit, require_kokoro_runtime, _raise_503.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.audio import probe as mod


class TestVerdict:
    def test_frozen_dataclass(self):
        v = mod._Verdict(ok=True)
        assert v.ok is True
        assert v.reason is None
        with pytest.raises(Exception):
            v.ok = False  # frozen


class TestResetProbeCache:
    def test_clears_caches(self, monkeypatch):
        monkeypatch.setattr(mod, "_cached_verdict", {"x": 1})
        monkeypatch.setattr(mod, "_LANE_STATUS", {"x": "ok"})
        mod._reset_probe_cache()
        assert mod._cached_verdict == {}
        assert mod._LANE_STATUS == {}


class TestAudioLaneStatus:
    def test_unknown_lane(self, monkeypatch):
        monkeypatch.setattr(mod, "_LANE_STATUS", {})
        monkeypatch.setattr(mod, "_LANE_REASON", {})
        result = mod.audio_lane_status("tts")
        assert result == {"status": "unknown", "reason": None}

    def test_known_lane(self, monkeypatch):
        monkeypatch.setattr(mod, "_LANE_STATUS", {"tts": "ok"})
        monkeypatch.setattr(mod, "_LANE_REASON", {})
        assert mod.audio_lane_status("tts") == {"status": "ok", "reason": None}

    def test_degraded_with_reason(self, monkeypatch):
        monkeypatch.setattr(mod, "_LANE_STATUS", {"stt": "degraded"})
        monkeypatch.setattr(mod, "_LANE_REASON", {"stt": "whisper missing"})
        result = mod.audio_lane_status("stt")
        assert result == {"status": "degraded", "reason": "whisper missing"}


class TestRecordLaneStatus:
    def test_records_status_no_reason(self, monkeypatch):
        monkeypatch.setattr(mod, "_LANE_STATUS", {})
        monkeypatch.setattr(mod, "_LANE_REASON", {"tts": "old"})
        mod._record_lane_status("tts", "ok", None)
        assert mod._LANE_STATUS["tts"] == "ok"
        assert "tts" not in mod._LANE_REASON

    def test_records_status_with_reason(self, monkeypatch):
        monkeypatch.setattr(mod, "_LANE_STATUS", {})
        monkeypatch.setattr(mod, "_LANE_REASON", {})
        mod._record_lane_status("tts", "degraded", "whisper fail")
        assert mod._LANE_STATUS["tts"] == "degraded"
        assert mod._LANE_REASON["tts"] == "whisper fail"


class TestProbeLane:
    def test_mlx_audio_missing_returns_verdict(self, monkeypatch):
        mod._reset_probe_cache()
        with patch("importlib.util.find_spec", return_value=None):
            v = mod._probe_lane("tts")
            assert v.ok is False
            assert "mlx-audio" in v.reason

    def test_unknown_lane_returns_verdict(self, monkeypatch):
        monkeypatch.setattr(mod, "_cached_verdict", {"": mod._Verdict(ok=True)})
        v = mod._probe_lane("unknown_lane")
        assert v.ok is False
        assert "unknown audio lane" in v.reason

    def test_submodule_import_fails(self, monkeypatch):
        monkeypatch.setattr(mod, "_cached_verdict", {"": mod._Verdict(ok=True)})
        with patch("builtins.__import__", side_effect=ImportError("no submodule")):
            v = mod._probe_lane("tts")
            assert v.ok is False
            assert "import failed" in v.reason

    def test_submodule_imports_ok(self, monkeypatch):
        monkeypatch.setattr(mod, "_cached_verdict", {"": mod._Verdict(ok=True)})
        with patch("builtins.__import__", return_value=MagicMock()):
            v = mod._probe_lane("tts")
            assert v.ok is True

    def test_caches_verdict(self, monkeypatch):
        monkeypatch.setattr(mod, "_cached_verdict", {"": mod._Verdict(ok=True)})
        with patch("builtins.__import__", return_value=MagicMock()):
            v1 = mod._probe_lane("tts")
            v2 = mod._probe_lane("tts")
            assert v1 is v2  # cached


class TestMlxAudioAvailable:
    def test_returns_probe_lane(self, monkeypatch):
        monkeypatch.setattr(mod, "_cached_verdict", {"": mod._Verdict(ok=True)})
        with patch("builtins.__import__", return_value=MagicMock()):
            v = mod.mlx_audio_available("tts")
            assert v.ok is True


class TestIsAudioModelAlias:
    def test_empty_returns_false(self):
        assert mod.is_audio_model_alias("") is False
        assert mod.is_audio_model_alias(None) is False

    def test_non_string_returns_false(self):
        assert mod.is_audio_model_alias(123) is False

    def test_alias_token_match(self):
        with patch("fusion_mlx.audio.registry.is_audio_name", return_value=False):
            assert mod.is_audio_model_alias("my-whisper-model") is True
            assert mod.is_audio_model_alias("parakeet-foo") is True
            assert mod.is_audio_model_alias("kokoro-v1") is True

    def test_registry_match(self):
        with patch("fusion_mlx.audio.registry.is_audio_name", return_value=True):
            assert mod.is_audio_model_alias("anything") is True

    def test_no_match(self):
        with patch("fusion_mlx.audio.registry.is_audio_name", return_value=False):
            assert mod.is_audio_model_alias("qwen2-7b") is False


class TestRequireKokoroRuntime:
    def test_misaki_present_returns(self):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            mod.require_kokoro_runtime()  # no raise

    def test_misaki_missing_raises_503(self):
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(Exception, match="503"):
                mod.require_kokoro_runtime()


class TestRequireAudioOrExit:
    def test_mlx_audio_present_returns(self):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            mod.require_audio_or_exit("whisper-1")  # no raise

    def test_mlx_audio_missing_exits_2(self):
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(SystemExit) as exc:
                mod.require_audio_or_exit("whisper-1")
            assert exc.value.code == 2


class TestDeepProbeAudioLane:
    def test_missing_returns_missing_status(self, monkeypatch):
        mod._reset_probe_cache()
        with patch("importlib.util.find_spec", return_value=None):
            result = mod.deep_probe_audio_lane("tts")
            assert result["status"] == "missing"

    def test_unknown_lane(self, monkeypatch):
        monkeypatch.setattr(mod, "_cached_verdict", {"": mod._Verdict(ok=True)})
        result = mod.deep_probe_audio_lane("unknown")
        # unknown lane → _probe_lane returns ok=False → recorded as "missing"
        assert result["status"] == "missing"

    def test_stt_dry_run_ok(self, monkeypatch):
        monkeypatch.setattr(
            mod,
            "_cached_verdict",
            {"": mod._Verdict(ok=True), "stt": mod._Verdict(ok=True)},
        )
        with patch.object(mod, "_dry_run_stt", return_value=(True, None)):
            result = mod.deep_probe_audio_lane("stt")
            assert result["status"] == "ok"

    def test_stt_dry_run_degraded(self, monkeypatch):
        monkeypatch.setattr(
            mod,
            "_cached_verdict",
            {"": mod._Verdict(ok=True), "stt": mod._Verdict(ok=True)},
        )
        with patch.object(mod, "_dry_run_stt", return_value=(False, "whisper fail")):
            result = mod.deep_probe_audio_lane("stt")
            assert result["status"] == "degraded"
            assert result["reason"] == "whisper fail"

    def test_tts_dry_run_ok(self, monkeypatch):
        monkeypatch.setattr(
            mod,
            "_cached_verdict",
            {"": mod._Verdict(ok=True), "tts": mod._Verdict(ok=True)},
        )
        with patch.object(mod, "_dry_run_tts", return_value=(True, None)):
            result = mod.deep_probe_audio_lane("tts")
            assert result["status"] == "ok"


class TestRaise503:
    def test_raises_http_exception(self):
        v = mod._Verdict(ok=False, reason="test reason")
        with pytest.raises(Exception, match="test reason"):
            mod._raise_503(v)


class TestRequireMlxAudio:
    def test_tts_ok_returns(self, monkeypatch):
        monkeypatch.setattr(
            mod,
            "_cached_verdict",
            {"": mod._Verdict(ok=True), "tts": mod._Verdict(ok=True)},
        )
        with patch("builtins.__import__", return_value=MagicMock()):
            mod.require_mlx_audio_tts()  # no raise

    def test_tts_missing_raises_503(self, monkeypatch):
        mod._reset_probe_cache()
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(Exception, match="503"):
                mod.require_mlx_audio_tts()

    def test_stt_ok_returns(self, monkeypatch):
        monkeypatch.setattr(
            mod,
            "_cached_verdict",
            {"": mod._Verdict(ok=True), "stt": mod._Verdict(ok=True)},
        )
        with patch("builtins.__import__", return_value=MagicMock()):
            mod.require_mlx_audio_stt()

    def test_stt_missing_raises_503(self, monkeypatch):
        mod._reset_probe_cache()
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(Exception, match="503"):
                mod.require_mlx_audio_stt()
