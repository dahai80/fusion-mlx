# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest


class TestSilenceReturnsEmpty:
    def test_pure_silence_returns_empty_string(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestRealSpeechStillTranscribes:
    def test_real_speech_still_transcribes(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestTrailingSilenceTrimmed:
    def test_speech_with_trailing_silence_stops_at_speech_end(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestVADDisabledViaEnv:
    def test_vad_disabled_via_env_pass_through(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    @pytest.mark.parametrize("value", ["false", "no", "off", "FALSE", "Off"])
    def test_env_disable_accepts_common_falsy_strings(self, value):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestAbsoluteTimestampsPreserved:
    def test_absolute_timestamps_preserved_when_vad_trims_leading_silence(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_word_level_timestamps_also_shifted(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestKwargOverrideDisables:
    def test_enable_vad_pretrim_false_kwarg_disables(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestVADImportFailureFallsBack:
    def test_vad_import_failure_falls_back(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestMalformedVADOutputFallsBack:
    def test_missing_start_key_falls_back(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_missing_end_key_falls_back(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_none_valued_timestamp_falls_back(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestTransientLoadFailureRetries:
    def test_transient_load_failure_retries_on_next_call(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_permanent_import_failure_is_cached(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestVADFalseNegativeGuard:
    def test_high_rms_no_speech_falls_back_to_whisper(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_pure_silence_below_rms_floor_still_returns_empty(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestVADThresholdKwargPropagates:
    def test_speech_threshold_kwarg_reaches_silero(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestRMSHelper:
    def test_pure_silence_below_floor(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_noisy_signal_above_floor(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_helper_swallows_bad_input_conservatively(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestVADLoadLock:
    def test_load_lock_serializes_first_load(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestUpstreamWhisperInputContract:
    def test_whisper_prepare_audio_signature_still_accepts_arrays(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_numpy_to_mx_conversion_still_works(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_whisper_log_mel_spectrogram_accepts_mx_array(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestParakeetEngineSkipsVAD:
    def test_parakeet_engine_skips_vad(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestEnvHelper:
    @pytest.mark.parametrize(
        "val,expected",
        [
            (None, False),
            ("", False),
            ("1", False),
            ("true", False),
            ("yes", False),
            ("on", False),
            ("0", True),
            ("false", True),
            ("FALSE", True),
            ("no", True),
            ("off", True),
            (" 0 ", True),
        ],
    )
    def test_env_helper(self, val, expected):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")


class TestShiftHelper:
    def test_shift_dict_segment(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_shift_dict_segment_with_words(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_shift_missing_keys_is_noop(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")

    def test_shift_object_segment(self):
        pytest.skip("STT VAD pretrim not migrated to fusion-mlx")
