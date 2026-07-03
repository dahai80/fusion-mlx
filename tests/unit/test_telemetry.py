import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fusion_mlx.telemetry.consent import maybe_prompt_for_consent
from fusion_mlx.telemetry.redact import (
    bucket_memory_gb,
    bucket_tokens,
    bucket_ttft_ms,
    bucket_tps,
    fingerprint_traceback,
    hash_flag_names,
    normalize_model_path,
    platform_info,
)
from fusion_mlx.telemetry.schema import (
    SCHEMA_VERSION,
    SessionPayload,
    TelemetryPayload,
    sample_preview_payload,
)
from fusion_mlx.telemetry.state import (
    ConsentState,
    consent_path,
    get_consent_state,
    is_enabled,
    record_consent,
    reset_state,
    set_cli_kill_switch,
)

import logging
logger = logging.getLogger(__name__)


class TestConsent(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._consent_patcher = patch(
            "fusion_mlx.telemetry.state.consent_path",
            return_value=Path(self._tmp) / "consent.yaml",
        )
        self._client_id_patcher = patch(
            "fusion_mlx.telemetry.state.client_id_path",
            return_value=Path(self._tmp) / "client-id",
        )
        self._consent_patcher.start()
        self._client_id_patcher.start()
        set_cli_kill_switch(False)
        # Reset the module-level kill switch
        import fusion_mlx.telemetry.state as _state_mod
        _state_mod._cli_kill_switch_active = False

    def tearDown(self):
        reset_state()
        self._consent_patcher.stop()
        self._client_id_patcher.stop()

    def test_no_consent_file_returns_none(self):
        self.assertIsNone(get_consent_state())

    def test_record_consent_true(self):
        state = record_consent(True, rapid_mlx_version="0.1.0")
        self.assertTrue(state.consent)
        loaded = get_consent_state()
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.consent)

    def test_record_consent_false(self):
        state = record_consent(False, rapid_mlx_version="0.1.0")
        self.assertFalse(state.consent)
        loaded = get_consent_state()
        self.assertIsNotNone(loaded)
        self.assertFalse(loaded.consent)

    def test_is_enabled_no_consent(self):
        self.assertFalse(is_enabled())

    def test_is_enabled_consent_true(self):
        record_consent(True, rapid_mlx_version="0.1.0")
        self.assertTrue(is_enabled())

    def test_is_enabled_consent_false(self):
        record_consent(False, rapid_mlx_version="0.1.0")
        self.assertFalse(is_enabled())

    def test_is_enabled_env_kill_switch(self):
        record_consent(True, rapid_mlx_version="0.1.0")
        with patch.dict(os.environ, {"RAPID_MLX_TELEMETRY": "0"}):
            self.assertFalse(is_enabled())

    def test_is_enabled_cli_kill_switch(self):
        record_consent(True, rapid_mlx_version="0.1.0")
        set_cli_kill_switch(True)
        self.assertFalse(is_enabled())
        set_cli_kill_switch(False)

    def test_reset_state_removes_files(self):
        record_consent(True, rapid_mlx_version="0.1.0")
        reset_state()
        self.assertIsNone(get_consent_state())


class TestRedact(unittest.TestCase):
    def test_bucket_tokens_small(self):
        self.assertEqual(bucket_tokens(0), "0-256")

    def test_bucket_tokens_boundary(self):
        self.assertEqual(bucket_tokens(256), "256-1k")

    def test_bucket_tokens_large(self):
        self.assertEqual(bucket_tokens(100000), "64k+")

    def test_bucket_tokens_negative(self):
        self.assertEqual(bucket_tokens(-5), "0-256")

    def test_bucket_ttft_ms(self):
        self.assertEqual(bucket_ttft_ms(50), "<100ms")
        self.assertEqual(bucket_ttft_ms(200), "100-500ms")
        self.assertEqual(bucket_ttft_ms(6000), ">5s")

    def test_bucket_tps(self):
        self.assertEqual(bucket_tps(5), "<10")
        self.assertEqual(bucket_tps(20), "10-30")
        self.assertEqual(bucket_tps(200), ">100")

    def test_bucket_memory_gb(self):
        self.assertEqual(bucket_memory_gb(0), 0)
        self.assertEqual(bucket_memory_gb(8 * 1024**3), 8)
        self.assertEqual(bucket_memory_gb(-100), 0)

    def test_normalize_model_path_hf_repo(self):
        self.assertEqual(
            normalize_model_path("mlx-community/Qwen3.5-9B-4bit"),
            "mlx-community/Qwen3.5-9B-4bit",
        )

    def test_normalize_model_path_local(self):
        self.assertEqual(normalize_model_path("/Users/alice/model"), "<local>")
        self.assertEqual(normalize_model_path("./model"), "<local>")
        self.assertEqual(normalize_model_path("~/model"), "<local>")

    def test_normalize_model_path_empty(self):
        self.assertEqual(normalize_model_path(""), "<empty>")

    def test_hash_flag_names_extracts_names(self):
        argv = ["--port", "8000", "--host", "0.0.0.0", "--verbose"]
        names = hash_flag_names(argv)
        self.assertIn("port", names)
        self.assertIn("host", names)
        self.assertIn("verbose", names)
        self.assertNotIn("8000", names)
        self.assertNotIn("0.0.0.0", names)

    def test_hash_flag_names_equals_syntax(self):
        argv = ["--api-key=sk-123"]
        names = hash_flag_names(argv)
        self.assertIn("api-key", names)
        self.assertNotIn("sk-123", names)

    def test_fingerprint_traceback_deterministic(self):
        try:
            raise ValueError("test error")
        except ValueError as exc:
            fp1 = fingerprint_traceback(exc)
            fp2 = fingerprint_traceback(exc)
            self.assertEqual(fp1, fp2)
            self.assertEqual(len(fp1), 16)

    def test_platform_info_has_keys(self):
        info = platform_info()
        for key in ("os", "os_version", "arch", "chip", "memory_gb", "python_version"):
            self.assertIn(key, info)


class TestSchema(unittest.TestCase):
    def test_schema_version_is_int(self):
        self.assertIsInstance(SCHEMA_VERSION, int)

    def test_session_payload_defaults(self):
        p = SessionPayload(subcommand="serve")
        self.assertEqual(p.subcommand, "serve")
        self.assertIsNone(p.duration_seconds)
        self.assertEqual(p.models_loaded, ())
        self.assertEqual(p.flag_names, ())

    def test_telemetry_payload_to_dict(self):
        info = platform_info()
        payload = TelemetryPayload(
            schema_version=SCHEMA_VERSION,
            client_id="test-cid",
            session_id="test-sid",
            rapid_mlx_version="0.1.0",
            platform=info,
            event="session_start",
            timestamp="2026-01-01T00:00:00Z",
            session=SessionPayload(subcommand="serve"),
        )
        d = payload.to_dict()
        self.assertIn("session", d)
        self.assertNotIn("request", d)
        self.assertNotIn("error", d)

    def test_sample_preview_payload(self):
        payload = sample_preview_payload(
            client_id="test-cid",
            rapid_mlx_version="0.1.0",
        )
        self.assertEqual(payload.client_id, "test-cid")
        self.assertEqual(payload.event, "session_start")
        self.assertIsNotNone(payload.platform)
        self.assertIsNotNone(payload.session)


if __name__ == "__main__":
    unittest.main()
