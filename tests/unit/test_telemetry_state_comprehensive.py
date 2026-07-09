"""telemetry/state 与 consent 测试。

覆盖 consent 状态机、env var kill switch、文件持久化路径。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fusion_mlx.telemetry import state
from fusion_mlx.telemetry.state import (
    ENV_VAR,
    client_id_path,
    consent_path,
    consent_source,
    get_consent_state,
    record_consent,
)


class TestConsentState(unittest.TestCase):

    def test_record_consent_yes(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                state, "consent_path", return_value=Path(td) / "consent.json"
            ):
                record_consent(True, fusion_mlx_version="0.4.1")
                cs = get_consent_state()
                self.assertIsNotNone(cs)
                self.assertTrue(cs.consent)

    def test_record_consent_no(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                state, "consent_path", return_value=Path(td) / "consent.json"
            ):
                record_consent(False, fusion_mlx_version="0.4.1")
                cs = get_consent_state()
                self.assertIsNotNone(cs)
                self.assertFalse(cs.consent)

    def test_no_consent_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                state, "consent_path", return_value=Path(td) / "nonexistent.json"
            ):
                self.assertIsNone(get_consent_state())

    def test_consent_source_returns_string(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                state, "consent_path", return_value=Path(td) / "consent.json"
            ):
                record_consent(True, fusion_mlx_version="0.4.1")
                src = consent_source()
                self.assertIsInstance(src, str)
                self.assertGreater(len(src), 0)


class TestEnvVarKillSwitch(unittest.TestCase):

    def test_env_var_name(self):
        self.assertIsInstance(ENV_VAR, str)
        self.assertGreater(len(ENV_VAR), 0)


class TestPaths(unittest.TestCase):

    def test_consent_path_is_path_or_str(self):
        p = consent_path()
        self.assertTrue(isinstance(p, (str, Path)))

    def test_client_id_path_is_path_or_str(self):
        p = client_id_path()
        self.assertTrue(isinstance(p, (str, Path)))


if __name__ == "__main__":
    unittest.main()
