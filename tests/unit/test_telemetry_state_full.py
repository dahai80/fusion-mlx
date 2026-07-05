"""telemetry/state 完整测试（importlib 绕过 mlx 链，覆盖全部函数）。

覆盖 get_consent_state/record_consent/get_or_create_client_id/
reset_state/is_enabled/consent_source/_env_kill_switch_active/
set_cli_kill_switch 全部函数 + ConsentState dataclass + 错误容忍。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_module():
    _touched = ("fusion_mlx", "fusion_mlx.telemetry", "fusion_mlx.telemetry.state")
    _prev = {k: sys.modules.get(k) for k in _touched}
    if "fusion_mlx" not in sys.modules:
        pkg = types.ModuleType("fusion_mlx")
        pkg.__path__ = ["fusion_mlx"]
        sys.modules["fusion_mlx"] = pkg
    if "fusion_mlx.telemetry" not in sys.modules:
        sub = types.ModuleType("fusion_mlx.telemetry")
        sub.__path__ = ["fusion_mlx/telemetry"]
        sys.modules["fusion_mlx.telemetry"] = sub
    spec = importlib.util.spec_from_file_location(
        "fusion_mlx.telemetry.state", "fusion_mlx/telemetry/state.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["fusion_mlx.telemetry.state"] = m
    spec.loader.exec_module(m)
    # Restore sys.modules so the standalone load does not leak stub packages or
    # a duplicate module instance into the rest of the pytest session (it broke
    # the other telemetry tests via "not in sys.modules" import errors).
    for k in _touched:
        if _prev[k] is not None:
            sys.modules[k] = _prev[k]
        else:
            sys.modules.pop(k, None)
    return m


state = _load_module()


class TestConsentStateDataclass(unittest.TestCase):

    def test_defaults(self):
        cs = state.ConsentState(
            consent=True, prompted_at="2026-07-04T00:00:00Z", prompted_version="0.4.1"
        )
        self.assertTrue(cs.consent)
        self.assertEqual(cs.schema_version, 1)

    def test_frozen(self):
        cs = state.ConsentState(consent=False, prompted_at="x", prompted_version="y")
        with self.assertRaises(Exception):
            cs.consent = True  # type: ignore


class TestPaths(unittest.TestCase):

    def test_consent_path_under_rapid_mlx_dir(self):
        p = state.consent_path()
        self.assertIn(".rapid-mlx", str(p))
        self.assertTrue(p.name == "telemetry-consent.yaml")

    def test_client_id_path_under_rapid_mlx_dir(self):
        p = state.client_id_path()
        self.assertIn(".rapid-mlx", str(p))
        self.assertTrue(p.name == "telemetry-client-id")

    def test_default_telemetry_dir_uses_home(self):
        d = state._default_telemetry_dir()
        self.assertEqual(d, Path.home() / ".rapid-mlx")


class TestGetConsentState(unittest.TestCase):

    def test_no_file_returns_none(self):
        with patch.object(
            state, "consent_path", return_value=Path("/nonexistent/consent.yaml")
        ):
            self.assertIsNone(state.get_consent_state())

    def test_valid_file_returns_state(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text(
                yaml.safe_dump(
                    {
                        "consent": True,
                        "prompted_at": "2026-07-04T00:00:00Z",
                        "prompted_version": "0.4.1",
                        "schema_version": 1,
                    }
                )
            )
            with patch.object(state, "consent_path", return_value=p):
                cs = state.get_consent_state()
            self.assertIsNotNone(cs)
            self.assertTrue(cs.consent)
            self.assertEqual(cs.prompted_version, "0.4.1")

    def test_consent_false_returns_state(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text(
                yaml.safe_dump(
                    {
                        "consent": False,
                        "prompted_at": "2026-07-04T00:00:00Z",
                        "prompted_version": "0.4.1",
                        "schema_version": 1,
                    }
                )
            )
            with patch.object(state, "consent_path", return_value=p):
                cs = state.get_consent_state()
            self.assertIsNotNone(cs)
            self.assertFalse(cs.consent)

    def test_corrupt_yaml_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text("{not: valid: yaml: [[[")
            with patch.object(state, "consent_path", return_value=p):
                self.assertIsNone(state.get_consent_state())

    def test_missing_consent_field_returns_none(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text(yaml.safe_dump({"prompted_at": "x", "prompted_version": "y"}))
            with patch.object(state, "consent_path", return_value=p):
                self.assertIsNone(state.get_consent_state())

    def test_consent_wrong_type_returns_none(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text(
                yaml.safe_dump(
                    {
                        "consent": "yes",  # 非 bool
                        "prompted_at": "x",
                        "prompted_version": "y",
                        "schema_version": 1,
                    }
                )
            )
            with patch.object(state, "consent_path", return_value=p):
                self.assertIsNone(state.get_consent_state())

    def test_prompted_at_wrong_type_returns_none(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text(
                yaml.safe_dump(
                    {
                        "consent": True,
                        "prompted_at": 123,  # 非 str
                        "prompted_version": "y",
                        "schema_version": 1,
                    }
                )
            )
            with patch.object(state, "consent_path", return_value=p):
                self.assertIsNone(state.get_consent_state())

    def test_wrong_schema_version_returns_none(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text(
                yaml.safe_dump(
                    {
                        "consent": True,
                        "prompted_at": "x",
                        "prompted_version": "y",
                        "schema_version": 99,  # 未来版本
                    }
                )
            )
            with patch.object(state, "consent_path", return_value=p):
                self.assertIsNone(state.get_consent_state())

    def test_non_int_schema_version_defaults_to_1(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text(
                yaml.safe_dump(
                    {
                        "consent": True,
                        "prompted_at": "x",
                        "prompted_version": "y",
                        "schema_version": "bad",  # 非整数
                    }
                )
            )
            with patch.object(state, "consent_path", return_value=p):
                # schema_version 非整数→默认 1→与 CURRENT 一致→返回 state
                cs = state.get_consent_state()
            self.assertIsNotNone(cs)
            self.assertEqual(cs.schema_version, 1)

    def test_empty_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            p.write_text("")
            with patch.object(state, "consent_path", return_value=p):
                self.assertIsNone(state.get_consent_state())


class TestRecordConsent(unittest.TestCase):

    def test_writes_valid_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            with patch.object(state, "consent_path", return_value=p):
                result = state.record_consent(True, rapid_mlx_version="0.4.1")
            self.assertTrue(result.consent)
            self.assertEqual(result.prompted_version, "0.4.1")
            self.assertTrue(p.exists())
            # 重新读回验证
            with patch.object(state, "consent_path", return_value=p):
                cs = state.get_consent_state()
            self.assertTrue(cs.consent)

    def test_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nested" / "deep" / "consent.yaml"
            with patch.object(state, "consent_path", return_value=p):
                state.record_consent(False, rapid_mlx_version="0.4.1")
            self.assertTrue(p.exists())

    def test_tmp_cleanup_no_leftover(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            with patch.object(state, "consent_path", return_value=p):
                state.record_consent(True, rapid_mlx_version="0.4.1")
            # 不应残留 .tmp 文件
            tmp = p.with_suffix(p.suffix + ".tmp")
            self.assertFalse(tmp.exists())

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "consent.yaml"
            with patch.object(state, "consent_path", return_value=p):
                state.record_consent(True, rapid_mlx_version="0.4.0")
                state.record_consent(False, rapid_mlx_version="0.4.1")
                cs = state.get_consent_state()
            self.assertFalse(cs.consent)
            self.assertEqual(cs.prompted_version, "0.4.1")


class TestGetOrCreateClientId(unittest.TestCase):

    def test_creates_new_id(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry-client-id"
            with patch.object(state, "client_id_path", return_value=p):
                cid = state.get_or_create_client_id()
            self.assertTrue(p.exists())
            self.assertGreater(len(cid), 8)  # UUID4 字符串

    def test_idempotent_returns_existing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry-client-id"
            p.write_text("existing-id-12345\n")
            with patch.object(state, "client_id_path", return_value=p):
                cid = state.get_or_create_client_id()
            self.assertEqual(cid, "existing-id-12345")

    def test_preserves_all_zero_uuid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry-client-id"
            p.write_text("00000000-0000-0000-0000-000000000000\n")
            with patch.object(state, "client_id_path", return_value=p):
                cid = state.get_or_create_client_id()
            self.assertEqual(cid, "00000000-0000-0000-0000-000000000000")

    def test_empty_file_creates_new(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry-client-id"
            p.write_text("   \n")
            with patch.object(state, "client_id_path", return_value=p):
                cid = state.get_or_create_client_id()
            self.assertGreater(len(cid), 8)


class TestResetState(unittest.TestCase):

    def test_removes_both_files(self):
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "consent.yaml"
            ip = Path(td) / "telemetry-client-id"
            cp.write_text("x")
            ip.write_text("y")
            with (
                patch.object(state, "consent_path", return_value=cp),
                patch.object(state, "client_id_path", return_value=ip),
            ):
                state.reset_state()
            self.assertFalse(cp.exists())
            self.assertFalse(ip.exists())

    def test_missing_files_no_error(self):
        with (
            patch.object(state, "consent_path", return_value=Path("/nonexistent/c")),
            patch.object(state, "client_id_path", return_value=Path("/nonexistent/i")),
        ):
            state.reset_state()  # 不抛异常


class TestEnvKillSwitch(unittest.TestCase):

    def test_no_env_returns_false(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(state._env_kill_switch_active())

    def test_zero_returns_true(self):
        with patch.dict(os.environ, {state.ENV_VAR: "0"}):
            self.assertTrue(state._env_kill_switch_active())

    def test_false_returns_true(self):
        with patch.dict(os.environ, {state.ENV_VAR: "false"}):
            self.assertTrue(state._env_kill_switch_active())

    def test_no_off_returns_true(self):
        for v in ("no", "off", "OFF", "False", "NO"):
            with patch.dict(os.environ, {state.ENV_VAR: v}):
                self.assertTrue(state._env_kill_switch_active())

    def test_empty_string_returns_true(self):
        with patch.dict(os.environ, {state.ENV_VAR: ""}):
            self.assertTrue(state._env_kill_switch_active())

    def test_one_returns_false(self):
        with patch.dict(os.environ, {state.ENV_VAR: "1"}):
            self.assertFalse(state._env_kill_switch_active())

    def test_truthy_ignored(self):
        # 故意无 force-on env，truthy 值不生效
        with patch.dict(os.environ, {state.ENV_VAR: "true"}):
            self.assertFalse(state._env_kill_switch_active())


class TestSetCliKillSwitch(unittest.TestCase):

    def setUp(self):
        state.set_cli_kill_switch(False)

    def tearDown(self):
        state.set_cli_kill_switch(False)

    def test_set_true(self):
        state.set_cli_kill_switch(True)
        self.assertTrue(state._cli_kill_switch_active)

    def test_set_false(self):
        state.set_cli_kill_switch(True)
        state.set_cli_kill_switch(False)
        self.assertFalse(state._cli_kill_switch_active)

    def test_idempotent(self):
        state.set_cli_kill_switch(True)
        state.set_cli_kill_switch(True)
        self.assertTrue(state._cli_kill_switch_active)


class TestIsEnabled(unittest.TestCase):

    def setUp(self):
        state.set_cli_kill_switch(False)

    def tearDown(self):
        state.set_cli_kill_switch(False)

    def test_cli_flag_disables(self):
        with patch.object(
            state, "get_consent_state", return_value=state.ConsentState(True, "x", "y")
        ):
            self.assertFalse(state.is_enabled(cli_no_telemetry=True))

    def test_cli_kill_switch_disables(self):
        state.set_cli_kill_switch(True)
        with patch.object(
            state, "get_consent_state", return_value=state.ConsentState(True, "x", "y")
        ):
            self.assertFalse(state.is_enabled())

    def test_env_kill_switch_disables(self):
        with patch.dict(os.environ, {state.ENV_VAR: "0"}):
            with patch.object(
                state,
                "get_consent_state",
                return_value=state.ConsentState(True, "x", "y"),
            ):
                self.assertFalse(state.is_enabled())

    def test_no_consent_file_disables(self):
        with patch.object(state, "get_consent_state", return_value=None):
            self.assertFalse(state.is_enabled())

    def test_consent_true_enables(self):
        with patch.object(
            state, "get_consent_state", return_value=state.ConsentState(True, "x", "y")
        ):
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(state.is_enabled())

    def test_consent_false_disables(self):
        with patch.object(
            state, "get_consent_state", return_value=state.ConsentState(False, "x", "y")
        ):
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(state.is_enabled())

    def test_precedence_cli_over_env(self):
        # cli flag 应优先于 env
        with patch.dict(os.environ, {state.ENV_VAR: "1"}):  # env 不阻断
            self.assertFalse(state.is_enabled(cli_no_telemetry=True))


class TestConsentSource(unittest.TestCase):

    def setUp(self):
        state.set_cli_kill_switch(False)

    def tearDown(self):
        state.set_cli_kill_switch(False)

    def test_cli_flag_source(self):
        self.assertIn("cli-flag", state.consent_source(cli_no_telemetry=True))

    def test_cli_kill_switch_source(self):
        state.set_cli_kill_switch(True)
        self.assertIn("cli-flag", state.consent_source())

    def test_env_var_source(self):
        with patch.dict(os.environ, {state.ENV_VAR: "0"}):
            src = state.consent_source()
        self.assertIn("env-var", src)
        self.assertIn(state.ENV_VAR, src)

    def test_no_consent_source(self):
        with patch.object(state, "get_consent_state", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                self.assertIn("default", state.consent_source())

    def test_consent_file_source(self):
        cs = state.ConsentState(True, "x", "y")
        with patch.object(state, "get_consent_state", return_value=cs):
            with patch.dict(os.environ, {}, clear=True):
                src = state.consent_source()
        self.assertIn("consent-file", src)


class TestEnvVarConstant(unittest.TestCase):

    def test_env_var_name(self):
        self.assertEqual(state.ENV_VAR, "RAPID_MLX_TELEMETRY")

    def test_schema_version_constant(self):
        self.assertEqual(state.CURRENT_CONSENT_SCHEMA_VERSION, 1)


if __name__ == "__main__":
    unittest.main()
