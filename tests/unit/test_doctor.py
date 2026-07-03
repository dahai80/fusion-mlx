import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fusion_mlx.doctor.env_health import (
    Check,
    CheckStatus,
    Report,
    Section,
    run_all,
    section_system,
    section_python,
    section_required_packages,
    section_optional_packages,
    section_hf_cache,
    section_network,
    section_shell_integration,
    section_optional_tools,
)

import logging
logger = logging.getLogger(__name__)


class TestCheckModel(unittest.TestCase):
    def test_check_defaults(self):
        c = Check(label="test", status=CheckStatus.OK)
        self.assertEqual(c.label, "test")
        self.assertEqual(c.status, CheckStatus.OK)
        self.assertEqual(c.detail, "")

    def test_check_with_detail(self):
        c = Check(label="test", status=CheckStatus.WARN, detail="some detail")
        self.assertEqual(c.detail, "some detail")


class TestSection(unittest.TestCase):
    def test_add_check(self):
        s = Section("Test Section")
        s.add("item 1", CheckStatus.OK)
        s.add("item 2", CheckStatus.FAIL, detail="broken")
        self.assertEqual(len(s.checks), 2)
        self.assertEqual(s.checks[0].label, "item 1")
        self.assertEqual(s.checks[1].status, CheckStatus.FAIL)


class TestReport(unittest.TestCase):
    def test_empty_report(self):
        r = Report()
        self.assertEqual(r.n_ok, 0)
        self.assertEqual(r.n_warn, 0)
        self.assertEqual(r.n_fail, 0)
        self.assertEqual(r.exit_code, 0)

    def test_exit_code_with_fail(self):
        r = Report()
        s = Section("test")
        s.add("ok item", CheckStatus.OK)
        s.add("fail item", CheckStatus.FAIL)
        r.sections.append(s)
        self.assertEqual(r.n_ok, 1)
        self.assertEqual(r.n_fail, 1)
        self.assertEqual(r.exit_code, 1)

    def test_exit_code_warn_only(self):
        r = Report()
        s = Section("test")
        s.add("warn item", CheckStatus.WARN)
        r.sections.append(s)
        self.assertEqual(r.n_warn, 1)
        self.assertEqual(r.exit_code, 0)


class TestSectionSystem(unittest.TestCase):
    def test_runs_without_error(self):
        s = section_system()
        self.assertIsInstance(s, Section)
        self.assertEqual(s.title, "System")
        self.assertTrue(len(s.checks) > 0)


class TestSectionPython(unittest.TestCase):
    def test_runs_without_error(self):
        s = section_python()
        self.assertIsInstance(s, Section)
        self.assertTrue(len(s.checks) > 0)
        # Python version check should pass on 3.10+
        import sys
        if sys.version_info >= (3, 10):
            self.assertEqual(s.checks[0].status, CheckStatus.OK)


class TestSectionRequiredPackages(unittest.TestCase):
    def test_runs_without_error(self):
        s = section_required_packages()
        self.assertIsInstance(s, Section)
        self.assertTrue(len(s.checks) > 0)


class TestSectionOptionalPackages(unittest.TestCase):
    def test_runs_without_error(self):
        s = section_optional_packages()
        self.assertIsInstance(s, Section)
        self.assertTrue(len(s.checks) > 0)


class TestSectionHfCache(unittest.TestCase):
    def test_runs_without_error(self):
        s = section_hf_cache()
        self.assertIsInstance(s, Section)
        self.assertTrue(len(s.checks) > 0)


class TestSectionNetwork(unittest.TestCase):
    def test_injected_probe(self):
        fake_status = CheckStatus.OK
        fake_detail = "HEAD https://huggingface.co -> HTTP 200"
        s = section_network(probe=lambda: (fake_status, fake_detail))
        self.assertIsInstance(s, Section)
        self.assertEqual(s.checks[0].status, CheckStatus.OK)

    def test_injected_probe_unreachable(self):
        fake_status = CheckStatus.WARN
        fake_detail = "unreachable (timeout)"
        s = section_network(probe=lambda: (fake_status, fake_detail))
        self.assertEqual(s.checks[0].status, CheckStatus.WARN)


class TestSectionShellIntegration(unittest.TestCase):
    def test_injected_which(self):
        s = section_shell_integration(
            which=lambda name: "/usr/local/bin/fusion-mlx" if name == "fusion-mlx" else None,
            rcs=[],
        )
        self.assertIsInstance(s, Section)
        self.assertTrue(len(s.checks) > 0)
        self.assertEqual(s.checks[0].status, CheckStatus.OK)

    def test_injected_which_missing(self):
        s = section_shell_integration(
            which=lambda name: None,
            rcs=[],
        )
        self.assertTrue(len(s.checks) > 0)
        self.assertEqual(s.checks[0].status, CheckStatus.FAIL)


class TestSectionOptionalTools(unittest.TestCase):
    def test_injected_which(self):
        s = section_optional_tools(
            which=lambda name: "/usr/local/bin/codex" if name == "codex" else None,
        )
        self.assertIsInstance(s, Section)
        self.assertEqual(s.checks[0].status, CheckStatus.OK)

    def test_injected_which_missing(self):
        s = section_optional_tools(
            which=lambda name: None,
        )
        self.assertEqual(s.checks[0].status, CheckStatus.WARN)


class TestRunAll(unittest.TestCase):
    def test_run_all_returns_report(self):
        report = run_all()
        self.assertIsInstance(report, Report)
        self.assertTrue(len(report.sections) > 0)

    def test_run_all_catches_crashes(self):
        def _crash():
            raise RuntimeError("boom")
        with patch(
            "fusion_mlx.doctor.env_health._SECTION_BUILDERS",
            (_crash,),
        ):
            report = run_all()
            self.assertEqual(len(report.sections), 1)
            self.assertEqual(report.n_fail, 1)


if __name__ == "__main__":
    unittest.main()
