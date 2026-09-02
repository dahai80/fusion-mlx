# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the code-sandbox reward endpoint (#743).

Two layers:

1. ``run_code_reward`` unit tests — the sandbox-exec isolation primitive.
   Run on macOS (``sandbox-exec`` present); skipped elsewhere. Each test
   writes real code into a real temp work dir and asserts the
   deny-by-default profile holds (fork denied, home write denied, tmp
   allowed) and the unittest pass-rate is parsed correctly.
2. ``POST /admin/api/fine-tune/reward/code`` route tests — the HTTP
   wrapper. ``FUSION_CODE_SANDBOX=on`` env-gate (default OFF -> 503),
   400 on missing/invalid body, 200 with the pass-rate payload when
   enabled. The route tests use a monkeypatched ``run_code_reward`` so
   they run on any platform (no sandbox-exec dependency).
"""

from __future__ import annotations

import os
import platform
import shutil
import textwrap

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.admin.routes import router as admin_router
from fusion_mlx.training.code_sandbox import (
    CodeRewardResult,
    _parse_unittest_results,
    run_code_reward,
)

_MACOS = platform.system() == "Darwin"
_HAS_SANDBOX = shutil.which("sandbox-exec") is not None
_SB_AVAILABLE = pytest.mark.skipif(
    not (_MACOS and _HAS_SANDBOX),
    reason="sandbox-exec isolation only on macOS",
)

# Minimal unittest test-suite the completion must satisfy. The
# ``solution`` module is the concatenation of ``code`` + ``tests``, so a
# passing completion defines ``def add(a, b): return a + b``.
_PASSING_CODE = "def add(a, b):\n    return a + b\n"
_FAILING_CODE = "def add(a, b):\n    return None\n"
_TEST_SUITE = textwrap.dedent("""\
    import unittest

    class TestSolution(unittest.TestCase):
        def test_add_positive(self):
            self.assertEqual(add(1, 2), 3)

        def test_add_zero(self):
            self.assertEqual(add(0, 0), 0)
    """)


# =============================================================================
# Layer 1: run_code_reward isolation + scoring (macOS sandbox-exec only)
# =============================================================================


@_SB_AVAILABLE
class TestSandboxIsolation:
    def test_passing_code_full_reward(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUSION_CODE_SANDBOX", "on")
        result = run_code_reward(
            _PASSING_CODE, _TEST_SUITE, work_dir=tmp_path, timeout=15.0
        )
        assert result.timed_out is False
        assert result.total == 2, result.stdout
        assert result.passed == 2
        assert result.reward == 1.0

    def test_failing_code_partial_reward(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUSION_CODE_SANDBOX", "on")
        result = run_code_reward(
            _FAILING_CODE, _TEST_SUITE, work_dir=tmp_path, timeout=15.0
        )
        assert result.total == 2
        assert result.passed == 0
        assert result.reward == 0.0

    def test_fork_is_denied(self, monkeypatch, tmp_path):
        # A poisoned completion must NOT be able to fork a sibling
        # process. ``subprocess.run`` raises PermissionError under the
        # (deny process-fork) profile; the test inside the sandbox
        # asserts the fork attempt was refused.
        code = textwrap.dedent("""\
            import subprocess

            def try_fork():
                try:
                    subprocess.run(["echo", "exfil"], capture_output=True)
                    return "forked"
                except PermissionError:
                    return "denied"
                except Exception:
                    return "other-error"
            """)
        tests = textwrap.dedent("""\
            import unittest

            class TestFork(unittest.TestCase):
                def test_fork_denied(self):
                    self.assertEqual(try_fork(), "denied")
            """)
        monkeypatch.setenv("FUSION_CODE_SANDBOX", "on")
        result = run_code_reward(code, tests, work_dir=tmp_path, timeout=15.0)
        assert result.total == 1, result.stdout + result.stderr
        assert result.passed == 1, result.stdout
        assert result.reward == 1.0

    def test_home_write_is_denied(self, monkeypatch, tmp_path):
        # Writing outside the work dir (into the user's home tree) must
        # be refused. The sandbox profile denies file-write* under the
        # home subpath; only the work dir is writable.
        home_marker = os.path.join(os.path.expanduser("~"), ".fusion_sb_probe")
        code = textwrap.dedent(f"""\
            import os

            def try_home_write():
                path = {home_marker!r}
                try:
                    with open(path, "w") as f:
                        f.write("pwned")
                    return "wrote-home"
                except PermissionError:
                    return "denied"
                except Exception:
                    return "other-error"
            """)
        tests = textwrap.dedent("""\
            import unittest

            class TestHomeWrite(unittest.TestCase):
                def test_home_denied(self):
                    self.assertEqual(try_home_write(), "denied")
            """)
        monkeypatch.setenv("FUSION_CODE_SANDBOX", "on")
        result = run_code_reward(code, tests, work_dir=tmp_path, timeout=15.0)
        assert result.passed == 1, result.stdout
        assert result.reward == 1.0
        # And no probe file was actually created in the home tree.
        assert not os.path.exists(home_marker)

    def test_work_dir_write_is_allowed(self, monkeypatch, tmp_path):
        # Sanity: the work dir itself IS writable (otherwise no test
        # could ever pass). ``unittest`` writes no files, so a passing
        # code+test that writes to cwd proves the allow-rule works.
        code = textwrap.dedent("""\
            import os

            def write_local():
                path = os.path.join(os.getcwd(), "local.txt")
                with open(path, "w") as f:
                    f.write("ok")
                return os.path.exists(path)
            """)
        tests = textwrap.dedent("""\
            import unittest

            class TestLocalWrite(unittest.TestCase):
                def test_local_allowed(self):
                    self.assertTrue(write_local())
            """)
        monkeypatch.setenv("FUSION_CODE_SANDBOX", "on")
        result = run_code_reward(code, tests, work_dir=tmp_path, timeout=15.0)
        assert result.passed == 1, result.stdout
        assert result.reward == 1.0

    def test_timeout_returns_zero_reward(self, monkeypatch, tmp_path):
        code = "while True:\n    pass\n"
        tests = textwrap.dedent("""\
            import unittest

            class TestTimeout(unittest.TestCase):
                def test_never(self):
                    self.assertTrue(True)
            """)
        monkeypatch.setenv("FUSION_CODE_SANDBOX", "on")
        result = run_code_reward(code, tests, work_dir=tmp_path, timeout=2.0)
        assert result.timed_out is True
        assert result.reward == 0.0
        assert "timed out" in result.error


# =============================================================================
# Layer 1b: env-gate + sandbox-absent fail-visible (no sandbox needed)
# =============================================================================


class TestEnvGate:
    def test_disabled_returns_error_result(self, monkeypatch):
        monkeypatch.delenv("FUSION_CODE_SANDBOX", raising=False)
        result = run_code_reward(_PASSING_CODE, _TEST_SUITE)
        assert result.reward == 0.0
        assert "disabled" in result.error

    def test_disabled_by_default(self, monkeypatch):
        # Default (env unset) must be OFF — untrusted exec never on by
        # default. ``FUSION_CODE_SANDBOX_TRUSTED`` posture mirrored.
        monkeypatch.delenv("FUSION_CODE_SANDBOX", raising=False)
        from fusion_mlx.training.code_sandbox import _is_enabled

        assert _is_enabled() is False

    @pytest.mark.skipif(
        _MACOS and _HAS_SANDBOX,
        reason="sandbox-exec present on this host; cannot test absence",
    )
    def test_sandbox_exec_absent_fail_visible(self, monkeypatch, tmp_path):
        # On a non-macOS host (or a macOS without sandbox-exec), the
        # sandbox-exec-absent path must surface a fail-visible error
        # rather than silently run untrusted code unsandboxed.
        monkeypatch.setenv("FUSION_CODE_SANDBOX", "on")
        result = run_code_reward(_PASSING_CODE, _TEST_SUITE, work_dir=tmp_path)
        assert result.reward == 0.0
        assert "sandbox-exec" in result.error


# =============================================================================
# Layer 1c: unittest output parsing (pure, no sandbox)
# =============================================================================


class TestParseUnittest:
    def test_all_pass(self):
        stdout = "test_add_positive (test.solution) ... ok\n"
        stdout += "test_add_zero (test.solution) ... ok\n"
        stdout += "Ran 2 tests in 0.001s\n\nOK\n"
        passed, total = _parse_unittest_results(stdout)
        assert passed == 2
        assert total == 2

    def test_partial_pass(self):
        stdout = "test_ok (test.solution) ... ok\n"
        stdout += "test_bad (test.solution) ... FAIL\n"
        stdout += "Ran 2 tests in 0.001s\n\nFAILED (failures=1)\n"
        passed, total = _parse_unittest_results(stdout)
        assert passed == 1
        assert total == 2

    def test_collection_error_no_summary(self):
        # Import error before any test runs: no ``Ran N`` line.
        stdout = "ERROR: test_solution\nTypeError: bad\n"
        passed, total = _parse_unittest_results(stdout)
        assert passed == 0
        assert total == 0

    def test_fail_word_in_body_not_miscounted(self):
        # An assertion message containing "FAIL" must not inflate the
        # passed count; only ``... ok`` markers count.
        stdout = "test_msg (test.solution) ... ok\n"
        stdout += "AssertionError: FAIL text in body\n"
        stdout += "Ran 1 tests in 0.001s\n\nOK\n"
        passed, total = _parse_unittest_results(stdout)
        assert passed == 1
        assert total == 1


# =============================================================================
# Layer 2: HTTP route wrapper
# =============================================================================


def _build_app():
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: True
    return app


def _stub_reward(monkeypatch, result=None):
    # Patch run_code_reward so route tests don't need sandbox-exec.
    result = result or CodeRewardResult(
        reward=1.0, passed=2, total=2, timed_out=False, stdout="ok", stderr=""
    )
    import fusion_mlx.training.code_sandbox as sb_mod

    called = {}

    def _fake(code, tests, **kw):
        called["code"] = code
        called["tests"] = tests
        called["kw"] = kw
        return result

    # The endpoint imports run_code_reward lazily inside the handler,
    # so patch the source module's attribute (the handler does
    # ``from fusion_mlx.training.code_sandbox import run_code_reward``).
    monkeypatch.setattr(sb_mod, "run_code_reward", _fake)
    return called


class TestCodeRewardRoute:
    def test_400_missing_code(self):
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"tests": _TEST_SUITE},
        )
        assert resp.status_code == 400
        assert "code" in resp.json()["detail"]

    def test_400_missing_tests(self):
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"code": _PASSING_CODE},
        )
        assert resp.status_code == 400
        assert "tests" in resp.json()["detail"]

    def test_400_empty_strings(self):
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"code": "   ", "tests": _TEST_SUITE},
        )
        assert resp.status_code == 400

    def test_400_bad_timeout(self):
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"code": _PASSING_CODE, "tests": _TEST_SUITE, "timeout": -5},
        )
        assert resp.status_code == 400
        assert "timeout" in resp.json()["detail"]

    def test_200_when_enabled(self, monkeypatch):
        called = _stub_reward(monkeypatch)
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"code": _PASSING_CODE, "tests": _TEST_SUITE, "timeout": 10},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reward"] == 1.0
        assert body["passed"] == 2
        assert body["total"] == 2
        assert body["timed_out"] is False
        # timeout threaded through to the runner.
        assert called["kw"].get("timeout") == 10.0

    def test_503_when_gate_disabled(self, monkeypatch):
        # When the runner returns a ``disabled`` error (gate OFF), the
        # route must surface 503 fail-visible — NOT a 200 reward 0.0,
        # which would read as "code ran, scored 0".
        disabled = CodeRewardResult(
            reward=0.0,
            passed=0,
            total=0,
            timed_out=False,
            error="code-sandbox reward disabled (set FUSION_CODE_SANDBOX=on)",
        )
        _stub_reward(monkeypatch, result=disabled)
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"code": _PASSING_CODE, "tests": _TEST_SUITE},
        )
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"]

    def test_500_on_runner_exception(self, monkeypatch):
        import fusion_mlx.training.code_sandbox as sb_mod

        def _boom(code, tests, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(sb_mod, "run_code_reward", _boom)
        app = _build_app()
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"code": _PASSING_CODE, "tests": _TEST_SUITE},
        )
        assert resp.status_code == 500
        assert "Code reward failed" in resp.json()["detail"]

    def test_admin_required(self):
        # Without the override, require_admin gates the route.
        app = FastAPI()
        app.include_router(admin_router)
        client = TestClient(app)
        resp = client.post(
            "/admin/api/fine-tune/reward/code",
            json={"code": _PASSING_CODE, "tests": _TEST_SUITE},
        )
        assert resp.status_code in (401, 403)
