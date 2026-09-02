# SPDX-License-Identifier: Apache-2.0
"""Container-isolated code-sandbox reward for GRPO (#743).

Executes model-generated ``code`` + dataset ``tests`` under a macOS
``sandbox-exec`` deny-by-default profile so a poisoned dataset or an
induced completion (``import os; os.system(...)``) cannot read/write the
user's home directory, reach the network, or fork a sibling process.

The reward is the unittest pass rate: the combined ``{code}\n{tests}`` is
run under ``python -m unittest`` (file-mode, no discovery), and the
``OK`` / ``FAILED`` lines are parsed into a pass/total count. The
endpoint that wraps this lives in ``admin/fine_tune_route.py``:

    POST /admin/api/fine-tune/reward/code  ->  {reward, passed, total, ...}

Gated behind ``FUSION_CODE_SANDBOX=on`` (default OFF) — mirrors the
trainer-side ``FUSION_CODE_SANDBOX_TRUSTED`` posture so the untrusted-exec
path is opt-in, never on by default.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Opt-in gate. Untrusted code execution is OFF unless the operator
# explicitly enables it (mirrors fusion-trainer C2 default-off).
_ENV_GATE = "FUSION_CODE_SANDBOX"

# Default per-run timeout (seconds). unittest itself has no wall-clock
# cap; the subprocess timeout is the only backstop against infinite loops.
_DEFAULT_TIMEOUT = 30.0

# macOS sandbox-exec profile: deny-by-default with the narrow allows a
# code-reward needs (run the interpreter, write only the work dir).
# - (allow default): start from the system default profile (process,
#   signal, sysctl, etc.) then layer explicit denials on top.
# - (deny network*): no outbound/inbound sockets (exfil / C2 block).
# - (deny file-write* (subpath "<home>")): home tree read-only. The work
#   dir is the only writable path.
# - (deny process-fork): blocks ``subprocess`` / ``os.system`` /
#   ``os.fork`` so a poisoned completion cannot spawn a sibling. The
#   interpreter itself is already running; this denies *further* forks.
_PROFILE_TEMPLATE = """\
(version 1)
(allow default)
(deny network*)
(deny file-write* (subpath "{home}"))
(allow file-write* (subpath "{work_dir}"))
(deny process-fork)
"""


def _is_enabled() -> bool:
    return os.environ.get(_ENV_GATE, "").strip().lower() in ("on", "1", "true")


def _sandbox_exec_path() -> str | None:
    return shutil.which("sandbox-exec")


@dataclass
class CodeRewardResult:
    reward: float
    passed: int
    total: int
    timed_out: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "passed": self.passed,
            "total": self.total,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


def _parse_unittest_results(stdout: str) -> tuple[int, int]:
    # ``python -m unittest`` verbose mode prints one line per test:
    #   ``test_func (test.solution) ... ok``  or  ``... FAIL`` / ``... ERROR``.
    # The trailing summary line is ``Ran N tests in T.tts``. We trust the
    # ``Ran N`` total (deterministic, written by the runner) and count the
    # explicit ``ok`` markers for passed — robust to a test that prints
    # the word "FAIL" inside an assertion message.
    total = 0
    m = re.search(r"Ran (\d+) tests?", stdout)
    if m:
        total = int(m.group(1))
    # Count ``... ok`` lines (unittest verbose marker). Anchor to the
    # trailing ``ok`` after the ellipsis to avoid matching body text.
    passed = len(re.findall(r"\.\.\.\s+ok\b", stdout))
    if total == 0:
        # No summary line (e.g. collection error before any test ran).
        # Fall back to the marker count alone.
        total = passed
    return passed, total


def run_code_reward(
    code: str,
    tests: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    work_dir: Path | None = None,
    python_exe: str | None = None,
) -> CodeRewardResult:
    # Fail-visible: every short-circuit returns a result with ``error``
    # set and reward 0.0 rather than raising — the caller (the GRPO
    # reward loop) treats reward 0 as "no signal", which is the safe
    # default for untrusted-exec failures.
    if not _is_enabled():
        msg = (
            f"code-sandbox reward disabled (set {_ENV_GATE}=on to enable); "
            "untrusted code execution is OFF by default"
        )
        logger.warning("code_sandbox: %s", msg)
        return CodeRewardResult(
            reward=0.0, passed=0, total=0, timed_out=False, error=msg
        )

    sandbox_exec = _sandbox_exec_path()
    if sandbox_exec is None:
        msg = "sandbox-exec not found (macOS only); cannot isolate untrusted code"
        logger.error("code_sandbox: %s", msg)
        return CodeRewardResult(
            reward=0.0, passed=0, total=0, timed_out=False, error=msg
        )

    own_dir = work_dir
    cleanup = False
    if own_dir is None:
        own_dir = Path(tempfile.mkdtemp(prefix="fusion_code_sb_"))
        cleanup = True
    own_dir = Path(own_dir)
    own_dir.mkdir(parents=True, exist_ok=True)

    py = python_exe or sys.executable
    home = str(Path.home())
    profile_path = own_dir / "fusion_sandbox.sb"
    profile_path.write_text(_PROFILE_TEMPLATE.format(home=home, work_dir=str(own_dir)))
    solution_path = own_dir / "solution.py"
    # Combine the completion + the test suite into one module so the
    # sandbox only forks ONE python process (process-fork is denied).
    # The tests are written as a ``unittest.TestCase`` whose body calls
    # the completion's functions; ``-m unittest`` imports ``solution``.
    solution_path.write_text(f"{code}\n\n{tests}")

    cmd = [
        sandbox_exec,
        "-f",
        str(profile_path),
        py,
        "-m",
        "unittest",
        "-v",
        "solution",
    ]
    logger.info("code_sandbox: running work_dir=%s timeout=%.1fs", own_dir, timeout)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(own_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        logger.warning("code_sandbox: timed out after %.1fs", timeout)
        return CodeRewardResult(
            reward=0.0,
            passed=0,
            total=0,
            timed_out=True,
            stdout=stdout,
            stderr=stderr,
            error=f"timed out after {timeout:.1f}s",
        )
    finally:
        if cleanup:
            shutil.rmtree(own_dir, ignore_errors=True)

    # ``python -m unittest`` writes the per-test lines + ``Ran N``
    # summary to STDERR by default (TextTestRunner stream=sys.stderr).
    # Parse the combined output so the markers land regardless of which
    # stream the runner chose.
    combined = f"{stdout}\n{stderr}"
    passed, total = _parse_unittest_results(combined)
    reward = (passed / total) if total > 0 else 0.0
    logger.info(
        "code_sandbox: passed=%d total=%d reward=%.4f stderr_len=%d",
        passed,
        total,
        reward,
        len(stderr),
    )
    return CodeRewardResult(
        reward=reward,
        passed=passed,
        total=total,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
    )
