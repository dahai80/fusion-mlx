# SPDX-License-Identifier: Apache-2.0
# Issue #1298: ?embedded=1 direct-open regression. The embedded/standalone
# decision lives in the TypeScript shell SDK (web/vendor/@fusion/web-sdk);
# bun runs the behavioral spec, this wrapper surfaces it in the pytest suite.
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_DIR = REPO_ROOT / "web" / "vendor" / "@fusion" / "web-sdk"
SPEC = "tests/embedded-param-1298.test.ts"
BUN = shutil.which("bun")


@pytest.mark.skipif(BUN is None, reason="bun not installed")
def test_embedded_param_sdk_regression():
    # Pre-#1298 SDK fails here: ?embedded=1 at top level logged
    # "standalone mode" and wired no host APIs.
    proc = subprocess.run(
        [BUN, "test", SPEC],
        cwd=SDK_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert " 0 fail" in out, out
    passed = re.search(r"(\d+)\s+pass", out)
    assert passed and int(passed.group(1)) >= 5, out
