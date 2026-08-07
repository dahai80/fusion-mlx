# Tests for #398: standalone deployments reject all /v1/* calls.
# start.sh (the standalone launcher) must default FUSION_ROUTE_WARN_ONLY=true
# so local /v1/* calls are not rejected by route_guard middleware (which
# enforces X-Fusion-Route by default since v0.7.0). The fix lives in the
# preflight() function of start.sh.
#
# Importers/callers: start.sh preflight() is run by do_start -> launch.
# Affected API: none (shell env). Schema: env var FUSION_ROUTE_WARN_ONLY
# ("true"/"false"); unset -> preflight defaults to "true" for standalone.
#
# Strategy: extract the preflight() function body from start.sh via sed
# (source only that function, not the top-level case dispatch), stub the
# helper functions it calls (resolve_hf_mirror, resolve_api_key, log_info,
# mkdir), invoke preflight, and assert the exported env var.

import os
import subprocess

_START_SH = os.path.join(os.path.dirname(__file__), "..", "..", "start.sh")


def _run_preflight(env_overrides: dict | None = None) -> dict:
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    env.update(env_overrides or {})
    # Inline bash: stub helpers, extract+source only preflight(), call it,
    # then print the resulting env var. Avoids executing start.sh main().
    script = r"""
set -euo pipefail
PORT=12345
LOG_DIR=/tmp
HF_MIRROR=https://hf-mirror.com
API_KEY=
# Stub every helper preflight() calls, so we can source it in isolation.
log_step() { :; }
log_info() { :; }
log_warn() { :; }
log_error() { :; }
ensure_venv() { :; }
resolve_hf_mirror() { :; }
resolve_api_key() { :; }
mkdir() { :; }
# Extract preflight() function definition only (up to the closing '}').
awk '/^preflight\(\)/{f=1} f{print} f&&/^}/{exit}' "$1" > /tmp/_pf_398.sh
source /tmp/_pf_398.sh
preflight
echo "WARN=${FUSION_ROUTE_WARN_ONLY:-UNSET}"
rm -f /tmp/_pf_398.sh
"""
    proc = subprocess.run(
        ["bash", "-c", script, "_", _START_SH],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(f"preflight failed: {proc.stderr}")
    line = [l for l in proc.stdout.splitlines() if l.startswith("WARN=")][0]
    return {"WARN": line.split("=", 1)[1]}


def test_preflight_defaults_warn_only_when_unset():
    result = _run_preflight({})  # FUSION_ROUTE_WARN_ONLY unset
    assert result["WARN"] == "true"


def test_preflight_respects_explicit_true():
    result = _run_preflight({"FUSION_ROUTE_WARN_ONLY": "true"})
    assert result["WARN"] == "true"


def test_preflight_respects_explicit_false_for_gateway():
    # Gateway deployments override with =false; preflight must NOT clobber.
    result = _run_preflight({"FUSION_ROUTE_WARN_ONLY": "false"})
    assert result["WARN"] == "false"


def test_preflight_does_not_overwrite_existing_value():
    # Any pre-set value (even non-bool) is preserved; standalone default
    # only applies when the var is unset.
    result = _run_preflight({"FUSION_ROUTE_WARN_ONLY": "enforce"})
    assert result["WARN"] == "enforce"
