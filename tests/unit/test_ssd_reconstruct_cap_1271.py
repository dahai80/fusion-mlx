# SPDX-License-Identifier: Apache-2.0
# Tests for #1271: the SSD-tier reconstruct cap must be env-overridable so
# agent deployments can raise it after measuring their own reconstruct vs
# re-prefill costs, and so the cap is measurable at all (a hardcoded 4096
# made a found 49216-token agent prefix unusable — full 69k re-prefill,
# 142s TTFT, client abort loop).
import os
import subprocess
import sys

_CAP_ENV = "FUSION_MLX_SSD_RECONSTRUCT_MAX_TOKENS"


class TestSsdReconstructCap:
    def test_default_cap_unchanged(self):
        code = (
            "import sys; sys.path.insert(0, '.'); "
            "from fusion_mlx.scheduler.sched_admission import "
            "_SSD_RECONSTRUCT_MAX_TOKENS as cap; print(cap)"
        )
        env = {k: v for k, v in os.environ.items() if k != _CAP_ENV}
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "4096"

    def test_env_override_raises_cap(self):
        code = (
            "import sys; sys.path.insert(0, '.'); "
            "from fusion_mlx.scheduler.sched_admission import "
            "_SSD_RECONSTRUCT_MAX_TOKENS as cap; print(cap)"
        )
        env = {k: v for k, v in os.environ.items() if k != _CAP_ENV}
        env[_CAP_ENV] = "100000"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "100000"

    def test_bad_env_value_falls_back_to_default(self):
        code = (
            "import sys; sys.path.insert(0, '.'); "
            "from fusion_mlx.scheduler.sched_admission import "
            "_SSD_RECONSTRUCT_MAX_TOKENS as cap; print(cap)"
        )
        env = {k: v for k, v in os.environ.items() if k != _CAP_ENV}
        env[_CAP_ENV] = "not-a-number"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "4096"
