# SPDX-License-Identifier: Apache-2.0
# Tests for #1271 / #1306: the SSD-tier reconstruct cap is env-overridable
# and defaults to unlimited (0). A hardcoded 4096 made a found 49216-token
# agent prefix unusable (full 69k re-prefill, 142s TTFT, client abort loop),
# and a stale 65536 setting rejected a warm 69632-token prefix, forcing a
# 3m56s re-prefill on every Claude Code turn.
import os
import subprocess
import sys

_CAP_ENV = "FUSION_MLX_SSD_RECONSTRUCT_MAX_TOKENS"


class TestSsdReconstructCap:
    def test_default_cap_is_unlimited(self):
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
        assert out.stdout.strip() == "0"

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
        assert out.stdout.strip() == "0"


class TestSsdReconstructCapGate:
    def test_zero_cap_never_exceeds(self):
        from fusion_mlx.scheduler.sched_admission import (
            _ssd_reconstruct_cap_exceeded,
        )

        assert _ssd_reconstruct_cap_exceeded(69632, 0) is False
        assert _ssd_reconstruct_cap_exceeded(1_000_000, 0) is False

    def test_negative_cap_never_exceeds(self):
        from fusion_mlx.scheduler.sched_admission import (
            _ssd_reconstruct_cap_exceeded,
        )

        assert _ssd_reconstruct_cap_exceeded(69632, -1) is False

    def test_positive_cap_still_gates(self):
        from fusion_mlx.scheduler.sched_admission import (
            _ssd_reconstruct_cap_exceeded,
        )

        assert _ssd_reconstruct_cap_exceeded(69632, 65536) is True
        assert _ssd_reconstruct_cap_exceeded(65536, 65536) is False
        assert _ssd_reconstruct_cap_exceeded(69216, 69632) is False
