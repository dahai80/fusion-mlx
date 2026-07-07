# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx._mlx_compat — M5 single-stream shim install().

install() is idempotent and a no-op when mlx.core can't be imported (Linux CI
/no mlx). The patched wrapper probes each device once and caches True/False.
Cover the no-mlx import path + idempotency guard directly; the probe branch
needs real mlx runtime so skip it when mlx is absent. Aims at lifting
_mlx_compat.py off 0% on the import-guard paths.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# conftest_stubs may shadow fusion_mlx._mlx_compat; load the real source.
_spec = importlib.util.spec_from_file_location(
    "fusion_mlx._mlx_compat_real",
    Path(__file__).resolve().parents[2] / "fusion_mlx" / "_mlx_compat.py",
)
_mlx_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mlx_compat)


class TestInstallNoMlx:
    def test_install_returns_none_when_mlx_absent(self):
        # mlx is absent in this venv → install() is a no-op returning None
        assert _mlx_compat.install() is None


class TestInstallIdempotent:
    def test_install_safe_to_call_multiple_times(self):
        for _ in range(5):
            assert _mlx_compat.install() is None  # no-mlx path each time


class TestLogger:
    def test_logger_defined(self):
        assert _mlx_compat.logger is not None
        # name may be fusion_mlx._mlx_compat or fusion_mlx._mlx_compat_real
        assert "_mlx_compat" in _mlx_compat.logger.name
