# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.hardware.cpu — CPU name + core detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fusion_mlx.hardware import cpu


class TestDetectCpuName:
    def test_sysctl_success_returns_stripped_stdout(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Apple M3 Pro\n")
            assert cpu.detect_cpu_name() == "Apple M3 Pro"

    def test_sysctl_success_strips_whitespace(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="  Intel i7  \n")
            assert cpu.detect_cpu_name() == "Intel i7"

    def test_sysctl_empty_stdout_falls_back_to_platform(self):
        with (
            patch("subprocess.run") as mock_run,
            patch("platform.processor", return_value="Fallback CPU"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="   \n")
            assert cpu.detect_cpu_name() == "Fallback CPU"

    def test_sysctl_nonzero_returncode_falls_back_to_platform(self):
        with (
            patch("subprocess.run") as mock_run,
            patch("platform.processor", return_value="Fallback"),
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="error")
            assert cpu.detect_cpu_name() == "Fallback"

    def test_sysctl_filenotfound_falls_back_to_platform(self):
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.processor", return_value="Plat"),
        ):
            assert cpu.detect_cpu_name() == "Plat"

    def test_sysctl_timeout_falls_back_to_platform(self):
        import subprocess as sp

        with (
            patch(
                "subprocess.run", side_effect=sp.TimeoutExpired(cmd="sysctl", timeout=5)
            ),
            patch("platform.processor", return_value="Plat"),
        ):
            assert cpu.detect_cpu_name() == "Plat"

    def test_platform_returns_none_falls_back_to_unknown(self):
        with (
            patch("subprocess.run") as mock_run,
            patch("platform.processor", return_value=None),
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert cpu.detect_cpu_name() == "Unknown"


class TestDetectCpuCores:
    def test_psutil_available_returns_cpu_count(self):
        with patch("psutil.cpu_count", return_value=8):
            assert cpu.detect_cpu_cores() == 8

    def test_psutil_returns_none_falls_to_zero(self):
        with patch("psutil.cpu_count", return_value=None):
            assert cpu.detect_cpu_cores() == 0

    def test_psutil_import_fails_returns_zero(self):
        # Simulate psutil not installed by making import fail
        real_import = (
            __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert cpu.detect_cpu_cores() == 0

    def test_psutil_exception_returns_zero(self):
        with patch("psutil.cpu_count", side_effect=Exception("boom")):
            assert cpu.detect_cpu_cores() == 0
