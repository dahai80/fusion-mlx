# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.hardware.detector — orchestrator."""

from __future__ import annotations

from unittest.mock import patch

from fusion_mlx.hardware import detector
from fusion_mlx.hardware.types import HardwareInfo


class TestDetectHardware:
    def test_returns_hardware_info(self):
        with (
            patch.object(detector, "detect_apple_gpu", return_value=[]),
            patch.object(detector, "detect_cpu_name", return_value="CPU"),
            patch.object(detector, "detect_cpu_cores", return_value=8),
            patch.object(detector, "detect_ram_bytes", return_value=100),
            patch.object(detector, "detect_disk_free_bytes", return_value=200),
        ):
            info = detector.detect_hardware()
            assert isinstance(info, HardwareInfo)
            assert info.cpu_name == "CPU"
            assert info.cpu_cores == 8
            assert info.ram_bytes == 100
            assert info.disk_free_bytes == 200
            assert info.os == "darwin"

    def test_unknown_os_normalized_to_darwin(self):
        with (
            patch("platform.system", return_value="Plan9"),
            patch.object(detector, "detect_apple_gpu", return_value=[]),
            patch.object(detector, "detect_cpu_name", return_value="C"),
            patch.object(detector, "detect_cpu_cores", return_value=0),
            patch.object(detector, "detect_ram_bytes", return_value=0),
            patch.object(detector, "detect_disk_free_bytes", return_value=0),
        ):
            info = detector.detect_hardware()
            assert info.os == "darwin"

    def test_linux_os_skips_apple_gpu(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch.object(detector, "detect_apple_gpu", return_value=[]) as gpu_mock,
            patch.object(detector, "detect_cpu_name", return_value="C"),
            patch.object(detector, "detect_cpu_cores", return_value=4),
            patch.object(detector, "detect_ram_bytes", return_value=0),
            patch.object(detector, "detect_disk_free_bytes", return_value=0),
        ):
            info = detector.detect_hardware()
            assert info.os == "linux"
            assert info.gpus == []
            gpu_mock.assert_not_called()

    def test_windows_os_keeps_windows(self):
        with (
            patch("platform.system", return_value="Windows"),
            patch.object(detector, "detect_apple_gpu", return_value=[]),
            patch.object(detector, "detect_cpu_name", return_value="C"),
            patch.object(detector, "detect_cpu_cores", return_value=4),
            patch.object(detector, "detect_ram_bytes", return_value=0),
            patch.object(detector, "detect_disk_free_bytes", return_value=0),
        ):
            info = detector.detect_hardware()
            assert info.os == "windows"
