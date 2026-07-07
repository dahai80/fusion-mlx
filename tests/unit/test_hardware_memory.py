# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.hardware.memory — RAM + disk detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fusion_mlx.hardware import memory


class TestDetectRamBytes:
    def test_returns_total_from_psutil(self):
        with patch("psutil.virtual_memory") as vm:
            vm.return_value = MagicMock(total=16384)
            assert memory.detect_ram_bytes() == 16384


class TestDetectAvailableRamBytes:
    def test_returns_available_from_psutil(self):
        with patch("psutil.virtual_memory") as vm:
            vm.return_value = MagicMock(available=8192)
            assert memory.detect_available_ram_bytes() == 8192


class TestEstimateUsableRam:
    def test_small_total_uses_min_4gib_reserve(self):
        total = 8 * 1024**3  # 8 GiB; 15% = 1.2 GiB < 4 GiB floor
        assert memory.estimate_usable_ram(total) == total - 4 * 1024**3

    def test_medium_total_uses_15_percent(self):
        total = 32 * 1024**3  # 32 GiB; 15% = 4.8 GiB (between floor/ceiling)
        assert memory.estimate_usable_ram(total) == total - int(total * 0.15)

    def test_large_total_uses_max_32gib_reserve(self):
        total = 256 * 1024**3  # 256 GiB; 15% = 38.4 GiB > 32 GiB cap
        assert memory.estimate_usable_ram(total) == total - 32 * 1024**3

    def test_zero_total_returns_zero(self):
        assert memory.estimate_usable_ram(0) == 0

    def test_total_below_reserve_returns_zero(self):
        assert memory.estimate_usable_ram(2 * 1024**3) == 0  # 2 GiB < 4 GiB reserve


class TestDetectDiskFreeBytes:
    def test_default_path_uses_home(self, tmp_path):
        with (
            patch("os.path.expanduser", return_value=str(tmp_path)),
            patch("shutil.disk_usage") as du,
        ):
            du.return_value = MagicMock(free=99999)
            assert memory.detect_disk_free_bytes() == 99999

    def test_explicit_path(self, tmp_path):
        with patch("shutil.disk_usage") as du:
            du.return_value = MagicMock(free=555)
            assert memory.detect_disk_free_bytes(str(tmp_path)) == 555

    def test_oserror_returns_zero(self, tmp_path):
        with patch("shutil.disk_usage", side_effect=OSError("no disk")):
            assert memory.detect_disk_free_bytes(str(tmp_path)) == 0
