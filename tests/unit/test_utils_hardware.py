# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.utils.hardware module."""

from unittest.mock import MagicMock, patch

from fusion_mlx.utils.hardware import (
    DEFAULT_MEMORY_BYTES,
    HardwareInfo,
    format_bytes,
    get_chip_name,
    get_max_working_set_bytes,
    get_total_memory_bytes,
    get_total_memory_gb,
    is_apple_silicon,
    is_mlx_available,
)


class TestHardwareInfo:
    def test_hardware_info_creation(self):
        info = HardwareInfo(
            chip_name="Apple M4 Pro",
            total_memory_gb=48.0,
            max_working_set_bytes=36 * 1024**3,
            mlx_device_name="Apple M4 Pro",
        )
        assert info.chip_name == "Apple M4 Pro"
        assert info.total_memory_gb == 48.0
        assert info.max_working_set_bytes == 36 * 1024**3
        assert info.mlx_device_name == "Apple M4 Pro"

    def test_hardware_info_default_mlx_device(self):
        info = HardwareInfo(
            chip_name="Apple M2",
            total_memory_gb=24.0,
            max_working_set_bytes=18 * 1024**3,
        )
        assert info.mlx_device_name is None

    def test_hardware_info_various_chips(self):
        chips = ["Apple M1", "Apple M2 Pro", "Apple M3 Max", "Apple M4 Ultra"]
        for chip in chips:
            info = HardwareInfo(
                chip_name=chip,
                total_memory_gb=16.0,
                max_working_set_bytes=12 * 1024**3,
            )
            assert info.chip_name == chip


class TestGetChipName:
    def test_get_chip_name_success(self):
        mock_result = MagicMock()
        mock_result.stdout = "Apple M4 Pro\n"
        with patch(
            "fusion_mlx.utils.hardware.subprocess.run", return_value=mock_result
        ):
            assert get_chip_name() == "Apple M4 Pro"

    def test_get_chip_name_fallback(self):
        with patch(
            "fusion_mlx.utils.hardware.subprocess.run", side_effect=OSError("no sysctl")
        ):
            assert get_chip_name() == "Apple Silicon"


class TestGetTotalMemoryBytes:
    def test_get_total_memory_bytes_sysctl_success(self):
        mock_result = MagicMock()
        mock_result.stdout = "34359738368\n"  # 32 GB
        with patch(
            "fusion_mlx.utils.hardware.subprocess.run", return_value=mock_result
        ):
            assert get_total_memory_bytes() == 34359738368

    def test_get_total_memory_bytes_default_fallback(self):
        with (
            patch(
                "fusion_mlx.utils.hardware.subprocess.run",
                side_effect=OSError("no sysctl"),
            ),
            patch("fusion_mlx.utils.hardware.HAS_MLX", False),
        ):
            assert get_total_memory_bytes() == DEFAULT_MEMORY_BYTES


class TestGetTotalMemoryGb:
    def test_get_total_memory_gb_conversion(self):
        bytes_val = 16 * 1024**3
        mock_result = MagicMock()
        mock_result.stdout = f"{bytes_val}\n"
        with patch(
            "fusion_mlx.utils.hardware.subprocess.run", return_value=mock_result
        ):
            assert get_total_memory_gb() == 16.0

    def test_get_total_memory_gb_fractional(self):
        bytes_val = 18 * 1024**3
        mock_result = MagicMock()
        mock_result.stdout = f"{bytes_val}\n"
        with patch(
            "fusion_mlx.utils.hardware.subprocess.run", return_value=mock_result
        ):
            assert get_total_memory_gb() == 18.0


class TestGetMaxWorkingSetBytes:
    def test_uses_mlx_max_working_set_when_available(self):
        mock_device_info = {"max_recommended_working_set_size": 27 * 1024**3}
        with patch("fusion_mlx.utils.hardware.HAS_MLX", True):
            with patch("fusion_mlx.utils.hardware.mx") as mock_mx:
                mock_mx.metal.is_available.return_value = True
                mock_mx.device_info.return_value = mock_device_info
                assert get_max_working_set_bytes() == 27 * 1024**3

    def test_falls_back_to_total_memory_without_psutil(self):
        bytes_val = 16 * 1024**3
        mock_result = MagicMock()
        mock_result.stdout = f"{bytes_val}\n"
        with (
            patch("fusion_mlx.utils.hardware.HAS_MLX", False),
            patch("fusion_mlx.utils.hardware.subprocess.run", return_value=mock_result),
        ):
            result = get_max_working_set_bytes()
            assert result == int(bytes_val * 0.75)


class TestIsAppleSilicon:
    def test_is_apple_silicon_true(self):
        with patch("fusion_mlx.utils.hardware.sys") as mock_sys:
            with patch("fusion_mlx.utils.hardware.platform") as mock_platform:
                mock_sys.platform = "darwin"
                mock_platform.machine.return_value = "arm64"
                assert is_apple_silicon() is True

    def test_is_apple_silicon_false_wrong_platform(self):
        with patch("fusion_mlx.utils.hardware.sys") as mock_sys:
            with patch("fusion_mlx.utils.hardware.platform") as mock_platform:
                mock_sys.platform = "linux"
                mock_platform.machine.return_value = "arm64"
                assert is_apple_silicon() is False

    def test_is_apple_silicon_false_wrong_arch(self):
        with patch("fusion_mlx.utils.hardware.sys") as mock_sys:
            with patch("fusion_mlx.utils.hardware.platform") as mock_platform:
                mock_sys.platform = "darwin"
                mock_platform.machine.return_value = "x86_64"
                assert is_apple_silicon() is False


class TestIsMlxAvailable:
    def test_is_mlx_available_not_apple_silicon(self):
        with patch("fusion_mlx.utils.hardware.is_apple_silicon", return_value=False):
            assert is_mlx_available() is False

    def test_is_mlx_available_import_error(self):
        with patch("fusion_mlx.utils.hardware.is_apple_silicon", return_value=True):
            with patch("fusion_mlx.utils.hardware.HAS_MLX", False):
                assert is_mlx_available() is False


class TestFormatBytesHardware:
    def test_format_bytes_gb(self):
        assert format_bytes(16 * 1024**3) == "16.00 GB"

    def test_format_bytes_mb(self):
        assert format_bytes(512 * 1024**2) == "512.00 MB"

    def test_format_bytes_kb(self):
        assert format_bytes(256 * 1024) == "256.00 KB"

    def test_format_bytes_small(self):
        assert format_bytes(42) == "42 B"


class TestDefaultMemoryBytes:
    def test_default_memory_bytes_value(self):
        assert DEFAULT_MEMORY_BYTES == 8 * 1024**3
