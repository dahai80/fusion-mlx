# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.hardware module — migrated to fusion-mlx.

NOTE: fusion-mlx restructured hardware into a top-level package
``fusion_mlx.hardware`` (not ``fusion_mlx.utils.hardware``).
The API surface is entirely different (detect_hardware(), GPUInfo,
HardwareInfo with different fields). All original omlx test classes
are skipped.
"""

import pytest

pytestmark = pytest.mark.skip(reason="omlx-only: omlx.utils.hardware has no fusion-mlx equivalent")


class TestHardwareInfo:
    def test_hardware_info_creation(self):
        pass

    def test_hardware_info_default_mlx_device(self):
        pass

    def test_hardware_info_various_chips(self):
        pass


class TestGetChipName:
    def test_get_chip_name_success(self):
        pass

    def test_get_chip_name_fallback(self):
        pass


class TestGetTotalMemoryBytes:
    def test_get_total_memory_bytes_sysctl_success(self):
        pass

    def test_get_total_memory_bytes_default_fallback(self):
        pass


class TestGetTotalMemoryGb:
    def test_get_total_memory_gb_conversion(self):
        pass

    def test_get_total_memory_gb_fractional(self):
        pass


class TestGetMaxWorkingSetBytes:
    def test_uses_mlx_max_working_set_when_available(self):
        pass

    def test_falls_back_to_total_memory_without_psutil(self):
        pass


class TestIsAppleSilicon:
    def test_is_apple_silicon_true(self):
        pass

    def test_is_apple_silicon_false_wrong_platform(self):
        pass

    def test_is_apple_silicon_false_wrong_arch(self):
        pass


class TestIsMlxAvailable:
    def test_is_mlx_available_not_apple_silicon(self):
        pass

    def test_is_mlx_available_import_error(self):
        pass


class TestFormatBytesHardware:
    def test_format_bytes_gb(self):
        pass

    def test_format_bytes_mb(self):
        pass

    def test_format_bytes_kb(self):
        pass

    def test_format_bytes_small(self):
        pass


class TestDefaultMemoryBytes:
    def test_default_memory_bytes_value(self):
        pass
