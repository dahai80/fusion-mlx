# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx.utils.formatting module."""

from fusion_mlx.utils.formatting import format_bytes


class TestFormatBytes:
    """Test cases for format_bytes function."""

    def test_format_gigabytes(self):
        assert format_bytes(1024**3) == "1.00 GB"
        assert format_bytes(2 * 1024**3) == "2.00 GB"
        assert format_bytes(16 * 1024**3) == "16.00 GB"

    def test_format_gigabytes_with_decimals(self):
        assert format_bytes(int(1.5 * 1024**3)) == "1.50 GB"
        assert format_bytes(int(2.75 * 1024**3)) == "2.75 GB"

    def test_format_megabytes(self):
        assert format_bytes(1024**2) == "1.00 MB"
        assert format_bytes(256 * 1024**2) == "256.00 MB"
        assert format_bytes(512 * 1024**2) == "512.00 MB"

    def test_format_megabytes_with_decimals(self):
        assert format_bytes(int(1.5 * 1024**2)) == "1.50 MB"
        assert format_bytes(int(100.25 * 1024**2)) == "100.25 MB"

    def test_format_kilobytes(self):
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(512 * 1024) == "512.00 KB"

    def test_format_kilobytes_with_decimals(self):
        assert format_bytes(int(1.5 * 1024)) == "1.50 KB"

    def test_format_bytes_small(self):
        assert format_bytes(0) == "0 B"
        assert format_bytes(1) == "1 B"
        assert format_bytes(512) == "512 B"
        assert format_bytes(1023) == "1023 B"

    def test_boundary_values(self):
        assert format_bytes(1023) == "1023 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024**2 - 1) == "1024.00 KB"
        assert format_bytes(1024**2) == "1.00 MB"
        assert format_bytes(1024**3 - 1) == "1024.00 MB"
        assert format_bytes(1024**3) == "1.00 GB"

    def test_large_values(self):
        assert format_bytes(1024**4) == "1024.00 GB"
        assert format_bytes(2 * 1024**4) == "2048.00 GB"

    def test_realistic_memory_sizes(self):
        assert format_bytes(8 * 1024**3) == "8.00 GB"
        assert format_bytes(16 * 1024**3) == "16.00 GB"
        assert format_bytes(32 * 1024**3) == "32.00 GB"
        assert format_bytes(64 * 1024**3) == "64.00 GB"
        assert format_bytes(128 * 1024**3) == "128.00 GB"
        assert format_bytes(192 * 1024**3) == "192.00 GB"
