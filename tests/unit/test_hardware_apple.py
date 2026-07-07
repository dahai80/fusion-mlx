# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.hardware.apple — Apple Silicon GPU detection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fusion_mlx.hardware import apple


class TestLookupBandwidth:
    def test_exact_m1(self):
        assert apple._lookup_bandwidth("M1") == 68.25

    def test_m1_max(self):
        assert apple._lookup_bandwidth("M1 Max") == 400.0

    def test_m4_pro(self):
        assert apple._lookup_bandwidth("M4 Pro") == 273.0

    def test_case_insensitive(self):
        assert apple._lookup_bandwidth("m1 max") == 400.0

    def test_chip_in_longer_string(self):
        assert apple._lookup_bandwidth("Apple M3 Pro (16-core)") == 150.0

    def test_unknown_chip_returns_none(self):
        assert apple._lookup_bandwidth("Intel i9") is None

    def test_empty_string_returns_none(self):
        assert apple._lookup_bandwidth("") is None


class TestDetectAppleGpu:
    def test_system_profiler_success(self):
        fake_json = json.dumps(
            {
                "SPHardwareDataType": [
                    {
                        "chip_type": "Apple M3 Pro",
                        "physical_memory": "18 GB",
                    }
                ]
            }
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_json)
            gpus = apple.detect_apple_gpu()
            assert len(gpus) == 1
            g = gpus[0]
            assert g.name == "Apple M3 Pro"
            assert g.vendor == "apple"
            assert g.vram_bytes == 18 * 1024**3
            assert g.memory_bandwidth_gbps == 150.0
            assert g.shared_memory is True

    def test_tb_unit_multiplier(self):
        fake_json = json.dumps(
            {
                "SPHardwareDataType": [
                    {"chip_type": "M2 Ultra", "physical_memory": "1 TB"}
                ]
            }
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_json)
            gpus = apple.detect_apple_gpu()
            assert gpus[0].vram_bytes == 1 * 1024**4

    def test_mb_unit_multiplier(self):
        fake_json = json.dumps(
            {"SPHardwareDataType": [{"chip_type": "M1", "physical_memory": "512 MB"}]}
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_json)
            gpus = apple.detect_apple_gpu()
            assert gpus[0].vram_bytes == 512 * 1024**2

    def test_unknown_unit_defaults_to_gb(self):
        fake_json = json.dumps(
            {"SPHardwareDataType": [{"chip_type": "M1", "physical_memory": "16 XB"}]}
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_json)
            gpus = apple.detect_apple_gpu()
            assert gpus[0].vram_bytes == 16 * 1024**3

    def test_nonzero_returncode_returns_empty(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert apple.detect_apple_gpu() == []

    def test_filenotfound_returns_empty(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert apple.detect_apple_gpu() == []

    def test_timeout_returns_empty(self):
        import subprocess as sp

        with patch(
            "subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="system_profiler", timeout=10),
        ):
            assert apple.detect_apple_gpu() == []

    def test_json_decode_error_returns_empty(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not json{")
            assert apple.detect_apple_gpu() == []

    def test_empty_chip_type_returns_empty(self):
        fake_json = json.dumps(
            {"SPHardwareDataType": [{"chip_type": "", "physical_memory": "16 GB"}]}
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_json)
            assert apple.detect_apple_gpu() == []

    def test_missing_sphardwaredatatype_returns_empty(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({}))
            assert apple.detect_apple_gpu() == []

    def test_malformed_memory_string_returns_empty(self):
        fake_json = json.dumps(
            {
                "SPHardwareDataType": [
                    {"chip_type": "M1", "physical_memory": "not a number GB"}
                ]
            }
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_json)
            assert apple.detect_apple_gpu() == []

    def test_unknown_chip_bandwidth_none(self):
        fake_json = json.dumps(
            {
                "SPHardwareDataType": [
                    {"chip_type": "Future Chip X", "physical_memory": "16 GB"}
                ]
            }
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_json)
            gpus = apple.detect_apple_gpu()
            assert gpus[0].memory_bandwidth_gbps is None
