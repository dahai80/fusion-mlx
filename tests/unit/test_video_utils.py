# SPDX-License-Identifier: Apache-2.0
"""Tests for video processing utilities."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fusion_mlx.utils.video import (
    FRAME_FACTOR,
    MAX_FRAMES,
    MIN_FRAMES,
    FileSizeExceededError,
    TempFileManager,
    ceil_by_factor,
    cleanup_temp_file,
    decode_base64_video,
    describe_video,
    extract_video_frames_smart,
    floor_by_factor,
    is_base64_video,
    is_url,
    process_video_input,
    round_by_factor,
    save_frames_to_temp,
    smart_nframes,
)


class TestSmartNframes:
    def test_short_video_min_frames(self):
        result = smart_nframes(total_frames=10, video_fps=30.0, target_fps=2.0)
        assert result >= MIN_FRAMES
        assert result % FRAME_FACTOR == 0

    def test_long_video_capped(self):
        result = smart_nframes(
            total_frames=10000, video_fps=30.0, target_fps=2.0, max_frames=32
        )
        assert result <= 32
        assert result % FRAME_FACTOR == 0

    def test_exact_duration(self):
        # 60s video at 2fps = 120 frames, capped at MAX_FRAMES=128
        result = smart_nframes(total_frames=1800, video_fps=30.0, target_fps=2.0)
        assert result <= MAX_FRAMES
        assert result % FRAME_FACTOR == 0

    def test_frame_factor_alignment(self):
        for total in (100, 500, 1000, 5000):
            result = smart_nframes(total_frames=total, video_fps=30.0)
            assert result % FRAME_FACTOR == 0

    def test_zero_fps(self):
        result = smart_nframes(total_frames=100, video_fps=0.0)
        assert result >= MIN_FRAMES

    def test_custom_params(self):
        result = smart_nframes(
            total_frames=300,
            video_fps=30.0,
            target_fps=1.0,
            min_frames=2,
            max_frames=16,
        )
        assert result >= 2
        assert result <= 16


class TestRounding:
    def test_round_by_factor(self):
        assert round_by_factor(27, 28) == 28
        assert round_by_factor(42, 28) == 56
        assert round_by_factor(56, 28) == 56

    def test_ceil_by_factor(self):
        assert ceil_by_factor(1, 28) == 28
        assert ceil_by_factor(28, 28) == 28
        assert ceil_by_factor(29, 28) == 56

    def test_floor_by_factor(self):
        assert floor_by_factor(27, 28) == 0
        assert floor_by_factor(28, 28) == 28
        assert floor_by_factor(55, 28) == 28


class TestIsUrl:
    def test_http(self):
        assert is_url("http://example.com/video.mp4")

    def test_https(self):
        assert is_url("https://example.com/video.mp4")

    def test_local_path(self):
        assert not is_url("/tmp/video.mp4")

    def test_base64(self):
        assert not is_url("data:video/mp4;base64,AAAA")


class TestIsBase64Video:
    def test_data_video_prefix(self):
        assert is_base64_video("data:video/mp4;base64,AAAA")

    def test_not_video(self):
        assert not is_base64_video("data:image/png;base64,AAAA")

    def test_url(self):
        assert not is_base64_video("https://example.com/video.mp4")


class TestProcessVideoInput:
    def test_local_path_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            path = f.name
        try:
            result = process_video_input(path)
            assert result == path
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_input(self):
        with pytest.raises(ValueError, match="Empty video input"):
            process_video_input("")

    def test_dict_with_url(self):
        with patch("fusion_mlx.utils.video.download_video") as mock_dl:
            mock_dl.return_value = "/tmp/video.mp4"
            result = process_video_input({"url": "https://example.com/test.mp4"})
            assert result == "/tmp/video.mp4"

    def test_dict_with_video_url(self):
        with patch("fusion_mlx.utils.video.download_video") as mock_dl:
            mock_dl.return_value = "/tmp/video.mp4"
            result = process_video_input(
                {"video_url": {"url": "https://example.com/test.mp4"}}
            )
            assert result == "/tmp/video.mp4"

    def test_base64_video(self):
        with patch("fusion_mlx.utils.video.decode_base64_video") as mock_dec:
            mock_dec.return_value = "/tmp/video.mp4"
            result = process_video_input("data:video/mp4;base64,AAAA")
            assert result == "/tmp/video.mp4"

    def test_unprocessable(self):
        with pytest.raises(ValueError, match="Cannot process video"):
            process_video_input("not_a_real_file_xyz.mp4")


class TestDecodeBase64Video:
    def test_data_uri_format(self):
        import base64

        video_bytes = b"\x00\x00\x00\x20ftypisom"
        b64_data = base64.b64encode(video_bytes).decode()
        data_uri = f"data:video/mp4;base64,{b64_data}"
        result = decode_base64_video(data_uri)
        try:
            assert Path(result).exists()
            assert Path(result).suffix == ".mp4"
            content = Path(result).read_bytes()
            assert content == video_bytes
        finally:
            cleanup_temp_file(result)

    def test_too_large(self):
        with pytest.raises(FileSizeExceededError):
            decode_base64_video("x" * (700 * 1024 * 1024 + 1))


class TestTempFileManager:
    def test_register_and_cleanup(self):
        manager = TempFileManager()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            path = f.name
        manager.register(path)
        assert Path(path).exists()
        manager.cleanup(path)
        assert not Path(path).exists()

    def test_cleanup_nonexistent(self):
        manager = TempFileManager()
        result = manager.cleanup("/tmp/nonexistent_file_xyz")
        assert result is False

    def test_cleanup_all(self):
        manager = TempFileManager()
        paths = []
        for _ in range(3):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"test")
                paths.append(f.name)
                manager.register(f.name)
        cleaned = manager.cleanup_all()
        assert cleaned == 3
        for p in paths:
            assert not Path(p).exists()


class TestSaveFramesToTemp:
    def test_save_frames(self):
        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]
        paths = save_frames_to_temp(frames)
        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists()
            assert Path(p).suffix == ".jpg"
            cleanup_temp_file(p)


class TestDescribeVideo:
    def test_describe_requires_cv2(self):
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError):
                describe_video("/tmp/test.mp4")


class TestExtractVideoFramesSmart:
    def test_requires_cv2(self):
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError):
                extract_video_frames_smart("/tmp/test.mp4")
