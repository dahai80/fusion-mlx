# SPDX-License-Identifier: Apache-2.0
# Tests for VideoGenEngine (mlx-video wrapper). mlx_video is mocked via
# sys.modules - no real mlx-video install or model loading required.

import sys
import types
from unittest.mock import MagicMock

import pytest

from fusion_mlx.engines.video import VideoGenEngine, _generate_one


def _install_mlx_video_stub(monkeypatch, generate_bytes=b"FAKE_MP4_DATA"):
    stub = types.ModuleType("mlx_video")
    load_calls = []
    gen_calls = []

    def _load(path):
        load_calls.append(path)
        return (MagicMock(name="vmodel"), MagicMock(name="vproc"))

    def _generate(
        model,
        processor,
        *,
        prompt,
        num_frames,
        height,
        width,
        fps,
        seed,
        output_path,
    ):
        gen_calls.append(
            {
                "prompt": prompt,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": fps,
                "seed": seed,
                "output_path": output_path,
            }
        )
        with open(output_path, "wb") as f:
            f.write(generate_bytes)

    stub.load = _load
    stub.generate = _generate
    stub._load_calls = load_calls
    stub._gen_calls = gen_calls
    monkeypatch.setitem(sys.modules, "mlx_video", stub)
    return stub


def _remove_mlx_video(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx_video", raising=False)


class TestVideoGenEngineStart:
    def test_repr_before_start_shows_stopped(self):
        engine = VideoGenEngine(model_name="ltx-2")
        assert "stopped" in repr(engine)
        assert "ltx-2" in repr(engine)

    @pytest.mark.asyncio
    async def test_start_loads_model_and_processor(self, monkeypatch):
        stub = _install_mlx_video_stub(monkeypatch)
        engine = VideoGenEngine(model_name="/models/ltx-2")
        await engine.start()
        assert stub._load_calls == ["/models/ltx-2"]
        assert engine._model is not None
        assert engine._processor is not None
        stats = engine.get_stats()
        assert stats["loaded"] is True
        assert stats["model_name"] == "/models/ltx-2"
        assert "running" in repr(engine)

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, monkeypatch):
        stub = _install_mlx_video_stub(monkeypatch)
        engine = VideoGenEngine(model_name="ltx-2")
        await engine.start()
        await engine.start()
        assert len(stub._load_calls) == 1

    @pytest.mark.asyncio
    async def test_start_raises_clear_error_when_mlx_video_missing(self, monkeypatch):
        _remove_mlx_video(monkeypatch)
        engine = VideoGenEngine(model_name="ltx-2")
        with pytest.raises(ImportError) as exc_info:
            await engine.start()
        assert "mlx-video" in str(exc_info.value)
        assert "pip install" in str(exc_info.value)


class TestVideoGenEngineGenerate:
    @pytest.mark.asyncio
    async def test_generate_not_started_raises_runtime_error(self):
        engine = VideoGenEngine(model_name="ltx-2")
        with pytest.raises(RuntimeError, match="not started"):
            await engine.generate(prompt="a cat")

    @pytest.mark.asyncio
    async def test_generate_returns_one_video_default(self, monkeypatch):
        _install_mlx_video_stub(monkeypatch, generate_bytes=b"MP4_AAAA")
        engine = VideoGenEngine(model_name="ltx-2")
        await engine.start()
        result = await engine.generate(prompt="a cat playing piano")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == b"MP4_AAAA"

    @pytest.mark.asyncio
    async def test_generate_n_increments_seed(self, monkeypatch):
        stub = _install_mlx_video_stub(monkeypatch)
        engine = VideoGenEngine(model_name="ltx-2")
        await engine.start()
        result = await engine.generate(prompt="a dog", n=3, seed=42)
        assert len(result) == 3
        seeds = [c["seed"] for c in stub._gen_calls]
        assert seeds == [42, 43, 44]

    @pytest.mark.asyncio
    async def test_generate_passes_dimensions_and_fps(self, monkeypatch):
        stub = _install_mlx_video_stub(monkeypatch)
        engine = VideoGenEngine(model_name="ltx-2")
        await engine.start()
        await engine.generate(
            prompt="sunset",
            num_frames=16,
            width=512,
            height=512,
            fps=12,
            seed=7,
        )
        call = stub._gen_calls[0]
        assert call["num_frames"] == 16
        assert call["width"] == 512
        assert call["height"] == 512
        assert call["fps"] == 12
        assert call["prompt"] == "sunset"


class TestGenerateOne:
    def test_reads_mp4_and_unlinks_temp(self, monkeypatch):
        stub = _install_mlx_video_stub(monkeypatch, generate_bytes=b"BINARY_MP4")
        model = MagicMock()
        processor = MagicMock()
        out = _generate_one(
            model,
            processor,
            prompt="p",
            num_frames=8,
            width=256,
            height=256,
            fps=10,
            seed=1,
        )
        assert out == b"BINARY_MP4"
        assert len(stub._gen_calls) == 1
        # temp file must be deleted after read
        written_path = stub._gen_calls[0]["output_path"]
        import os

        assert not os.path.exists(written_path)


class TestVideoGenEngineStop:
    @pytest.mark.asyncio
    async def test_stop_nulls_model_and_processor(self, monkeypatch):
        _install_mlx_video_stub(monkeypatch)
        engine = VideoGenEngine(model_name="ltx-2")
        await engine.start()
        assert engine._model is not None
        await engine.stop()
        assert engine._model is None
        assert engine._processor is None
        assert engine.get_stats()["loaded"] is False
        assert "stopped" in repr(engine)

    @pytest.mark.asyncio
    async def test_stop_when_not_started_is_noop(self):
        engine = VideoGenEngine(model_name="ltx-2")
        await engine.stop()
        assert engine._model is None
