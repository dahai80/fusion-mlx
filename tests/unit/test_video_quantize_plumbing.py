# SPDX-License-Identifier: Apache-2.0
# Issue #586: VideoGen quantize knob must be reachable through the engine API.
# Tests the full plumbing chain: VideoGenParams field -> VideoGenEngine.generate
# kwargs forwarding -> H3 backend generate() passes quantize=params.quantize
# into generate_video(). Default "none" preserves all other backends.
from pathlib import Path

from fusion_mlx.engines.video_backends.base import VideoGenParams


class TestVideoGenParamsQuantize:
    def test_quantize_field_default_none(self):
        p = VideoGenParams(prompt="test")
        assert p.quantize == "none"

    def test_quantize_field_set(self):
        p = VideoGenParams(prompt="test", quantize="dit8_te4")
        assert p.quantize == "dit8_te4"

    def test_quantize_field_dit8(self):
        p = VideoGenParams(prompt="test", quantize="dit8")
        assert p.quantize == "dit8"


class TestEngineForwardsQuantize:
    def _make_engine(self, monkeypatch, captured):
        # Use the real __init__ (sets up activity tracker) but inject a fake
        # backend via resolve_backend so no model loading occurs.
        from fusion_mlx.engines import video as video_mod

        class FakeBackend:
            name = "fake"
            _loaded = True

            async def start(self, model_path, **kwargs):
                pass

            async def stop(self):
                pass

            async def generate(self, params):
                captured["quantize"] = params.quantize
                return [b"FAKEMP4"]

            def constraints(self):
                from fusion_mlx.engines.video_backends.base import VideoConstraints

                return VideoConstraints()

            def get_stats(self):
                return {}

            def last_denoise_stats(self):
                return {}

        monkeypatch.setattr(video_mod, "resolve_backend", lambda *a, **k: FakeBackend())
        return video_mod.VideoGenEngine("fake-model")

    async def test_generate_forwards_quantize_kwarg_to_params(self, monkeypatch):
        # VideoGenEngine.generate must read quantize from **kwargs and set it
        # on VideoGenParams so the backend can read params.quantize.
        captured = {}
        engine = self._make_engine(monkeypatch, captured)

        await engine.generate(
            prompt="p",
            num_frames=17,
            width=512,
            height=512,
            quantize="dit8_te4",
        )
        assert captured["quantize"] == "dit8_te4"

    async def test_generate_defaults_quantize_none_when_unset(self, monkeypatch):
        captured = {}
        engine = self._make_engine(monkeypatch, captured)

        await engine.generate(
            prompt="p",
            num_frames=17,
            width=512,
            height=512,
        )
        assert captured["quantize"] == "none"


class TestEngineForwardsLastFrameImage:
    # Issue #687: VideoGenEngine.generate must read last_frame_image from
    # **kwargs and set it on VideoGenParams so the H3 backend's l2va/fl2va
    # last-frame keyframe path fires. Without this, engine-layer callers
    # (ComfyUI/SDK) had last_frame_image silently dropped → always first-frame.
    def _make_engine(self, monkeypatch, captured):
        from fusion_mlx.engines import video as video_mod

        class FakeBackend:
            name = "fake"
            _loaded = True

            async def start(self, model_path, **kwargs):
                pass

            async def stop(self):
                pass

            async def generate(self, params):
                captured["last_frame_image"] = params.last_frame_image
                captured["image"] = params.image
                return [b"FAKEMP4"]

            def constraints(self):
                from fusion_mlx.engines.video_backends.base import VideoConstraints

                return VideoConstraints()

            def get_stats(self):
                return {}

            def last_denoise_stats(self):
                return {}

        monkeypatch.setattr(video_mod, "resolve_backend", lambda *a, **k: FakeBackend())
        return video_mod.VideoGenEngine("fake-model")

    async def test_generate_forwards_last_frame_image_kwarg_to_params(
        self, monkeypatch
    ):
        captured = {}
        engine = self._make_engine(monkeypatch, captured)

        await engine.generate(
            prompt="p",
            num_frames=17,
            width=512,
            height=512,
            last_frame_image="/tmp/last.png",
        )
        assert captured["last_frame_image"] == "/tmp/last.png"

    async def test_generate_defaults_last_frame_image_none_when_unset(
        self, monkeypatch
    ):
        captured = {}
        engine = self._make_engine(monkeypatch, captured)

        await engine.generate(
            prompt="p",
            num_frames=17,
            width=512,
            height=512,
        )
        assert captured["last_frame_image"] is None

    async def test_generate_forwards_both_first_and_last_frame(self, monkeypatch):
        # fl2va joint: image (first-frame) + last_frame_image (last-frame)。
        captured = {}
        engine = self._make_engine(monkeypatch, captured)

        await engine.generate(
            prompt="p",
            num_frames=17,
            width=512,
            height=512,
            image="/tmp/first.png",
            last_frame_image="/tmp/last.png",
        )
        assert captured["image"] == "/tmp/first.png"
        assert captured["last_frame_image"] == "/tmp/last.png"


class TestH3BackendPassesQuantize:
    def _make_backend(self, monkeypatch, captured, tmp_path):
        # Stub generate_video (records kwargs) + is_safe_local_path (allow the
        # tmp_path model dir) + pre-set _loaded=True so start() is skipped.
        from fusion_mlx.engines.video_backends import minimax_h3 as h3_mod

        def fake_generate_video(**kwargs):
            captured.update(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"FAKEMP4")
            return None

        monkeypatch.setattr(
            "fusion_mlx.video.minimax_h3.generate.generate_video",
            fake_generate_video,
        )
        monkeypatch.setattr(h3_mod, "is_safe_local_path", lambda p: True)

        b = h3_mod.MiniMaxH3Backend(str(tmp_path / "h3-model"))
        b._loaded = True
        return b

    async def test_backend_generate_passes_quantize_to_generate_video(
        self, monkeypatch, tmp_path
    ):
        # minimax_h3.py backend.generate() must pass quantize=params.quantize
        # into the generate_video(...) call.
        captured = {}
        b = self._make_backend(monkeypatch, captured, tmp_path)
        p = VideoGenParams(
            prompt="test",
            num_frames=49,
            width=768,
            height=448,
            n=1,
            quantize="dit8_te4",
        )
        result = await b.generate(p)
        assert result == [b"FAKEMP4"]
        assert captured["quantize"] == "dit8_te4"

    async def test_backend_generate_defaults_quantize_none(self, monkeypatch, tmp_path):
        captured = {}
        b = self._make_backend(monkeypatch, captured, tmp_path)
        p = VideoGenParams(
            prompt="test",
            num_frames=49,
            width=768,
            height=448,
            n=1,
        )
        await b.generate(p)
        assert captured["quantize"] == "none"
