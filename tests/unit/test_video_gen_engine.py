# SPDX-License-Identifier: Apache-2.0
# Tests for VideoGenEngine. Stubs the vendored pure-MLX LTX-2 port
# (fusion_mlx.video.ltx2: get_model_path + generate_video) - no real model
# loading or generation. (Phase 4: LTX-2 no longer routes through mlx-video.)
import sys
from pathlib import Path

import pytest

from fusion_mlx.engines.video import VideoGenEngine


def _install_ltx2_port_stub(monkeypatch, *, generate_side_effect=None):
    calls = {"resolve": [], "generate": []}

    from fusion_mlx.video.ltx2 import generate as port_gen
    from fusion_mlx.video.ltx2 import utils as port_utils

    monkeypatch.setattr(
        port_utils,
        "get_model_path",
        lambda repo: calls["resolve"].append(repo) or Path("/tmp/fake-ltx-2"),
    )

    def generate_video(model_repo, text_encoder_repo, prompt, **kwargs):
        calls["generate"].append(
            {
                "model_repo": model_repo,
                "text_encoder_repo": text_encoder_repo,
                "prompt": prompt,
                **kwargs,
            }
        )
        if generate_side_effect is not None:
            generate_side_effect(kwargs)
        else:
            with open(kwargs["output_path"], "wb") as f:
                f.write(b"FAKEMP4" + str(kwargs.get("seed", 0)).encode())
        return None

    monkeypatch.setattr(port_gen, "generate_video", generate_video)
    return calls


@pytest.fixture
def stub(monkeypatch):
    return _install_ltx2_port_stub(monkeypatch)


class TestStart:
    async def test_start_resolves_model_path(self, stub):
        engine = VideoGenEngine("ltx-video/ltx-2-13b-distilled")
        await engine.start()
        assert engine._loaded is True
        assert stub["resolve"] == ["ltx-video/ltx-2-13b-distilled"]

    async def test_start_idempotent(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        await engine.start()
        assert len(stub["resolve"]) == 1

    async def test_start_works_without_mlx_video(self, stub, monkeypatch):
        # Phase 4: LTX-2 runs on the vendored pure-MLX port, so start() must
        # succeed even with mlx_video absent (no hard mlx-video dependency).
        monkeypatch.setitem(sys.modules, "mlx_video", None)
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        assert engine._loaded is True


class TestGenerate:
    async def test_seed_increment(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        await engine.generate(
            prompt="p", n=3, seed=42, num_frames=17, width=512, height=512
        )
        seeds = [c["seed"] for c in stub["generate"]]
        assert seeds == [42, 43, 44]

    async def test_seed_none_is_random_not_zero(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        await engine.generate(
            prompt="p", n=2, seed=None, num_frames=17, width=512, height=512
        )
        seeds = [c["seed"] for c in stub["generate"]]
        assert all(s != 0 for s in seeds)
        assert len(set(seeds)) == 2

    async def test_returns_bytes_written_by_generate(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        result = await engine.generate(
            prompt="p", n=1, seed=7, num_frames=17, width=512, height=512
        )
        assert len(result) == 1
        assert result[0].startswith(b"FAKEMP4")

    async def test_temp_file_unlinked_after_generate(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        await engine.generate(
            prompt="p", n=1, seed=1, num_frames=17, width=512, height=512
        )
        out_path = Path(stub["generate"][0]["output_path"])
        assert not out_path.exists()

    async def test_pipeline_passed_through(self, stub):
        engine = VideoGenEngine("ltx-2", pipeline="dev")
        await engine.start()
        await engine.generate(
            prompt="p", n=1, seed=1, num_frames=17, width=512, height=512
        )
        assert stub["generate"][0]["pipeline"].value == "dev"

    async def test_text_encoder_repo_passed_through(self, stub):
        engine = VideoGenEngine("ltx-2", text_encoder_repo="google/gemma-3-12b-it")
        await engine.start()
        await engine.generate(
            prompt="p", n=1, seed=1, num_frames=17, width=512, height=512
        )
        assert stub["generate"][0]["text_encoder_repo"] == "google/gemma-3-12b-it"

    async def test_generate_before_start_raises(self, stub):
        engine = VideoGenEngine("ltx-2")
        with pytest.raises(RuntimeError, match="not started"):
            await engine.generate(prompt="p")

    async def test_generate_calls_generate_video_per_n(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        await engine.generate(
            prompt="p", n=4, seed=100, num_frames=17, width=512, height=512
        )
        assert len(stub["generate"]) == 4


class TestStop:
    async def test_stop_clears_loaded(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        assert engine._loaded is True
        await engine.stop()
        assert engine._loaded is False

    async def test_stop_when_not_loaded_is_noop(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.stop()
        assert engine._loaded is False


class TestGetStats:
    def test_get_stats_not_loaded(self, stub):
        engine = VideoGenEngine("ltx-2")
        stats = engine.get_stats()
        assert stats == {"model_name": "ltx-2", "loaded": False}

    async def test_get_stats_loaded(self, stub):
        engine = VideoGenEngine("ltx-2")
        await engine.start()
        assert engine.get_stats()["loaded"] is True
