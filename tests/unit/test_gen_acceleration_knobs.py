# SPDX-License-Identifier: Apache-2.0
# Verifies the diffusion-acceleration knobs (num_inference_steps / cfg_scale /
# tiling / no_compile / scheduler / negative_prompt / quantize) actually reach
# the backend generate calls. Stubs the pure-MLX wan2/ltx2 ports + mflux so no
# weights/compute are needed - this is a wiring test, not a quality/perf test.
import sys
import types
from pathlib import Path

import pytest

from fusion_mlx.engines.video_backends import LTX2Backend, VideoGenParams, Wan2Backend


def _install_wan2_stub(monkeypatch):
    # Phase 5: Wan2 runs on the vendored pure-MLX port (fusion_mlx.video.wan2).
    calls = {"generate": []}

    from fusion_mlx.video.wan2 import generate as port_gen
    from fusion_mlx.video.wan2 import utils as port_utils

    monkeypatch.setattr(
        port_utils, "get_model_path", lambda repo: Path("/tmp/fake-wan2")
    )

    def generate_video(model_dir, prompt, **kwargs):
        calls["generate"].append({"model_dir": model_dir, "prompt": prompt, **kwargs})
        with open(kwargs["output_path"], "wb") as f:
            f.write(b"WANMP4")

    monkeypatch.setattr(port_gen, "generate_video", generate_video)
    return calls


def _install_ltx2_stub(monkeypatch):
    # Phase 4: LTX-2 runs on the vendored pure-MLX port (fusion_mlx.video.ltx2).
    # Stub the port's get_model_path + generate_video (PipelineType stays the
    # real port enum) - no real 19B weights or model loading.
    calls = {"generate": []}

    from fusion_mlx.video.ltx2 import generate as port_gen
    from fusion_mlx.video.ltx2 import utils as port_utils

    monkeypatch.setattr(
        port_utils, "get_model_path", lambda repo: Path("/tmp/fake-ltx2")
    )

    def generate_video(model_repo, text_encoder_repo, prompt, **kwargs):
        calls["generate"].append({"model_repo": model_repo, "prompt": prompt, **kwargs})
        with open(kwargs["output_path"], "wb") as f:
            f.write(b"LTXMP4")

    monkeypatch.setattr(port_gen, "generate_video", generate_video)
    return calls


class TestLTX2KnobFlow:
    @pytest.fixture
    def stub(self, monkeypatch):
        return _install_ltx2_stub(monkeypatch)

    async def test_steps_cfg_tiling_forwarded(self, stub):
        backend = LTX2Backend("ltx-2")
        await backend.start("ltx-2")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=97,
            width=768,
            height=512,
            num_inference_steps=10,
            cfg_scale=3.0,
            tiling="auto",
        )
        await backend.generate(params)
        call = stub["generate"][0]
        assert call["num_inference_steps"] == 10
        assert call["cfg_scale"] == 3.0
        assert call["tiling"] == "auto"

    async def test_steps_omitted_when_none(self, stub):
        # Regression guard: before the fix, steps/cfg were NEVER passed (hardcoded
        # to mlx-video's 40-step default). Now they pass only when set.
        backend = LTX2Backend("ltx-2")
        await backend.start("ltx-2")
        params = VideoGenParams(prompt="a cat", num_frames=97, width=768, height=512)
        await backend.generate(params)
        call = stub["generate"][0]
        assert "num_inference_steps" not in call
        assert "cfg_scale" not in call

    async def test_enhance_prompt_forwarded(self, stub):
        backend = LTX2Backend("ltx-2")
        await backend.start("ltx-2")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=97,
            width=768,
            height=512,
            enhance_prompt=True,
        )
        await backend.generate(params)
        assert stub["generate"][0]["enhance_prompt"] is True


class TestWan2KnobFlow:
    @pytest.fixture
    def stub(self, monkeypatch):
        return _install_wan2_stub(monkeypatch)

    async def test_no_compile_opt_in_false(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        params = VideoGenParams(
            prompt="a cat",
            num_frames=41,
            width=768,
            height=512,
            no_compile=False,
            tiling="none",
        )
        await backend.generate(params)
        call = stub["generate"][0]
        assert call["no_compile"] is False
        assert call["tiling"] == "none"

    async def test_no_compile_defaults_true_when_none(self, stub):
        # Safe default: compile OFF unless the caller explicitly opts in.
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        params = VideoGenParams(prompt="a cat", num_frames=41, width=768, height=512)
        await backend.generate(params)
        assert stub["generate"][0]["no_compile"] is True

    async def test_tiling_omitted_when_none(self, stub):
        backend = Wan2Backend("wan2.1")
        await backend.start("wan2.1")
        params = VideoGenParams(prompt="a cat", num_frames=41, width=768, height=512)
        await backend.generate(params)
        assert "tiling" not in stub["generate"][0]


class TestImageGenKnobFlow:
    def _make_engine(self, monkeypatch, capture):
        from fusion_mlx.engines import image_gen as mod

        class FakeImage:
            def __init__(self):
                from types import SimpleNamespace

                self.image = SimpleNamespace(save=lambda buf, format=None: None)

        class FakeFlux:
            def __init__(self, *args, **kwargs):
                capture["init_kwargs"] = kwargs

            def generate_image(self, **kwargs):
                capture["generate_calls"].append(kwargs)
                return FakeImage()

        monkeypatch.setattr(mod, "_infer_model_config_label", lambda p: "schnell")

        fake_cfg_mod = types.ModuleType("mflux.models.common.config.model_config")

        class ModelConfig:
            @staticmethod
            def schnell():
                return "schnell_cfg"

            @staticmethod
            def dev():
                return "dev_cfg"

        fake_cfg_mod.ModelConfig = ModelConfig
        fake_flux_mod = types.ModuleType("mflux.models.flux.variants.txt2img.flux")
        fake_flux_mod.Flux1 = FakeFlux
        fake_mflux = types.ModuleType("mflux")
        fake_mflux_pkg_models = types.ModuleType("mflux.models")
        fake_mflux_pkg_common = types.ModuleType("mflux.models.common")
        monkeypatch.setitem(sys.modules, "mflux", fake_mflux)
        monkeypatch.setitem(sys.modules, "mflux.models", fake_mflux_pkg_models)
        monkeypatch.setitem(sys.modules, "mflux.models.common", fake_mflux_pkg_common)
        monkeypatch.setitem(
            sys.modules,
            "mflux.models.common.config.model_config",
            fake_cfg_mod,
        )
        monkeypatch.setitem(
            sys.modules,
            "mflux.models.flux.variants.txt2img.flux",
            fake_flux_mod,
        )
        engine = mod.ImageGenEngine("flux-schnell", quantize=4)
        return engine, capture

    async def test_quantize_passed_at_load(self, monkeypatch):
        capture = {"init_kwargs": None, "generate_calls": []}
        engine, _ = self._make_engine(monkeypatch, capture)
        await engine.start()
        assert capture["init_kwargs"].get("quantize") == 4

    async def test_scheduler_and_negative_prompt_forwarded(self, monkeypatch):
        capture = {"init_kwargs": None, "generate_calls": []}
        engine, _ = self._make_engine(monkeypatch, capture)
        await engine.start()
        await engine.generate(
            prompt="a cat",
            steps=4,
            scheduler="sdrm",
            negative_prompt="blurry",
        )
        call = capture["generate_calls"][0]
        assert call["scheduler"] == "sdrm"
        assert call["negative_prompt"] == "blurry"

    async def test_scheduler_omitted_when_none(self, monkeypatch):
        capture = {"init_kwargs": None, "generate_calls": []}
        engine, _ = self._make_engine(monkeypatch, capture)
        await engine.start()
        await engine.generate(prompt="a cat", steps=4)
        call = capture["generate_calls"][0]
        assert "scheduler" not in call
        assert "negative_prompt" not in call


class TestVideoGenParamsCarriesNewKnobs:
    def test_defaults_none(self):
        p = VideoGenParams(prompt="x")
        assert p.tiling is None
        assert p.no_compile is None
        assert p.enhance_prompt is None

    def test_set_round_trip(self):
        p = VideoGenParams(
            prompt="x", tiling="auto", no_compile=False, enhance_prompt=True
        )
        assert p.tiling == "auto"
        assert p.no_compile is False
        assert p.enhance_prompt is True
