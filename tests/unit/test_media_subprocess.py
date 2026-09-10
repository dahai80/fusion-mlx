# SPDX-License-Identifier: Apache-2.0
# Tests for S3 image gen subprocess isolation: gen_kwargs builder +
# MediaJobManager spec building. Worker end-to-end needs real model
# (real_model marker).
import os

import pytest

from fusion_mlx.engines._gen_kwargs import build_gen_kwargs
from fusion_mlx.media.job_manager import MediaJobManager


class TestBuildGenKwargs:
    def test_txt2img_basic(self):
        gk = build_gen_kwargs(
            variant="txt2img",
            seed=42,
            prompt="a cat",
            steps=4,
            height=1024,
            width=1024,
            guidance=1.0,
        )
        assert gk["seed"] == 42
        assert gk["prompt"] == "a cat"
        assert gk["num_inference_steps"] == 4
        assert gk["height"] == 1024
        assert gk["width"] == 1024
        assert gk["guidance"] == 1.0
        assert "negative_prompt" not in gk

    def test_qwen_image_negative_prompt(self):
        gk = build_gen_kwargs(
            variant="qwen_image",
            seed=0,
            prompt="test",
            steps=30,
            height=1024,
            width=1024,
            guidance=4.0,
            negative_prompt="blurry",
        )
        assert gk["negative_prompt"] == "blurry"

    def test_controlnet_requires_control_image(self):
        with pytest.raises(ValueError, match="requires control_image"):
            build_gen_kwargs(
                variant="controlnet_canny",
                seed=0,
                prompt="test",
                steps=28,
                height=1024,
                width=1024,
                guidance=4.0,
            )

    def test_controlnet_with_image(self):
        gk = build_gen_kwargs(
            variant="controlnet_canny",
            seed=0,
            prompt="test",
            steps=28,
            height=1024,
            width=1024,
            guidance=4.0,
            control_image="/tmp/canny.png",
            controlnet_strength=0.8,
        )
        assert gk["controlnet_image_path"] == "/tmp/canny.png"
        assert gk["controlnet_strength"] == 0.8

    def test_fill_requires_edit_and_mask(self):
        with pytest.raises(ValueError, match="requires edit_image and mask_image"):
            build_gen_kwargs(
                variant="fill",
                seed=0,
                prompt="test",
                steps=28,
                height=1024,
                width=1024,
                guidance=4.0,
            )

    def test_sd3_shift_from_extra_kwargs(self):
        gk = build_gen_kwargs(
            variant="sd3",
            seed=0,
            prompt="test",
            steps=28,
            height=1024,
            width=1024,
            guidance=4.0,
            extra_kwargs={"shift": 3.0},
        )
        assert gk["shift"] == 3.0

    def test_flux_negative_prompt_ignored(self):
        # Flux variants warn + don't include negative_prompt
        gk = build_gen_kwargs(
            variant="flux1_dev",
            seed=0,
            prompt="test",
            steps=28,
            height=1024,
            width=1024,
            guidance=4.0,
            negative_prompt="blurry",
        )
        assert "negative_prompt" not in gk

    def test_scheduler_passed_through(self):
        gk = build_gen_kwargs(
            variant="txt2img",
            seed=0,
            prompt="test",
            steps=4,
            height=1024,
            width=1024,
            guidance=1.0,
            scheduler="linear",
        )
        assert gk["scheduler"] == "linear"


class TestMediaJobManagerSpec:
    def test_build_spec_creates_output_dir(self):
        mgr = MediaJobManager()
        output_dir, spec = mgr._build_spec(
            variant="qwen_image",
            model_path="mlx-community/Qwen-Image-2512-4bit",
            quantize=4,
            config_label=None,
            output_format="PNG",
            n_images=2,
            gen_params={
                "seed": 0,
                "prompt": "test",
                "steps": 30,
                "height": 1024,
                "width": 1024,
                "guidance": 4.0,
            },
        )
        assert os.path.isdir(output_dir)
        assert spec["variant"] == "qwen_image"
        assert spec["n_images"] == 2
        assert spec["output_format"] == "PNG"
        assert spec["quantize"] == 4
        # Cleanup
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)


class TestSubprocessEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("FUSION_IMAGE_SUBPROCESS", raising=False)
        from fusion_mlx.engines.image_gen import _subprocess_enabled

        assert _subprocess_enabled() is False

    def test_env_on(self, monkeypatch):
        monkeypatch.setenv("FUSION_IMAGE_SUBPROCESS", "1")
        from fusion_mlx.engines.image_gen import _subprocess_enabled

        assert _subprocess_enabled() is True
