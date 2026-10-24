# SPDX-License-Identifier: Apache-2.0
# Issues #1375 / #1376: Qwen-Image-Edit multi-image wiring + image LoRA.
# #1375: mflux's QwenImageEdit.generate_image takes `image_paths` (a list). The
# old wiring passed `image_path`, which that method does not accept, so the
# variant could not consume the edit target or any reference image.
# #1376: per-request LoRA adapters must reach the mflux model constructor, and
# the path must be contained by FUSION_LORA_ALLOWED_DIRS.

import logging
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from fusion_mlx.api.images import LoRAAdapter, _validate_image_loras
from fusion_mlx.engines._gen_kwargs import build_gen_kwargs
from fusion_mlx.engines.image_gen import ImageGenEngine
from fusion_mlx.media.image_worker import _load_model


def _gk(variant, **kw):
    base = dict(
        variant=variant,
        seed=7,
        prompt="p",
        steps=20,
        height=1024,
        width=1024,
        guidance=1.0,
    )
    base.update(kw)
    return build_gen_kwargs(**base)


class TestQwenImageEditImagePaths:
    def test_edit_and_references_become_image_paths(self):
        gk = _gk(
            "qwen_image_edit",
            edit_image="target.png",
            reference_images=["char.png", "scene.png"],
        )
        assert gk["image_paths"] == ["target.png", "char.png", "scene.png"]
        assert "image_path" not in gk

    def test_references_only_is_allowed(self):
        gk = _gk("qwen_image_edit", reference_images=["char.png"])
        assert gk["image_paths"] == ["char.png"]

    def test_no_images_raises(self):
        with pytest.raises(ValueError, match="requires edit_image or reference"):
            _gk("qwen_image_edit")

    def test_image_strength_warns_instead_of_silently_ignored(self, caplog):
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.engines._gen_kwargs"):
            gk = _gk(
                "qwen_image_edit",
                edit_image="target.png",
                image_strength=0.6,
            )
        assert "image_paths" in gk
        assert "image_strength=0.60 ignored" in caplog.text

    def test_other_variants_keep_image_path(self):
        gk = _gk("kontext", edit_image="target.png")
        assert gk["image_path"] == "target.png"


class TestReferenceVariants:
    def test_qwen_edit_is_not_warned_about_references(self, caplog):
        engine = ImageGenEngine(model_name="Qwen/Qwen-Image-Edit-2509")
        engine._admit_generation = AsyncMock()
        engine._generate_subprocess = AsyncMock(return_value=[b"png"])
        with (
            patch(
                "fusion_mlx.engines.image_gen._subprocess_enabled", return_value=True
            ),
            caplog.at_level(logging.WARNING, logger="fusion_mlx.engines.image_gen"),
        ):
            import asyncio

            asyncio.run(
                engine.generate(
                    prompt="same character",
                    edit_image="target.png",
                    reference_images=["char.png"],
                )
            )
        assert engine.variant == "qwen_image_edit"
        assert "ignores" not in caplog.text


class TestImageLoraValidation:
    def test_disabled_without_allowed_dirs(self, monkeypatch):
        monkeypatch.delenv("FUSION_LORA_ALLOWED_DIRS", raising=False)
        with pytest.raises(HTTPException) as exc:
            _validate_image_loras([LoRAAdapter(path="x.safetensors")])
        assert exc.value.status_code == 400
        assert "FUSION_LORA_ALLOWED_DIRS" in exc.value.detail

    def test_path_outside_allowed_dir_rejected(self, tmp_path, monkeypatch):
        allowed = tmp_path / "loras"
        allowed.mkdir()
        outside = tmp_path / "elsewhere.safetensors"
        outside.write_bytes(b"x")
        monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", str(allowed))
        with pytest.raises(HTTPException) as exc:
            _validate_image_loras([LoRAAdapter(path=str(outside))])
        assert exc.value.status_code == 400
        assert "outside" in exc.value.detail

    def test_non_safetensors_rejected(self, tmp_path, monkeypatch):
        allowed = tmp_path / "loras"
        allowed.mkdir()
        bad = allowed / "weights.bin"
        bad.write_bytes(b"x")
        monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", str(allowed))
        with pytest.raises(HTTPException) as exc:
            _validate_image_loras([LoRAAdapter(path=str(bad))])
        assert "not safetensors" in exc.value.detail

    def test_missing_file_rejected(self, tmp_path, monkeypatch):
        allowed = tmp_path / "loras"
        allowed.mkdir()
        monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", str(allowed))
        with pytest.raises(HTTPException):
            _validate_image_loras([LoRAAdapter(path=str(allowed / "nope.safetensors"))])

    def test_valid_adapter_resolved_to_realpath(self, tmp_path, monkeypatch):
        allowed = tmp_path / "loras"
        allowed.mkdir()
        lora = allowed / "char.safetensors"
        lora.write_bytes(b"x")
        monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", str(allowed))
        out = _validate_image_loras([LoRAAdapter(path=str(lora), scale=0.9)])
        assert out == [{"path": os.path.realpath(lora), "scale": 0.9}]


class TestLoraLoadPath:
    def test_native_pipeline_rejects_lora(self):
        with pytest.raises(ValueError, match="not supported for variant 'sd15'"):
            _load_model("sd15", "/tmp/model", None, None, lora_paths=["a.safetensors"])

    @pytest.mark.asyncio
    async def test_in_process_mode_rejects_lora(self):
        engine = ImageGenEngine(model_name="black-forest-labs/FLUX.1-dev")
        engine._flux = object()
        engine._admit_generation = AsyncMock()
        with patch(
            "fusion_mlx.engines.image_gen._subprocess_enabled", return_value=False
        ):
            with pytest.raises(ValueError, match="require the subprocess worker"):
                await engine.generate(
                    prompt="p", lora=[{"path": "a.safetensors", "scale": 1.0}]
                )

    def test_mflux_ctor_receives_adapters(self, monkeypatch):
        # mflux ships only with the [image] extra; CI installs [dev].
        mc_mod = pytest.importorskip("mflux.models.common.config.model_config")

        seen = {}

        class _FakeModel:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        monkeypatch.setattr(mc_mod.ModelConfig, "dev", staticmethod(lambda: "CFG"))
        monkeypatch.setitem(
            __import__(
                "fusion_mlx.engines.image_gen", fromlist=["VARIANT_MAP"]
            ).VARIANT_MAP,
            "flux1_dev",
            ("fake.mfluxmod", "FakeModel", "dev", 4.0),
        )
        import sys
        import types

        fake = types.ModuleType("fake.mfluxmod")
        fake.FakeModel = _FakeModel
        monkeypatch.setitem(sys.modules, "fake.mfluxmod", fake)

        _load_model(
            "flux1_dev",
            "/tmp/model",
            None,
            None,
            lora_paths=["/tmp/a.safetensors"],
            lora_scales=[0.8],
        )
        assert seen["lora_paths"] == ["/tmp/a.safetensors"]
        assert seen["lora_scales"] == [0.8]
        assert seen["model_config"] == "CFG"
