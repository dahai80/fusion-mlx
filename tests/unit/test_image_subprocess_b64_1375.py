# SPDX-License-Identifier: Apache-2.0
# Issue #1375: the subprocess worker takes file paths, but the API layer sends
# base64 blobs. The in-process branch decoded them; the subprocess branch did
# not, so any b64 edit_image (multi-image Qwen-Image-Edit included) reached
# mflux as open(<b64 string>) and the worker died.

import base64
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_mlx.engines.image_gen import ImageGenEngine

PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000a49444154789c6300010000050001"
        "0d0a2db40000000049454e44ae426082"
    )
).decode()


class _FakeManager:
    def __init__(self):
        self.calls = []
        self.output_dir = None

    async def run_image_job(self, **kwargs):
        self.calls.append(kwargs)
        params = kwargs.get("gen_params", {})
        self.inputs_existed = {
            "edit": os.path.isfile(params.get("edit_image") or ""),
            "refs": [
                os.path.isfile(ref) for ref in (params.get("reference_images") or [])
            ],
        }
        os.makedirs(self.output_dir, exist_ok=True)
        out = os.path.join(self.output_dir, "0.png")
        with open(out, "wb") as handle:
            handle.write(b"fake-png")
        return SimpleNamespace(outputs=[out])


@pytest.fixture(autouse=True)
def _reset_media_manager():
    ImageGenEngine._media_mgr = None
    yield
    ImageGenEngine._media_mgr = None


def _engine(variant: str = "qwen_image_edit") -> ImageGenEngine:
    engine = ImageGenEngine(model_name="Qwen/Qwen-Image-Edit-2509")
    engine._variant = variant
    engine._update_activity = MagicMock()
    engine._finish_activity = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_b64_inputs_become_files_for_worker(tmp_path):
    engine = _engine()
    manager = _FakeManager()
    manager.output_dir = str(tmp_path / "out")
    payload = f"data:image/png;base64,{PNG_1PX}"

    with (
        patch("fusion_mlx.engines.image_gen.wait_for_llm_idle", new=AsyncMock()),
        patch("fusion_mlx.media.job_manager.MediaJobManager", return_value=manager),
    ):
        images = await engine._generate_subprocess(
            prompt="put the refs in the scene",
            width=512,
            height=512,
            steps=2,
            seed=1,
            guidance=1.0,
            n_images=1,
            output_format="PNG",
            scheduler=None,
            negative_prompt=None,
            denoising_end=None,
            on_step=None,
            control_image=None,
            controlnet_strength=None,
            reference_images=[payload, payload],
            reference_strengths=None,
            edit_image=payload,
            mask_image=None,
            depth_image=None,
            image_strength=None,
            kwargs={},
            t0=0.0,
            activity_id="test-1375",
        )

    assert images == [b"fake-png"]
    params = manager.calls[0]["gen_params"]
    assert manager.inputs_existed["edit"] is True
    assert len(params["reference_images"]) == 2
    assert all(manager.inputs_existed["refs"])
    # The worker is gone, so the decoded inputs must be cleaned up.
    assert not os.path.exists(params["edit_image"])
    for ref in params["reference_images"]:
        assert not os.path.exists(ref)


@pytest.mark.asyncio
async def test_existing_paths_pass_through(tmp_path):
    engine = _engine("kontext")
    manager = _FakeManager()
    manager.output_dir = str(tmp_path / "out")
    base = tmp_path / "base.png"
    base.write_bytes(b"png")

    with (
        patch("fusion_mlx.engines.image_gen.wait_for_llm_idle", new=AsyncMock()),
        patch("fusion_mlx.media.job_manager.MediaJobManager", return_value=manager),
    ):
        await engine._generate_subprocess(
            prompt="edit",
            width=512,
            height=512,
            steps=2,
            seed=1,
            guidance=2.5,
            n_images=1,
            output_format="PNG",
            scheduler=None,
            negative_prompt=None,
            denoising_end=None,
            on_step=None,
            control_image=None,
            controlnet_strength=None,
            reference_images=None,
            reference_strengths=None,
            edit_image=str(base),
            mask_image=None,
            depth_image=None,
            image_strength=None,
            kwargs={},
            t0=0.0,
            activity_id="test-1375-paths",
        )

    params = manager.calls[0]["gen_params"]
    assert params["edit_image"] == str(base)
    assert os.path.isfile(params["edit_image"])


@pytest.mark.asyncio
async def test_qwen_edit_keeps_multi_image_order(tmp_path, caplog):
    engine = _engine()
    manager = _FakeManager()
    manager.output_dir = str(tmp_path / "out")
    payload = f"data:image/png;base64,{PNG_1PX}"

    with (
        patch("fusion_mlx.engines.image_gen.wait_for_llm_idle", new=AsyncMock()),
        patch("fusion_mlx.media.job_manager.MediaJobManager", return_value=manager),
        caplog.at_level(logging.INFO, logger="fusion_mlx.engines.image_gen"),
    ):
        await engine._generate_subprocess(
            prompt="compose",
            width=512,
            height=512,
            steps=2,
            seed=1,
            guidance=1.0,
            n_images=1,
            output_format="PNG",
            scheduler=None,
            negative_prompt=None,
            denoising_end=None,
            on_step=None,
            control_image=None,
            controlnet_strength=None,
            reference_images=[payload, payload],
            reference_strengths=None,
            edit_image=payload,
            mask_image=None,
            depth_image=None,
            image_strength=None,
            kwargs={},
            t0=0.0,
            activity_id="test-1375-order",
        )

    params = manager.calls[0]["gen_params"]
    assert params["edit_image"] not in params["reference_images"]
    assert len(set(params["reference_images"])) == 2
    assert "resolved 3 base64 image input(s)" in caplog.text
