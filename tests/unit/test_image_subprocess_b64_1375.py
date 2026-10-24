# SPDX-License-Identifier: Apache-2.0
# Issue #1375: the subprocess worker takes file paths, but the API layer sends
# base64 blobs. The in-process branch decoded them; the subprocess branch did
# not, so any b64 edit_image (multi-image Qwen-Image-Edit included) reached
# mflux as open(<b64 string>) and the worker died.

import base64
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    def __init__(self, output_dir):
        self.calls = []
        self.output_dir = output_dir

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


def _fake_manager(manager, monkeypatch):
    # Install the double on the engine class itself: _generate_subprocess
    # reads ImageGenEngine._media_mgr before falling back to constructing a
    # MediaJobManager, and a real manager cached by an earlier test in the
    # same session would otherwise spawn a live worker.
    monkeypatch.setattr(ImageGenEngine, "_media_mgr", manager, raising=False)
    monkeypatch.setattr("fusion_mlx.engines.image_gen.wait_for_llm_idle", AsyncMock())


def _engine(variant: str = "qwen_image_edit") -> ImageGenEngine:
    engine = ImageGenEngine(model_name="Qwen/Qwen-Image-Edit-2509")
    engine._variant = variant
    engine._update_activity = MagicMock()
    engine._finish_activity = AsyncMock()
    return engine


async def _run(
    engine,
    tmp_path,
    *,
    activity_id,
    edit_image,
    reference_images=None,
):
    return await engine._generate_subprocess(
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
        reference_images=reference_images,
        reference_strengths=None,
        edit_image=edit_image,
        mask_image=None,
        depth_image=None,
        image_strength=None,
        kwargs={},
        t0=0.0,
        activity_id=activity_id,
    )


@pytest.mark.asyncio
async def test_b64_inputs_become_files_for_worker(tmp_path, monkeypatch):
    manager = _FakeManager(str(tmp_path / "out"))
    _fake_manager(manager, monkeypatch)
    payload = f"data:image/png;base64,{PNG_1PX}"

    images = await _run(
        _engine(),
        tmp_path,
        activity_id="test-1375",
        edit_image=payload,
        reference_images=[payload, payload],
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
async def test_existing_paths_pass_through(tmp_path, monkeypatch):
    manager = _FakeManager(str(tmp_path / "out"))
    _fake_manager(manager, monkeypatch)
    base = tmp_path / "base.png"
    base.write_bytes(b"png")

    await _run(
        _engine("kontext"),
        tmp_path,
        activity_id="test-1375-paths",
        edit_image=str(base),
    )

    params = manager.calls[0]["gen_params"]
    assert params["edit_image"] == str(base)
    assert os.path.isfile(params["edit_image"])


@pytest.mark.asyncio
async def test_qwen_edit_keeps_multi_image_order(tmp_path, monkeypatch, caplog):
    manager = _FakeManager(str(tmp_path / "out"))
    _fake_manager(manager, monkeypatch)
    payload = f"data:image/png;base64,{PNG_1PX}"

    with caplog.at_level(logging.INFO, logger="fusion_mlx.engines.image_gen"):
        await _run(
            _engine(),
            tmp_path,
            activity_id="test-1375-order",
            edit_image=payload,
            reference_images=[payload, payload],
        )

    params = manager.calls[0]["gen_params"]
    assert params["edit_image"] not in params["reference_images"]
    assert len(set(params["reference_images"])) == 2
    assert "resolved 3 base64 image input(s)" in caplog.text
