# SPDX-License-Identifier: Apache-2.0
"""Worker weight preflight accepts both backbone layouts (audit 1019 E2E).

FLUX/Qwen-Image ship the denoiser as transformer/ (or transformer-00001),
SD1.5/SD2/SDXL ship unet/. The preflight used to require "transformer", so
every SD request through the subprocess worker died with
"weight preflight failed: missing transformer shard" before loading.
"""

import pathlib

import pytest

from fusion_mlx.media.image_worker import _preflight_components


def _touch(root: pathlib.Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")


def test_mlx_sd_layout_passes(tmp_path):
    _touch(tmp_path, "unet/model.safetensors")
    _touch(tmp_path, "vae/model.safetensors")
    _touch(tmp_path, "text_encoder/model.safetensors")
    assert _preflight_components(str(tmp_path)) == []


def test_diffusers_sd_layout_passes(tmp_path):
    _touch(tmp_path, "unet/diffusion_pytorch_model.safetensors")
    _touch(tmp_path, "vae/diffusion_pytorch_model.safetensors")
    _touch(tmp_path, "text_encoder/model.safetensors")
    assert _preflight_components(str(tmp_path)) == []


@pytest.mark.parametrize(
    "transformer_shard", ["transformer-00001.safetensors", "transformer/0.safetensors"]
)
def test_flux_and_qwen_layouts_still_pass(tmp_path, transformer_shard):
    _touch(tmp_path, transformer_shard)
    _touch(tmp_path, "vae/diffusion_pytorch_model.safetensors")
    _touch(tmp_path, "text_encoder/model.safetensors")
    assert _preflight_components(str(tmp_path)) == []


def test_missing_backbone_is_reported(tmp_path):
    _touch(tmp_path, "vae/model.safetensors")
    _touch(tmp_path, "text_encoder/model.safetensors")
    assert _preflight_components(str(tmp_path)) == ["unet/transformer shard"]


def test_partial_download_still_fails(tmp_path):
    _touch(tmp_path, "unet/model.safetensors")
    assert _preflight_components(str(tmp_path)) == ["vae shard", "text_encoder shard"]


def test_empty_dir_reports_no_safetensors(tmp_path):
    assert _preflight_components(str(tmp_path)) == ["safetensors (none found)"]


def test_missing_dir_is_reported():
    assert _preflight_components("/nonexistent/model/path") == ["model_dir"]
