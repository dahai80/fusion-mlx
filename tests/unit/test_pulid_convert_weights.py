# SPDX-License-Identifier: Apache-2.0
# Unit tests for PuLID/EVA-CLIP weight converter.
# Loads convert_weights.py via importlib.util to avoid the
# fusion_mlx.__init__ -> server -> gui_compat -> mlx_whisper chain.

import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

_CONVERT_MOD_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "fusion_mlx" / "video" / "pulid_mlx" / "convert_weights.py"
)


def _load_convert_module():
    spec = importlib.util.spec_from_file_location(
        "pulid_convert_weights", str(_CONVERT_MOD_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cw = _load_convert_module()


class FakeTensor:
    def __init__(self, data, ndim=2):
        self._data = np.array(data, dtype=np.float32)
        self.ndim = ndim
        self.shape = (
            self._data.shape
            if ndim == 2
            else (1, 1, 1, 1) if ndim == 4
            else self._data.shape
        )

    def float(self):
        return self

    def numpy(self):
        return self._data

    def permute(self, *dims):
        return FakeTensor(np.transpose(self._data, dims))

    def contiguous(self):
        return self


def _make_fake_conv_tensor(out_ch=3, in_ch=3, kh=3, kw=3):
    data = np.random.randn(out_ch, in_ch, kh, kw).astype(np.float32)
    t = FakeTensor.__new__(FakeTensor)
    t._data = data
    t.ndim = 4
    t.shape = data.shape
    t.float = lambda: t
    t.numpy = lambda: data
    t.permute = lambda *dims: FakeTensor(np.transpose(data, dims))
    t.contiguous = lambda: t
    return t


class TestConvertEvaClip:
    def test_strips_visual_prefix(self):
        pt = {
            "visual.patch_embed.proj.weight": FakeTensor(np.random.randn(16, 3, 2, 2)),
            "visual.blocks.0.norm.weight": FakeTensor(np.random.randn(16)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eva.safetensors"
            result = cw.convert_eva_clip(pt, str(out))
            assert "patch_embed.proj.weight" in result
            assert "blocks.0.norm.weight" in result
            assert not any(k.startswith("visual.") for k in result)

    def test_skips_text_encoder_keys(self):
        pt = {
            "visual.blocks.0.norm.weight": FakeTensor(np.random.randn(16)),
            "text.encoder.weight": FakeTensor(np.random.randn(16)),
            "logit_scale": FakeTensor(np.array(1.0)),
            "mask_token": FakeTensor(np.random.randn(16)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eva.safetensors"
            result = cw.convert_eva_clip(pt, str(out))
            assert len(result) == 1
            assert "blocks.0.norm.weight" in result

    def test_conv2d_transposed(self):
        pt = {
            "visual.patch_embed.proj.weight": _make_fake_conv_tensor(16, 3, 14, 14),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eva.safetensors"
            result = cw.convert_eva_clip(pt, str(out))
            w = result["patch_embed.proj.weight"]
            assert w.shape == (16, 14, 14, 3)


class TestConvertIdformer:
    def test_remaps_mapping_keys(self):
        pt = {
            "mapping.0.weight": FakeTensor(np.random.randn(16, 16)),
            "mapping.0.bias": FakeTensor(np.random.randn(16)),
            "mapping.1.weight": FakeTensor(np.random.randn(16)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "idformer.safetensors"
            result = cw.convert_idformer(pt, str(out))
            assert "mappings.0.net.0.weight" in result
            assert "mappings.0.net.0.bias" in result
            assert "mappings.0.net.1.weight" in result

    def test_remaps_id_embedding_mapping(self):
        pt = {
            "id_embedding_mapping.0.weight": FakeTensor(np.random.randn(16, 16)),
            "id_embedding_mapping.4.bias": FakeTensor(np.random.randn(16)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "idformer.safetensors"
            result = cw.convert_idformer(pt, str(out))
            assert "id_embedding_mapping.net.0.weight" in result
            assert "id_embedding_mapping.net.4.bias" in result

    def test_strips_model_prefix(self):
        pt = {
            "model.layers.0.weight": FakeTensor(np.random.randn(16, 16)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "idformer.safetensors"
            result = cw.convert_idformer(pt, str(out))
            assert "layers.0.weight" in result
            assert not any(k.startswith("model.") for k in result)

    def test_passes_through_unknown_keys(self):
        pt = {
            "layers.0.attn.to_q.weight": FakeTensor(np.random.randn(16, 16)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "idformer.safetensors"
            result = cw.convert_idformer(pt, str(out))
            assert "layers.0.attn.to_q.weight" in result


class TestConvertPulidMeta:
    def test_writes_conversion_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            pulid_dir = Path(tmp) / "source"
            pulid_dir.mkdir()
            (pulid_dir / "eva_clip").mkdir()
            (pulid_dir / "eva_clip" / "dummy.bin").write_text("fake")
            out_dir = Path(tmp) / "output"

            original_load = cw._load_pytorch_weights
            cw._load_pytorch_weights = lambda p: {
                "visual.blocks.0.weight": FakeTensor(np.random.randn(16)),
            }
            try:
                cw.convert_pulid(str(pulid_dir), str(out_dir))
            finally:
                cw._load_pytorch_weights = original_load

            meta_path = out_dir / "conversion_meta.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert "eva_clip" in meta["components"]
