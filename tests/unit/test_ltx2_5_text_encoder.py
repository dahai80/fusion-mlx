# SPDX-License-Identifier: Apache-2.0
# P2 checkpoint: LTX-2.5 Gemma4 text encoder wrapper + projection.
# Uses a stub language model (no 12B weights) mirroring Gemma4TextModel's
# __call__(inputs, mask=, skip_final_norm=) contract.
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.video.ltx2_5.text_encoder import (
    LTX2_5_CAPTION_CHANNELS,
    LTX2_5TextEncoder,
    LTX2_5TextProjection,
)


class _StubGemma4(nn.Module):
    # Mimics Gemma4TextModel.__call__: accepts skip_final_norm + mask kwargs,
    # returns hidden state of shape (batch, seq, hidden_size).
    def __init__(self, hidden_size: int = 16, num_layers: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_tokens = nn.Embedding(100, hidden_size)
        self.layers = [_StubLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.Identity()
        self.embed_scale = hidden_size**0.5

    def __call__(self, inputs, mask=None, skip_final_norm=False, **kwargs):
        h = self.embed_tokens(inputs) * self.embed_scale
        for layer in self.layers:
            h = layer(h)
        if skip_final_norm:
            return h
        return h
        # norm is Identity so no difference in stub

    def sanitize(self, weights):
        return weights

    def load_weights(self, items, strict=False):
        return None


class _StubLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.linear(x) + x


class _StubGemma4Tuple(_StubGemma4):
    # Variant that returns a tuple (hidden, extra) to test tuple-unwrapping.
    def __call__(self, inputs, mask=None, skip_final_norm=False, **kwargs):
        h = self.embed_tokens(inputs) * self.embed_scale
        return h, "extra"


class TestLTX2_5TextProjection:
    def test_output_dim_3840(self):
        proj = LTX2_5TextProjection(in_features=16, out_features=3840)
        x = mx.zeros((1, 5, 16))
        out = proj(x)
        assert out.shape == (1, 5, 3840)

    def test_default_caption_channels(self):
        proj = LTX2_5TextProjection(in_features=16)
        assert proj.linear2.weight.shape[0] == LTX2_5_CAPTION_CHANNELS

    def test_custom_hidden(self):
        proj = LTX2_5TextProjection(in_features=8, out_features=32, hidden_size=64)
        assert proj.linear1.weight.shape[0] == 64
        assert proj.linear2.weight.shape[0] == 32

    def test_forward_shape(self):
        proj = LTX2_5TextProjection(in_features=16, out_features=20)
        out = proj(mx.random.normal((2, 7, 16)))
        assert out.shape == (2, 7, 20)


class TestLTX2_5TextEncoder:
    def test_encode_no_projection(self):
        lm = _StubGemma4(hidden_size=16)
        enc = LTX2_5TextEncoder(lm, projection=None)
        ids = mx.array([[1, 2, 3]])
        out = enc.encode(ids)
        assert out.shape == (1, 3, 16)

    def test_encode_with_projection(self):
        lm = _StubGemma4(hidden_size=16)
        proj = LTX2_5TextProjection(in_features=16, out_features=3840)
        enc = LTX2_5TextEncoder(lm, projection=proj)
        ids = mx.array([[1, 2, 3, 4]])
        out = enc.encode(ids)
        assert out.shape == (1, 4, 3840)

    def test_call_alias(self):
        lm = _StubGemma4(hidden_size=16)
        enc = LTX2_5TextEncoder(lm, projection=None)
        ids = mx.array([[1, 2]])
        assert enc(ids).shape == enc.encode(ids).shape

    def test_caption_channels_field(self):
        enc = LTX2_5TextEncoder(_StubGemma4(), projection=None, caption_channels=3840)
        assert enc.caption_channels == 3840

    def test_tuple_hidden_unwrapped(self):
        lm = _StubGemma4Tuple(hidden_size=16)
        enc = LTX2_5TextEncoder(lm, projection=None)
        ids = mx.array([[1, 2, 3]])
        out = enc.encode(ids)
        assert out.shape == (1, 3, 16)

    def test_attention_mask_passed(self):
        class _MaskCapture(_StubGemma4):
            def __init__(self):
                super().__init__(hidden_size=16)
                self.received_mask = None

            def __call__(self, inputs, mask=None, skip_final_norm=False, **kwargs):
                self.received_mask = mask
                return super().__call__(inputs, mask=mask, skip_final_norm=skip_final_norm)

        lm = _MaskCapture()
        enc = LTX2_5TextEncoder(lm, projection=None)
        ids = mx.array([[1, 2, 3]])
        mask = mx.array([[1, 1, 0]])
        enc.encode(ids, attention_mask=mask)
        assert lm.received_mask is not None

    def test_skip_final_norm_default_true(self):
        class _NormCapture(_StubGemma4):
            def __init__(self):
                super().__init__(hidden_size=16)
                self.skip_received = None

            def __call__(self, inputs, mask=None, skip_final_norm=False, **kwargs):
                self.skip_received = skip_final_norm
                return super().__call__(inputs, mask=mask, skip_final_norm=skip_final_norm)

        lm = _NormCapture()
        enc = LTX2_5TextEncoder(lm, projection=None)
        enc.encode(mx.array([[1]]))
        assert lm.skip_received is True


class TestSplitProjectionWeights:
    def test_split(self):
        from fusion_mlx.video.ltx2_5.text_encoder import _split_projection_weights

        weights = {
            "projection.linear1.weight": mx.zeros((4, 8)),
            "projection.linear2.weight": mx.zeros((4, 4)),
            "model.embed_tokens.weight": mx.zeros((10, 8)),
            "model.layers.0.weight": mx.zeros((8, 8)),
        }
        lang, proj = _split_projection_weights(weights)
        assert "model.embed_tokens.weight" in lang
        assert "model.layers.0.weight" in lang
        assert "linear1.weight" in proj
        assert "linear2.weight" in proj
        assert len(proj) == 2
        assert len(lang) == 2

    def test_no_projection_keys(self):
        from fusion_mlx.video.ltx2_5.text_encoder import _split_projection_weights

        lang, proj = _split_projection_weights({"model.weight": mx.zeros((2, 2))})
        assert len(proj) == 0
        assert len(lang) == 1


class TestLoadTextEncoderMissingFiles:
    def test_missing_weights_raises(self, tmp_path):
        from fusion_mlx.video.ltx2_5.text_encoder import load_text_encoder

        with pytest.raises(FileNotFoundError, match="weights not found"):
            load_text_encoder(tmp_path / "nope.safetensors")

    def test_missing_config_raises(self, tmp_path):
        from fusion_mlx.video.ltx2_5.text_encoder import load_text_encoder

        wpath = tmp_path / "enc.safetensors"
        wpath.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="config not found"):
            load_text_encoder(wpath)
