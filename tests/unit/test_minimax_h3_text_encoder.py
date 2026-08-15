# SPDX-License-Identifier: Apache-2.0
# P4 Text Encoder checkpoint：MiniMaxH3TextEncoder 第 50 层截断 + 不接 final norm。
# 无 66GB Qwen3-VL 权重 → 用 stub language_model 复刻 mlx-vlm Qwen3VLModel 结构验证逻辑。
import mlx.core as mx
import mlx.nn as nn
import pytest

from fusion_mlx.video.minimax_h3.text_encoder import (
    H3_TEXT_ENCODER_LAYER,
    MiniMaxH3TextEncoder,
)


class _StubArgs:
    def __init__(self, hidden_size, num_hidden_layers):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class _StubLayer(nn.Module):
    # 每层给 hidden 加一个层标记偏置，便于断言"截断在第 N 层"。
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def __call__(self, h, mask=None, cache=None, position_ids=None):
        return h + float(self.idx)


class _StubQwen3VLModel(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, vocab_size=32):
        super().__init__()
        self.args = _StubArgs(hidden_size, num_hidden_layers)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [_StubLayer(i) for i in range(num_hidden_layers)]
        self.norm = nn.Identity()

    def __call__(self, *a, **kw):
        raise AssertionError("不应调用 final-norm 路径")


class _StubLanguageModel(nn.Module):
    # 复刻 mlx-vlm LanguageModel: .model + .args
    def __init__(self, hidden_size, num_hidden_layers):
        super().__init__()
        self.args = _StubArgs(hidden_size, num_hidden_layers)
        self.model = _StubQwen3VLModel(hidden_size, num_hidden_layers)


def _make_encoder(num_layers=60, hidden=5120, layer=H3_TEXT_ENCODER_LAYER):
    lm = _StubLanguageModel(hidden, num_layers)
    return MiniMaxH3TextEncoder(lm, layer=layer)


class TestConstruction:
    def test_layer_constant(self):
        assert H3_TEXT_ENCODER_LAYER == 49

    def test_hidden_size_captured(self):
        enc = _make_encoder(num_layers=60, hidden=5120)
        assert enc.hidden_size == 5120
        assert enc.layer == 49

    def test_layer_too_large_rejected(self):
        # layer >= num_hidden_layers → 拒绝（50 层模型，layer=50 越界）。
        with pytest.raises(ValueError):
            _make_encoder(num_layers=50, hidden=5120, layer=50)

    def test_layer_ok_at_boundary(self):
        # 50 层，layer=49 是最后一层 → 合法。
        enc = _make_encoder(num_layers=50, hidden=64, layer=49)
        assert enc.layer == 49


class TestForward:
    def test_truncates_at_layer(self):
        # embed + sum(idx for idx in 0..49) = embed + 49*50/2 = embed + 1225。
        enc = _make_encoder(num_layers=60, hidden=8, layer=49)
        input_ids = mx.array([[1, 2, 3, 4]])
        out = enc(input_ids)
        assert out.shape == (1, 4, 8)
        embed = enc.language_model.model.embed_tokens(input_ids)
        expected = embed + 1225.0
        assert mx.allclose(out, expected, atol=1e-4)

    def test_no_final_norm(self):
        # 改用层数=layer+1 验证截断后不会继续累加后续层标记。
        enc = _make_encoder(num_layers=60, hidden=8, layer=2)
        input_ids = mx.array([[1, 2]])
        out = enc(input_ids)
        embed = enc.language_model.model.embed_tokens(input_ids)
        # 只加 idx 0+1+2 = 3，不加 3..59。
        expected = embed + 3.0
        assert mx.allclose(out, expected, atol=1e-4)

    def test_output_shape_5120(self):
        enc = _make_encoder(num_layers=60, hidden=5120, layer=49)
        input_ids = mx.array([[5, 6, 7, 8, 9]])
        out = enc(input_ids)
        assert out.shape == (1, 5, 5120)

    def test_output_seq_len_matches_input(self):
        enc = _make_encoder(num_layers=60, hidden=16, layer=49)
        input_ids = mx.array([[1, 2, 3]])
        out = enc(input_ids)
        assert out.shape[1] == 3

    def test_attention_mask_padding(self):
        enc = _make_encoder(num_layers=60, hidden=8, layer=0)
        input_ids = mx.array([[1, 2, 3, 4]])
        am = mx.array([[1, 1, 0, 0]])
        out = enc(input_ids, attention_mask=am)
        assert out.shape == (1, 4, 8)

    def test_batch_gt_one(self):
        enc = _make_encoder(num_layers=60, hidden=8, layer=49)
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        out = enc(input_ids)
        assert out.shape == (2, 3, 8)


class TestLoadEncoder:
    def test_missing_model_dir(self, tmp_path):
        from fusion_mlx.video.minimax_h3.text_encoder import load_text_encoder

        with pytest.raises(Exception):
            load_text_encoder(tmp_path / "nonexistent")
