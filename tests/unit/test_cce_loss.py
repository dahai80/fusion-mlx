"""Chunked CE loss (cce.py) correctness tests: parity with default-style
fp32 CE, mask handling, chunk boundaries, fallback path."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.trainer import default_loss

from fusion_mlx.training.cce import make_chunked_ce_loss


class _Body(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.embed = nn.Embedding(64, d)

    def __call__(self, x):
        return self.embed(x)


class _Model(nn.Module):
    def __init__(self, d=32, v=64):
        super().__init__()
        self.model = _Body(d)
        self.lm_head = nn.Linear(d, v)

    def __call__(self, x):
        return self.lm_head(self.model(x))


def _batch(T=17, B=2):
    mx.random.seed(0)
    batch = mx.random.randint(0, 64, (B, T + 1))
    lengths = mx.array([[0, T], [0, T // 2 + 3]])
    return batch, lengths


def test_parity_with_default_loss_unmasked():
    mx.random.seed(1)
    model = _Model()
    batch, lengths = _batch()
    lengths = mx.array([[0, batch.shape[1] - 1], [0, batch.shape[1] - 1]])
    l_def, n_def = default_loss(model, batch, lengths)
    l_cce, n_cce = make_chunked_ce_loss(chunk_tokens=8)(model, batch, lengths)
    mx.eval(l_def, l_cce)
    assert int(n_def) == int(n_cce) == 2 * (batch.shape[1] - 1)
    assert abs(float(l_def) - float(l_cce)) < 1e-5


def test_mask_excludes_padding_positions():
    mx.random.seed(1)
    model = _Model()
    batch, _ = _batch()
    T = batch.shape[1] - 1
    full = mx.array([[0, T], [0, T]])
    partial = mx.array([[0, T], [0, 5]])
    l_full, n_full = make_chunked_ce_loss(chunk_tokens=8)(model, batch, full)
    l_part, n_part = make_chunked_ce_loss(chunk_tokens=8)(model, batch, partial)
    mx.eval(l_full, l_part)
    assert int(n_full) == 2 * T
    assert int(n_part) == T + 5
    assert float(l_part) != float(l_full)


def test_chunk_boundary_tail():
    mx.random.seed(2)
    model = _Model()
    batch, _ = _batch(T=17)
    lengths = mx.array([[0, 17], [0, 17]])
    l1, _ = make_chunked_ce_loss(chunk_tokens=8)(model, batch, lengths)
    l2, _ = make_chunked_ce_loss(chunk_tokens=32)(model, batch, lengths)
    mx.eval(l1, l2)
    assert abs(float(l1) - float(l2)) < 1e-6


def test_fallback_without_lm_head():
    mx.random.seed(1)

    class _Headless(nn.Module):
        def __init__(self, d=32, v=64):
            super().__init__()
            self.model = _Body(d)
            self.out_proj = nn.Linear(d, v)

        def __call__(self, x):
            return self.out_proj(self.model(x))

    headless = _Headless()
    batch, lengths = _batch()
    lengths = mx.array([[0, batch.shape[1] - 1], [0, batch.shape[1] - 1]])
    l_def, _ = default_loss(headless, batch, lengths)
    l_fb, _ = make_chunked_ce_loss()(headless, batch, lengths)
    mx.eval(l_def, l_fb)
    assert abs(float(l_def) - float(l_fb)) < 1e-5


def test_validate_rejects_pack_combo():
    from fusion_mlx.training.service import FineTuneConfig

    cfg = FineTuneConfig(
        chunked_ce_loss=True,
        pack_sequences=True,
    )
    try:
        cfg.validate()
        raise AssertionError("expected ValueError for pack+chunked combo")
    except ValueError as e:
        assert "chunked_ce_loss" in str(e)
