# SPDX-License-Identifier: Apache-2.0
"""#1400: sequence packing for SFT — batch building, mask, loss, injection."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from fusion_mlx.training import packing


class _ListDataset:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def _items():
    return [
        ([1, 2, 3], 1),
        ([4, 5], 0),
        ([6, 7, 8, 9], 2),
        ([10], 0),
        ([11, 12], 1),
        ([13, 14, 15], 0),
    ]


class TestPackSamples:
    def test_rows_respect_max_seq_length(self):
        rows = packing.pack_samples(_items(), 8)
        assert rows
        for row in rows:
            assert len(row["tokens"]) <= 8

    def test_every_token_preserved_exactly_once(self):
        rows = packing.pack_samples(_items(), 64)
        tokens = [t for row in rows for t in row["tokens"]]
        assert sorted(tokens) == sorted(t for t, _ in _items() for t in t)

    def test_segments_contiguous_and_offsets_kept(self):
        rows = packing.pack_samples(_items(), 64)
        for row in rows:
            cursor = 0
            for start, end, offset in row["segments"]:
                assert start == cursor
                assert end > start
                cursor = end
            assert cursor == len(row["tokens"])

    def test_long_sample_truncated(self):
        rows = packing.pack_samples([(list(range(100)), 0)], 16)
        assert len(rows) == 1
        assert len(rows[0]["tokens"]) == 16

    def test_empty_sample_skipped(self):
        rows = packing.pack_samples([([], 0), ([1, 2], 0)], 8)
        assert len(rows) == 1
        assert rows[0]["tokens"] == [1, 2]


class TestBuildBatch:
    def test_shapes_and_padding(self):
        rows = packing.pack_samples(_items(), 16)
        batch, segments, loss_mask = packing._build_batch(rows, 16)
        assert batch.shape == segments.shape == loss_mask.shape
        assert batch.shape[1] == min(16, 33)
        seg = np.asarray(segments)
        assert (seg == -1).any()
        assert (seg >= 0).any()

    def test_loss_mask_honours_prompt_offset(self):
        batch, segments, loss_mask = packing._build_batch(
            [{"tokens": [1, 2, 3, 4], "segments": [(0, 4, 2)]}], 32
        )
        mask = np.asarray(loss_mask)[0]
        assert mask.tolist()[:4] == [False, False, True, True]
        assert not mask[4:].any()


class TestIteratePackedBatches:
    def test_yields_triples_and_stops(self):
        batches = list(
            packing.iterate_packed_batches(
                _ListDataset(_items()), 2, 8, loop=False, seed=0
            )
        )
        assert len(batches) == 1
        for batch, segments, loss_mask in batches:
            assert batch.shape == segments.shape == loss_mask.shape

    def test_rows_cached_across_iterators(self):
        ds = _ListDataset(_items())
        first = packing._packed_rows(ds, 8)
        second = packing._packed_rows(ds, 8)
        assert first is second

    def test_raises_when_rows_below_batch_size(self):
        with pytest.raises(ValueError, match="at least batch_size"):
            next(packing.iterate_packed_batches(_ListDataset(_items()), 100, 16))

    def test_distributed_rejected(self):
        class _World:
            def size(self):
                return 2

        with pytest.raises(ValueError, match="not supported in distributed"):
            next(
                packing.iterate_packed_batches(
                    _ListDataset(_items()), 2, 16, comm_group=_World()
                )
            )


class TestBlockDiagonalMask:
    def test_blocks_do_not_cross_segments(self):
        segments = mx.array([[0, 0, 1, 1]])
        mask = np.asarray(packing.block_diagonal_mask(segments))
        assert mask.shape == (1, 1, 4, 4)
        assert mask[0, 0, 0, 1] and mask[0, 0, 2, 3]
        assert not mask[0, 0, 0, 2]
        assert not mask[0, 0, 3, 1]


class _FakeModel(nn.Module):
    def __init__(self, vocab=8, seen=None):
        super().__init__()
        self.vocab = vocab
        self.seen = seen if seen is not None else []

    def __call__(self, inputs):
        self.seen.append(None)
        return mx.zeros(inputs.shape + (self.vocab,))


class TestPackedLoss:
    def test_prompt_tokens_excluded_from_token_count(self):
        model = _FakeModel()
        batch = mx.array([[1, 2, 3, 4, 5]])
        segments = mx.array([[0, 0, 0, 0, 0]])
        loss_mask = mx.array([[False, False, True, True, False]])
        loss, ntoks = packing.packed_loss(model, batch, segments, loss_mask)
        assert int(ntoks) == 2
        assert abs(float(loss) - np.log(8)) < 1e-4

    def test_mask_context_cleared_after_call(self):
        class _Recording(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def __call__(self, inputs):
                self.calls.append(packing._CURRENT_MASK.get())
                return mx.zeros(inputs.shape + (8,))

        model = _Recording()
        batch = mx.array([[1, 2, 3]])
        segments = mx.array([[0, 0, 1]])
        loss_mask = mx.array([[False, True, True]])
        packing.packed_loss(model, batch, segments, loss_mask)
        assert model.calls[0] is not None
        assert packing._CURRENT_MASK.get() is None


class TestInstallMaskInjection:
    def test_injects_mask_into_blocks(self, monkeypatch):
        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.masks = []

            def __call__(self, x, mask=None, cache=None):
                self.masks.append(mask)
                return x

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_Block(), _Block()]

        monkeypatch.setattr(packing, "_INSTALLED_CLASSES", set())
        model = _Model()
        packing.install_mask_injection(model)

        packed = mx.zeros((1, 1, 2, 2), dtype=mx.bool_)
        token = packing._CURRENT_MASK.set(packed)
        try:
            for layer in model.layers:
                layer(mx.zeros((1, 2, 3)))
        finally:
            packing._CURRENT_MASK.reset(token)
        assert model.layers[0].masks[0] is packed
        assert model.layers[1].masks[0] is packed

        for layer in model.layers:
            layer(mx.zeros((1, 2, 3)), mask="causal")
        assert model.layers[0].masks[-1] == "causal"

    def test_incompatible_block_raises(self, monkeypatch):
        class _Weird(nn.Module):
            def __call__(self, x, something_else=None):
                return x

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_Weird()]

        monkeypatch.setattr(packing, "_INSTALLED_CLASSES", set())
        with pytest.raises(RuntimeError, match="do not accept"):
            packing.install_mask_injection(_Model())

    def test_model_without_layers_raises(self, monkeypatch):
        monkeypatch.setattr(packing, "_INSTALLED_CLASSES", set())
        with pytest.raises(RuntimeError, match="cannot find model.layers"):
            packing.install_mask_injection(nn.Module())
