# SPDX-License-Identifier: Apache-2.0
"""Tests for #245: embedding batch dedup + chunking optimization."""

import pytest

from fusion_mlx.api.embeddings_routes import _dedup_inputs


class TestDedupInputs:
    def test_no_duplicates_returns_identity(self):
        unique, mapping = _dedup_inputs(["a", "b", "c"])
        assert unique == ["a", "b", "c"]
        assert mapping == [0, 1, 2]

    def test_all_duplicates_returns_single(self):
        unique, mapping = _dedup_inputs(["x", "x", "x"])
        assert unique == ["x"]
        assert mapping == [0, 0, 0]

    def test_partial_duplicates(self):
        unique, mapping = _dedup_inputs(["hello", "world", "hello", "world", "foo"])
        assert unique == ["hello", "world", "foo"]
        assert mapping == [0, 1, 0, 1, 2]

    def test_empty_list(self):
        unique, mapping = _dedup_inputs([])
        assert unique == []
        assert mapping == []

    def test_single_item(self):
        unique, mapping = _dedup_inputs(["only"])
        assert unique == ["only"]
        assert mapping == [0]

    def test_mapping_reconstructs_original(self):
        original = ["a", "b", "a", "c", "b", "a"]
        unique, mapping = _dedup_inputs(original)
        reconstructed = [unique[idx] for idx in mapping]
        assert reconstructed == original

    def test_preserves_insertion_order(self):
        unique, mapping = _dedup_inputs(["z", "a", "z", "m", "a"])
        assert unique == ["z", "a", "m"]
        assert mapping == [0, 1, 0, 2, 1]
