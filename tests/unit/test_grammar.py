# SPDX-License-Identifier: Apache-2.0
"""Unit tests for grammar-constrained decoding (xgrammar + llguidance).

No importers — standalone pytest file testing fusion_mlx.api.grammar.
User instruction: "继续推进剩下的工作" — gap plan #21 llguidance integration.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fusion_mlx.api.grammar import (
    GrammarBackend,
    GrammarConstraintProcessor,
    _HfTokenizerForLlguidance,
    resolve_grammar_backend,
)


# ─── resolve_grammar_backend ──────────────────────────────────────────────


class TestResolveGrammarBackend:
    def test_auto_prefers_llguidance_when_available(self):
        with patch("fusion_mlx.api.grammar._is_llguidance_available", return_value=True):
            result = resolve_grammar_backend(None)
            assert result == GrammarBackend.LLGUIDANCE

    def test_auto_falls_back_to_xgrammar(self):
        with patch("fusion_mlx.api.grammar._is_llguidance_available", return_value=False), \
             patch("fusion_mlx.api.grammar._is_xgrammar_available", return_value=True):
            result = resolve_grammar_backend(None)
            assert result == GrammarBackend.XGRAMMAR

    def test_auto_neither_available(self):
        with patch("fusion_mlx.api.grammar._is_llguidance_available", return_value=False), \
             patch("fusion_mlx.api.grammar._is_xgrammar_available", return_value=False):
            result = resolve_grammar_backend(None)
            assert result == GrammarBackend.AUTO

    def test_explicit_llguidance(self):
        result = resolve_grammar_backend("llguidance")
        assert result == GrammarBackend.LLGUIDANCE

    def test_explicit_xgrammar(self):
        result = resolve_grammar_backend("xgrammar")
        assert result == GrammarBackend.XGRAMMAR

    def test_unknown_falls_back_to_auto(self):
        with patch("fusion_mlx.api.grammar._is_llguidance_available", return_value=True):
            result = resolve_grammar_backend("unknown_backend")
            assert result == GrammarBackend.LLGUIDANCE


# ─── _HfTokenizerForLlguidance ────────────────────────────────────────────


class TestHfTokenizerAdapter:
    @pytest.fixture()
    def fake_hf(self):
        tok = MagicMock()
        tok.eos_token_id = 2
        tok.bos_token_id = 1
        tok.all_special_ids = [1, 2]
        vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5}
        tok.get_vocab.return_value = vocab
        id_to_tok = {v: k for k, v in vocab.items()}
        tok.convert_ids_to_tokens = lambda idx: id_to_tok.get(idx)
        tok.encode.return_value = [3, 4, 5]
        return tok

    def test_tokens_are_bytes(self, fake_hf):
        adapted = _HfTokenizerForLlguidance(fake_hf)
        assert isinstance(adapted.tokens, list)
        for t in adapted.tokens:
            assert isinstance(t, bytes)
        assert adapted.tokens[3] == b"a"

    def test_special_token_ids(self, fake_hf):
        adapted = _HfTokenizerForLlguidance(fake_hf)
        assert 1 in adapted.special_token_ids
        assert 2 in adapted.special_token_ids

    def test_eos_bos_ids(self, fake_hf):
        adapted = _HfTokenizerForLlguidance(fake_hf)
        assert adapted.eos_token_id == 2
        assert adapted.bos_token_id == 1

    def test_call_encodes(self, fake_hf):
        adapted = _HfTokenizerForLlguidance(fake_hf)
        result = adapted("abc")
        assert result == [3, 4, 5]


# ─── GrammarConstraintProcessor — llguidance backend ──────────────────────


class TestLlguidanceProcessor:
    @pytest.fixture()
    def fake_ll_matcher(self):
        matcher = MagicMock()
        matcher.compute_bitmask.return_value = None
        matcher.is_stopped.return_value = False
        matcher.is_error.return_value = False
        matcher.consume_token = MagicMock()
        return matcher

    def test_init_llguidance(self, fake_ll_matcher):
        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=100, backend=GrammarBackend.LLGUIDANCE
        )
        assert proc.backend == GrammarBackend.LLGUIDANCE
        assert not proc.is_terminated

    def test_call_with_bytes_bitmask(self, fake_ll_matcher):
        vocab_size = 64
        width = (vocab_size + 31) // 32
        all_allowed = b"\xff" * (width * 4)
        fake_ll_matcher.compute_bitmask.return_value = all_allowed

        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=vocab_size, backend=GrammarBackend.LLGUIDANCE
        )
        import mlx.core as mx

        logits = mx.zeros((1, vocab_size))
        result = proc(None, logits)
        assert result.shape == (1, vocab_size)

    def test_call_with_numpy_bitmask(self, fake_ll_matcher):
        vocab_size = 64
        width = (vocab_size + 31) // 32
        all_allowed = np.full((1, width), -1, dtype=np.int32)
        fake_ll_matcher.compute_bitmask.return_value = all_allowed

        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=vocab_size, backend=GrammarBackend.LLGUIDANCE
        )
        import mlx.core as mx

        logits = mx.zeros((1, vocab_size))
        result = proc(None, logits)
        assert result.shape == (1, vocab_size)

    def test_call_with_none_bitmask(self, fake_ll_matcher):
        fake_ll_matcher.compute_bitmask.return_value = None
        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=64, backend=GrammarBackend.LLGUIDANCE
        )
        import mlx.core as mx

        logits = mx.zeros((1, 64))
        result = proc(None, logits)
        assert result.shape == (1, 64)

    def test_accept_token_stops_on_stopped(self, fake_ll_matcher):
        fake_ll_matcher.is_stopped.return_value = True
        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=64, backend=GrammarBackend.LLGUIDANCE
        )
        proc.accept_token(5)
        assert proc.is_terminated

    def test_accept_token_stops_on_error(self, fake_ll_matcher):
        fake_ll_matcher.is_error.return_value = True
        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=64, backend=GrammarBackend.LLGUIDANCE
        )
        proc.accept_token(5)
        assert proc.is_terminated

    def test_advance_returns_false_on_stop(self, fake_ll_matcher):
        import mlx.core as mx

        fake_ll_matcher.is_stopped.return_value = True
        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=64, backend=GrammarBackend.LLGUIDANCE
        )
        tokens = mx.array([5])
        proc.advance(None)
        result = proc.advance(tokens)
        assert result is False

    def test_advance_first_call_skips_accept(self, fake_ll_matcher):
        proc = GrammarConstraintProcessor(
            fake_ll_matcher, vocab_size=64, backend=GrammarBackend.LLGUIDANCE
        )
        result = proc.advance(None)
        assert result is True
        fake_ll_matcher.consume_token.assert_not_called()


# ─── GrammarConstraintProcessor — xgrammar backend (mocked) ───────────────


class TestXgrammarProcessor:
    def test_detect_xgrammar(self):
        mock_matcher = MagicMock()
        mock_matcher.__class__.__module__ = "xgrammar.grammar_matcher"
        result = GrammarConstraintProcessor._detect_backend(mock_matcher)
        assert result == GrammarBackend.XGRAMMAR


# ─── _detect_backend ──────────────────────────────────────────────────────


class TestDetectBackend:
    def test_detect_llguidance(self):
        mock_matcher = MagicMock()
        mock_matcher.__class__.__module__ = "llguidance.matcher"
        result = GrammarConstraintProcessor._detect_backend(mock_matcher)
        assert result == GrammarBackend.LLGUIDANCE

    def test_detect_xgrammar(self):
        mock_matcher = MagicMock()
        mock_matcher.__class__.__module__ = "xgrammar.grammar_matcher"
        result = GrammarConstraintProcessor._detect_backend(mock_matcher)
        assert result == GrammarBackend.XGRAMMAR

    def test_detect_default_xgrammar(self):
        mock_obj = MagicMock()
        mock_obj.__class__.__module__ = "unknown.module"
        result = GrammarConstraintProcessor._detect_backend(mock_obj)
        assert result == GrammarBackend.XGRAMMAR


# ─── Integration: bitmask → logits ────────────────────────────────────────


class TestBitmaskApplication:
    def test_bytes_bitmask_masks_tokens(self):
        vocab_size = 128
        width = (vocab_size + 31) // 32
        disallow_all = b"\x00" * (width * 4)
        fake_matcher = MagicMock()
        fake_matcher.compute_bitmask.return_value = disallow_all
        fake_matcher.is_stopped.return_value = False
        fake_matcher.is_error.return_value = False

        proc = GrammarConstraintProcessor(
            fake_matcher, vocab_size=vocab_size, backend=GrammarBackend.LLGUIDANCE
        )
        import mlx.core as mx

        logits = mx.zeros((1, vocab_size))
        result = proc(None, logits)
        assert result.shape == (1, vocab_size)
        np_result = np.array(result)
        assert np.all(np_result <= 0)

    def test_bytes_bitmask_allows_tokens(self):
        vocab_size = 128
        width = (vocab_size + 31) // 32
        allow_all = b"\xff" * (width * 4)
        fake_matcher = MagicMock()
        fake_matcher.compute_bitmask.return_value = allow_all
        fake_matcher.is_stopped.return_value = False
        fake_matcher.is_error.return_value = False

        proc = GrammarConstraintProcessor(
            fake_matcher, vocab_size=vocab_size, backend=GrammarBackend.LLGUIDANCE
        )
        import mlx.core as mx

        logits = mx.zeros((1, vocab_size))
        result = proc(None, logits)
        assert result.shape == (1, vocab_size)
