from unittest.mock import MagicMock, patch

import pytest

from fusion_mlx.speculative.prompt_lookup import (
    PromptLookupDecoder,
)


class TestDraftAcceptanceRate:
    """Test acceptance rate tracking in PromptLookupDecoder."""

    def _make_decoder(self, **kwargs):
        return PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, **kwargs)

    def test_zero_acceptance(self):
        dec = self._make_decoder()
        dec.add_prompt_tokens([1, 2, 1, 2, 1, 2])
        drafts = dec.get_draft_tokens()
        # No acceptance recorded
        stats = dec.get_stats()
        assert stats["accepted_tokens"] == 0
        assert stats["acceptance_rate"] == 0.0

    def test_full_acceptance(self):
        dec = self._make_decoder()
        dec.add_prompt_tokens([1, 2, 3, 1, 2, 3, 1, 2])
        drafts = dec.get_draft_tokens()
        if drafts:
            dec.record_accepted(len(drafts))
            stats = dec.get_stats()
            assert stats["accepted_tokens"] == len(drafts)
            assert stats["acceptance_rate"] == 1.0

    def test_partial_acceptance(self):
        dec = self._make_decoder()
        dec.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4, 6])
        drafts = dec.get_draft_tokens()
        if drafts:
            dec.record_accepted(2)
            stats = dec.get_stats()
            assert stats["accepted_tokens"] == 2
            assert 0 < stats["acceptance_rate"] < 1.0

    def test_multiple_draft_rounds(self):
        dec = self._make_decoder()
        dec.add_prompt_tokens([1, 2, 3, 1, 2, 3, 1, 2, 3])
        drafts1 = dec.get_draft_tokens()
        if drafts1:
            dec.record_accepted(2)
        # Add more tokens, generate another draft
        dec.add_generated_token(99)
        drafts2 = dec.get_draft_tokens()
        if drafts2:
            dec.record_accepted(1)

        stats = dec.get_stats()
        assert stats["total_drafts"] >= 1
        assert stats["successful_drafts"] >= 1

    def test_stats_history_size(self):
        dec = self._make_decoder()
        dec.add_prompt_tokens([1, 2, 3, 4, 5])
        stats = dec.get_stats()
        assert stats["history_size"] == 5

    def test_no_drafts_below_ngram(self):
        dec = self._make_decoder(ngram_size=4)
        dec.add_prompt_tokens([1, 2, 3])
        assert dec.get_draft_tokens() == []


class TestFallbackOnRejection:
    """Test behavior when draft tokens are rejected."""

    def test_no_match_returns_empty(self):
        dec = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=2)
        # All unique tokens, no n-gram repetition
        dec.add_prompt_tokens([10, 20, 30, 40, 50, 60])
        drafts = dec.get_draft_tokens()
        assert drafts == []

    def test_min_matches_gate(self):
        dec = PromptLookupDecoder(num_draft_tokens=4, ngram_size=2, min_matches=3)
        # Pattern [1, 2] repeats but continuation is only 1 token long
        dec.add_prompt_tokens([1, 2, 3, 1, 2])
        drafts = dec.get_draft_tokens()
        # Continuation after [1,2] is [3], length=1 < min_matches=3
        assert drafts == []

    def test_single_match_skipped(self):
        dec = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=2)
        # Pattern appears only once, no continuation
        dec.add_prompt_tokens([1, 2, 3, 4, 5])
        drafts = dec.get_draft_tokens()
        assert drafts == []

    def test_current_occurrence_skipped(self):
        dec = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=1)
        # Only one occurrence of [1,2,3], the query itself
        dec.add_prompt_tokens([1, 2, 3, 4])
        drafts = dec.get_draft_tokens()
        # The only position for [1,2,3] is the current one, so it's skipped
        assert drafts == []

    def test_repeated_pattern_yields_draft(self):
        dec = PromptLookupDecoder(num_draft_tokens=3, ngram_size=2, min_matches=1)
        dec.add_prompt_tokens([1, 2, 3, 4, 1, 2])
        drafts = dec.get_draft_tokens()
        # [1,2] appeared at pos 0 and 4. At pos 0, continuation is [3,4].
        # At pos 4, continuation is empty. Best is [3,4], length=2 >= min_matches=1
        assert len(drafts) >= 1
        assert drafts[0] == 3

    def test_rejection_does_not_mutate_history(self):
        dec = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=2)
        dec.add_prompt_tokens([10, 20, 30, 40, 50])
        before_len = len(dec._token_history)
        drafts = dec.get_draft_tokens()
        after_len = len(dec._token_history)
        # get_draft_tokens should not modify history
        assert before_len == after_len


class TestNggramMatching:
    """Test n-gram index construction and lookup."""

    def test_ngram_index_populated(self):
        dec = PromptLookupDecoder(ngram_size=3)
        dec.add_prompt_tokens([1, 2, 3, 4])
        # n-grams: (1,), (1,2), (1,2,3), (2,), (2,3), (2,3,4), (3,), (3,4), (4,)
        assert (1, 2, 3) in dec._ngram_index
        assert (2, 3, 4) in dec._ngram_index

    def test_ngram_positions_correct(self):
        dec = PromptLookupDecoder(ngram_size=2)
        dec.add_prompt_tokens([1, 2, 1, 2])
        # (1,2) should appear at positions 0 and 2
        positions = dec._ngram_index.get((1, 2), [])
        assert 0 in positions
        assert 2 in positions

    def test_reset_clears_index(self):
        dec = PromptLookupDecoder(ngram_size=2)
        dec.add_prompt_tokens([1, 2, 3])
        dec.reset()
        assert dec._token_history == []
        assert len(dec._ngram_index) == 0

    def test_generated_token_extends_index(self):
        dec = PromptLookupDecoder(ngram_size=2)
        dec.add_prompt_tokens([1, 2])
        dec.add_generated_token(3)
        assert (2, 3) in dec._ngram_index

    def test_best_continuation_selected(self):
        dec = PromptLookupDecoder(num_draft_tokens=4, ngram_size=2, min_matches=1)
        # [1,2] at pos 0 -> continuation [3,4,5]
        # [1,2] at pos 4 -> continuation [3]
        dec.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3])
        drafts = dec.get_draft_tokens()
        # Best continuation is [3,4,5] (length 3)
        assert drafts == [3, 4, 5]

    def test_draft_capped_at_num_draft_tokens(self):
        dec = PromptLookupDecoder(num_draft_tokens=2, ngram_size=2, min_matches=1)
        dec.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2])
        drafts = dec.get_draft_tokens()
        assert len(drafts) <= 2
