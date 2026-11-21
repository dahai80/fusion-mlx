# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM-parity min_tokens EOS suppression (#1683)."""

import math

import mlx.core as mx
import pytest

from fusion_mlx.request import SamplingParams
from fusion_mlx.utils.sampling import MinTokensProcessor

EOS = 248046
# Qwen3.5-class vocab (EOS id sits above the legacy 128256 Qwen2.5 size —
# an out-of-range index silently no-ops in mlx, which is why VOCAB must
# actually contain the EOS id).
VOCAB = 262144


def _logits_2d():
    logits = mx.zeros((1, VOCAB))
    return logits


def _logits_1d():
    return mx.zeros((VOCAB,))


def _eos_logit(logits):
    if logits.ndim > 1:
        return float(logits[0, EOS])
    return float(logits[EOS])


class TestMinTokensProcessor:
    def test_suppresses_eos_until_min_reached(self):
        proc = MinTokensProcessor(3, [EOS])
        tokens = [1, 2, 3]  # prompt history, 3 tokens
        # First decode call: history still prompt-sized (last prompt token
        # re-fed), 0 generated tokens -> EOS must be masked.
        logits = proc(tokens, _logits_2d())
        assert _eos_logit(logits) == float("-inf")
        # After 2 generated tokens appended to history, still suppressed
        # (generated=2 < 3).
        logits = proc(tokens + [10, 11], _logits_2d())
        assert _eos_logit(logits) == float("-inf")
        # Third generated token: generated=3 >= min -> released.
        logits = proc(tokens + [10, 11, 12], _logits_2d())
        assert _eos_logit(logits) == 0.0
        # Done is latched: later calls are a no-op even after re-growth.
        logits = proc(tokens + [10, 11, 12, 13], _logits_2d())
        assert _eos_logit(logits) == 0.0

    def test_counts_multi_token_jump(self):
        proc = MinTokensProcessor(5, [EOS])
        tokens = [1, 2]
        logits = proc(tokens, _logits_2d())
        assert _eos_logit(logits) == float("-inf")
        # Spec-decode style jump: 4 tokens accepted at once.
        logits = proc(tokens + [7, 8, 9, 10], _logits_2d())
        # generated=4 < 5 -> still suppressed
        assert _eos_logit(logits) == float("-inf")
        logits = proc(tokens + [7, 8, 9, 10, 11], _logits_2d())
        assert _eos_logit(logits) == 0.0

    def test_suppresses_all_eos_ids(self):
        proc = MinTokensProcessor(2, [EOS, 15])
        logits = proc([1, 2], _logits_2d())
        assert float(logits[0, EOS]) == float("-inf")
        assert float(logits[0, 15]) == float("-inf")
        assert float(logits[0, 16]) == 0.0

    def test_1d_logits_supported(self):
        proc = MinTokensProcessor(1, [EOS])
        logits = proc([1, 2], _logits_1d())
        assert _eos_logit(logits) == float("-inf")
        logits = proc([1, 2, 9], _logits_1d())
        assert _eos_logit(logits) == 0.0

    def test_released_logits_are_finite_elsewhere(self):
        proc = MinTokensProcessor(1, [EOS])
        logits = proc([1, 2], _logits_2d())
        arr = logits[0]
        # EOS masked, a non-EOS entry stays finite.
        assert float(arr[EOS]) == float("-inf")
        assert math.isfinite(float(arr[123]))

    def test_invalid_min_tokens_rejected(self):
        with pytest.raises(ValueError):
            MinTokensProcessor(0, [EOS])
        with pytest.raises(ValueError):
            MinTokensProcessor(-1, [EOS])

    def test_empty_eos_ids_rejected(self):
        with pytest.raises(ValueError):
            MinTokensProcessor(2, [])


class TestSamplingParamsMinTokens:
    def test_default_zero(self):
        sp = SamplingParams()
        assert sp.min_tokens == 0

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            SamplingParams(min_tokens=-1)

    def test_positive_accepted(self):
        sp = SamplingParams(min_tokens=8)
        assert sp.min_tokens == 8
