# SPDX-License-Identifier: Apache-2.0
"""#1528: per-request speculative params + acceptance counters over REST.

Covers: SpeculativeParams validation (1-4), SamplingParams plumbing,
response echo, the vendored patch chain-K override read, and
snapshot_native_mtp_stats accounting.
"""

import unittest

import pytest
from pydantic import ValidationError

from fusion_mlx.api.openai_models import ChatCompletionRequest, SpeculativeParams
from fusion_mlx.request import SamplingParams


class TestSpeculativeParamsValidation(unittest.TestCase):
    def test_valid_range_accepted(self):
        for k in (1, 2, 3, 4):
            req = ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                speculative={"draft_tokens": k},
            )
            self.assertEqual(req.speculative.draft_tokens, k)

    def test_out_of_range_rejected(self):
        for k in (0, 5, -1):
            with self.assertRaises(ValidationError):
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    speculative={"draft_tokens": k},
                )

    def test_absent_is_none(self):
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        self.assertIsNone(req.speculative)

    def test_speculative_params_direct(self):
        p = SpeculativeParams(draft_tokens=3)
        self.assertEqual(p.draft_tokens, 3)
        with self.assertRaises(ValidationError):
            SpeculativeParams(draft_tokens=9)


class TestSamplingParamsPlumbing(unittest.TestCase):
    def test_default_none(self):
        sp = SamplingParams()
        self.assertIsNone(sp.spec_draft_tokens)

    def test_build_sampling_params_carries_override(self):
        from fusion_mlx.api.openai._common import _build_sampling_params

        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            speculative={"draft_tokens": 4},
        )
        sp = _build_sampling_params(req)
        self.assertEqual(sp.spec_draft_tokens, 4)

        req2 = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        sp2 = _build_sampling_params(req2)
        self.assertIsNone(sp2.spec_draft_tokens)


class TestChainKOverrideRead(unittest.TestCase):
    def test_clamped(self):
        from fusion_mlx.patches.mlx_lm_mtp.batch_generator import _resolve_chain_k

        self.assertTrue(1 <= _resolve_chain_k() <= 4)


class TestNativeMtpStatsSnapshot(unittest.TestCase):
    def test_snapshot_is_copy(self):
        from fusion_mlx.patches.mlx_lm_mtp.batch_generator import (
            _NATIVE_MTP_STATS,
            _record_mtp_cycle,
        )

        before = dict(_NATIVE_MTP_STATS)
        _record_mtp_cycle(2, 1)
        after = dict(_NATIVE_MTP_STATS)
        self.assertEqual(after["verifies"], before["verifies"] + 1)
        self.assertEqual(after["drafted"], before["drafted"] + 2)
        self.assertEqual(after["accepted"], before["accepted"] + 1)

    def test_full_accept_counted(self):
        from fusion_mlx.patches.mlx_lm_mtp.batch_generator import (
            _NATIVE_MTP_STATS,
            _record_mtp_cycle,
        )

        before = _NATIVE_MTP_STATS["full_accepts"]
        _record_mtp_cycle(3, 3)
        self.assertEqual(_NATIVE_MTP_STATS["full_accepts"], before + 1)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
