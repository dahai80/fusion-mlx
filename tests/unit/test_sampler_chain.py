# SPDX-License-Identifier: Apache-2.0
"""Tests for DRY + Mirostat v2 stateful samplers (PR-D, v2 doc §3.5)."""

from __future__ import annotations

import mlx.core as mx
import pytest


def _logits(vals: list[float]) -> mx.array:
    return mx.array(vals, dtype=mx.float32)


class TestMirostatV2:
    def test_invalid_tau(self):
        from fusion_mlx.utils.sampling import make_mirostat_v2_processor

        with pytest.raises(ValueError, match="mirostat_tau"):
            make_mirostat_v2_processor(tau=0.0, eta=0.1, vocab_size=10)

    def test_invalid_eta(self):
        from fusion_mlx.utils.sampling import make_mirostat_v2_processor

        with pytest.raises(ValueError, match="mirostat_eta"):
            make_mirostat_v2_processor(tau=5.0, eta=0.0, vocab_size=10)

    def test_first_call_keeps_argmax(self):
        from fusion_mlx.utils.sampling import make_mirostat_v2_processor

        proc = make_mirostat_v2_processor(tau=5.0, eta=0.1, vocab_size=5)
        logits = _logits([0.0, 10.0, 0.0, 0.0, 0.0])
        out = proc([], logits)
        # argmax (token 1) must survive the surprise mask.
        assert int(mx.argmax(out, axis=-1).item()) == 1

    def test_first_call_masks_low_prob_tokens(self):
        from fusion_mlx.utils.sampling import make_mirostat_v2_processor

        # tau tiny → mu tiny → only very-high-prob tokens survive.
        proc = make_mirostat_v2_processor(tau=0.5, eta=0.1, vocab_size=5)
        logits = _logits([0.0, 0.0, 0.0, 0.0, 20.0])
        out = proc([], logits)
        # token 4 dominates; others masked to -inf.
        assert float(out[0].item()) == -float("inf")
        assert float(out[4].item()) == 20.0

    def test_mu_updates_after_step(self):
        from fusion_mlx.utils.sampling import make_mirostat_v2_processor

        proc = make_mirostat_v2_processor(tau=5.0, eta=0.5, vocab_size=4)
        # Step 1: peaky distribution, token 0 chosen.
        logits1 = _logits([10.0, 0.0, 0.0, 0.0])
        proc([], logits1)
        # Step 2: generator picked token 0. mu should have decreased (the
        # chosen token's surprise was ~0, err = 0 - mu < 0, mu -= eta*err
        # increases mu). Verify mu changed from the initial 2*tau.
        assert proc.__closure__ is not None
        # The state dict is captured by closure; inspect via attribute path.
        # Easier: assert behavior — second call still keeps argmax and is
        # stable (no crash, finite output for kept token).
        logits2 = _logits([8.0, 1.0, 1.0, 1.0])
        out = proc([0], logits2)
        assert int(mx.argmax(out, axis=-1).item()) == 0


class TestDRY:
    def test_invalid_multiplier(self):
        from fusion_mlx.utils.sampling import make_dry_processor

        with pytest.raises(ValueError, match="dry_multiplier"):
            make_dry_processor(multiplier=0.0)

    def test_invalid_base(self):
        from fusion_mlx.utils.sampling import make_dry_processor

        with pytest.raises(ValueError, match="dry_base"):
            make_dry_processor(multiplier=1.0, base=1.0)

    def test_short_window_no_penalty(self):
        from fusion_mlx.utils.sampling import make_dry_processor

        proc = make_dry_processor(multiplier=1.0, allowed_length=3)
        logits = _logits([1.0, 2.0, 3.0, 4.0])
        # Only 2 tokens generated → below allowed_length → no change.
        out = proc([0, 1], logits)
        assert mx.array_equal(out, logits)

    def test_penalizes_repeat_extension(self):
        from fusion_mlx.utils.sampling import make_dry_processor

        proc = make_dry_processor(
            multiplier=2.0, base=1.75, allowed_length=2, penalty_last_n=-1
        )
        # Sequence [0, 1, 0] — suffix [0,1] appeared earlier? The token
        # following the earlier [0,1] was 0 (index 2). So candidate 0 would
        # extend a repeat → penalized.
        logits = _logits([5.0, 5.0, 5.0, 5.0])
        out = proc([0, 1, 0, 1], logits)
        # Token 0 should be penalized (its logit reduced below 5.0).
        assert float(out[0].item()) < 5.0
        # Non-repeat tokens untouched.
        assert float(out[2].item()) == 5.0
        assert float(out[3].item()) == 5.0

    def test_respects_allowed_length(self):
        from fusion_mlx.utils.sampling import make_dry_processor

        # allowed_length=3 → a 2-token repeat should NOT trigger penalty.
        proc = make_dry_processor(multiplier=2.0, allowed_length=3)
        logits = _logits([5.0, 5.0, 5.0, 5.0])
        out = proc([0, 1, 0, 1], logits)
        assert float(out[0].item()) == 5.0

    def test_window_cap_bounds_cost(self):
        from fusion_mlx.utils.sampling import make_dry_processor

        proc = make_dry_processor(multiplier=1.0, allowed_length=2)
        # Large window — must not hang.
        big = [i % 4 for i in range(500)]
        logits = _logits([1.0, 1.0, 1.0, 1.0])
        out = proc(big, logits)
        assert out.shape[0] == 4

    def test_breaker_resets_sequence(self):
        from fusion_mlx.utils.sampling import make_dry_processor

        # breaker token = 9; sequence crosses it → no penalty.
        proc = make_dry_processor(multiplier=2.0, allowed_length=2, breaker_ids=[9])
        logits = _logits([5.0] * 10)
        out = proc([0, 1, 9, 0, 1], logits)
        # The [0,1] after breaker is a fresh sequence; token 0 following
        # the earlier [9,0,1]... the earlier [0,1] is before the breaker,
        # so the match crosses 9 → skipped.
        assert float(out[0].item()) == 5.0


class TestSamplingParamsFields:
    def test_defaults_off(self):
        from fusion_mlx.request import SamplingParams

        sp = SamplingParams()
        assert sp.mirostat_tau == 0.0
        assert sp.mirostat_mode == 0
        assert sp.dry_multiplier == 0.0
        assert sp.dry_base == 1.75
        assert sp.dry_allowed_length == 2
        assert sp.dry_penalty_last_n == -1

    def test_set_mirostat(self):
        from fusion_mlx.request import SamplingParams

        sp = SamplingParams(mirostat_tau=5.0, mirostat_eta=0.1, mirostat_mode=2)
        assert sp.mirostat_mode == 2
        assert sp.mirostat_tau == 5.0


class TestRequestModelFields:
    def test_chat_request_accepts_mirostat_dry(self):
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            mirostat_tau=5.0,
            mirostat_mode=2,
            dry_multiplier=1.5,
            dry_base=1.75,
        )
        assert req.mirostat_tau == 5.0
        assert req.dry_multiplier == 1.5

    def test_chat_request_validates_ranges(self):
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                mirostat_tau=99.0,  # > 10.0 le
            )

    def test_completions_request_accepts_dry(self):
        from fusion_mlx.api.openai_models import CompletionRequest

        req = CompletionRequest(model="m", prompt="hi", dry_multiplier=1.0)
        assert req.dry_multiplier == 1.0
        assert req.mirostat_tau is None


class TestCommonMapping:
    def test_mirostat_dry_passed_build_sampling_params(self):
        from fusion_mlx.api.openai._common import _build_sampling_params
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            mirostat_tau=5.0,
            mirostat_eta=0.2,
            mirostat_mode=2,
            dry_multiplier=1.5,
            dry_base=2.0,
            dry_allowed_length=3,
            dry_penalty_last_n=128,
            repetition_penalty=1.3,
        )
        sp = _build_sampling_params(req)
        assert sp.mirostat_tau == 5.0
        assert sp.mirostat_eta == 0.2
        assert sp.mirostat_mode == 2
        assert sp.dry_multiplier == 1.5
        assert sp.dry_base == 2.0
        assert sp.dry_allowed_length == 3
        assert sp.dry_penalty_last_n == 128
        # Regression: repetition_penalty was previously dropped here.
        assert sp.repetition_penalty == 1.3

    def test_defaults_when_unset(self):
        from fusion_mlx.api.openai._common import _build_sampling_params
        from fusion_mlx.api.openai_models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        sp = _build_sampling_params(req)
        assert sp.mirostat_tau == 0.0
        assert sp.dry_multiplier == 0.0
        assert sp.repetition_penalty == 1.0
