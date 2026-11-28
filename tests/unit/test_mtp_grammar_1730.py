# SPDX-License-Identifier: Apache-2.0
# #1730 (GaaD): grammar-forced tokens compose with the MTP draft path.
# Covers the compose primitives and the fail-visible gates:
#   - _single_allowed_token: bitmask popcount == 1 -> forced token splice
#   - _grammar_processor_of: llguidance-only detection, ring unwrap,
#     per-row list flattening
#   - mtp_generate_step gates: non-llguidance matchers and non-greedy
#     grammar requests raise instead of silently running shared-state
#     masking under speculation

import unittest
from unittest.mock import MagicMock

import mlx.core as mx

from fusion_mlx.api.grammar import GrammarBackend
from fusion_mlx.speculative.mtp.generator import (
    _grammar_processor_of,
    _single_allowed_token,
    mtp_generate_step,
    mtp_grammar_compose_enabled,
)


def _llguidance_proc():
    p = MagicMock()
    p.accept_token = MagicMock()
    p.matcher = MagicMock()
    p._backend = GrammarBackend.LLGUIDANCE
    return p


def _xgrammar_like_proc():
    p = MagicMock()
    p.accept_token = MagicMock()
    p.matcher = MagicMock()
    p._backend = "xgrammar"
    return p


class TestSingleAllowedToken(unittest.TestCase):
    def test_single_bit_returns_token(self):
        import numpy as np

        words = np.zeros(10, dtype=np.int32)
        words[3] = 1 << 7
        self.assertEqual(_single_allowed_token(words, 320), 3 * 32 + 7)

    def test_multiple_allowed_returns_none(self):
        import numpy as np

        words = np.zeros(10, dtype=np.int32)
        words[0] = 0b11
        self.assertIsNone(_single_allowed_token(words, 320))

    def test_all_ones_word_returns_none(self):
        import numpy as np

        words = np.array([-1], dtype=np.int32)
        self.assertIsNone(_single_allowed_token(words, 320))

    def test_none_bitmask_returns_none(self):
        self.assertIsNone(_single_allowed_token(None, 320))

    def test_bits_beyond_vocab_ignored(self):
        import numpy as np

        # vocab 40: word 1 covers ids 32..63, only ids < 40 decodable.
        # Single in-vocab bit (id 35) + bits past vocab -> forced id 35.
        words = np.zeros(2, dtype=np.int32)
        words[1] = np.int32(-65528)  # 0xFFFF0008: id 35 + past-vocab bits 48..63
        self.assertEqual(_single_allowed_token(words, 40), 35)


class TestGrammarProcessorOf(unittest.TestCase):
    def test_llguidance_proc_detected(self):
        p = _llguidance_proc()
        self.assertIs(_grammar_processor_of([[p]]), p)

    def test_flat_list_detected(self):
        p = _llguidance_proc()
        self.assertIs(_grammar_processor_of([p]), p)

    def test_ring_wrapper_unwrapped(self):
        inner = _llguidance_proc()
        ring = MagicMock(spec=["_wrapped", "accept_token", "matcher"])
        ring._wrapped = inner
        ring.matcher = inner.matcher
        ring.accept_token = MagicMock()
        self.assertIs(_grammar_processor_of([[ring]]), ring)

    def test_xgrammar_shaped_returns_none(self):
        self.assertIsNone(_grammar_processor_of([[_xgrammar_like_proc()]]))

    def test_empty_returns_none(self):
        self.assertIsNone(_grammar_processor_of(None))
        self.assertIsNone(_grammar_processor_of([[None]]))


class TestMtpGrammarComposeEnv(unittest.TestCase):
    def test_default_on(self):
        import os

        old = os.environ.pop("FUSION_MLX_MTP_GRAMMAR", None)
        try:
            self.assertTrue(mtp_grammar_compose_enabled())
        finally:
            if old is not None:
                os.environ["FUSION_MLX_MTP_GRAMMAR"] = old

    def test_env_off(self):
        import os

        old = os.environ.get("FUSION_MLX_MTP_GRAMMAR")
        os.environ["FUSION_MLX_MTP_GRAMMAR"] = "0"
        try:
            self.assertFalse(mtp_grammar_compose_enabled())
        finally:
            if old is None:
                os.environ.pop("FUSION_MLX_MTP_GRAMMAR", None)
            else:
                os.environ["FUSION_MLX_MTP_GRAMMAR"] = old


class TestMtpGenerateStepGrammarGates(unittest.TestCase):
    def _fake_model_cache(self):
        class _C:
            def is_trimmable(self):
                return False

        model = MagicMock()
        model.layers = [_C()]
        model.mtp_forward = MagicMock()
        return model, [_C(), _C()]

    def _run(self, procs, temp=0.0):
        model, cache = self._fake_model_cache()
        prompt = mx.array([1, 2, 3], mx.uint32)
        list(
            mtp_generate_step(
                prompt,
                model,
                max_tokens=4,
                logits_processors=procs,
                prompt_cache=cache,
                temp=temp,
            )
        )

    def test_non_llguidance_matcher_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._run([[_xgrammar_like_proc()]])
        self.assertIn("llguidance", str(ctx.exception))

    def test_non_greedy_grammar_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._run([[_llguidance_proc()]], temp=0.7)
        self.assertIn("greedy=False", str(ctx.exception))

    def test_compose_disabled_raises(self):
        import os

        old = os.environ.get("FUSION_MLX_MTP_GRAMMAR")
        os.environ["FUSION_MLX_MTP_GRAMMAR"] = "0"
        try:
            with self.assertRaises(ValueError) as ctx:
                self._run([[_llguidance_proc()]])
            self.assertIn("compose unavailable", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("FUSION_MLX_MTP_GRAMMAR", None)
            else:
                os.environ["FUSION_MLX_MTP_GRAMMAR"] = old

    def test_grammar_with_other_processors_raises(self):
        other = object()  # no accept_token attr -> non-grammar processor
        with self.assertRaises(ValueError) as ctx:
            self._run([[_llguidance_proc(), other]])
        self.assertIn("cannot coexist", str(ctx.exception))


class TestGrammarDrain(unittest.TestCase):
    # GrammarMaskRing prefetches masks on a background worker; compose
    # calls matcher methods directly from the batch thread. Without the
    # drain the two overlap and llguidance raises "Already borrowed"
    # (RefCell double borrow, #1730 live finding).

    def test_drain_calls_ring_drain_job(self):
        from fusion_mlx.patches.mlx_lm_mtp.batch_generator import _grammar_drain

        calls = []
        proc = MagicMock()
        proc._drain_job = lambda: calls.append(1)
        _grammar_drain(proc)
        assert calls == [1]

    def test_drain_noop_without_ring(self):
        from fusion_mlx.patches.mlx_lm_mtp.batch_generator import _grammar_drain

        _grammar_drain(object())  # must not raise

    def test_mask_logits_drains_before_bitmask(self):
        import fusion_mlx.shim.grammar_ring as ring
        from fusion_mlx.patches.mlx_lm_mtp import batch_generator as bg

        order = []
        proc = MagicMock()

        def _drain():
            order.append("drain")

        proc._drain_job = _drain
        proc.matcher.compute_bitmask = MagicMock(
            side_effect=lambda: order.append("bitmask") or b"\x00" * 64
        )
        logits = MagicMock()
        logits.shape = (1, 512)

        orig = ring.apply_bitmask
        ring.apply_bitmask = lambda mask, lg, vocab: order.append("apply")
        try:
            bg._grammar_mask_logits(proc, logits)
        finally:
            ring.apply_bitmask = orig
        assert order == ["drain", "bitmask", "apply"]


if __name__ == "__main__":
    unittest.main()
