# SPDX-License-Identifier: Apache-2.0
import unittest

from fusion_mlx.model_aliases import AliasProfile


class TestCapabilitiesProperty(unittest.TestCase):
    def test_empty_profile_only_spec_decode(self):
        p = AliasProfile()
        self.assertEqual(p.capabilities, frozenset({"spec_decode"}))

    def test_dflash_with_spec(self):
        p = AliasProfile(supports_dflash=True)
        self.assertEqual(p.capabilities, frozenset({"dflash", "spec_decode"}))

    def test_mllm_with_tools(self):
        p = AliasProfile(
            supports_mllm=True,
            tool_call_parser="hermes",
            supports_spec_decode=True,
        )
        self.assertEqual(
            p.capabilities, frozenset({"vision", "tool_call", "spec_decode"})
        )

    def test_moe_hybrid(self):
        p = AliasProfile(is_moe=True, is_hybrid=True, supports_dspark=True)
        self.assertEqual(
            p.capabilities, frozenset({"moe", "hybrid", "dspark", "spec_decode"})
        )

    def test_audio_reasoning(self):
        p = AliasProfile(is_audio=True, reasoning_parser="deepseek")
        self.assertEqual(
            p.capabilities, frozenset({"audio", "reasoning", "spec_decode"})
        )

    def test_frozen_dataclass_immutability(self):
        p = AliasProfile(supports_dflash=True)
        with self.assertRaises(AttributeError):
            p.supports_dflash = False

    def test_no_spec_decode_excluded(self):
        p = AliasProfile(supports_spec_decode=False)
        self.assertNotIn("spec_decode", p.capabilities)


if __name__ == "__main__":
    unittest.main()
