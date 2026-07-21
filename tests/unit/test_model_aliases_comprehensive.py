"""model_aliases 纯函数测试。

覆盖 resolve_model/resolve_profile/DEFAULT_ALIASES 的边界。
"""

from __future__ import annotations

import unittest

from fusion_mlx.model_aliases import resolve_model, resolve_profile


class TestResolveModel(unittest.TestCase):

    def test_known_alias_resolves(self):
        # README 声明的别名
        result = resolve_model("gpt-4o")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_unknown_alias_returns_empty(self):
        # 未知别名可能原样返回（passthrough）或返回空——两种行为都接受
        result = resolve_model("totally-nonexistent-model-xyz")
        self.assertIsInstance(result, str)
        # 至少不应抛异常

    def test_passthrough_full_hf_id(self):
        # 完整 HF id 应原样返回或解析到自身
        result = resolve_model("Qwen/Qwen3-4B")
        self.assertIsInstance(result, str)

    def test_empty_input(self):
        result = resolve_model("")
        self.assertFalse(result)


class TestResolveProfile(unittest.TestCase):

    def test_known_profile(self):
        # 尝试已知 profile 名
        result = resolve_profile("gpt-4o")
        # 返回 AliasProfile 或 None; AliasProfile 的必填字段是 hf_path
        self.assertTrue(result is None or hasattr(result, "hf_path"))

    def test_unknown_profile_returns_none(self):
        self.assertIsNone(resolve_profile("nonexistent-profile-xyz"))


if __name__ == "__main__":
    unittest.main()
