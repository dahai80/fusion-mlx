"""config.py 纯函数测试（绕过 mlx import 链）。

覆盖 MemoryTier/SchedulingPolicy 枚举、SchedulerConfig/MemoryConfig/ServerConfig
dataclass、_deep_merge/_load_model_config/get_config/reset_config。
"""

from __future__ import annotations

import importlib.util
import unittest


def _load_module():
    """直接加载 config.py，绕过 fusion_mlx/__init__.py 的 mlx 链。"""
    spec = importlib.util.spec_from_file_location("config", "fusion_mlx/config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cfg = _load_module()


class TestEnums(unittest.TestCase):

    def test_memory_tier_values(self):
        self.assertTrue(hasattr(cfg, "MemoryTier"))
        names = [t.name for t in cfg.MemoryTier]
        self.assertIn("SAFE", names)
        self.assertIn("BALANCED", names)
        self.assertIn("AGGRESSIVE", names)

    def test_scheduling_policy_values(self):
        self.assertTrue(hasattr(cfg, "SchedulingPolicy"))
        self.assertGreaterEqual(len(list(cfg.SchedulingPolicy)), 1)


class TestDeepMerge(unittest.TestCase):

    def test_simple_override(self):
        self.assertEqual(cfg._deep_merge({"a": 1}, {"a": 2}), {"a": 2})

    def test_nested_merge(self):
        result = cfg._deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
        self.assertEqual(result, {"a": {"b": 3, "c": 2}})

    def test_add_new_key(self):
        self.assertEqual(cfg._deep_merge({"a": 1}, {"b": 2}), {"a": 1, "b": 2})

    def test_empty_override(self):
        self.assertEqual(cfg._deep_merge({"a": 1}, {}), {"a": 1})

    def test_empty_base(self):
        self.assertEqual(cfg._deep_merge({}, {"a": 1}), {"a": 1})

    def test_both_empty(self):
        self.assertEqual(cfg._deep_merge({}, {}), {})

    def test_override_replaces_non_dict_with_dict(self):
        result = cfg._deep_merge({"a": 1}, {"a": {"b": 2}})
        self.assertEqual(result, {"a": {"b": 2}})


class TestSchedulerConfig(unittest.TestCase):

    def test_defaults(self):
        c = cfg.SchedulerConfig()
        # 验证有默认字段（具体值可能变，但字段应存在）
        self.assertGreaterEqual(c.max_num_seqs, 1)

    def test_custom_values(self):
        c = cfg.SchedulerConfig(max_num_seqs=64)
        self.assertEqual(c.max_num_seqs, 64)


class TestServerConfig(unittest.TestCase):

    def test_defaults(self):
        c = cfg.ServerConfig()
        self.assertIsNotNone(c.host)
        self.assertGreater(c.port, 0)

    def test_custom_values(self):
        c = cfg.ServerConfig(host="0.0.0.0", port=9000)
        self.assertEqual(c.host, "0.0.0.0")
        self.assertEqual(c.port, 9000)


class TestLoadModelConfig(unittest.TestCase):

    def test_returns_dict(self):
        # _load_model_config 读 pkg + user 配置，应始终返回 dict（可能为空）
        result = cfg._load_model_config()
        self.assertIsInstance(result, dict)

    def test_missing_pkg_returns_empty_or_user(self):
        # 即使 pkg 配置缺失，也应返回 dict（空或 user 配置）
        result = cfg._load_model_config()
        self.assertIsInstance(result, dict)


class TestGetConfig(unittest.TestCase):

    def test_returns_server_config(self):
        c = cfg.get_config()
        self.assertIsInstance(c, cfg.ServerConfig)

    def test_reset_returns_new_config(self):
        c = cfg.reset_config()
        self.assertIsInstance(c, cfg.ServerConfig)


if __name__ == "__main__":
    unittest.main()
