"""model_aliases 完整测试（绕过 mlx import 链）。

覆盖 _load_aliases/list_aliases/list_profiles/resolve_model/
resolve_profile/suggest_similar 全部函数 + AliasProfile dataclass。
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_module():
    """直接按文件路径加载，绕过 fusion_mlx/__init__.py 的 mlx 依赖链。"""
    spec = importlib.util.spec_from_file_location(
        "model_aliases", "fusion_mlx/model_aliases.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ma = _load_module()


class TestAliasProfile(unittest.TestCase):

    def test_defaults(self):
        p = ma.AliasProfile(name="x", hf_path="org/x")
        self.assertEqual(p.name, "x")
        self.assertEqual(p.hf_path, "org/x")
        self.assertFalse(p.supports_dflash)
        self.assertFalse(p.is_moe)
        self.assertIsNone(p.drafter_hf_path)
        self.assertEqual(p.description, "")
        self.assertIsNone(p.tool_call_parser)
        self.assertIsNone(p.reasoning_parser)
        self.assertFalse(p.is_hybrid)
        self.assertTrue(p.supports_spec_decode)
        self.assertFalse(p.supports_mllm)
        self.assertFalse(p.is_audio)

    def test_all_fields(self):
        p = ma.AliasProfile(
            name="qwen",
            hf_path="Qwen/Qwen3",
            supports_dflash=True,
            is_moe=True,
            drafter_hf_path="Qwen/drafter",
            description="test",
            tool_call_parser="hermes",
            reasoning_parser="deepseek",
            is_hybrid=True,
            supports_spec_decode=False,
            supports_mllm=True,
            is_audio=True,
        )
        self.assertTrue(p.supports_dflash)
        self.assertTrue(p.is_moe)
        self.assertEqual(p.drafter_hf_path, "Qwen/drafter")
        self.assertFalse(p.supports_spec_decode)
        self.assertTrue(p.supports_mllm)
        self.assertTrue(p.is_audio)


class TestLoadAliases(unittest.TestCase):

    def test_missing_file_returns_empty(self):
        with patch.object(ma, "_ALIASES_FILE", Path("/nonexistent/aliases.json")):
            self.assertEqual(ma._load_aliases(), {})

    def test_valid_file_loads(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"gpt-4o": "Qwen/Qwen3-32B"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                result = ma._load_aliases()
                self.assertEqual(result, {"gpt-4o": "Qwen/Qwen3-32B"})

    def test_corrupt_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                f.write("{invalid json")
            with patch.object(ma, "_ALIASES_FILE", af):
                self.assertEqual(ma._load_aliases(), {})

    def test_non_dict_root_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump(["list", "not", "dict"], f)
            with patch.object(ma, "_ALIASES_FILE", af):
                self.assertEqual(ma._load_aliases(), {})


class TestListAliases(unittest.TestCase):

    def test_returns_sorted_keys(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"zeta": "z", "alpha": "a", "mid": "m"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                self.assertEqual(ma.list_aliases(), ["alpha", "mid", "zeta"])

    def test_empty_when_no_file(self):
        with patch.object(ma, "_ALIASES_FILE", Path("/nonexistent")):
            self.assertEqual(ma.list_aliases(), [])


class TestListProfiles(unittest.TestCase):

    def test_string_entry(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"gpt-4o": "Qwen/Qwen3-32B"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                profiles = ma.list_profiles()
                self.assertEqual(len(profiles), 1)
                self.assertEqual(profiles[0].name, "gpt-4o")
                self.assertEqual(profiles[0].hf_path, "Qwen/Qwen3-32B")

    def test_dict_entry_with_all_fields(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump(
                    {
                        "qwen": {
                            "hf_path": "Qwen/Qwen3",
                            "supports_dflash": True,
                            "is_moe": True,
                            "drafter_hf_path": "Qwen/d",
                            "description": "test",
                            "tool_call_parser": "hermes",
                            "reasoning_parser": "deepseek",
                            "is_hybrid": True,
                            "supports_spec_decode": False,
                            "supports_mllm": True,
                        }
                    },
                    f,
                )
            with patch.object(ma, "_ALIASES_FILE", af):
                p = ma.list_profiles()[0]
                self.assertTrue(p.supports_dflash)
                self.assertTrue(p.is_moe)
                self.assertEqual(p.drafter_hf_path, "Qwen/d")
                self.assertFalse(p.supports_spec_decode)
                self.assertTrue(p.supports_mllm)

    def test_dict_entry_with_path_key_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"x": {"path": "org/x"}}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                p = ma.list_profiles()[0]
                self.assertEqual(p.hf_path, "org/x")

    def test_dict_entry_missing_hf_path_uses_empty_or_name(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"x": {"description": "no path"}}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                p = ma.list_profiles()[0]
                # hf_path 缺失时回退到空字符串或 name（两种实现都接受）
                self.assertTrue(p.hf_path in ("", "x"))


class TestResolveModel(unittest.TestCase):

    def test_string_alias_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"gpt-4o": "Qwen/Qwen3-32B"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                self.assertEqual(ma.resolve_model("gpt-4o"), "Qwen/Qwen3-32B")

    def test_dict_alias_resolves_to_hf_path(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"x": {"hf_path": "org/x"}}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                self.assertEqual(ma.resolve_model("x"), "org/x")

    def test_hf_id_with_slash_passthrough(self):
        with patch.object(ma, "_ALIASES_FILE", Path("/nonexistent")):
            self.assertEqual(ma.resolve_model("Qwen/Qwen3-4B"), "Qwen/Qwen3-4B")

    def test_existing_path_passthrough(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model")
            os.makedirs(p)
            with patch.object(ma, "_ALIASES_FILE", Path("/nonexistent")), \
                 patch.object(ma, "_check_path_allowed"):
                self.assertEqual(ma.resolve_model(p), p)

    def test_unknown_returns_input(self):
        with patch.object(ma, "_ALIASES_FILE", Path("/nonexistent")):
            self.assertEqual(ma.resolve_model("unknown-model"), "unknown-model")


class TestResolveProfile(unittest.TestCase):

    def test_known_profile(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"gpt-4o": "Qwen/Qwen3"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                p = ma.resolve_profile("gpt-4o")
                self.assertIsNotNone(p)
                self.assertEqual(p.hf_path, "Qwen/Qwen3")

    def test_unknown_returns_none(self):
        with patch.object(ma, "_ALIASES_FILE", Path("/nonexistent")):
            self.assertIsNone(ma.resolve_profile("nonexistent"))


class TestSuggestSimilar(unittest.TestCase):

    def test_returns_close_matches(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"qwen3.5-4b": "a", "qwen3.5-9b": "b", "llama4-8b": "c"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                suggestions = ma.suggest_similar("qwen3.5-4b")
                self.assertIn("qwen3.5-4b", suggestions)

    def test_no_match_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"qwen": "a"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                self.assertEqual(ma.suggest_similar("zzzzzzzzz"), [])

    def test_custom_n_and_cutoff(self):
        with tempfile.TemporaryDirectory() as td:
            af = Path(td) / "aliases.json"
            with open(af, "w") as f:
                json.dump({"abc": "a", "abd": "b", "abe": "c", "abf": "d"}, f)
            with patch.object(ma, "_ALIASES_FILE", af):
                result = ma.suggest_similar("abc", n=2, cutoff=0.5)
                self.assertLessEqual(len(result), 2)


class TestPopularAliasesConstant(unittest.TestCase):

    def test_popular_aliases_list(self):
        self.assertIsInstance(ma.POPULAR_ALIASES, list)
        self.assertGreater(len(ma.POPULAR_ALIASES), 0)
        self.assertIn("qwen3.5-4b-4bit", ma.POPULAR_ALIASES)


if __name__ == "__main__":
    unittest.main()
