# SPDX-License-Identifier: Apache-2.0
"""
force_text wiring gates (#393 escape hatch, now live).

Covers the b+c landing: ServerConfig.force_text default, EnginePool
storage + VLM flip condition referencing it, and the config_resolve
staging on the --model-dir pool boot path.
"""

from __future__ import annotations

import ast
import importlib.util
import unittest


def _load_config_module():
    spec = importlib.util.spec_from_file_location("config", "fusion_mlx/config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestServerConfigForceText(unittest.TestCase):
    def test_default_false(self):
        m = _load_config_module()
        self.assertFalse(m.ServerConfig().force_text)

    def test_explicit_true(self):
        m = _load_config_module()
        self.assertTrue(m.ServerConfig(force_text=True).force_text)


class TestEnginePoolForceText(unittest.TestCase):
    def test_stores_flag(self):
        from fusion_mlx.pool.engine_pool import EnginePool

        pool = EnginePool(force_text=True)
        self.assertTrue(pool._force_text)

    def test_default_false(self):
        from fusion_mlx.pool.engine_pool import EnginePool

        pool = EnginePool()
        self.assertFalse(pool._force_text)


class TestForceTextFlipCondition(unittest.TestCase):
    def test_load_engine_flip_references_pool_flag(self):
        src = open("fusion_mlx/pool/engine_pool.py").read()
        tree = ast.parse(src)
        flip_src = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_load_engine":
                flip_src = ast.get_source_segment(src, node) or ""
        self.assertIn(
            '(force_lm or self._force_text) and effective_type == "vlm"',
            flip_src,
        )


class TestConfigResolveStaging(unittest.TestCase):
    def test_model_dir_path_maps_no_mllm_to_force_text(self):
        src = open("fusion_mlx/cli_serve/config_resolve.py").read()
        self.assertIn("config.force_text = True", src)
        tree = ast.parse(src)
        staged = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_serve_from_model_dir"
            ):
                staged = [
                    ast.get_source_segment(src, n) or ""
                    for n in ast.walk(node)
                    if isinstance(n, ast.Assign)
                    and "force_text" in (ast.get_source_segment(src, n) or "")
                ]
        self.assertTrue(staged, "force_text staging missing from _serve_from_model_dir")


if __name__ == "__main__":
    unittest.main()
