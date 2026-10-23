# SPDX-License-Identifier: Apache-2.0
"""
Pool-mode contention suppression (#contention-pool) + spec-route flood
fix (#spec-route-flood) gates.

Covers the cross-scheduler decode registry and the AST-gated wiring in
sched_step.py: sibling-engine suppression of the contention flag, and the
`is None` sentinel that stops per-step spec-method re-decisions.
"""

from __future__ import annotations

import ast
import unittest

from fusion_mlx.scheduler import _contention


class TestContentionRegistry(unittest.TestCase):
    def setUp(self):
        _contention.reset()
        _contention.set_pool_shared(False)

    def tearDown(self):
        _contention.reset()
        _contention.set_pool_shared(False)

    def test_pool_shared_flag_roundtrip(self):
        self.assertFalse(_contention.pool_shared())
        _contention.set_pool_shared(True)
        self.assertTrue(_contention.pool_shared())

    def test_self_decode_not_counted_as_sibling(self):
        _contention.mark_decode(1, now=100.0)
        self.assertEqual(_contention.others_decoding(1, now=100.0), 0)

    def test_sibling_within_window_counted(self):
        _contention.mark_decode(1, now=100.0)
        _contention.mark_decode(2, now=100.5)
        self.assertEqual(_contention.others_decoding(1, now=101.0), 1)

    def test_sibling_outside_window_not_counted(self):
        _contention.mark_decode(1, now=100.0)
        _contention.mark_decode(2, now=100.0)
        self.assertEqual(_contention.others_decoding(1, now=110.0), 0)

    def test_stale_entries_pruned(self):
        _contention.mark_decode(2, now=1.0)
        for i in range(20):
            _contention.mark_decode(100 + i, now=1000.0 + i)
        self.assertNotIn(2, _contention._last_decode)


class TestSchedStepGates(unittest.TestCase):
    def _load_sched_step_src(self):
        return open("fusion_mlx/scheduler/sched_step.py").read()

    def test_contention_flag_gated_on_no_siblings(self):
        src = self._load_sched_step_src()
        self.assertIn(
            "_siblings = others_decoding(id(self), window_s=_span + 15.0)", src
        )
        self.assertIn("_cv > self._contention_cv_threshold", src)
        self.assertIn("_siblings == 0", src)

    def test_contention_flag_gated_on_slowdown_ratio(self):
        src = self._load_sched_step_src()
        self.assertIn("_mean >= 3.0 * self._decode_baseline_dt", src)
        self.assertIn("_slowdown", src)

    def test_contention_flag_gated_on_pool_mode(self):
        src = self._load_sched_step_src()
        self.assertIn("not pool_shared()", src)
        pool_src = open("fusion_mlx/pool/engine_pool.py").read()
        self.assertIn("_mark_pool_shared_if_multi()", pool_src)
        self.assertIn("set_pool_shared(True)", pool_src)

    def test_decode_steps_mark_registry(self):
        src = self._load_sched_step_src()
        self.assertIn("mark_decode(id(self))", src)

    def test_spec_method_sentinel_is_none_check(self):
        src = self._load_sched_step_src()
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            seg = ast.get_source_segment(src, node) or ""
            if (
                "request._active_spec_method" in seg
                and "_decide_spec_method(request)" in seg
            ):
                self.assertIn("is None", seg)
                found = True
        self.assertTrue(found, "_decide_spec_method call site not found")


if __name__ == "__main__":
    unittest.main()
