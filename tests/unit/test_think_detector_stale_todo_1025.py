# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1025.

#1025: model_auto_config/core.py had a stale TODO claiming
fusion_mlx/reasoning/think_detector.py "does not exist yet" — but the
module DOES exist (ThinkDetector class + looks_like_autonomous_think).

Fix (#1025): updated the stale comment to reflect that the module exists
but is UNWIRED (legacy regex dispatch still owns parser selection).
"""

from __future__ import annotations

import pytest


class TestThinkDetectorModuleExists:
    """The referenced module exists and is importable."""

    def test_think_detector_importable(self):
        from fusion_mlx.reasoning.think_detector import ThinkDetector

        assert ThinkDetector is not None

    def test_looks_like_autonomous_think_callable(self):
        from fusion_mlx.reasoning.think_detector import looks_like_autonomous_think

        assert looks_like_autonomous_think("<think>hello") is True
        assert looks_like_autonomous_think("hello") is False

    def test_stale_todo_removed(self):
        # The stale "does not exist yet" TODO must be gone from core.py.
        import inspect

        from fusion_mlx.model_auto_config import core

        source = inspect.getsource(core)
        assert (
            "think_detector.py does not exist" not in source
        ), "stale TODO claiming think_detector.py does not exist must be removed (#1025)"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
