# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1024.

#1024: flywheel was hard-banned via _FORBIDDEN_UNTIL_FIXED in server.py —
incomplete feature, dead code chain (forbidden gate + route registry entry +
lazy route module mapping). Removed the mounting; kept bench/flywheel.py
library functions + api/flywheel_routes.py for programmatic use with an
explicit runner.

Fix (#1024): removed flywheel from _LAZY_ROUTES + _ROUTE_REGISTRY +
_FORBIDDEN_UNTIL_FIXED mechanism. Library code retained.
"""

from __future__ import annotations

import pytest


class TestFlywheelRouteRemoved:
    """Flywheel routes no longer mounted on the server."""

    def test_flywheel_not_in_lazy_routes(self):
        from fusion_mlx.server import _LAZY_ROUTES

        assert (
            "flywheel" not in _LAZY_ROUTES
        ), "flywheel must not be in _LAZY_ROUTES (dead route removed #1024)"

    def test_flywheel_routes_module_still_importable(self):
        # Library code retained for programmatic use with explicit runner.
        from fusion_mlx.api.flywheel_routes import router

        assert router is not None

    def test_flywheel_bench_library_still_importable(self):
        from fusion_mlx.bench.flywheel import flywheel, recommend, store_result

        assert callable(flywheel)
        assert callable(recommend)
        assert callable(store_result)

    def test_forbidden_until_fixed_mechanism_gone(self):
        # The _FORBIDDEN_UNTIL_FIXED set was only used for flywheel.
        # Verify it's no longer referenced in server.py source.
        import inspect

        from fusion_mlx import server

        source = inspect.getsource(server)
        assert (
            "_FORBIDDEN_UNTIL_FIXED" not in source
        ), "_FORBIDDEN_UNTIL_FIXED mechanism must be removed (dead code #1024)"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
