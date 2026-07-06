# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _version.py version string.

Covers:
- __version__ format (semver 3-part)
- No dev/rc/beta suffixes in release
"""

from __future__ import annotations

import importlib.util


def _load_version_module():
    """Load _version.py directly to avoid fusion_mlx/__init__.py MLX chain."""
    spec = importlib.util.spec_from_file_location(
        "_version", "fusion_mlx/_version.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestVersionFormat:
    """__version__ must be a valid semver string."""

    def setup_method(self):
        self.vmod = _load_version_module()

    def test_version_is_string(self):
        assert isinstance(self.vmod.__version__, str)

    def test_version_has_three_parts(self):
        parts = self.vmod.__version__.split(".")
        assert len(parts) == 3, f"Expected 3-part semver, got {self.vmod.__version__}"

    def test_version_parts_are_numeric(self):
        parts = self.vmod.__version__.split(".")
        for p in parts:
            assert p.isdigit(), f"Non-numeric version part: {p}"

    def test_version_has_no_dev_suffix(self):
        """Release versions should not carry dev/rc/beta suffixes."""
        assert "dev" not in self.vmod.__version__
        assert "rc" not in self.vmod.__version__
        assert "beta" not in self.vmod.__version__
        assert "alpha" not in self.vmod.__version__

    def test_version_greater_than_minimum(self):
        parts = tuple(int(p) for p in self.vmod.__version__.split("."))
        assert parts >= (0, 4, 0), f"Version too low: {self.vmod.__version__}"
