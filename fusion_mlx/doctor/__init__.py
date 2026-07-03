# SPDX-License-Identifier: Apache-2.0
"""
Fusion-MLX Doctor — environment-health check.

``fusion-mlx doctor`` is a fast (≤ 5 s) self-diagnostic that answers one
question: *is my install/env broken?*  It probes hardware, Python, packages,
HuggingFace cache, network, shell integration, and optional tooling.  It
never loads a model, boots a server, or runs benchmarks.

Model-validation tiers that used to live here (``smoke / check / full /
benchmark``) moved to ``fusion-mlx bench --tier ...`` as of v0.7.22.

Entry point: ``fusion-mlx doctor [--verbose]``.

Exit codes:
  0 — everything ok or only warnings
  1 — one or more ✗ issues
"""

from .env_health import Check, CheckStatus, Report, Section, run_all

# Deprecated compatibility re-exports. The internal consumer
# (``fusion_mlx.bench.tiers.*``) was removed; these are kept solely so
# external PyPI users with ``from fusion_mlx.doctor import DoctorRunner``
# (or any of the others) don't break across the upgrade. Prefer
# importing from ``fusion_mlx.doctor.runner`` directly. May be dropped in
# a future major-version bump.
# TODO: fusion_mlx.bench.tiers may not exist in fusion-mlx; verify availability
from .runner import (  # noqa: F401  # public surface, deprecated
    CheckResult,
    DoctorRunner,
    Status,
    TierResult,
    python_executable,
    run_subprocess,
)

__all__ = [
    # New env-health surface (the public face of `fusion-mlx doctor`).
    "Check",
    "CheckStatus",
    "Report",
    "Section",
    "run_all",
    # Deprecated — kept for back-compat with external imports. See note
    # above the ``.runner`` re-export block.
    "CheckResult",
    "DoctorRunner",
    "Status",
    "TierResult",
    "python_executable",
    "run_subprocess",
]
