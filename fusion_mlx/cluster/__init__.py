# SPDX-License-Identifier: Apache-2.0
"""Cluster — mDNS advertising + peer node registry / self-healing."""

from .platform import Platform, detect_platform
from .router import (
    Backend,
    ClusterRouter,
    Snapshot,
    bootstrap_weighted,
    build_backends_from_config,
    get_router,
    set_router,
)

__all__ = [
    "Platform",
    "detect_platform",
    "Backend",
    "ClusterRouter",
    "Snapshot",
    "bootstrap_weighted",
    "build_backends_from_config",
    "get_router",
    "set_router",
]
