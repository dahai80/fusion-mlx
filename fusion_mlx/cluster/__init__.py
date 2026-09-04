# SPDX-License-Identifier: Apache-2.0
"""Cluster — mDNS advertising + peer node registry / self-healing."""

from .platform import Platform, detect_platform

__all__ = ["Platform", "detect_platform"]
