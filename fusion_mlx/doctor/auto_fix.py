# SPDX-License-Identifier: Apache-2.0
"""Doctor --fix self-healing module.

Inspects the env-health report and applies safe automatic fixes for
3 classes of drift:
  1. HF mirror unreachable → switch to backup mirror
  2. Port conflict detected → suggest alternative
  3. Cache quota exceeds physical memory → downgrade tier

Each fix is logged and reported back to the caller.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BACKUP_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]

_DEFAULT_PORT = 11434


@dataclass
class FixAction:
    description: str
    success: bool
    detail: str = ""


def run_auto_fix(report) -> list[FixAction]:
    actions: list[FixAction] = []

    for section in report.sections:
        section_name = section.title.lower()
        for row in section.rows:
            if row.status.value == "fail":
                _try_fix(section_name, row.label, row.detail or "", actions)

    return actions


def _try_fix(
    section_name: str,
    label: str,
    detail: str,
    actions: list[FixAction],
) -> None:
    label_lower = label.lower()
    detail_lower = detail.lower()

    if "mirror" in label_lower or "mirror" in detail_lower or "network" in section_name:
        if "unreachable" in detail_lower or "timeout" in detail_lower:
            actions.append(_fix_mirror())
            return

    if "port" in label_lower or "port" in detail_lower:
        actions.append(_fix_port(detail))
        return

    if "cache" in label_lower and ("quota" in detail_lower or "exceed" in detail_lower):
        actions.append(_fix_cache_quota())
        return


def _fix_mirror() -> FixAction:
    current = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
    for mirror in _BACKUP_MIRRORS:
        if mirror != current:
            os.environ["HF_MIRROR"] = mirror
            logger.info("auto-fix: HF mirror switched %s → %s", current, mirror)
            return FixAction(
                description=f"HF mirror unreachable: switched to {mirror}",
                success=True,
                detail=f"previous={current}",
            )
    return FixAction(
        description="HF mirror unreachable: no backup available",
        success=False,
    )


def _fix_port(detail: str) -> FixAction:
    import socket

    for candidate in range(_DEFAULT_PORT, _DEFAULT_PORT + 10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", candidate))
            return FixAction(
                description=f"Port {_DEFAULT_PORT} in use: suggest --port {candidate}",
                success=True,
                detail=f"port {candidate} is free",
            )
        except OSError:
            continue
    return FixAction(
        description="Port conflict: no free port in range "
        f"{_DEFAULT_PORT}-{_DEFAULT_PORT + 9}",
        success=False,
    )


def _fix_cache_quota() -> FixAction:
    from ..config import MemoryTier, auto_detect_memory_tier

    tier = auto_detect_memory_tier()
    if tier != MemoryTier.AGGRESSIVE:
        logger.info(
            "auto-fix: cache quota exceeds physical → tier downgrade to %s", tier.value
        )
        return FixAction(
            description=f"Cache quota exceeds physical: downgrade tier to {tier.value}",
            success=True,
            detail=f"restart with --memory-tier {tier.value}",
        )
    return FixAction(
        description="Cache quota exceeds physical but already at SAFE tier",
        success=False,
        detail="reduce ssd_cache_max_bytes manually",
    )
