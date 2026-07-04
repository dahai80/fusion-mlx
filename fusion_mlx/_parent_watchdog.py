# SPDX-License-Identifier: Apache-2.0
"""Parent watchdog stub — no parent-process monitoring in this build."""

import logging

logger = logging.getLogger(__name__)


def install_parent_watchdog(ppid: int, *, interval: float = 2.0) -> None:
    logger.debug("Parent watchdog skipped (stub), ppid=%s", ppid)


def resolve_expected_ppid(ppid: int | None) -> int | None:
    return ppid
