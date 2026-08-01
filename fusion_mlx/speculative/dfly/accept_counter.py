# SPDX-License-Identifier: Apache-2.0
"""DFly accept-rate counter — mirrors DFlashAcceptCounter pattern."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DFlyAcceptSnapshot:
    accepted: int = 0
    drafted: int = 0

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted > 0 else 0.0


class DFlyAcceptCounter:
    def __init__(self) -> None:
        self._accepted = 0
        self._drafted = 0
        self._lock = threading.Lock()

    def record(self, accepted: int, drafted: int) -> None:
        with self._lock:
            self._accepted += accepted
            self._drafted += drafted

    def snapshot(self) -> DFlyAcceptSnapshot:
        with self._lock:
            return DFlyAcceptSnapshot(
                accepted=self._accepted,
                drafted=self._drafted,
            )

    def reset(self) -> None:
        with self._lock:
            self._accepted = 0
            self._drafted = 0


_global_counter: DFlyAcceptCounter | None = None
_global_lock = threading.Lock()


def get_global_counter() -> DFlyAcceptCounter:
    global _global_counter
    with _global_lock:
        if _global_counter is None:
            _global_counter = DFlyAcceptCounter()
        return _global_counter


def reset_global_counter_for_tests() -> None:
    global _global_counter
    with _global_lock:
        _global_counter = None
