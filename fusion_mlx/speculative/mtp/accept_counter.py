# SPDX-License-Identifier: Apache-2.0
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MTPAcceptSnapshot:
    attempts: int
    accepts: int
    tokens_saved: int

    @property
    def accept_ratio(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.accepts / self.attempts


class MTPAcceptCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts = 0
        self._accepts = 0
        self._tokens_saved = 0

    def record_attempt(self) -> None:
        with self._lock:
            self._attempts += 1

    def record_accept(self, tokens_saved: int = 1) -> None:
        with self._lock:
            self._accepts += 1
            self._tokens_saved += tokens_saved

    def record_reject(self) -> None:
        pass

    def snapshot(self) -> MTPAcceptSnapshot:
        with self._lock:
            return MTPAcceptSnapshot(
                attempts=self._attempts,
                accepts=self._accepts,
                tokens_saved=self._tokens_saved,
            )

    def reset(self) -> None:
        with self._lock:
            self._attempts = 0
            self._accepts = 0
            self._tokens_saved = 0


_global_counter: MTPAcceptCounter | None = None
_global_lock = threading.Lock()


def get_global_counter() -> MTPAcceptCounter:
    global _global_counter
    with _global_lock:
        if _global_counter is None:
            _global_counter = MTPAcceptCounter()
        return _global_counter


def reset_global_counter_for_tests() -> None:
    global _global_counter
    with _global_lock:
        _global_counter = None
