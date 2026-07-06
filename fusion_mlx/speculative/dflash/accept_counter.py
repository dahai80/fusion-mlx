# SPDX-License-Identifier: Apache-2.0
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class DFlashAcceptSnapshot:
    attempts: int
    accepts: int
    tokens_saved: int

    @property
    def accept_ratio(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.accepts / self.attempts

    @property
    def mean_tokens_per_attempt(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.tokens_saved / self.attempts


class DFlashAcceptCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts = 0
        self._accepts = 0
        self._tokens_saved = 0

    def record_attempt(self) -> None:
        with self._lock:
            self._attempts += 1

    def record_accept(self, tokens_saved: int = 1) -> None:
        if tokens_saved < 0:
            raise ValueError(f"tokens_saved must be non-negative; got {tokens_saved}")
        with self._lock:
            self._accepts += 1
            self._tokens_saved += tokens_saved

    def record_reject(self) -> None:
        return None

    def snapshot(self) -> DFlashAcceptSnapshot:
        with self._lock:
            return DFlashAcceptSnapshot(
                attempts=self._attempts,
                accepts=self._accepts,
                tokens_saved=self._tokens_saved,
            )

    def reset(self) -> None:
        with self._lock:
            self._attempts = 0
            self._accepts = 0
            self._tokens_saved = 0


_global_counter = DFlashAcceptCounter()


def get_global_counter() -> DFlashAcceptCounter:
    return _global_counter


def reset_global_counter_for_tests() -> None:
    _global_counter.reset()
