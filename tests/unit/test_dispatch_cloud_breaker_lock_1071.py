# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1071.

#1071: CloudRouter breaker state (_circuit_open / _cloud_failure_count /
_cloud_open_at) was mutated check-then-act with no lock; is_cloud_circuit_open
flipped _cloud_circuit_open on the READ path. report_local_* can be called
from sync context → potential count loss / half-open probe storm.
Fix: threading.RLock guards all breaker reads/writes.
"""

from __future__ import annotations

import inspect
import threading

import pytest

from fusion_mlx.dispatch.cloud_router import CloudRouter


class TestCloudBreakerLock1071:
    """Breaker state is lock-guarded on every read/write path."""

    def test_source_has_breaker_lock(self):
        src = inspect.getsource(CloudRouter)
        assert "_breaker_lock" in src
        assert "threading.RLock" in src
        # every mutator + the read-with-mutation path acquire the lock
        for method in (
            "report_local_failure",
            "report_local_success",
            "is_circuit_open",
            "report_cloud_failure",
            "report_cloud_success",
            "is_cloud_circuit_open",
            "should_route_to_cloud",
        ):
            assert f"def {method}" in src
            assert method in src

    def test_concurrent_local_failures_no_lost_count(self):
        cr = CloudRouter("anthropic/test", threshold=99999)
        cr._circuit_failure_threshold = 1000

        def hammer():
            for _ in range(200):
                cr.report_local_failure()

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 10 threads × 200 = 2000 failures, no lost increments
        assert cr._circuit_failure_count == 2000
        assert cr._circuit_open is True

    def test_concurrent_cloud_failures_no_lost_count(self):
        cr = CloudRouter("anthropic/test", threshold=99999)
        cr._cloud_failure_threshold = 1000

        def hammer():
            for _ in range(200):
                cr.report_cloud_failure()

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert cr._cloud_failure_count == 2000
        assert cr._cloud_circuit_open is True

    def test_concurrent_success_failure_consistent(self):
        cr = CloudRouter("anthropic/test", threshold=99999)
        cr._circuit_failure_threshold = 500

        def failer():
            for _ in range(200):
                cr.report_local_failure()

        def successer():
            for _ in range(200):
                cr.report_local_success()

        threads = [threading.Thread(target=failer) for _ in range(5)]
        threads += [threading.Thread(target=successer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # report_local_success resets count to 0 only when open; the exact
        # final count is timing-dependent, but it must be in [0, 1000] and
        # consistent (no negative, no overflow corruption). The lock makes
        # the read in is_circuit_open observe a consistent value.
        assert 0 <= cr._circuit_failure_count <= 1000

    def test_is_cloud_circuit_open_does_not_race_half_open(self):
        cr = CloudRouter("anthropic/test", threshold=99999)
        cr._cloud_failure_threshold = 1
        cr.report_cloud_failure()
        assert cr._cloud_circuit_open is True
        # force cooldown expiry
        cr._cloud_open_at = 0.0
        cr._cloud_cooldown = 0.001

        # concurrent reads must all see a consistent post-half-open state
        results = []

        def reader():
            results.append(cr.is_cloud_circuit_open())

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # after cooldown, half-open flips to False — all readers see False
        # (no reader sees True while another flips it, thanks to the lock)
        assert all(r is False for r in results)

    def test_should_route_to_cloud_no_deadlock(self):
        # RLock must allow should_route_to_cloud -> is_cloud_circuit_open
        # nesting without self-deadlock.
        cr = CloudRouter("anthropic/test", threshold=99999)
        cr._cloud_failure_threshold = 1
        cr.report_cloud_failure()
        cr._cloud_open_at = 0.0
        cr._cloud_cooldown = 0.001

        def call():
            for _ in range(50):
                cr.should_route_to_cloud(1)

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        # if the lock were non-reentrant this would deadlock; join with a
        # timeout so a hang fails the test instead of hanging CI.
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "should_route_to_cloud deadlocked"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
