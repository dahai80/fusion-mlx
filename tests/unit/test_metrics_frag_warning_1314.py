# SPDX-License-Identifier: Apache-2.0
# #1314: the fragmentation warning must not fire on every /metrics scrape.

import logging

import mlx.core as mx
import pytest

from fusion_mlx.routes_internal import metrics

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _reset_frag_throttle():
    metrics._last_frag_warning_at = 0.0
    yield
    metrics._last_frag_warning_at = 0.0


def _force_high_fragmentation(monkeypatch, phys_bytes=64 * 1024**3):
    monkeypatch.setattr(
        "fusion_mlx.utils.proc_memory.get_phys_footprint",
        lambda: phys_bytes,
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)


def _frag_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "High memory fragmentation" in record.getMessage()
    ]


def test_fragmentation_warning_is_throttled(monkeypatch, caplog):
    _force_high_fragmentation(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.routes_internal.metrics"):
        for _ in range(10):
            metrics._render_uptime_metal()
    assert len(_frag_warnings(caplog)) == 1
    logger.info("fragmentation warning throttled across 10 renders")


class _Clock:
    def __init__(self, now):
        self.now = now

    def monotonic(self):
        return self.now


def test_fragmentation_warning_repeats_after_interval(monkeypatch, caplog):
    _force_high_fragmentation(monkeypatch)
    clock = _Clock(metrics._FRAG_WARN_INTERVAL_S + 1.0)
    monkeypatch.setattr(metrics, "time", clock)
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.routes_internal.metrics"):
        metrics._render_uptime_metal()
        clock.now += metrics._FRAG_WARN_INTERVAL_S
        metrics._render_uptime_metal()
    assert len(_frag_warnings(caplog)) == 2
    logger.info("fragmentation warning repeats once per throttle interval")


def test_fragmentation_gauge_still_rendered_every_scrape(monkeypatch):
    _force_high_fragmentation(monkeypatch)
    for _ in range(3):
        body = "\n".join(metrics._render_uptime_metal())
        assert "fusion_mlx_memory_fragmentation_ratio 1.0" in body
    logger.info("fragmentation gauge unaffected by the warning throttle")


def test_no_warning_below_threshold(monkeypatch, caplog):
    monkeypatch.setattr(
        "fusion_mlx.utils.proc_memory.get_phys_footprint",
        lambda: 64 * 1024**3,
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 60 * 1024**3)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 4 * 1024**3)
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.routes_internal.metrics"):
        metrics._render_uptime_metal()
    assert _frag_warnings(caplog) == []
    logger.info("low fragmentation emits no warning")
