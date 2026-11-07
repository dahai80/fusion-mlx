# 1M prefill abort #2 diagnosis needs to know which prefill attention
# route (fast kernel / quantized chunked / dequant+SDPA) runs at deep
# context - the ~28 GB transient was inferred, not observed.
import logging

import pytest

from fusion_mlx.patches import turboquant_attention as ta


def test_route_log_emits_first_call_at_or_above_threshold(caplog):
    ta._route_log_count = 0
    with caplog.at_level(logging.INFO, logger="fusion_mlx.patches.turboquant_attention"):
        ta._log_prefill_route("fast", 131072, 2048, 16384)
    assert any("[tq-prefill-route]" in r.message for r in caplog.records)


def test_route_log_throttles_below_threshold_and_beyond_first(caplog):
    ta._route_log_count = 0
    with caplog.at_level(logging.INFO, logger="fusion_mlx.patches.turboquant_attention"):
        ta._log_prefill_route("fast", 65535, 2048, 16384)
        ta._log_prefill_route("quantized", 131072, 2048, 16384)
        ta._log_prefill_route("quantized", 131072, 2048, 16384)
    route_lines = [r for r in caplog.records if "[tq-prefill-route]" in r.message]
    # 65535 is below threshold (not counted); the first eligible call
    # emits, the immediate repeat is inside the same 64-window
    assert len(route_lines) == 1
    assert "route=quantized" in route_lines[0].message


def test_route_log_every_n_emits_once_per_window(caplog, monkeypatch):
    monkeypatch.setattr(ta, "_ROUTE_LOG_EVERY", 3)
    ta._route_log_count = 0
    with caplog.at_level(logging.INFO, logger="fusion_mlx.patches.turboquant_attention"):
        for _ in range(6):
            ta._log_prefill_route("quantized", 986000, 2048, 16384)
    route_lines = [r for r in caplog.records if "[tq-prefill-route]" in r.message]
    assert len(route_lines) == 2
    assert "route=quantized" in route_lines[0].message
    assert "T=986000" in route_lines[0].message
