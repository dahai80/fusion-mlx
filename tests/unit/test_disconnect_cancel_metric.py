# SPDX-License-Identifier: Apache-2.0
"""Tests for the client-disconnect cancel counter (#645)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from fusion_mlx.server_metrics import (
    ServerMetrics,
    get_server_metrics,
    record_llm_disconnect_cancel,
)


def _fresh() -> ServerMetrics:
    return ServerMetrics()


def test_record_disconnect_cancel_bumps_counter():
    sm = _fresh()
    assert sm.cancelled_requests == 0
    sm.record_disconnect_cancel()
    sm.record_disconnect_cancel()
    assert sm.cancelled_requests == 2


def test_cancel_counter_in_to_dict():
    sm = _fresh()
    sm.record_disconnect_cancel()
    d = sm.to_dict()
    assert d["cancelled_requests"] == 1


def test_cancel_counter_resets_on_clear_metrics():
    sm = _fresh()
    sm.record_disconnect_cancel()
    sm.record_disconnect_cancel()
    sm.clear_metrics()
    assert sm.cancelled_requests == 0


def test_cancel_counter_thread_safe():
    sm = _fresh()
    n_threads = 20
    n_each = 50

    def _bump():
        for _ in range(n_each):
            sm.record_disconnect_cancel()

    threads = [threading.Thread(target=_bump) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sm.cancelled_requests == n_threads * n_each


def test_record_llm_disconnect_cancel_swallows_errors(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "fusion_mlx.server_metrics.get_server_metrics", lambda: _boom()
    )
    # Must not raise even though get_server_metrics blows up.
    record_llm_disconnect_cancel()


def test_metrics_renders_cancelled_total():
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics

    sm = get_server_metrics()
    before = sm.cancelled_requests
    sm.record_disconnect_cancel()
    body = render_prometheus_metrics()
    assert "# TYPE fusion_mlx_requests_cancelled_total counter" in body
    assert "# HELP fusion_mlx_requests_cancelled_total" in body
    # Global singleton — assert the delta lands in the rendered value, not an
    # absolute number (other tests may have bumped the counter).
    assert f"fusion_mlx_requests_cancelled_total {before + 1}" in body


@pytest.mark.asyncio
async def test_wait_with_disconnect_ticks_counter_on_disconnect():
    from fusion_mlx.service.disconnect_guard import _wait_with_disconnect

    class _Disconnects:
        async def is_disconnected(self) -> bool:
            return True

    async def _noop():
        await asyncio.sleep(10)

    sm = get_server_metrics()
    before = sm.cancelled_requests
    result = await _wait_with_disconnect(_noop(), _Disconnects(), timeout=5.0)
    assert result is None
    assert sm.cancelled_requests == before + 1


def test_streaming_cancel_tick_is_wired_in_cancellederror_handler():
    """AST-verify the streaming CancelledError handler calls the cancel-counter
    tick. A real mid-stream TestClient cancel is fragile across Starlette/uvicorn
    versions (the CancelledError path depends on live ASGI disconnect), so this
    pins the wiring deterministically instead of asserting behavior through a
    brittle harness. If the tick call is removed or moved out of the handler,
    this test fails — which is exactly the regression that matters."""
    import ast
    from pathlib import Path

    src = Path("fusion_mlx/api/openai_routes.py").read_text()
    tree = ast.parse(src)
    wired = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        exc = node.type
        if exc is None:
            continue
        # Match `asyncio.CancelledError` or bare `CancelledError`.
        name = exc.id if isinstance(exc, ast.Name) else getattr(exc, "attr", None)
        if name != "CancelledError":
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "record_llm_disconnect_cancel"
            ):
                wired = True
    assert wired, "record_llm_disconnect_cancel() must be called inside the streaming CancelledError handler"
