# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ANE / Metal wired / IOSurface /metrics exposure (P0底座)."""

from fusion_mlx.routes_internal.metrics import render_prometheus_metrics


class _FakeEnforcer:
    def __init__(self, ane=0, wired=0, iosurface=0):
        self._ane = ane
        self._wired = wired
        self._iosurface = iosurface

    def get_ane_resident_bytes(self):
        return self._ane

    def get_metal_wired_bytes(self):
        return self._wired

    def get_iosurface_shared_bytes(self):
        return self._iosurface


class _FakePool:
    def __init__(self, enforcer):
        self.process_memory_enforcer = enforcer

    @property
    def model_count(self):
        return 0

    @property
    def loaded_model_count(self):
        return 0

    @property
    def current_model_memory(self):
        return 0


def _render_lines():
    return render_prometheus_metrics().splitlines()


def test_metrics_contains_ane_resident_gauge(monkeypatch):
    enf = _FakeEnforcer(ane=2 * 1024**3, wired=0, iosurface=0)
    import fusion_mlx.server as srv

    monkeypatch.setitem(srv._server_state, "engine_pool", _FakePool(enf))
    body = render_prometheus_metrics()
    assert "fusion_mlx_ane_resident_bytes_total" in body
    assert "fusion_mlx_metal_wired_bytes" in body
    assert "fusion_mlx_iosurface_shared_bytes" in body


def test_metrics_ane_value_reflects_enforcer(monkeypatch):
    enf = _FakeEnforcer(ane=5 * 1024**3, wired=3 * 1024**3, iosurface=1 * 1024**3)
    import fusion_mlx.server as srv

    monkeypatch.setitem(srv._server_state, "engine_pool", _FakePool(enf))
    body = render_prometheus_metrics()
    assert f"fusion_mlx_ane_resident_bytes_total {5 * 1024**3}" in body
    assert f"fusion_mlx_metal_wired_bytes {3 * 1024**3}" in body
    assert f"fusion_mlx_iosurface_shared_bytes {1 * 1024**3}" in body


def test_metrics_enforcer_none_returns_zero(monkeypatch):
    import fusion_mlx.server as srv

    monkeypatch.setitem(srv._server_state, "engine_pool", _FakePool(None))
    body = render_prometheus_metrics()
    assert "fusion_mlx_ane_resident_bytes_total 0" in body
    assert "fusion_mlx_metal_wired_bytes 0" in body


def test_metrics_pool_none_does_not_crash(monkeypatch):
    import fusion_mlx.server as srv

    monkeypatch.setitem(srv._server_state, "engine_pool", None)
    body = render_prometheus_metrics()
    # gauges still emitted at 0
    assert "fusion_mlx_ane_resident_bytes_total 0" in body
