# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

logger = logging.getLogger(__name__)


class _StubQueue:
    def __init__(self, payloads):
        self.payloads = payloads

    def enqueue(self, payload):
        self.payloads.append(payload)


@pytest.fixture
def tmp_telemetry_dir(tmp_path, monkeypatch):
    import fusion_mlx.telemetry.state as state

    monkeypatch.setattr(state, "_default_telemetry_dir", lambda: tmp_path)
    monkeypatch.setattr(state, "_activation_latch", set())
    return tmp_path


@pytest.fixture
def captured(monkeypatch):
    payloads = []
    import fusion_mlx.telemetry.emit as emit

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue(payloads))
    return payloads


@pytest.fixture
def enabled(monkeypatch):
    import fusion_mlx.telemetry.emit as emit
    import fusion_mlx.telemetry.state as state

    monkeypatch.setattr(emit, "is_enabled", lambda *, cli_no_telemetry=False: True)
    monkeypatch.setattr(state, "is_enabled", lambda *, cli_no_telemetry=False: True)


@pytest.fixture
def disabled(monkeypatch):
    import fusion_mlx.telemetry.emit as emit
    import fusion_mlx.telemetry.state as state

    monkeypatch.setattr(emit, "is_enabled", lambda *, cli_no_telemetry=False: False)
    monkeypatch.setattr(state, "is_enabled", lambda *, cli_no_telemetry=False: False)


def _patch_watchdog(monkeypatch):
    import fusion_mlx._parent_watchdog as wd

    monkeypatch.setattr(wd, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(wd, "write_pid_file", lambda: None)
    monkeypatch.setattr(wd, "write_status", lambda status: None)
    monkeypatch.setattr(wd, "record_crash", lambda: 1)
    monkeypatch.setattr(wd, "clear_crash_counter", lambda: None)
    monkeypatch.setattr(wd, "remove_pid_file", lambda: None)
    monkeypatch.setattr(wd, "write_exit_status", lambda status: None)


def _patch_metrics(monkeypatch):
    import fusion_mlx.server_metrics as sm

    fake = MagicMock()
    monkeypatch.setattr(sm, "get_server_metrics", lambda: fake)


def _make_lifespan_server(monkeypatch, *, startup_raises):
    import fusion_mlx.server as server_mod

    srv = server_mod.Server.__new__(server_mod.Server)

    async def _boom():
        raise RuntimeError("boom")

    async def _noop():
        pass

    srv._startup = _boom if startup_raises else _noop
    srv._shutdown = _noop
    return srv


async def _drive_lifespan(srv):
    gen = srv._lifespan()
    try:
        await gen.__anext__()
    finally:
        try:
            await gen.aclose()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_lifespan_startup_error_emits_when_enabled(
    monkeypatch, enabled, captured, tmp_telemetry_dir
):
    _patch_watchdog(monkeypatch)
    _patch_metrics(monkeypatch)
    srv = _make_lifespan_server(monkeypatch, startup_raises=True)
    with pytest.raises(RuntimeError):
        await _drive_lifespan(srv)
    error_payloads = [p for p in captured if p.get("event") == "error"]
    assert len(error_payloads) == 1, f"expected 1 error payload, got {captured}"
    err = error_payloads[0]["error"]
    assert err["category"] == "lifespan_failure"
    assert err["phase"] == "startup"
    assert "fingerprint" in err
    logger.info("lifespan error payload captured: %s", err)


@pytest.mark.asyncio
async def test_lifespan_startup_error_no_emit_when_disabled(
    monkeypatch, disabled, captured, tmp_telemetry_dir
):
    _patch_watchdog(monkeypatch)
    _patch_metrics(monkeypatch)
    srv = _make_lifespan_server(monkeypatch, startup_raises=True)
    with pytest.raises(RuntimeError):
        await _drive_lifespan(srv)
    error_payloads = [p for p in captured if p.get("event") == "error"]
    assert error_payloads == [], f"expected no emit when disabled, got {error_payloads}"


def test_emit_error_contract_when_enabled(
    monkeypatch, enabled, captured, tmp_telemetry_dir
):
    import fusion_mlx.telemetry.emit as emit

    emit.error(category="lifespan_failure", phase="startup", exc=RuntimeError("boom"))
    assert len(captured) == 1
    err = captured[0]["error"]
    assert err["category"] == "lifespan_failure"
    assert err["phase"] == "startup"
    assert "fingerprint" in err


def test_emit_error_no_emit_when_disabled(
    monkeypatch, disabled, captured, tmp_telemetry_dir
):
    import fusion_mlx.telemetry.emit as emit

    emit.error(category="lifespan_failure", phase="startup", exc=RuntimeError("boom"))
    assert captured == []


def test_model_pull_activation_contract_when_enabled(
    monkeypatch, enabled, captured, tmp_telemetry_dir
):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.activation_spec import ACTIVATION_MODEL_PULL, SURFACE_CLI

    monkeypatch.setattr(emit, "claim_activation_marker", lambda kind: True)
    emit.activation(activation_kind=ACTIVATION_MODEL_PULL, surface=SURFACE_CLI)
    assert len(captured) == 1
    act = captured[0]["activation"]
    assert act["activation_kind"] == "model_pull"
    assert act["surface"] == "cli"
    logger.info("model_pull activation payload captured: %s", act)


def test_model_pull_activation_no_emit_when_disabled(
    monkeypatch, disabled, captured, tmp_telemetry_dir
):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.activation_spec import ACTIVATION_MODEL_PULL, SURFACE_CLI

    emit.activation(activation_kind=ACTIVATION_MODEL_PULL, surface=SURFACE_CLI)
    assert captured == []
