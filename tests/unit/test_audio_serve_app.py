# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the 0917 audio-serve None-app fix.

``cli_serve/audio_mode.py`` boots uvicorn with ``server.get_app()`` rather
than the module-level ``app`` snapshot (which is ``None`` until the first
``get_app()`` call). The old shape handed uvicorn a ``None`` app and every
request 500'd with "'NoneType' object is not callable". These tests pin
the contract: ``get_app()`` never returns ``None``.
"""

from __future__ import annotations

import threading


def _reset_server_module():
    import fusion_mlx.server as srv

    saved_inst = srv._server_instance
    saved_app = srv.app
    srv._server_instance = None
    srv.app = None
    return srv, saved_inst, saved_app


def _restore(srv, saved_inst, saved_app):
    srv._server_instance = saved_inst
    srv.app = saved_app


def test_get_app_returns_callable_not_none(monkeypatch):
    # Server() construction is heavy (loads config); patch it to a stub
    # whose .app is a sentinel. The contract under test is that get_app()
    # resolves the None snapshot to Server().app, never returning None.
    srv, si, sa = _reset_server_module()
    try:
        sentinel = object()

        class _StubServer:
            app = sentinel

        monkeypatch.setattr(srv, "Server", lambda: _StubServer())
        got = srv.get_app()
        assert got is sentinel, "get_app() returned None — audio serve would 500"
    finally:
        _restore(srv, si, sa)


def test_get_app_double_checked_lock_single_construction(monkeypatch):
    # Two concurrent callers must not both construct Server() (orphan app).
    srv, si, sa = _reset_server_module()
    try:
        constructs = []

        class _StubServer:
            def __init__(self):
                constructs.append(threading.current_thread().name)
                self.app = object()

        monkeypatch.setattr(srv, "Server", _StubServer)
        results = []

        def worker():
            results.append(srv.get_app())

        t1 = threading.Thread(target=worker, name="g1")
        t2 = threading.Thread(target=worker, name="g2")
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)
        assert (
            len(constructs) == 1
        ), f"Server constructed {len(constructs)}x, expected 1"
        assert results[0] is results[1], "callers got different app objects"
    finally:
        _restore(srv, si, sa)


def test_audio_mode_source_uses_get_app_not_snapshot():
    # Source-level guard: audio_mode must resolve via get_app(), not bind
    # the module-level None snapshot. Catches a revert that reintroduces
    # the 500-on-every-request regression.
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2]
        / "fusion_mlx"
        / "cli_serve"
        / "audio_mode.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "server.get_app()" in text, "audio_mode no longer uses server.get_app()"
    # The raw module-level app import alone (without get_app) is the bug.
    assert "from ..server import app" not in text or "get_app" in text
