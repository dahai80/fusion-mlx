# SPDX-License-Identifier: Apache-2.0
# Regression guard for issue #357: GET /v1/models/status was shadowed by the
# gui_compat router's GET /v1/models/{model_name} catch-all. Starlette matches
# routes in registration order, so the specific /status path MUST be registered
# before the parameterized {model_name} route or it is unreachable (captured as
# model_name="status" -> 404 "Model 'status' not found").
#
# Two tests:
#   1. Functional: a minimal app with the #357 fix order (status route before
#      the gui_compat include) returns 200 for GET /v1/models/status.
#   2. Source-order guard: server.py itself registers models_status BEFORE the
#      gui_compat include - catches a future reorder of the real file, not just
#      a reconstruction.

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import require_admin
from fusion_mlx.gui_compat.server import get_gui_compat_router


def _build_app_status_first():
    # Mirrors server.py's #357 fix order: specific route before the catch-all.
    app = FastAPI()

    @app.get("/v1/models/status")
    async def models_status(is_admin: bool = Depends(require_admin)):
        return {"models": []}

    app.include_router(get_gui_compat_router())
    # Auth is not what #357 tests; bypass require_admin so the request reaches
    # the handler instead of being bounced by missing credentials.
    app.dependency_overrides[require_admin] = lambda: True
    return app


def _build_app_status_last():
    # The bug order: catch-all registered first shadows the specific path.
    app = FastAPI()
    app.include_router(get_gui_compat_router())

    @app.get("/v1/models/status")
    async def models_status(is_admin: bool = Depends(require_admin)):
        return {"models": []}

    app.dependency_overrides[require_admin] = lambda: True
    return app


class TestModelsStatusRouteOrdering:
    # Functional proof: status-first -> 200 (the fix); status-last -> 404
    # (captured by {model_name}, reproducing the original #357 symptom).

    def test_status_first_is_reachable(self):
        app = _build_app_status_first()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models/status")
        assert resp.status_code == 200, (
            f"expected 200 (status registered before catch-all), got "
            f"{resp.status_code}: {resp.text}"
        )

    def test_status_last_is_shadowed(self):
        # Documents the trap: if the catch-all is registered first, /status is
        # captured as model_name="status" -> 404. This is the pre-fix symptom.
        app = _build_app_status_last()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models/status")
        assert resp.status_code == 404, (
            f"expected 404 (status shadowed by {{model_name}}), got "
            f"{resp.status_code}: {resp.text}"
        )


def test_server_py_registers_status_before_gui_compat_include():
    # Guards the ACTUAL server.py against a reorder, not just a reconstruction.
    # If someone moves models_status below the gui_compat include, #357 returns.
    server_py = Path(__file__).resolve().parents[2] / "fusion_mlx" / "server.py"
    source = server_py.read_text()
    lines = source.splitlines()

    status_line = None
    include_line = None
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        # The status route decorator. Strip whitespace so indentation drift
        # (method body nesting) doesn't break the match.
        if (
            status_line is None
            and stripped.startswith("@app.get(")
            and "/v1/models/status" in stripped
        ):
            status_line = i
        if (
            include_line is None
            and "app.include_router(" in stripped
            and "get_gui_compat_router" in stripped
        ):
            include_line = i

    assert status_line is not None, (
        "could not locate @app.get('/v1/models/status') in server.py - "
        "the route handler was removed? #357 regressed."
    )
    assert include_line is not None, (
        "could not locate app.include_router(get_gui_compat_router()) in "
        "server.py - the gui_compat include was removed or renamed?"
    )
    assert status_line < include_line, (
        f"#357 regression: /v1/models/status (line {status_line}) is "
        f"registered AFTER the gui_compat include (line {include_line}). "
        "Starlette matches in registration order, so the {model_name} "
        "catch-all shadows /status -> 404. Move the status route above the "
        "gui_compat include."
    )
