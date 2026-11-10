# SPDX-License-Identifier: Apache-2.0
"""#1526: verify_scoped_api_key must accept a valid admin session cookie.

The macOS app (FusionClient) authenticates by logging in once and sending
the session cookie — it never attaches an Authorization header. Scoped
routes (/admin/api/models/{id}/load, list, unload) rejected every such
request with 401 while downloads (require_admin, cookie-aware) worked.
"""

import unittest

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.admin.auth import (
    SESSION_COOKIE_NAME,
    create_session_token,
)
from fusion_mlx.middleware.auth import verify_scoped_api_key


def _make_app():
    app = FastAPI()

    @app.get("/scoped-ping")
    async def scoped_ping(role: str = Depends(verify_scoped_api_key)):
        return {"role": role}

    return app


class TestScopedSessionAuth(unittest.TestCase):
    def test_session_cookie_grants_admin_role(self):
        token = create_session_token()
        client = TestClient(_make_app())
        resp = client.get("/scoped-ping", cookies={SESSION_COOKIE_NAME: token})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], "admin")

    def test_no_cookie_no_bearer_is_401(self):
        client = TestClient(_make_app())
        resp = client.get("/scoped-ping")
        self.assertEqual(resp.status_code, 401)

    def test_invalid_cookie_is_401(self):
        client = TestClient(_make_app())
        resp = client.get("/scoped-ping", cookies={SESSION_COOKIE_NAME: "deadbeef"})
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])
