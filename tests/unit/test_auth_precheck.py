import pytest

from fusion_mlx.middleware.auth_precheck import AuthPrecheckMiddleware


def _scope(path="/v1/chat/completions", method="POST"):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"content-length", b"4")],
    }


@pytest.mark.asyncio
async def test_drain_breaks_on_disconnect(monkeypatch):
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth_precheck._get_configured_api_key",
        lambda: "sk-test",
    )
    receive_count = 0

    async def receive():
        nonlocal receive_count
        receive_count += 1
        if receive_count > 20:
            raise AssertionError("drain loop spun past 20 receives on disconnect")
        return {"type": "http.disconnect"}

    sent = []

    async def send(msg):
        sent.append(msg)

    async def app(scope, receive, send):
        raise AssertionError("downstream must not run on rejected auth")

    mw = AuthPrecheckMiddleware(app)
    await mw(_scope(), receive, send)
    assert sent and sent[0]["status"] == 401


@pytest.mark.asyncio
async def test_drain_completes_on_full_body(monkeypatch):
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth_precheck._get_configured_api_key",
        lambda: "sk-test",
    )
    bodies = [
        {"type": "http.request", "body": b"abcd", "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
    ]

    async def receive():
        if bodies:
            return bodies.pop(0)
        raise AssertionError("drain loop kept receiving after full body")

    sent = []

    async def send(msg):
        sent.append(msg)

    async def app(scope, receive, send):
        raise AssertionError("downstream must not run on rejected auth")

    mw = AuthPrecheckMiddleware(app)
    await mw(_scope(), receive, send)
    assert sent and sent[0]["status"] == 401


@pytest.mark.asyncio
async def test_valid_key_passes_through(monkeypatch):
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth_precheck._get_configured_api_key",
        lambda: "sk-test",
    )
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True

    mw = AuthPrecheckMiddleware(app)

    async def receive():
        raise AssertionError("body must not be read on pass-through")

    scope = _scope()
    scope["headers"] = [(b"authorization", b"Bearer sk-test")]
    await mw(scope, receive, None)
    assert called


@pytest.mark.asyncio
async def test_setup_api_key_excluded_from_precheck(monkeypatch):
    # #1046: fresh install (no key, no anonymous) — the bootstrap setup
    # endpoint must not be pre-checked, else it 401s the very endpoint that
    # sets the initial key (request=None defeats the loopback exemption).
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth_precheck._get_configured_api_key",
        lambda: None,
    )
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth_precheck._anonymous_access_allowed",
        lambda request: False,
    )
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        raise AssertionError("body must not be read on pass-through")

    mw = AuthPrecheckMiddleware(app)
    scope = _scope(path="/admin/api/setup-api-key", method="POST")
    await mw(scope, receive, None)
    assert called


@pytest.mark.asyncio
async def test_other_admin_path_still_prechecked_when_no_key(monkeypatch):
    # #1046: only the setup endpoint is excluded — other /admin/ body-bearing
    # paths are still pre-checked (no key + no anonymous -> 401, no buffer).
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth_precheck._get_configured_api_key",
        lambda: None,
    )
    monkeypatch.setattr(
        "fusion_mlx.middleware.auth_precheck._anonymous_access_allowed",
        lambda request: False,
    )
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {"type": "http.disconnect"}

    sent = []

    async def send(msg):
        sent.append(msg)

    mw = AuthPrecheckMiddleware(app)
    scope = _scope(path="/admin/api/login", method="POST")
    await mw(scope, receive, send)
    assert not called
    assert sent and sent[0]["status"] == 401
