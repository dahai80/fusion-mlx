# SPDX-License-Identifier: Apache-2.0
"""Tests for ``serve --model-dir`` UDS / fd dispatch (issue #569).

The ``--model-dir`` multi-model path (``_serve_from_model_dir``) used to
call a bare ``uvicorn.run(app, host=host, port=port, ...)`` that treated
a ``unix:/path`` host as a TCP hostname and died with Errno 8. #569
routes it through the same ``_run_uvicorn`` dispatch the single-model
serve path uses, and stamps ``get_config().bind_uds`` /
``bind_host`` / ``bind_port`` / ``bind_listen_fd`` so the lifespan
``Ready:`` banner reports the real listener.

These tests pin that contract by driving the real
``_serve_from_model_dir`` with the heavyweight ``create_app`` stubbed,
and asserting on the dispatch chokepoint + bind-config state.
"""

from __future__ import annotations

import types

import pytest

from fusion_mlx import cli_serve as cli_serve_mod


def _make_args(**overrides):
    ns = types.SimpleNamespace(
        model_dir="/tmp/fake-model-dir",
        host="0.0.0.0",
        port=None,
        log_level="INFO",
        listen_fd=None,
        api_key=None,
    )
    ns.__dict__.update(overrides)
    return ns


@pytest.fixture
def stub_model_dir_deps(monkeypatch):
    """Stub the heavyweight deps ``_serve_from_model_dir`` touches so the
    test drives it through to the ``_run_uvicorn`` dispatch without
    building a real FastAPI app or booting engines.
    """
    from fusion_mlx import server as server_mod
    from fusion_mlx.config import get_config

    sentinel_app = object()
    monkeypatch.setattr(server_mod, "create_app", lambda config: sentinel_app)

    cfg = get_config()
    cfg.bind_host = None
    cfg.bind_port = None
    cfg.bind_listen_fd = None
    cfg.bind_uds = None
    return monkeypatch, sentinel_app


def _patch_dispatch(monkeypatch):
    """Patch ``_run_uvicorn`` to record it was called (UDS-aware path)
    and patch ``uvicorn.run`` to record it was NOT called (bare path)."""
    called = {"run_uvicorn": False, "uvicorn_run": False}

    def fake_run_uvicorn(app, args, log_level):
        called["run_uvicorn"] = True
        called["app"] = app

    def fake_uvicorn_run(app, **kwargs):
        called["uvicorn_run"] = True

    monkeypatch.setattr(cli_serve_mod, "_run_uvicorn", fake_run_uvicorn)
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    return called


def test_model_dir_uds_host_dispatches_via_run_uvicorn(
    stub_model_dir_deps, monkeypatch
):
    """``--model-dir --host unix:/path`` MUST go through ``_run_uvicorn``
    (UDS-aware) and MUST NOT call bare ``uvicorn.run`` — the #569 bug."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(
        _make_args(host="unix:/tmp/fusion-test-569.sock")
    )

    assert called["run_uvicorn"] is True
    assert called["app"] is sentinel_app
    assert (
        called["uvicorn_run"] is False
    ), "bare uvicorn.run must NOT be called for unix: host (#569)"

    from fusion_mlx.config import get_config

    cfg = get_config()
    assert cfg.bind_uds == "/tmp/fusion-test-569.sock"
    assert cfg.bind_host is None
    assert cfg.bind_port is None
    assert cfg.bind_listen_fd is None


def test_model_dir_listen_fd_dispatches_via_run_uvicorn(
    stub_model_dir_deps, monkeypatch
):
    """``--model-dir --listen-fd N`` MUST go through ``_run_uvicorn`` and
    stamp ``bind_listen_fd``, not bare ``uvicorn.run``."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(_make_args(listen_fd=9, port=8000))

    assert called["run_uvicorn"] is True
    assert called["uvicorn_run"] is False

    from fusion_mlx.config import get_config

    cfg = get_config()
    assert cfg.bind_listen_fd == 9
    assert cfg.bind_uds is None
    assert cfg.bind_host is None
    assert cfg.bind_port is None


def test_model_dir_default_host_port_dispatches_via_run_uvicorn(
    stub_model_dir_deps, monkeypatch
):
    """Default ``--model-dir`` (no unix:, no fd) MUST go through
    ``_run_uvicorn`` with ``bind_host``/``bind_port`` stamped, not bare
    ``uvicorn.run``. ``0.0.0.0`` rewrites to ``localhost`` for the banner."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(_make_args(host="0.0.0.0", port=12345))

    assert called["run_uvicorn"] is True
    assert called["uvicorn_run"] is False

    from fusion_mlx.config import get_config

    cfg = get_config()
    assert cfg.bind_host == "localhost"
    assert cfg.bind_port == 12345
    assert cfg.bind_uds is None
    assert cfg.bind_listen_fd is None


def test_model_dir_explicit_host_kept_verbatim(stub_model_dir_deps, monkeypatch):
    """A non-``0.0.0.0`` host is kept verbatim in ``bind_host`` (not
    rewritten to localhost)."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(_make_args(host="127.0.0.1", port=7788))

    assert called["run_uvicorn"] is True
    assert called["uvicorn_run"] is False

    from fusion_mlx.config import get_config

    cfg = get_config()
    assert cfg.bind_host == "127.0.0.1"
    assert cfg.bind_port == 7788


def test_model_dir_uds_clears_stale_host_port(stub_model_dir_deps, monkeypatch):
    """A prior ``bind_host``/``bind_port`` MUST be cleared when a UDS
    run follows, so the lifespan banner reports the real listener —
    mirrors the single-model reset contract pinned in
    ``test_serve_listen_fd.py``."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    from fusion_mlx.config import get_config

    cfg = get_config()
    cfg.bind_host = "127.0.0.1"
    cfg.bind_port = 9999
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(
        _make_args(host="unix:/tmp/fusion-test-569b.sock")
    )

    assert called["run_uvicorn"] is True
    assert cfg.bind_host is None
    assert cfg.bind_port is None
    assert cfg.bind_uds == "/tmp/fusion-test-569b.sock"


def test_model_dir_stages_cli_api_key(stub_model_dir_deps, monkeypatch):
    """Issue #636: ``serve --model-dir <dir> --api-key <CLI>`` MUST stage
    the CLI key into the ``server._api_key`` module global before
    ``create_app`` constructs ``Server()``, whose ``__init__`` auth-sync
    reads that global. The text/audio paths stage it (cli_serve.py:102 /
    1154); the model-dir path previously imported only ``create_app`` and
    skipped the staging, so ``--api-key X`` was dropped and /v1/* rejected
    X with 401 while enforcing the settings.json key."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    from fusion_mlx import server as server_mod

    monkeypatch_fixture.setattr(server_mod, "_api_key", None)
    monkeypatch_fixture.delenv("FUSION_MLX_API_KEY", raising=False)
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(_make_args(api_key="CLI_KEY_636"))

    assert called["run_uvicorn"] is True
    assert server_mod._api_key == "CLI_KEY_636", (
        "model-dir path must stage --api-key into server._api_key so "
        "Server.__init__ auth-sync honors the CLI key over settings.json (#636)"
    )


def test_model_dir_no_api_key_leaves_global_unset(stub_model_dir_deps, monkeypatch):
    """Without ``--api-key`` (and no env), the model-dir path leaves
    ``server._api_key`` unset so auth falls back to settings.json —
    the anonymous-dev / persisted-key contract (priority CLI > env >
    settings). Pins that the #636 staging is not over-eager."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    from fusion_mlx import server as server_mod

    monkeypatch_fixture.setattr(server_mod, "_api_key", None)
    monkeypatch_fixture.delenv("FUSION_MLX_API_KEY", raising=False)
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(_make_args(api_key=None))

    assert called["run_uvicorn"] is True
    assert server_mod._api_key is None


def test_model_dir_cli_key_beats_env(stub_model_dir_deps, monkeypatch):
    """``--api-key <CLI>`` on the model-dir path must win over a set
    ``FUSION_MLX_API_KEY`` env, matching ``_resolve_api_key``'s CLI >
    env order."""
    monkeypatch_fixture, sentinel_app = stub_model_dir_deps
    from fusion_mlx import server as server_mod

    monkeypatch_fixture.setattr(server_mod, "_api_key", None)
    monkeypatch_fixture.setenv("FUSION_MLX_API_KEY", "ENV_KEY_636")
    called = _patch_dispatch(monkeypatch_fixture)

    cli_serve_mod._serve_from_model_dir(_make_args(api_key="CLI_KEY_636"))

    assert called["run_uvicorn"] is True
    assert server_mod._api_key == "CLI_KEY_636"
