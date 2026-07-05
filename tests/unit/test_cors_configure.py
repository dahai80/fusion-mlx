# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion_mlx/server.py CORS configuration.

Covers ``configure_cors_from_env`` + ``_resolve_cors_origins`` (the
resolver that ``--cors-origins`` / ``FUSION_MLX_CORS_ALLOW_ORIGINS``
feed) and the ``_create_app`` wiring that reads the resolved module
global. Default (no flag, no env) must stay ``["*"]`` for forward
compatibility with the friendly single-machine UX.
"""

from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware

from fusion_mlx import server


def _reset_server_singleton() -> None:
    server._server_instance = None
    server.app = None


def _cors_middleware_kwargs(app):
    for m in app.user_middleware:
        if m.cls is CORSMiddleware:
            return m.kwargs
    return None


class TestResolveCorsOrigins:
    def test_none_when_no_flag_no_env(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_CORS_ALLOW_ORIGINS", raising=False)
        assert server._resolve_cors_origins(None) is None

    def test_cli_list_returned(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_CORS_ALLOW_ORIGINS", raising=False)
        result = server._resolve_cors_origins(["https://a.com", "https://b.com"])
        assert result == ["https://a.com", "https://b.com"]

    def test_cli_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_CORS_ALLOW_ORIGINS", "https://env.com")
        result = server._resolve_cors_origins(["https://cli.com"])
        assert result == ["https://cli.com"]

    def test_env_used_when_cli_none(self, monkeypatch):
        monkeypatch.setenv(
            "FUSION_MLX_CORS_ALLOW_ORIGINS", "https://a.com, https://b.com"
        )
        result = server._resolve_cors_origins(None)
        assert result == ["https://a.com", "https://b.com"]

    def test_env_whitespace_and_empty_segments_cleaned(self, monkeypatch):
        monkeypatch.setenv(
            "FUSION_MLX_CORS_ALLOW_ORIGINS", " , https://a.com , , https://b.com, "
        )
        result = server._resolve_cors_origins(None)
        assert result == ["https://a.com", "https://b.com"]

    def test_empty_cli_list_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_CORS_ALLOW_ORIGINS", "https://env.com")
        assert server._resolve_cors_origins([]) == ["https://env.com"]

    def test_whitespace_only_env_returns_none(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_CORS_ALLOW_ORIGINS", "   ,  , ")
        assert server._resolve_cors_origins(None) is None


class TestConfigureCorsFromEnv:
    def test_returns_and_sets_global_none(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_CORS_ALLOW_ORIGINS", raising=False)
        prev = server._cors_origins
        try:
            result = server.configure_cors_from_env(None)
            assert result is None
            assert server._cors_origins is None
        finally:
            server._cors_origins = prev

    def test_returns_and_sets_global_list(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_CORS_ALLOW_ORIGINS", raising=False)
        prev = server._cors_origins
        try:
            result = server.configure_cors_from_env(["https://x.com"])
            assert result == ["https://x.com"]
            assert server._cors_origins == ["https://x.com"]
        finally:
            server._cors_origins = prev

    def test_env_drives_resolution_when_cli_none(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_CORS_ALLOW_ORIGINS", "https://env.com")
        prev = server._cors_origins
        try:
            result = server.configure_cors_from_env(None)
            assert result == ["https://env.com"]
            assert server._cors_origins == ["https://env.com"]
        finally:
            server._cors_origins = prev


class TestCreateAppCorsWiring:
    def test_default_wildcard_when_unset(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_CORS_ALLOW_ORIGINS", raising=False)
        prev_cors = server._cors_origins
        prev_inst = server._server_instance
        prev_app = server.app
        try:
            server._cors_origins = None
            _reset_server_singleton()
            app = server.get_app()
            kwargs = _cors_middleware_kwargs(app)
            assert kwargs is not None, "CORSMiddleware not registered"
            assert kwargs["allow_origins"] == ["*"]
        finally:
            server._cors_origins = prev_cors
            server._server_instance = prev_inst
            server.app = prev_app

    def test_pinned_origins_when_set(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_CORS_ALLOW_ORIGINS", raising=False)
        prev_cors = server._cors_origins
        prev_inst = server._server_instance
        prev_app = server.app
        try:
            server._cors_origins = ["https://a.com", "https://b.com"]
            _reset_server_singleton()
            app = server.get_app()
            kwargs = _cors_middleware_kwargs(app)
            assert kwargs is not None, "CORSMiddleware not registered"
            assert kwargs["allow_origins"] == ["https://a.com", "https://b.com"]
        finally:
            server._cors_origins = prev_cors
            server._server_instance = prev_inst
            server.app = prev_app
