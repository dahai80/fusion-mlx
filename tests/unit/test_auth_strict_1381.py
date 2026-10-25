# SPDX-License-Identifier: Apache-2.0
# #1381: strict auth mode for gateway-as-sole-entry deployments, plus the
# rejection-visibility log and the route auth-coverage audit.

import json
import logging

import pytest
from starlette.requests import Request

from fusion_mlx.middleware import auth, auth_audit


def _request(path="/v1/chat/completions", host="203.0.113.7"):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "client": (host, 54321),
        "query_string": b"",
    }
    return Request(scope)


class TestStrictRequested:
    def test_env_flag(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_STRICT_AUTH", "1")
        monkeypatch.setattr(auth, "_settings_strict_auth", lambda: False)
        assert auth.strict_auth_requested() is True

    def test_cli_flag(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_STRICT_AUTH", raising=False)
        monkeypatch.setattr(auth, "_settings_strict_auth", lambda: False)
        assert auth.strict_auth_requested(cli_value=True) is True

    def test_settings_section(self, tmp_path, monkeypatch):
        cfg = tmp_path / "settings.json"
        cfg.write_text(json.dumps({"auth": {"strict": True}}))
        monkeypatch.delenv("FUSION_MLX_STRICT_AUTH", raising=False)
        monkeypatch.setattr(auth, "_SETTINGS_PATH", (str(cfg),))
        assert auth.strict_auth_requested() is True

    def test_default_off(self, tmp_path, monkeypatch):
        cfg = tmp_path / "settings.json"
        cfg.write_text(json.dumps({"auth": {"api_key": "x"}}))
        monkeypatch.delenv("FUSION_MLX_STRICT_AUTH", raising=False)
        monkeypatch.setattr(auth, "_SETTINGS_PATH", (str(cfg),))
        assert auth.strict_auth_requested() is False

    def test_unreadable_settings_is_not_strict(self, tmp_path, monkeypatch):
        # A corrupt settings.json must not silently turn strict mode ON (that
        # would take a healthy deployment down) nor be read as consent.
        cfg = tmp_path / "settings.json"
        cfg.write_text("{not json")
        monkeypatch.delenv("FUSION_MLX_STRICT_AUTH", raising=False)
        monkeypatch.setattr(auth, "_SETTINGS_PATH", (str(cfg),))
        assert auth.strict_auth_requested() is False


class TestEnforceStrict:
    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
        with pytest.raises(RuntimeError, match="requires an API key"):
            auth.enforce_strict_auth(None, cli_value=True)

    def test_raises_with_anonymous_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_ALLOW_ANONYMOUS", "true")
        with pytest.raises(RuntimeError, match="FUSION_ALLOW_ANONYMOUS"):
            auth.enforce_strict_auth("k", cli_value=True)

    def test_passes_with_key(self, monkeypatch):
        monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
        assert auth.enforce_strict_auth("k", cli_value=True) is True

    def test_noop_when_not_strict(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_STRICT_AUTH", raising=False)
        monkeypatch.setattr(auth, "_settings_strict_auth", lambda: False)
        assert auth.enforce_strict_auth(None) is False


class TestRejectionVisibility:
    def test_logs_host_and_path(self, monkeypatch, caplog):
        auth._auth_reject_last.clear()
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.middleware.auth"):
            auth._log_auth_rejection(
                _request("/v1/chat/completions"), "missing API key"
            )
        assert "host=203.0.113.7" in caplog.text
        assert "/v1/chat/completions" in caplog.text

    def test_rate_limited_per_host_and_path(self, monkeypatch, caplog):
        auth._auth_reject_last.clear()
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.middleware.auth"):
            auth._log_auth_rejection(_request(), "invalid API key")
            auth._log_auth_rejection(_request(), "invalid API key")
            auth._log_auth_rejection(_request(path="/v1/embeddings"), "invalid API key")
        assert caplog.text.count("/v1/chat/completions") == 1
        assert caplog.text.count("/v1/embeddings") == 1

    def test_verify_failure_is_logged(self, monkeypatch, caplog):
        auth._auth_reject_last.clear()
        monkeypatch.setattr(auth, "_get_configured_api_key", lambda: "real-key")
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.middleware.auth"):
            with pytest.raises(Exception):
                auth._verify_api_key_values("wrong", request=_request())
        assert "invalid API key" in caplog.text


class TestRouteAudit:
    def test_every_non_exempt_route_is_guarded(self):
        from fusion_mlx.server import create_app

        findings = auth_audit.audit_route_auth(create_app())
        assert findings == [], (
            "routes reachable without an API key: "
            f"{findings}; add the auth dependency or an explicit "
            "UNGUARDED_ALLOWLIST entry with a reason (#1381)"
        )

    def test_exempt_probes_are_documented(self):
        for path in ("/health", "/healthz", "/livez", "/readyz"):
            assert path in auth_audit.EXEMPT_PATHS
