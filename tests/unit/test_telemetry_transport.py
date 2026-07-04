# SPDX-License-Identifier: Apache-2.0
import logging
from unittest import mock
from urllib.error import HTTPError, URLError

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_TELEMETRY_DEBUG", raising=False)
    monkeypatch.delenv("RAPID_MLX_TELEMETRY_ENDPOINT", raising=False)


def test_empty_batch_is_success_no_network():
    from fusion_mlx.telemetry import transport

    with mock.patch.object(transport, "urlopen") as urlopen:
        assert transport.post_batch([]) is True
        urlopen.assert_not_called()


def test_post_batch_returns_true_on_2xx():
    from fusion_mlx.telemetry import transport

    resp = mock.MagicMock()
    resp.status = 200
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with mock.patch.object(transport, "urlopen", return_value=resp) as urlopen:
        assert transport.post_batch([{"x": 1}]) is True
        assert urlopen.call_count == 1


def test_4xx_is_immediate_drop_no_retry():
    from fusion_mlx.telemetry import transport

    resp = mock.MagicMock()
    resp.status = 400
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        mock.patch.object(transport, "urlopen", return_value=resp) as urlopen,
        mock.patch.object(transport.time, "sleep") as sleep,
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 1
        sleep.assert_not_called()


def test_5xx_retries_then_gives_up():
    from fusion_mlx.telemetry import transport

    resp = mock.MagicMock()
    resp.status = 503
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        mock.patch.object(transport, "urlopen", return_value=resp) as urlopen,
        mock.patch.object(transport.time, "sleep") as sleep,
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 3
        assert sleep.call_count == 2


def test_url_error_treated_as_distinct_from_timeout():
    from fusion_mlx.telemetry import transport

    with (
        mock.patch.object(
            transport, "urlopen", side_effect=URLError("connection reset")
        ),
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False


def test_timeout_error_caught():
    from fusion_mlx.telemetry import transport

    with (
        mock.patch.object(transport, "urlopen", side_effect=TimeoutError("slow")),
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False


def test_os_error_caught():
    from fusion_mlx.telemetry import transport

    with (
        mock.patch.object(
            transport, "urlopen", side_effect=OSError("name resolution failed")
        ),
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False


def test_http_error_4xx_does_not_retry():
    from fusion_mlx.telemetry import transport

    exc = HTTPError(
        url="https://x",
        code=400,
        msg="bad",
        hdrs=None,
        fp=None,
    )
    with (
        mock.patch.object(transport, "urlopen", side_effect=exc) as urlopen,
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 1


def test_http_error_response_body_closed():
    from fusion_mlx.telemetry import transport

    closed_4xx = mock.MagicMock()
    err_4xx = HTTPError(
        url="https://x",
        code=400,
        msg="bad",
        hdrs=None,
        fp=None,
    )
    err_4xx.close = closed_4xx

    closed_5xx_a = mock.MagicMock()
    closed_5xx_b = mock.MagicMock()
    closed_5xx_c = mock.MagicMock()
    err_5xx_attempts = []
    for c in (closed_5xx_a, closed_5xx_b, closed_5xx_c):
        e = HTTPError(
            url="https://x",
            code=503,
            msg="busy",
            hdrs=None,
            fp=None,
        )
        e.close = c
        err_5xx_attempts.append(e)

    with (
        mock.patch.object(transport, "urlopen", side_effect=[err_4xx]),
        mock.patch.object(transport.time, "sleep"),
    ):
        transport.post_batch([{"x": 1}])
    closed_4xx.assert_called_once()

    with (
        mock.patch.object(transport, "urlopen", side_effect=err_5xx_attempts),
        mock.patch.object(transport.time, "sleep"),
    ):
        transport.post_batch([{"x": 1}])
    closed_5xx_a.assert_called_once()
    closed_5xx_b.assert_called_once()
    closed_5xx_c.assert_called_once()


def test_http_error_5xx_retries():
    from fusion_mlx.telemetry import transport

    exc = HTTPError(
        url="https://x",
        code=503,
        msg="busy",
        hdrs=None,
        fp=None,
    )
    with (
        mock.patch.object(transport, "urlopen", side_effect=exc) as urlopen,
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 3


def test_oversized_payload_dropped_locally():
    from fusion_mlx.telemetry import transport

    big = {"x": "a" * (transport.MAX_BODY_BYTES + 100)}
    with mock.patch.object(transport, "urlopen") as urlopen:
        assert transport.post_batch([big]) is False
        urlopen.assert_not_called()


def test_non_https_non_loopback_override_fails_closed(monkeypatch):
    from fusion_mlx.telemetry import transport

    monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", "http://insecure.example/v1")
    assert transport.endpoint() is None


def test_endpoint_override_only_accepts_localhost(monkeypatch):
    from fusion_mlx.telemetry import transport

    for ok in (
        "http://localhost:8787/v1/events",
        "http://127.0.0.1:8787/v1/events",
        "https://localhost/v1/events",
        "http://[::1]:8787/v1/events",
    ):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", ok)
        assert transport.endpoint() == ok, f"{ok} should be accepted"

    for bad in (
        "https://attacker.example/v1/events",
        "https://localhost.attacker.example/v1/events",
        "https://telemetry.attacker.example/v1/events",
        "ftp://localhost/v1/events",
        "not-a-url",
    ):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", bad)
        assert transport.endpoint() is None, f"{bad} should be refused (fail closed)"

    monkeypatch.delenv("RAPID_MLX_TELEMETRY_ENDPOINT", raising=False)
    assert transport.endpoint() == transport.DEFAULT_ENDPOINT


def test_post_batch_fails_closed_on_rejected_override(monkeypatch):
    from fusion_mlx.telemetry import transport

    monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", "https://attacker.example/v1")

    called = []

    def fake_urlopen(req, timeout):
        called.append(req)
        raise AssertionError(
            "post_batch hit the network even though the override was rejected"
        )

    with mock.patch.object(transport, "urlopen", fake_urlopen):
        assert transport.post_batch([{"x": 1}]) is False
    assert called == [], "fail-closed contract violated: a request was sent"


def test_malformed_url_request_does_not_raise(monkeypatch):
    from fusion_mlx.telemetry import transport

    bad_url = "https://example.com/v1/\x00events"
    with (
        mock.patch.object(transport, "endpoint", return_value=bad_url),
        mock.patch.object(transport, "_is_localhost_override", return_value=False),
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False


def test_malformed_port_localhost_override_does_not_raise(monkeypatch):
    from fusion_mlx.telemetry import transport

    monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", "http://localhost:bad/v1")
    assert transport.endpoint() is None


def test_last_attempt_5xx_log_says_giving_up_not_will_retry():
    from fusion_mlx.telemetry import transport

    captured: list[str] = []

    def fake_log(msg):
        captured.append(msg)

    resp = mock.MagicMock()
    resp.status = 503
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        mock.patch.object(transport, "urlopen", return_value=resp),
        mock.patch.object(transport.time, "sleep"),
        mock.patch.object(transport, "_log", fake_log),
    ):
        assert transport.post_batch([{"x": 1}]) is False

    last = captured[-1]
    assert "giving up" in last, captured
    assert "will retry" not in last


def test_non_serializable_payload_returns_false_not_raise():
    from fusion_mlx.telemetry import transport

    class _NotSerializable:
        pass

    with mock.patch.object(transport, "urlopen") as urlopen:
        assert transport.post_batch([{"x": _NotSerializable()}]) is False
        urlopen.assert_not_called()


def test_retry_constants_are_finite():
    from fusion_mlx.telemetry import transport

    assert isinstance(transport.TIMEOUT_S, float)
    assert 0 < transport.TIMEOUT_S < 10
    assert isinstance(transport.RETRY_BACKOFFS_S, tuple)
    assert all(
        isinstance(b, (int, float)) and b >= 0 for b in transport.RETRY_BACKOFFS_S
    )


def test_user_agent_is_self_identifying():
    import re

    from fusion_mlx.telemetry import transport

    ua = transport._user_agent()
    assert re.search(r"\brapid-mlx/\S+", ua), (
        f"UA must follow 'rapid-mlx/<version>' shape, got {ua!r}"
    )
    assert "Python-urllib" not in ua


def test_post_sends_self_identifying_user_agent():
    import re

    from fusion_mlx.telemetry import transport

    captured: dict = {}

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.headers)
        resp = mock.MagicMock()
        resp.status = 200
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    with mock.patch.object(transport, "urlopen", fake_urlopen):
        assert transport.post_batch([{"x": 1}]) is True
    ua = {k.lower(): v for k, v in captured["headers"].items()}["user-agent"]
    assert re.search(r"\brapid-mlx/\S+", ua), (
        f"UA must follow 'rapid-mlx/<version>' shape, got {ua!r}"
    )
    assert "Python-urllib" not in ua


def test_debug_env_truthy_off_by_default(monkeypatch):
    from fusion_mlx.telemetry import transport

    monkeypatch.delenv("RAPID_MLX_TELEMETRY_DEBUG", raising=False)
    assert transport.debug_enabled() is False

    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_DEBUG", falsy)
        assert transport.debug_enabled() is False

    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_DEBUG", truthy)
        assert transport.debug_enabled() is True
