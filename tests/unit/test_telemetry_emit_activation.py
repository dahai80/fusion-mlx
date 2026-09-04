# SPDX-License-Identifier: Apache-2.0
import pytest

from fusion_mlx.telemetry import emit, state


@pytest.fixture
def tmp_telemetry_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "_default_telemetry_dir", lambda: tmp_path)
    monkeypatch.setattr(state, "_activation_latch", set())
    return tmp_path


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(emit, "is_enabled", lambda *, cli_no_telemetry=False: True)
    monkeypatch.setattr(state, "is_enabled", lambda *, cli_no_telemetry=False: True)


@pytest.fixture
def captured(monkeypatch):
    payloads = []
    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue(payloads))
    return payloads


class _StubQueue:
    def __init__(self, payloads):
        self.payloads = payloads

    def enqueue(self, payload):
        self.payloads.append(payload)


def test_server_surface_cli_when_chat_spawn(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_CHAT_SPAWN", "1")
    assert emit.server_surface() == "cli"
    monkeypatch.delenv("FUSION_MLX_CHAT_SPAWN", raising=False)
    assert emit.server_surface() == "api"


def test_request_sample_rate_env(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", "0.0")
    assert emit._request_sample_rate() == 0.0
    assert emit._should_sample_request() is False
    monkeypatch.setenv("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", "1.0")
    assert emit._request_sample_rate() == 1.0
    assert emit._should_sample_request() is True


def test_request_sample_rate_bad_value_defaults(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", "not-a-number")
    assert emit._request_sample_rate() == 0.1
    monkeypatch.delenv("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", raising=False)
    assert emit._request_sample_rate() == 0.1


def test_request_widened_fields(enabled, captured, monkeypatch):
    monkeypatch.setenv("FUSION_MLX_TELEMETRY_REQUEST_SAMPLE", "1.0")
    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="org/model",
        stream=False,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=5,
        ttft_ms=200.0,
        tps=42.0,
        status=200,
        caller_agent="claude-cli/x",
        output_degenerate=True,
        completion_empty=False,
        completion_abnormally_short=True,
    )
    assert len(captured) == 1
    req = captured[0]["request"]
    assert req["caller_agent"] == "claude-code"
    assert req["output_degenerate"] is True
    assert req["completion_empty"] is False
    assert req["completion_abnormally_short"] is True


def test_activation_rejects_bad_pair(enabled, captured, tmp_telemetry_dir):
    emit.activation("bogus", "api")
    assert captured == []


def test_activation_no_fire_when_disabled(monkeypatch, captured, tmp_telemetry_dir):
    monkeypatch.setattr(emit, "is_enabled", lambda *, cli_no_telemetry=False: False)
    emit.activation("first_inference", "api")
    assert captured == []


def test_activation_emits_when_enabled(
    enabled, captured, tmp_telemetry_dir, monkeypatch
):
    monkeypatch.setattr(emit, "claim_activation_marker", lambda kind: True)
    emit.activation("first_inference", "api")
    assert len(captured) == 1
    act = captured[0]["activation"]
    assert act["activation_kind"] == "first_inference"
    assert act["surface"] == "api"
    assert act["spec_version"] == 3
