# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry.schema import (
    ActivationPayload,
    RequestPayload,
    TelemetryPayload,
    sample_request_preview_payload,
)


def test_activation_payload_fields():
    ap = ActivationPayload(
        activation_kind="first_inference",
        surface="api",
        client_id="abc",
        spec_version=3,
        occurred_at_epoch=1700000000,
    )
    assert ap.activation_kind == "first_inference"
    assert ap.surface == "api"


def test_request_payload_new_fields_default():
    rp = RequestPayload(
        endpoint="/v1/chat/completions",
        model_alias="m",
        stream=False,
        tool_call_used=False,
        prompt_tokens_bucket="0",
        completion_tokens_bucket="1-100",
        ttft_ms_bucket="100-500",
        tps_bucket="10-50",
        status=200,
        caller_agent="claude-code",
        output_degenerate=False,
        completion_empty=False,
        completion_abnormally_short=False,
    )
    assert rp.caller_agent == "claude-code"


def test_sample_request_preview_payload_event():
    p = sample_request_preview_payload(client_id="x", fusion_mlx_version="0.9.0")
    assert p.event == "request"
    assert p.request is not None
    assert p.request.endpoint == "/v1/chat/completions"


def test_telemetry_payload_has_activation_slot():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(TelemetryPayload)}
    assert "activation" in fields
