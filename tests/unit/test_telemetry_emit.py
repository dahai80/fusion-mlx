# SPDX-License-Identifier: Apache-2.0
import importlib
import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FUSION_MLX_TELEMETRY", raising=False)

    import fusion_mlx.telemetry.state as state

    importlib.reload(state)

    import fusion_mlx.telemetry.emit as emit

    importlib.reload(emit)
    emit._reset_for_tests()
    return tmp_path


@pytest.fixture
def opted_in(fake_home):
    from fusion_mlx.telemetry.state import record_consent

    record_consent(True, rapid_mlx_version="0.0.0+test")
    return fake_home


@pytest.fixture
def stub_queue(monkeypatch):
    from fusion_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    return captured


def test_session_start_no_op_when_disabled(fake_home, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.session_start(subcommand="serve")
    assert stub_queue == []


def test_session_end_no_op_when_disabled(fake_home, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.session_end(subcommand="serve", duration_seconds=42)
    assert stub_queue == []


def test_request_no_op_when_disabled(fake_home, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="qwen3.5-9b-4bit",
        stream=True,
        tool_call_used=False,
        prompt_tokens=100,
        completion_tokens=400,
        ttft_ms=250.0,
        tps=42.0,
        status=200,
    )
    assert stub_queue == []


def test_error_no_op_when_disabled(fake_home, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.error(category="model_load_failure", exc=RuntimeError("x"), phase="startup")
    assert stub_queue == []


def test_session_id_is_stable_under_concurrent_first_callers(fake_home):
    import threading

    from fusion_mlx.telemetry import emit

    emit._reset_for_tests()

    results: list[str] = []
    started = threading.Event()
    barrier = threading.Barrier(32)

    def racer():
        barrier.wait(timeout=5.0)
        results.append(emit.session_id())

    threads = [threading.Thread(target=racer) for _ in range(32)]
    for t in threads:
        t.start()
    started.set()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == 32
    assert len(set(results)) == 1, (
        f"concurrent first callers generated {len(set(results))} distinct "
        f"session_ids: {set(results)}"
    )


def test_cli_kill_switch_overrides_opt_in(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.state import set_cli_kill_switch

    set_cli_kill_switch(True)
    try:
        emit.session_start(subcommand="serve")
        emit.session_end(subcommand="serve", duration_seconds=42)
        emit.request(
            endpoint="/v1/chat/completions",
            model_alias="qwen3.5-9b-4bit",
            stream=True,
            tool_call_used=False,
            prompt_tokens=100,
            completion_tokens=400,
            ttft_ms=250.0,
            tps=42.0,
            status=200,
        )
        emit.error(
            category="model_load_failure",
            exc=RuntimeError("x"),
            phase="startup",
        )
    finally:
        set_cli_kill_switch(False)
    assert stub_queue == []


def test_subcommand_normalized_to_allowlist(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.session_start(subcommand="serve")
    assert stub_queue[-1]["session"]["subcommand"] == "serve"

    emit.session_end(subcommand="serve", duration_seconds=10)
    assert stub_queue[-1]["session"]["subcommand"] == "serve"

    leak = "internal:dump?path=/Users/alice/secrets.txt"
    emit.session_start(subcommand=leak)
    assert stub_queue[-1]["session"]["subcommand"] == "other"
    assert "alice" not in repr(stub_queue[-1])


def test_runtime_payload_carries_every_schema_v1_field(opted_in, stub_queue):
    from dataclasses import fields as _fields

    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.schema import SessionPayload

    expected_keys = {f.name for f in _fields(SessionPayload)}

    emit.session_start(subcommand="serve", flag_names=[])
    session = stub_queue[-1]["session"]
    missing = expected_keys - set(session)
    assert not missing, f"session_start dropped v1 keys: {missing}"

    emit.session_end(subcommand="serve", duration_seconds=42)
    session = stub_queue[-1]["session"]
    missing = expected_keys - set(session)
    assert not missing, f"session_end dropped v1 keys: {missing}"


def test_session_start_envelope_when_enabled(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.session_start(
        subcommand="serve",
        flag_names=["host", "port"],
        models_loaded=["mlx-community/Qwen3.5-9B-4bit"],
    )
    assert len(stub_queue) == 1
    payload = stub_queue[0]

    assert payload["schema_version"] == 1
    assert payload["event"] == "session_start"
    assert payload["timestamp"].endswith("Z")
    assert payload["platform"]["os"] in {"darwin", "linux", "windows"}
    assert "chip" in payload["platform"]

    assert set(payload["session"]["flag_names"]) == {"host", "port"}
    blob = repr(payload)
    assert "0.0.0.0" not in blob
    assert "8000" not in blob


def test_session_start_models_loaded_redacted(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.session_start(
        subcommand="serve",
        models_loaded=[
            "mlx-community/Qwen3.5-9B-4bit",
            "/Users/alice/secret-checkout",
        ],
    )
    loaded = stub_queue[0]["session"]["models_loaded"]
    assert "mlx-community/Qwen3.5-9B-4bit" in loaded
    assert "<local>" in loaded
    assert "alice" not in repr(loaded)


def test_session_start_models_loaded_capped_at_32(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.session_start(
        subcommand="serve",
        models_loaded=[f"org/model-{i}" for i in range(50)],
    )
    assert len(stub_queue[0]["session"]["models_loaded"]) == 32


def test_request_buckets_not_raw_numbers(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="qwen3.5-9b-4bit",
        stream=True,
        tool_call_used=False,
        prompt_tokens=137,
        completion_tokens=1729,
        ttft_ms=432.5,
        tps=58.2,
        status=200,
    )
    r = stub_queue[0]["request"]
    assert r["prompt_tokens_bucket"] == "0-256"
    assert r["completion_tokens_bucket"] == "1k-4k"
    assert r["ttft_ms_bucket"] == "100-500ms"
    assert r["tps_bucket"] == "50-100"
    request_blob = repr(r)
    for raw in ("137", "1729", "432.5", "58.2"):
        assert raw not in request_blob, f"{raw!r} survived into request payload: {r}"


def test_error_category_and_phase_normalised_to_allowlist(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    try:
        raise RuntimeError("synthetic")
    except RuntimeError as exc:
        emit.error(category="model_load_failure", exc=exc, phase="startup")
    e = stub_queue[-1]["error"]
    assert e["category"] == "model_load_failure"
    assert e["phase"] == "startup"

    leak = "user typed: please summarize Q3 numbers"
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        emit.error(category=leak, exc=exc, phase=leak)
    e = stub_queue[-1]["error"]
    assert e["category"] == "other"
    assert e["phase"] == "other"
    blob = repr(stub_queue[-1])
    assert "summarize" not in blob
    assert "Q3" not in blob


def test_error_carries_fingerprint_no_message(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    try:
        raise ValueError("/Users/alice/secret.txt: not found")
    except ValueError as exc:
        emit.error(category="model_load_failure", exc=exc, phase="startup")

    err = stub_queue[0]["error"]
    assert len(err["fingerprint"]) == 16
    blob = repr(stub_queue[0])
    assert "/Users/alice/secret.txt" not in blob
    assert "not found" not in blob


def test_session_start_swallows_internal_bug(opted_in, monkeypatch, stub_queue):
    from fusion_mlx.telemetry import emit

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic redact failure")

    monkeypatch.setattr(emit, "normalize_model_path", boom)
    emit.session_start(subcommand="serve", models_loaded=["org/model"])


def test_emit_does_not_catch_keyboard_interrupt(opted_in, monkeypatch, stub_queue):
    from fusion_mlx.telemetry import emit

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(emit, "platform_info", interrupt)
    with pytest.raises(KeyboardInterrupt):
        emit.session_start(subcommand="serve")


def test_public_emit_signatures_have_no_prompt_or_completion_fields():
    import inspect

    from fusion_mlx.telemetry import emit

    forbidden = {
        "prompt",
        "prompt_text",
        "prompts",
        "messages",
        "user_message",
        "completion",
        "completion_text",
        "completions",
        "generated_text",
        "response_text",
        "input_text",
        "output_text",
        "content",
        "text",
        "system_prompt",
        "api_key",
        "auth_token",
        "bearer",
        "file_path",
        "filepath",
        "path",
        "url",
        "stream_url",
        "ip",
        "ip_address",
        "hostname",
        "engine",
    }
    for fn_name in ("session_start", "session_end", "request", "error"):
        fn = getattr(emit, fn_name)
        params = set(inspect.signature(fn).parameters.keys())
        leak = params & forbidden
        assert not leak, (
            f"emit.{fn_name} exposes prompt-like parameter(s) {leak!r}; "
            "free-form text must go through redact.py first"
        )


def test_request_endpoint_constrained_to_allowlist(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="qwen3.5-9b-4bit",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    assert stub_queue[-1]["request"]["endpoint"] == "/v1/chat/completions"

    emit.request(
        endpoint="/v1/chat/completions?api_key=sk-PROD-SECRET#anchor",
        model_alias="qwen3.5-9b-4bit",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "/v1/chat/completions"
    blob = repr(last)
    assert "sk-PROD-SECRET" not in blob
    assert "anchor" not in blob

    emit.request(
        endpoint="/internal/dump?path=/Users/alice/secrets.txt",
        model_alias="qwen3.5-9b-4bit",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "other"
    assert "alice" not in repr(last)
    assert "secrets.txt" not in repr(last)


def test_request_endpoint_normalizes_full_url_to_path(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.request(
        endpoint="https://api.example.com/v1/chat/completions",
        model_alias="qwen3.5-9b-4bit",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "/v1/chat/completions"
    blob = repr(last)
    assert "api.example.com" not in blob
    assert "https://" not in blob

    emit.request(
        endpoint="https://host/v1/chat/completions?key=sk-PROD-LEAK#frag",
        model_alias="qwen3.5-9b-4bit",
        stream=True,
        tool_call_used=False,
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=100.0,
        tps=10.0,
        status=200,
    )
    last = stub_queue[-1]
    assert last["request"]["endpoint"] == "/v1/chat/completions"
    blob = repr(last)
    assert "sk-PROD-LEAK" not in blob
    assert "frag" not in blob
    assert "host" not in blob


def test_session_models_loaded_does_not_materialize_full_input(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    pulled: list[int] = []

    def big_gen():
        for i in range(10_000):
            pulled.append(i)
            yield f"mlx-community/model-{i}"

    emit.session_start(subcommand="serve", models_loaded=big_gen())
    last = stub_queue[-1]
    assert len(pulled) == 32
    assert len(last["session"]["models_loaded"]) == 32

    pulled.clear()
    emit.session_end(subcommand="serve", duration_seconds=1, models_loaded=big_gen())
    last = stub_queue[-1]
    assert len(pulled) == 32
    assert len(last["session"]["models_loaded"]) == 32


def test_session_end_hook_fires_exactly_once(fake_home):
    from fusion_mlx.telemetry import emit

    emit._reset_for_tests()
    calls: list[int] = []

    def hook():
        calls.append(1)

    emit.register_session_end_hook(hook)
    emit.fire_session_end_hook()
    emit.fire_session_end_hook()
    assert calls == [1], f"hook fired {len(calls)} time(s) -- the latch is broken"


def test_session_end_hook_swallows_callable_exceptions(fake_home):
    from fusion_mlx.telemetry import emit

    emit._reset_for_tests()

    def boom():
        raise RuntimeError("synthetic")

    emit.register_session_end_hook(boom)
    emit.fire_session_end_hook()


def test_session_end_hook_no_op_without_registration(fake_home):
    from fusion_mlx.telemetry import emit

    emit._reset_for_tests()
    emit.fire_session_end_hook()


def test_safe_does_not_swallow_signature_mismatch(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    with pytest.raises(TypeError):
        emit.session_start(subkommand="serve")

    with pytest.raises(TypeError):
        emit.request(
            model_alias="qwen3.5-9b-4bit",
            stream=True,
            tool_call_used=False,
            prompt_tokens=10,
            completion_tokens=10,
            ttft_ms=100.0,
            tps=10.0,
            status=200,
        )

    assert stub_queue == []


def test_request_model_alias_local_path_redacted(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="/Users/alice/private-model-checkout",
        stream=True,
        tool_call_used=False,
        prompt_tokens=100,
        completion_tokens=400,
        ttft_ms=250.0,
        tps=42.0,
        status=200,
    )
    r = stub_queue[0]["request"]
    assert r["model_alias"] == "<local>"
    assert "alice" not in repr(stub_queue[0])


def test_flag_values_never_cross_telemetry_boundary(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit
    from fusion_mlx.telemetry.redact import hash_flag_names

    secret = "sk-prod-XXXXXXXXXXXXXXXXXXXXXXXX"
    bearer = "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
    prompt = "summarize this confidential email about Q3 numbers"
    argv = [
        "serve",
        "qwen3.5-9b-4bit",
        "--api-key",
        secret,
        "--auth-header",
        bearer,
        "--initial-prompt",
        prompt,
    ]
    flag_names = hash_flag_names(argv)
    emit.session_start(subcommand="serve", flag_names=flag_names)

    blob = repr(stub_queue[0])
    assert secret not in blob
    assert bearer not in blob
    assert prompt not in blob
    assert set(stub_queue[0]["session"]["flag_names"]) == {
        "api-key",
        "auth-header",
        "initial-prompt",
    }
    flag_names_set = set(stub_queue[0]["session"]["flag_names"])
    assert {"api-key", "auth-header", "initial-prompt"} <= flag_names_set


def test_error_fingerprint_does_not_echo_exception_message(opted_in, stub_queue):
    from fusion_mlx.telemetry import emit

    prompt_in_exc = "summarize this confidential email about Q3 numbers"
    try:
        raise ValueError(f"parser failed on: {prompt_in_exc}")
    except ValueError as exc:
        emit.error(category="parser_failure", exc=exc, phase="chat")

    blob = repr(stub_queue[0])
    assert prompt_in_exc not in blob
    assert "parser failed" not in blob
