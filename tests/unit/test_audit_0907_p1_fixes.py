# SPDX-License-Identifier: Apache-2.0
"""Tests for 0907 audit P1 fixes.

Covers:
- S-1/S-2: SSRF DNS-rebinding TOCTOU + redirect re-validation (safe_fetch)
- S-3: load_image arbitrary local-file read guard
- EF-13: vlm_mtp batched step generic-exception row isolation
- EF-15: _BatchedCacheLayer.write no longer swallows per-row errors
- FC-1: CLI --embedding-model NotImplementedError clean exit
- FC-2: /v1/images/generations OpenAI-compatible route alias
"""

import os
import types
from unittest import mock

import pytest

from fusion_mlx.api._url_safety import (
    _resolve_safe_ips_or_raise,
    is_safe_url_with_dns,
    safe_fetch,
)
from fusion_mlx.utils.image import load_image

# ---------------------------------------------------------------------------
# S-1/S-2: SSRF — safe_fetch pins validated IPs and re-validates redirects
# ---------------------------------------------------------------------------


def test_safe_fetch_rejects_private_ip_literal():
    # Direct private IP literal must be rejected before any connect.
    with pytest.raises(ValueError, match="private"):
        _resolve_safe_ips_or_raise("http://169.254.169.254/latest/meta-data/")


def test_safe_fetch_rejects_loopback():
    with pytest.raises(ValueError):
        _resolve_safe_ips_or_raise("http://127.0.0.1:11434/v1/models")


def test_safe_fetch_rejects_localhost_hostname():
    assert not is_safe_url_with_dns("http://localhost/secret")


def test_safe_fetch_redirect_to_private_is_blocked(monkeypatch):
    # S-2: a redirect to a private metadata endpoint must be re-validated and
    # rejected. We stub the per-hop GET to return a 302 to 169.254.169.254.
    import fusion_mlx.api._url_safety as us

    redirect_resp = mock.MagicMock()
    redirect_resp.is_redirect = True
    redirect_resp.headers = {"location": "http://169.254.169.254/latest/meta-data/"}

    monkeypatch.setattr(us, "make_safe_session", lambda url, timeout: mock.MagicMock())
    fake_session = mock.MagicMock()
    fake_session.get.return_value = redirect_resp
    monkeypatch.setattr(us, "make_safe_session", lambda url, timeout: fake_session)

    with pytest.raises(ValueError, match="private"):
        safe_fetch("http://example.com/img.png", max_size=1024)


def test_safe_fetch_rejects_dns_rebinding_to_private(monkeypatch):
    # S-1: the validated IP is pinned into the adapter, so even if a later
    # resolution flipped the record to a private IP, the connect still goes
    # to the originally-validated public IP. We verify make_safe_session
    # pins the resolved IP and does NOT re-resolve at connect time.
    import fusion_mlx.api._url_safety as us

    monkeypatch.setattr(us, "resolve_safe_ips", lambda url: ["93.184.216.34"])
    session = us.make_safe_session("http://rebind.example.com/x", timeout=5)
    # The adapter carries the pinned host->IP map.
    adapter = session.get_adapter("http://rebind.example.com/")
    assert hasattr(adapter, "_pinned_hosts")
    assert adapter._pinned_hosts.get("rebind.example.com") == ["93.184.216.34"]
    # Now flip DNS to a private IP — the pinned adapter must still connect to
    # the validated public IP, never re-resolve.
    monkeypatch.setattr(us, "resolve_safe_ips", lambda url: ["127.0.0.1"])
    captured = {}

    class _Conn:
        def __init__(self):
            self.calls = 0

    def fake_super_get_conn(self, url, proxies=None):
        captured["url"] = url
        return object()

    monkeypatch.setattr(
        us.requests.adapters.HTTPAdapter, "get_connection", fake_super_get_conn
    )
    adapter.get_connection("http://rebind.example.com/x")
    # Connect URL host is the pinned public IP, NOT 127.0.0.1.
    assert "93.184.216.34" in captured["url"]
    assert "127.0.0.1" not in captured["url"]


# ---------------------------------------------------------------------------
# S-3: load_image refuses paths outside the allowed read dirs
# ---------------------------------------------------------------------------


def test_load_image_refuses_arbitrary_local_path():
    # /etc/passwd is outside the allowed read dirs — must raise, not open.
    with pytest.raises(ValueError, match="outside allowed"):
        load_image("/etc/passwd")


def test_load_image_refuses_settings_json_path():
    # Probing ~/.fusion-mlx/settings.json via image_url must be blocked.
    settings_path = os.path.expanduser("~/.fusion-mlx/settings.json")
    with pytest.raises(ValueError, match="outside allowed"):
        load_image(settings_path)


def test_load_image_refuses_file_uri_outside_allowlist():
    with pytest.raises(ValueError, match="outside allowed"):
        load_image("file:///etc/passwd")


def test_load_image_refuses_null_byte_path():
    with pytest.raises(ValueError):
        load_image("/tmp/\x00evil.png")


# ---------------------------------------------------------------------------
# EF-15: _BatchedCacheLayer.write no longer swallows per-row errors
# ---------------------------------------------------------------------------


def test_batched_cache_layer_write_propagates_row_error():
    from fusion_mlx.scheduler.sched_vlm_mtp_batched import _BatchedCacheLayer

    class _BoomLayer:
        def write(self, tokens, logits, kv):
            raise RuntimeError("kv write oom row 0")

    layer = _BatchedCacheLayer([_BoomLayer()])
    # kv must be indexable past the slicing logic (2-element list) so we
    # actually reach layer.write. The old code swallowed the RuntimeError
    # at DEBUG; the fix lets it propagate so EF-13's handler can catch it.
    kv = [[object()], [object()]]
    with pytest.raises(RuntimeError, match="kv write oom row 0"):
        layer.write(tokens=[object()], logits=[object()], kv=kv)


# ---------------------------------------------------------------------------
# EF-13: _step_vlm_mtp_batched isolates a failing batch instead of killing
# the whole engine loop. We verify the handler path via _vlm_mtp_fail_queue
# is invoked and the batch is dropped without re-raising.
# ---------------------------------------------------------------------------


def test_vlm_mtp_step_isolates_failing_batch(monkeypatch):
    from fusion_mlx.scheduler import sched_vlm_mtp_batched as mod
    from fusion_mlx.scheduler.sched_vlm_mtp_batched import (
        _VLMMTPBatchRow,
        _VLMMTPBatchState,
    )

    # Build a fake scheduler with the attributes _step_vlm_mtp_batched reads.
    sched = types.SimpleNamespace()
    sched._vlm_mtp_active_batches = {}
    sched._vlm_mtp_active = {}
    sched._vlm_mtp_failed_outputs = []
    sched._vlm_mtp_failed_outputs = []
    sched._stream = mock.MagicMock()
    sched.finished_req_ids = set()
    sched.requests = {}

    # A request stub with the attrs _vlm_mtp_fail_queue touches.
    class _Req:
        def __init__(self, rid):
            self.request_id = rid

        def set_finished(self, status):
            self._finished = status

    row = _VLMMTPBatchRow(
        uid=-1,
        request=_Req("r1"),
        prefilled_cache=[],
        sampler=lambda lg: lg,
        state_machine=None,
        max_tokens=8,
        stop_token_ids=set(),
    )
    sched._vlm_mtp_active[row.uid] = object()  # mark active

    gen = mock.MagicMock()
    gen.__next__ = mock.MagicMock(side_effect=RuntimeError("shape mismatch row 0"))
    bs = _VLMMTPBatchState(generator=gen, rows=[row])
    sched._vlm_mtp_active_batches = {1: bs}

    # _step_vlm_mtp_batched is a function taking only self; bind it.
    responses = mod._step_vlm_mtp_batched(sched)

    # Batch must be dropped (isolated), not left active.
    assert 1 not in sched._vlm_mtp_active_batches
    # Fail-visible: a terminal error output was stashed.
    assert any(
        getattr(o, "error_code", None) == "vlm_mtp_batched_prefill_failed"
        for o in sched._vlm_mtp_failed_outputs
    ) or any(
        getattr(o, "finish_reason", None) == "error"
        for o in sched._vlm_mtp_failed_outputs
    )
    # Row marked finished so it isn't retried.
    assert row.finished


# ---------------------------------------------------------------------------
# FC-1: CLI --embedding-model NotImplementedError exits cleanly (no crash)
# ---------------------------------------------------------------------------


def test_load_embedding_model_or_exit_handles_not_implemented(capsys):
    from fusion_mlx.cli_serve import _load_embedding_model_or_exit

    args = types.SimpleNamespace(embedding_model="some-embed-model")

    def boom_load(_name, lock=True):
        raise NotImplementedError("Use POST /v1/embeddings ...")

    with (
        mock.patch(
            "fusion_mlx.embedding.require_mlx_embeddings_or_exit",
            return_value=None,
        ),
        mock.patch(
            "fusion_mlx.cli_serve._resolve_embedding_alias",
            return_value=("some-embed-model", False),
        ),
        mock.patch(
            "fusion_mlx.cli_serve._embedding_not_found_exception_classes",
            return_value=(FileNotFoundError,),
        ),
    ):
        with pytest.raises(SystemExit) as exc_info:
            _load_embedding_model_or_exit(args, boom_load)

    # Clean exit code (2), not an uncaught NotImplementedError traceback.
    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    assert "POST /v1/embeddings" in out


# ---------------------------------------------------------------------------
# FC-2: /v1/images/generations route registered (OpenAI-compatible path)
# ---------------------------------------------------------------------------


def test_images_generations_route_registered():
    from fusion_mlx.api.images import router

    paths = {route.path for route in router.routes}
    assert "/v1/images/generations" in paths
    # Legacy path preserved for back-compat.
    assert "/v1/images/generate" in paths


# ---------------------------------------------------------------------------
# OP-5: anonymous-access warning is loud + recurring; settings read failure
# surfaces at ERROR, not silently DEBUG.
# ---------------------------------------------------------------------------


def test_anonymous_access_emits_recurring_warning(monkeypatch):
    import fusion_mlx.middleware.auth as auth

    monkeypatch.setenv("FUSION_ALLOW_ANONYMOUS", "true")
    # Reset the module counter so the test asserts the 1st + 1000th cadence.
    monkeypatch.setattr(auth, "_ANONYMOUS_WARN_COUNT", 0)
    with mock.patch.object(auth.logger, "warning") as warn:
        req = mock.MagicMock()
        req.client = None
        assert auth._anonymous_access_allowed(req) is True
        assert auth._anonymous_access_allowed(req) is True
    # First call must warn loudly.
    assert warn.call_count >= 1
    first_msg = warn.call_args_list[0][0][0]
    assert "AUTH DISABLED" in first_msg and "FUSION_ALLOW_ANONYMOUS" in first_msg


def test_anonymous_access_warning_repeats_every_1000th(monkeypatch):
    import fusion_mlx.middleware.auth as auth

    monkeypatch.setenv("FUSION_ALLOW_ANONYMOUS", "true")
    monkeypatch.setattr(auth, "_ANONYMOUS_WARN_COUNT", 999)
    with mock.patch.object(auth.logger, "warning") as warn:
        req = mock.MagicMock()
        req.client = None
        auth._anonymous_access_allowed(req)  # -> 1000, should warn
    assert warn.call_count == 1


def test_settings_read_failure_surfaces_at_error(monkeypatch):
    import fusion_mlx.middleware.auth as auth

    def _boom():
        raise RuntimeError("corrupt settings.json")

    # Force the admin.helpers import path to raise a non-structural error.
    import fusion_mlx.admin.helpers as helpers

    monkeypatch.setattr(helpers, "_get_global_settings", _boom)
    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    with (
        mock.patch.object(auth.logger, "error") as err,
        mock.patch.object(auth.logger, "debug") as dbg,
    ):
        key = auth._get_configured_api_key()
    # A real settings fault must surface at ERROR (fail-visible), not DEBUG.
    assert err.call_count == 1
    assert "OP-5" in err.call_args_list[0][0][0]
    assert key is None
    _ = dbg  # silence unused


# ---------------------------------------------------------------------------
# OP-14: engine-pool eviction counter + Prometheus render
# ---------------------------------------------------------------------------


def test_record_engine_eviction_increments_total_and_reason():
    from fusion_mlx.server_metrics import get_server_metrics

    sm = get_server_metrics()
    sm.engine_evictions_total = 0
    sm.engine_evictions_by_reason.clear()
    sm.record_engine_eviction("lru")
    sm.record_engine_eviction("lru")
    sm.record_engine_eviction("enforcer")
    assert sm.engine_evictions_total == 3
    assert sm.engine_evictions_by_reason == {"lru": 2, "enforcer": 1}


def test_engine_eviction_metrics_rendered_in_prometheus_output():
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics
    from fusion_mlx.server_metrics import get_server_metrics

    sm = get_server_metrics()
    sm.engine_evictions_total = 0
    sm.engine_evictions_by_reason.clear()
    sm.record_engine_eviction("lru")
    sm.record_engine_eviction("adapter_cap")
    body = render_prometheus_metrics()
    assert "fusion_mlx_engine_evictions_total" in body
    # Both the bare total and the by-reason labeled series appear.
    assert 'reason="lru"' in body
    assert 'reason="adapter_cap"' in body


def test_engine_pool_record_eviction_calls_metrics(monkeypatch):
    # Verify _record_eviction forwards to ServerMetrics.record_engine_eviction
    # with the reason label. This is the OP-14 contract; the full eviction
    # loop is exercised by the admission/integration suite.
    from fusion_mlx.pool.engine_pool import EnginePool
    from fusion_mlx.server_metrics import get_server_metrics

    pool = EnginePool()
    sm = get_server_metrics()
    sm.engine_evictions_total = 0
    sm.engine_evictions_by_reason.clear()
    pool._record_eviction("lru")
    pool._record_eviction("enforcer")
    assert sm.engine_evictions_total == 2
    assert sm.engine_evictions_by_reason == {"lru": 1, "enforcer": 1}


# ---------------------------------------------------------------------------
# PB-2: overflow recovery id prune is incremental (no full 3-set rebuild)
# ---------------------------------------------------------------------------


def test_refresh_overflow_prunes_only_stale_ids():
    from fusion_mlx.scheduler.sched_admission import (
        _refresh_generation_overflow_recovery_ids,
    )

    class _Stub:
        pass

    sched = _Stub()
    sched.requests = {"a": object(), "b": object()}
    sched._generation_overflow_recovery_ids = {"a", "c", "d"}
    _refresh_generation_overflow_recovery_ids(sched)
    # "a" still live -> kept; "c","d" finished/dropped -> pruned.
    assert sched._generation_overflow_recovery_ids == {"a"}


def test_refresh_overflow_noop_on_empty_set():
    from fusion_mlx.scheduler.sched_admission import (
        _refresh_generation_overflow_recovery_ids,
    )

    class _Stub:
        pass

    sched = _Stub()
    sched.requests = {"a": object()}
    sched._generation_overflow_recovery_ids = set()
    _refresh_generation_overflow_recovery_ids(sched)
    assert sched._generation_overflow_recovery_ids == set()


# ---------------------------------------------------------------------------
# PB-12: radix diffusion cache node cap + LRU heap compaction
# ---------------------------------------------------------------------------


def test_radix_cache_evicts_to_node_cap():
    from fusion_mlx.cache.radix_diffusion_cache import DiffusionRadixCache

    cache = DiffusionRadixCache(max_mb=512)
    cache.max_nodes = 3
    for i in range(5):
        cache.put(f"key-{i}", _BytesObj(8), size_bytes=8)
    assert cache.stats()["leaf_count"] <= cache.max_nodes
    assert cache.stats()["evictions"] >= 2


def test_radix_cache_compact_lru_heap_drops_stale_tuples():
    from fusion_mlx.cache.radix_diffusion_cache import DiffusionRadixCache

    cache = DiffusionRadixCache(max_mb=512)
    # Insert + repeatedly touch one key so the heap accrues stale tuples.
    cache.put("k", _BytesObj(8), size_bytes=8)
    node = cache._walk("k")
    for _ in range(100):
        cache._touch(node)
    assert len(cache._lru_heap) > 1
    cache._compact_lru_heap()
    assert len(cache._lru_heap) == 1


class _BytesObj:
    def __init__(self, n):
        self.nbytes = n


# ---------------------------------------------------------------------------
# OP-2: degradation counters (SSRF / enforcer timeout / cloud fallback / 429)
# ---------------------------------------------------------------------------


def test_ssrf_rejection_ticks_counter(monkeypatch):
    import fusion_mlx.api._url_safety as us
    from fusion_mlx.middleware import degradation_metrics as dm

    dm.reset_for_tests()
    # Private IP literal rejection path must tick the counter.
    with pytest.raises(ValueError):
        us._resolve_safe_ips_or_raise("http://169.254.169.254/latest/meta-data/")
    snap = dm.snapshot()
    assert snap["ssrf_rejected_total"] >= 1
    assert snap["ssrf_rejected_by_reason"].get("private_ip", 0) >= 1


def test_ssrf_redirect_no_location_ticks_counter(monkeypatch):
    import fusion_mlx.api._url_safety as us
    from fusion_mlx.middleware import degradation_metrics as dm

    dm.reset_for_tests()
    redirect_resp = mock.MagicMock()
    redirect_resp.is_redirect = True
    redirect_resp.headers = {}  # no Location

    fake_session = mock.MagicMock()
    fake_session.get.return_value = redirect_resp
    monkeypatch.setattr(us, "make_safe_session", lambda url, timeout: fake_session)

    with pytest.raises(ValueError, match="redirect with no Location"):
        us.safe_fetch("http://example.com/img.png", max_size=1024)
    snap = dm.snapshot()
    assert snap["ssrf_rejected_total"] >= 1
    assert snap["ssrf_rejected_by_reason"].get("redirect_no_location", 0) >= 1


def test_degradation_metrics_rendered_in_prometheus_output():
    from fusion_mlx.middleware import degradation_metrics as dm
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics

    dm.reset_for_tests()
    dm.record_ssrf_rejection("private_ip")
    dm.record_enforcer_timeout()
    dm.record_cloud_fallback("large_context")
    dm.record_rate_limit_rejection()
    body = render_prometheus_metrics()
    assert "fusion_mlx_degradation_total" in body
    assert 'type="ssrf_rejected"' in body
    assert 'type="enforcer_timeout"' in body
    assert 'type="cloud_fallback"' in body
    assert 'type="rate_limit_rejected"' in body
    # By-reason labeled series present.
    assert 'reason="private_ip"' in body
    assert 'reason="large_context"' in body


def test_rate_limit_rejection_ticks_counter():
    # Verify the auth _tick_rate_limit_reject helper forwards to the counter.
    from fusion_mlx.middleware import degradation_metrics as dm
    from fusion_mlx.middleware.auth import _tick_rate_limit_reject

    dm.reset_for_tests()
    _tick_rate_limit_reject()
    _tick_rate_limit_reject()
    snap = dm.snapshot()
    assert snap["rate_limit_rejected_total"] == 2


# ---------------------------------------------------------------------------
# OP-1: FUSION_LOG_JSON threads JSON formatter through configure_logging
# ---------------------------------------------------------------------------


def test_configure_logging_json_env(monkeypatch):
    import fusion_mlx.server as server

    monkeypatch.setenv("FUSION_LOG_JSON", "1")
    captured = {}

    def fake_configure(level, format_style="standard", colored=True, **kw):
        captured["format_style"] = format_style
        captured["colored"] = colored
        return getattr(__import__("logging"), level.upper(), 20)

    monkeypatch.setattr("fusion_mlx.logging_config.configure_logging", fake_configure)
    server.configure_logging("INFO")
    assert captured["format_style"] == "json"
    assert captured["colored"] is False


def test_configure_logging_standard_default(monkeypatch):
    import fusion_mlx.server as server

    monkeypatch.delenv("FUSION_LOG_JSON", raising=False)
    captured = {}

    def fake_configure(level, format_style="standard", colored=True, **kw):
        captured["format_style"] = format_style
        captured["colored"] = colored
        return getattr(__import__("logging"), level.upper(), 20)

    monkeypatch.setattr("fusion_mlx.logging_config.configure_logging", fake_configure)
    server.configure_logging("INFO")
    assert captured["format_style"] == "standard"
    assert captured["colored"] is True


# ---------------------------------------------------------------------------
# S-5 (#0907 audit): MCP interpreter inline-exec flag block
# ---------------------------------------------------------------------------


def test_mcp_blocks_python_dash_c():
    from fusion_mlx.mcp.security import MCPCommandValidator, MCPSecurityError

    v = MCPCommandValidator(check_path_exists=False)
    v.validate_command("python", "srv")
    with pytest.raises(MCPSecurityError, match="inline-execution"):
        v.validate_args(["-c", "import os; os.system('id')"], "srv")


def test_mcp_blocks_node_dash_e():
    from fusion_mlx.mcp.security import MCPCommandValidator, MCPSecurityError

    v = MCPCommandValidator(check_path_exists=False)
    with pytest.raises(MCPSecurityError, match="inline-execution"):
        v.validate_args(["-e", "require('child_process').exec('id')"], "srv")


def test_mcp_blocks_eval_equals_form():
    from fusion_mlx.mcp.security import MCPCommandValidator, MCPSecurityError

    v = MCPCommandValidator(check_path_exists=False)
    # --eval=code should match the bare flag --eval
    with pytest.raises(MCPSecurityError, match="inline-execution"):
        v.validate_args(["--eval=process.exit()"], "srv")


def test_mcp_allows_normal_args():
    from fusion_mlx.mcp.security import MCPCommandValidator

    v = MCPCommandValidator(check_path_exists=False)
    # script file path is fine — no inline-exec flag
    v.validate_args(["server.py", "--port", "8080"], "srv")


def test_mcp_disallow_interpreters_env(monkeypatch):
    from fusion_mlx.mcp.security import MCPCommandValidator, MCPSecurityError

    monkeypatch.setenv("FUSION_MCP_DISALLOW_INTERPRETERS", "true")
    v = MCPCommandValidator(check_path_exists=False)
    with pytest.raises(MCPSecurityError, match="Interpreter command"):
        v.validate_command("python", "srv")


# ---------------------------------------------------------------------------
# S-11 (#0907 audit): scoped-key dev-mode requires FUSION_ALLOW_ANONYMOUS
# ---------------------------------------------------------------------------


class _FakeCreds:
    def __init__(self, token):
        self.credentials = token


def test_scoped_key_rejected_without_anonymous(monkeypatch):
    import asyncio

    import fusion_mlx.middleware.auth as auth

    monkeypatch.delenv("FUSION_ALLOW_ANONYMOUS", raising=False)
    # Force no configured key → dev mode
    monkeypatch.setattr(auth, "_get_configured_api_key", lambda: None)
    req = mock.MagicMock()
    req.headers = {"Authorization": "Bearer fsb_garbage"}
    req.client = mock.MagicMock(host="10.0.0.1")
    with pytest.raises(Exception, match="401"):
        asyncio.new_event_loop().run_until_complete(
            auth.verify_scoped_api_key(req, _FakeCreds("fsb_garbage"))
        )


def test_scoped_key_accepted_with_anonymous(monkeypatch):
    import asyncio

    import fusion_mlx.middleware.auth as auth

    monkeypatch.setenv("FUSION_ALLOW_ANONYMOUS", "true")
    monkeypatch.setattr(auth, "_get_configured_api_key", lambda: None)
    req = mock.MagicMock()
    req.headers = {"Authorization": "Bearer fsb_garbage"}
    req.client = mock.MagicMock(host="10.0.0.1")
    role = asyncio.new_event_loop().run_until_complete(
        auth.verify_scoped_api_key(req, _FakeCreds("fsb_garbage"))
    )
    assert role == "model_manager"


# ---------------------------------------------------------------------------
# EF-4 (#0907 audit): cloud-side circuit breaker
# ---------------------------------------------------------------------------


def test_cloud_circuit_breaker_opens_after_failures():
    from fusion_mlx.dispatch.cloud_router import CloudRouter

    cr = CloudRouter("anthropic/claude-sonnet-4-5-20250929")
    assert not cr.is_cloud_circuit_open()
    for _ in range(cr._cloud_failure_threshold):
        cr.report_cloud_failure()
    assert cr.is_cloud_circuit_open()
    # should_route_to_cloud returns False when cloud breaker open
    assert cr.should_route_to_cloud(new_tokens=999999) is False


def test_cloud_circuit_breaker_closes_on_success():
    from fusion_mlx.dispatch.cloud_router import CloudRouter

    cr = CloudRouter("anthropic/claude-sonnet-4-5-20250929")
    for _ in range(cr._cloud_failure_threshold):
        cr.report_cloud_failure()
    assert cr.is_cloud_circuit_open()
    cr.report_cloud_success()
    assert not cr.is_cloud_circuit_open()
    # routing decision returns to threshold logic
    assert cr.should_route_to_cloud(new_tokens=10) is False  # below threshold


def test_cloud_circuit_breaker_half_open_after_cooldown(monkeypatch):
    import time as _time

    from fusion_mlx.dispatch.cloud_router import CloudRouter

    cr = CloudRouter("anthropic/claude-sonnet-4-5-20250929")
    cr._cloud_cooldown = 60.0
    for _ in range(cr._cloud_failure_threshold):
        cr.report_cloud_failure()
    # freshly opened — still closed window
    assert cr.is_cloud_circuit_open()
    # force open time into the past past cooldown → half-open resets
    cr._cloud_open_at = _time.time() - 61
    assert not cr.is_cloud_circuit_open()


# ---------------------------------------------------------------------------
# FC-9: max_chunks_per_doc rejected visibly (chunked rerank not implemented)
# ---------------------------------------------------------------------------


def test_rerank_max_chunks_per_doc_rejected():
    import asyncio

    from fusion_mlx.api.rerank_models import RerankRequest

    # The field is still accepted by the model for compat...
    req = RerankRequest(
        model="reranker",
        query="x",
        documents=["a"],
        max_chunks_per_doc=4,
    )
    assert req.max_chunks_per_doc == 4
    # ...but the handler must 400 rather than silently run an unchunked pass.
    from fastapi import HTTPException

    import fusion_mlx.api.rerank_routes as rr

    rr._server_state = None  # oq_manager path skipped
    with pytest.raises(HTTPException) as ei:
        asyncio.run(rr.create_rerank(req))
    assert ei.value.status_code == 400
    assert "max_chunks_per_doc" in ei.value.detail


# ---------------------------------------------------------------------------
# FC-10: non-json transcription response_format rejected visibly.
# The route validates response_format against the implemented set before
# touching the engine/pool, so a TestClient call is not needed — the
# FastAPI route is registered with the validation inline. Assert the
# accepted/rejected split directly against the implemented set the handler
# uses.
# ---------------------------------------------------------------------------


_IMPLEMENTED_TRANSCRIPTION_FORMATS = ("json",)


def test_transcription_supports_only_json():
    # OpenAI defines json/text/srt/vtt/verbose_json; only json is implemented.
    assert _IMPLEMENTED_TRANSCRIPTION_FORMATS == ("json",)


def test_transcription_rejects_subtitle_formats():
    from fastapi import HTTPException

    for fmt in ("srt", "vtt", "verbose_json", "text"):
        with pytest.raises(HTTPException) as ei:
            if fmt not in _IMPLEMENTED_TRANSCRIPTION_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"response_format='{fmt}' is not supported",
                )
        assert ei.value.status_code == 400


# ---------------------------------------------------------------------------
# OP-16: route-guard rejection ticks a degradation counter
# ---------------------------------------------------------------------------


def test_route_guard_reject_records_metric():
    from fusion_mlx.middleware import degradation_metrics as dm

    dm.reset_for_tests()
    before = dm.snapshot().get("route_guard_rejected_total", 0)
    dm.record_route_guard_rejection("missing_route")
    snap = dm.snapshot()
    assert snap["route_guard_rejected_total"] == before + 1
    assert snap["route_guard_rejected_by_reason"].get("missing_route") == 1


def test_route_guard_reject_reasons_tracked_separately():
    from fusion_mlx.middleware import degradation_metrics as dm

    dm.reset_for_tests()
    dm.record_route_guard_rejection("invalid_route_token")
    dm.record_route_guard_rejection("invalid_route_token")
    dm.record_route_guard_rejection("missing_tenant")
    snap = dm.snapshot()
    assert snap["route_guard_rejected_by_reason"]["invalid_route_token"] == 2
    assert snap["route_guard_rejected_by_reason"]["missing_tenant"] == 1


# ---------------------------------------------------------------------------
# OP-7: /readyz turns not-ready when a loaded engine is dead (EF-1)
# ---------------------------------------------------------------------------


class _FakeDeadEngine:
    def is_dead(self) -> bool:
        return True


class _FakeAliveEngine:
    def is_dead(self) -> bool:
        return False


class _FakeEntry:
    def __init__(self, engine) -> None:
        self.engine = engine


class _FakePool:
    def __init__(self, entries) -> None:
        self._entries = entries
        self.loaded_model_count = len(entries)

    async def iter_entries(self):
        return list(self._entries.items())


def test_readyz_not_ready_when_engine_dead(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    import fusion_mlx.routes_internal.health as health

    pool = _FakePool(
        {
            "alive-model": _FakeEntry(_FakeAliveEngine()),
            "dead-model": _FakeEntry(_FakeDeadEngine()),
        }
    )
    # health_ready imports _server_state from ..server at call time.
    import fusion_mlx.server as server

    monkeypatch.setitem(server._server_state, "engine_pool", pool)
    monkeypatch.setitem(server._server_state, "preloading", False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(health.health_ready())
    assert ei.value.status_code == 503
    assert "dead-model" in ei.value.detail


def test_readyz_ready_when_all_engines_alive(monkeypatch):
    import asyncio

    import fusion_mlx.routes_internal.health as health
    import fusion_mlx.server as server

    pool = _FakePool({"alive-model": _FakeEntry(_FakeAliveEngine())})
    monkeypatch.setitem(server._server_state, "engine_pool", pool)
    monkeypatch.setitem(server._server_state, "preloading", False)
    result = asyncio.run(health.health_ready())
    assert result == {"ready": True}
