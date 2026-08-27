# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest
from starlette.routing import compile_path


def test_current_load_route_rejects_slash_ids():
    rx, _, _ = compile_path("/v1/models/{model_id}/load")
    assert rx.match("/v1/models/a/b/load") is None


def test_path_converter_load_route_matches_slash_ids():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    assert rx.match("/v1/models/a/b/load") is not None
    assert rx.match("/v1/models/a/b/load").groupdict()["model_id"] == "a/b"


def test_load_route_path_converter_matches_slash_ids():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    m = rx.match("/v1/models/mlx-community/Llama-3.2-1B-Instruct-4bit/load")
    assert m is not None
    assert m.groupdict()["model_id"] == "mlx-community/Llama-3.2-1B-Instruct-4bit"


def test_load_route_path_converter_matches_hyphen_ids():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    m = rx.match("/v1/models/mlx-community-Llama-3.2-1B-Instruct-4bit/load")
    assert m is not None
    assert m.groupdict()["model_id"] == "mlx-community-Llama-3.2-1B-Instruct-4bit"


def test_load_route_path_converter_does_not_swallow_status():
    rx, _, _ = compile_path("/v1/models/{model_id:path}/load")
    assert rx.match("/v1/models/status") is None


def test_slash_to_hyphen_retry_resolves_pool_entry():
    class FakeEntry:
        engine = None

    class FakePool:
        def __init__(self):
            self._entries = {"mlx-community-Foo-4bit": FakeEntry()}

        def get_entry(self, mid):
            return self._entries.get(mid)

    pool = FakePool()
    requested = "mlx-community/Foo-4bit"
    resolved = requested
    entry = pool.get_entry(resolved)
    if entry is None and "/" in resolved:
        resolved = resolved.replace("/", "-")
        entry = pool.get_entry(resolved)
    assert entry is not None
    assert resolved == "mlx-community-Foo-4bit"


def test_slash_to_hyphen_retry_not_applied_when_slash_entry_exists():
    class FakeEntry:
        engine = None

    class FakePool:
        def __init__(self):
            self._entries = {"a/b": FakeEntry()}

        def get_entry(self, mid):
            return self._entries.get(mid)

    pool = FakePool()
    resolved = "a/b"
    entry = pool.get_entry(resolved)
    if entry is None and "/" in resolved:
        resolved = resolved.replace("/", "-")
        entry = pool.get_entry(resolved)
    assert entry is not None
    assert resolved == "a/b"


REAL_MODEL = os.environ.get("FUSION_MLX_REAL_MODEL_TESTS") == "1"


@pytest.mark.real_model
def test_load_slash_id_real_server():
    if not REAL_MODEL:
        pytest.skip("set FUSION_MLX_REAL_MODEL_TESTS=1")
    import json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        "http://127.0.0.1:11434/v1/models/mlx-community/Llama-3.2-1B-Instruct-4bit/load",
        method="POST",
        data=b"{}",
        headers={"Content-Type": "application/json", "X-Fusion-Source": "model-hub"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            assert resp.status == 200
            assert body["status"] == "ok"
    except urllib.error.HTTPError as e:
        assert "Model not found" in str(
            e.detail
        ), f"route did not match: {e.code} {e.detail}"
