# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /v1/node/load node-level load snapshot (Issue #264).

Covers the ``_node_load_snapshot`` helper that backs the
``GET /v1/node/load`` endpoint. Exercises schema shape, node identity,
model extraction, capacity headroom math, and graceful handling when no
pool is attached.
"""

from __future__ import annotations

from types import SimpleNamespace

from fusion_mlx.server import _node_load_snapshot


class _FakePool:
    def __init__(self, status):
        self._status = status

    def get_status(self):
        return self._status


def _fake_config(host="0.0.0.0", port=11434):
    return SimpleNamespace(host=host, port=port, bind_host=None, bind_port=None)


def _fake_status():
    return {
        "current_model_memory": 4_000_000_000,
        "final_ceiling": 16_000_000_000,
        "models": [
            {
                "id": "qwen3-4b",
                "loaded": True,
                "is_loading": False,
                "estimated_size": 4_000_000_000,
            },
            {
                "id": "qwen3-8b",
                "loaded": False,
                "is_loading": False,
                "estimated_size": 8_000_000_000,
            },
        ],
    }


class TestNodeLoadSnapshot:
    def test_schema_fields_present(self):
        snap = _node_load_snapshot(_FakePool(_fake_status()), _fake_config())
        for key in (
            "node_id",
            "host",
            "port",
            "uptime_seconds",
            "active_requests",
            "memory",
            "models",
            "capacity",
            "throughput",
        ):
            assert key in snap, f"missing top-level key: {key}"
        for key in ("total_bytes", "available_bytes", "used_bytes", "available_percent"):
            assert key in snap["memory"], f"missing memory key: {key}"
        for key in ("free_memory_bytes", "can_load_estimate_bytes"):
            assert key in snap["capacity"], f"missing capacity key: {key}"
        for key in ("avg_prefill_tps", "avg_generation_tps"):
            assert key in snap["throughput"], f"missing throughput key: {key}"

    def test_node_id_format_and_port(self):
        snap = _node_load_snapshot(_FakePool(_fake_status()), _fake_config(port=11434))
        assert snap["port"] == 11434
        assert snap["node_id"].endswith(":11434")
        assert snap["host"] == "0.0.0.0"

    def test_models_extracted_with_resident_bytes(self):
        snap = _node_load_snapshot(_FakePool(_fake_status()), _fake_config())
        ids = {m["id"] for m in snap["models"]}
        assert ids == {"qwen3-4b", "qwen3-8b"}
        loaded = [m for m in snap["models"] if m["loaded"]]
        assert len(loaded) == 1
        assert loaded[0]["id"] == "qwen3-4b"
        assert loaded[0]["resident_bytes"] == 4_000_000_000

    def test_can_load_estimate_uses_ceiling_headroom(self):
        snap = _node_load_snapshot(_FakePool(_fake_status()), _fake_config())
        # final_ceiling(16G) - current_model_memory(4G) = 12G
        assert snap["capacity"]["can_load_estimate_bytes"] == 12_000_000_000

    def test_no_ceiling_falls_back_to_available_memory(self):
        status = _fake_status()
        status["final_ceiling"] = None
        snap = _node_load_snapshot(_FakePool(status), _fake_config())
        # No ceiling => can_load = available system memory (>=0, real machine)
        assert snap["capacity"]["can_load_estimate_bytes"] >= 0
        assert snap["capacity"]["can_load_estimate_bytes"] == snap["memory"]["available_bytes"]

    def test_no_pool_is_safe(self):
        snap = _node_load_snapshot(None, _fake_config())
        assert snap["models"] == []
        assert snap["capacity"]["can_load_estimate_bytes"] >= 0
        assert snap["active_requests"] == 0

    def test_memory_available_percent_in_range(self):
        snap = _node_load_snapshot(_FakePool(_fake_status()), _fake_config())
        pct = snap["memory"]["available_percent"]
        assert 0.0 <= pct <= 100.0
        assert snap["memory"]["total_bytes"] > 0
        assert snap["memory"]["available_bytes"] >= 0
        assert snap["memory"]["used_bytes"] >= 0

    def test_pool_get_status_failure_is_safe(self):
        class _BrokenPool:
            def get_status(self):
                raise RuntimeError("boom")

        snap = _node_load_snapshot(_BrokenPool(), _fake_config())
        assert snap["models"] == []
        assert snap["capacity"]["can_load_estimate_bytes"] >= 0
