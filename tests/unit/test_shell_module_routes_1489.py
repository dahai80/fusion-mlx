# SPDX-License-Identifier: Apache-2.0
# #1489: shell-module manifest endpoints served by the engine. Covers the
# search/notifications behavior contract and the manifest drift contract
# (registered upstream must equal the gateway static upstream).

import json
from pathlib import Path

from fusion_mlx.api import shell_module_routes as smr

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSearchModels:
    async def test_filter_and_cap(self, monkeypatch):
        models = [{"id": f"model-{i}", "engine_type": "llm"} for i in range(20)]
        monkeypatch.setattr(smr, "_engine_models", lambda: _async(models))
        resp = await smr.shell_search_models(q="model-1")
        body = json.loads(resp.body)
        assert len(body["items"]) == 8
        assert body["degraded"] is False
        assert all("model-1" in it["title"] for it in body["items"])

    async def test_query_match_on_other_keys(self, monkeypatch):
        models = [{"id": "x", "model_type": "qwen2"}]
        monkeypatch.setattr(smr, "_engine_models", lambda: _async(models))
        resp = await smr.shell_search_models(q="qwen")
        body = json.loads(resp.body)
        assert body["items"] == [{"title": "x", "route": "/models"}]

    async def test_engine_failure_degrades_explicitly(self, monkeypatch):
        async def _boom():
            raise RuntimeError("pool down")

        monkeypatch.setattr(smr, "_engine_models", _boom)
        resp = await smr.shell_search_models(q="")
        body = json.loads(resp.body)
        assert body["items"] == []
        assert body["degraded"] is True

    async def test_no_match_empty(self, monkeypatch):
        monkeypatch.setattr(smr, "_engine_models", lambda: _async([{"id": "a"}]))
        resp = await smr.shell_search_models(q="zzz")
        assert json.loads(resp.body)["items"] == []


class TestNotifications:
    async def test_oom_imminent_maps_to_error_item(self, monkeypatch):
        async def _health(_auth=True):
            return {"oom_risk": "imminent"}

        monkeypatch.setattr(smr, "_engine_models", lambda: _async([]))
        monkeypatch.setattr("fusion_mlx.routes_internal.health.v1_health", _health)
        resp = await smr.shell_notifications()
        body = json.loads(resp.body)
        assert body["count"] == 1
        assert body["items"][0]["id"] == "mlx-oom-imminent"
        assert body["items"][0]["level"] == "error"
        assert body["degraded"] is False

    async def test_loaded_models_produce_info_items(self, monkeypatch):
        async def _health(_auth=True):
            return {"oom_risk": "none"}

        models = [{"id": "m1", "loaded": True}, {"id": "m2", "loaded": False}]
        monkeypatch.setattr(smr, "_engine_models", lambda: _async(models))
        monkeypatch.setattr("fusion_mlx.routes_internal.health.v1_health", _health)
        resp = await smr.shell_notifications()
        body = json.loads(resp.body)
        assert body["count"] == 1
        assert body["items"][0]["id"] == "mlx-model-loaded:m1"

    async def test_health_failure_degrades_flag(self, monkeypatch):
        async def _boom(_auth=True):
            raise RuntimeError("health down")

        monkeypatch.setattr("fusion_mlx.routes_internal.health.v1_health", _boom)
        monkeypatch.setattr(smr, "_engine_models", lambda: _async([]))
        resp = await smr.shell_notifications()
        body = json.loads(resp.body)
        assert body["count"] == 0
        assert body["degraded"] is True


class TestManifestDriftContract:
    def test_manifest_upstream_matches_engine(self):
        # #1489: the self-registered upstream must equal the gateway static
        # upstream (engine :11434) or the gateway drift guard warns on every
        # heartbeat. The web bridge :11471 is opt-in and must never be
        # declared as the module upstream.
        manifest = json.loads(
            (REPO_ROOT / "web" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["upstream"].rstrip("/") == "http://127.0.0.1:11434"
        assert manifest["id"] == "fusion-mlx"


async def _async(value):
    return value
