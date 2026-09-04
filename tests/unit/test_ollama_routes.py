# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Ollama-compatible API routes."""

from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import ollama_routes
from fusion_mlx.api.ollama_routes import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaGenerateRequest,
    _build_openai_messages_chat,
    _build_openai_messages_generate,
    _options_to_params,
)
from fusion_mlx.middleware.auth import verify_api_key


@dataclass
class _FakeEntry:
    model_id: str = "test-model"
    model_path: str = "/tmp/__ollama_test_models/test-model"
    model_type: str = "llm"
    estimated_size: int = 1000
    actual_size: int | None = None
    last_observed_size: int | None = 1200
    config_model_type: str = "llama"
    engine: object | None = None


class _FakePool:
    def __init__(self, entries: dict[str, _FakeEntry] | None = None):
        self._entries = entries or {"test-model": _FakeEntry()}

    def list_models(self):
        return list(self._entries.keys())

    def get_entry(self, mid):
        return self._entries.get(mid)

    def get_loaded_model_ids(self):
        return [k for k, v in self._entries.items() if v.engine is not None]

    @property
    def loaded_model_count(self):
        return len(self.get_loaded_model_ids())


@pytest.fixture
def app_client(monkeypatch):
    app = FastAPI()

    async def _fake_auth():
        return True

    app.dependency_overrides[verify_api_key] = _fake_auth
    app.include_router(ollama_routes.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_pool():
    orig = ollama_routes._pool
    yield
    ollama_routes._pool = orig


class TestOptionsToParams:
    def test_empty_options(self):
        assert _options_to_params(None) == {}
        assert _options_to_params({}) == {}

    def test_temperature_mapping(self):
        assert _options_to_params({"temperature": 0.5}) == {"temperature": 0.5}

    def test_num_predict_to_max_tokens(self):
        assert _options_to_params({"num_predict": 512}) == {"max_tokens": 512}

    def test_repeat_penalty_mapping(self):
        assert _options_to_params({"repeat_penalty": 1.2}) == {
            "repetition_penalty": 1.2
        }

    def test_unknown_keys_ignored(self):
        assert _options_to_params({"unknown_key": 42}) == {}

    def test_none_values_skipped(self):
        assert _options_to_params({"temperature": None}) == {}


class TestBuildOpenAIMessagesGenerate:
    def test_prompt_only(self):
        msgs = _build_openai_messages_generate("hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_prompt_with_system(self):
        msgs = _build_openai_messages_generate("hello", system="you are helpful")
        assert msgs == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]


class TestBuildOpenAIMessagesChat:
    def test_simple_messages(self):
        msgs = [
            OllamaChatMessage(role="user", content="hi"),
            OllamaChatMessage(role="assistant", content="hello"),
        ]
        result = _build_openai_messages_chat(msgs)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_images_generate_content_parts(self):
        msgs = [OllamaChatMessage(role="user", content="describe", images=["abc123"])]
        result = _build_openai_messages_chat(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"


class TestPydanticModels:
    def test_generate_request_defaults(self):
        req = OllamaGenerateRequest()
        assert req.model == "default"
        assert req.prompt == ""
        assert req.stream is True

    def test_chat_request_defaults(self):
        req = OllamaChatRequest(messages=[OllamaChatMessage(role="user", content="hi")])
        assert req.model == "default"
        assert req.stream is True

    def test_chat_message_defaults(self):
        msg = OllamaChatMessage()
        assert msg.role == "user"
        assert msg.content == ""


class TestShowEndpoint:
    def test_show_returns_model_details(self, app_client, monkeypatch):
        monkeypatch.setattr(
            ollama_routes, "_pool", _FakePool({"test-model": _FakeEntry()})
        )
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: ("test-model", {}),
        )
        r = app_client.post("/api/show", json={"name": "test-model"})
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "test-model"
        assert body["details"]["family"] == "llama"
        assert body["size"] == 1200

    def test_show_404_unknown(self, app_client, monkeypatch):
        monkeypatch.setattr(ollama_routes, "_pool", _FakePool({}))
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: (m, {}),
        )
        r = app_client.post("/api/show", json={"name": "nope"})
        assert r.status_code == 404


class TestPsEndpoint:
    def test_ps_lists_loaded(self, app_client, monkeypatch):
        entry = _FakeEntry(engine=object())
        monkeypatch.setattr(
            ollama_routes, "_pool", _FakePool({"loaded-model": entry})
        )
        r = app_client.get("/api/ps")
        assert r.status_code == 200
        models = r.json()["models"]
        assert len(models) == 1
        assert models[0]["name"] == "loaded-model"

    def test_ps_empty_when_pool_none(self, app_client, monkeypatch):
        monkeypatch.setattr(ollama_routes, "_pool", None)
        r = app_client.get("/api/ps")
        assert r.status_code == 200
        assert r.json()["models"] == []


class TestPullEndpoint:
    def test_pull_nostream_returns_success(self, app_client, monkeypatch):
        monkeypatch.setattr(ollama_routes, "_pool", _FakePool({}))
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: (m, {}),
        )
        r = app_client.post("/api/pull", json={"name": "new-model", "stream": False})
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_pull_stream_emits_success(self, app_client, monkeypatch):
        monkeypatch.setattr(ollama_routes, "_pool", _FakePool({}))
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: (m, {}),
        )
        with app_client.stream(
            "POST", "/api/pull", json={"name": "new-model", "stream": True}
        ) as resp:
            assert resp.status_code == 200
            lines = [
                ln for ln in resp.iter_lines() if ln and ln.startswith("{")
            ]
        assert any('"status": "success"' in ln for ln in lines)


class TestDeleteEndpoint:
    def test_delete_404_unknown(self, app_client, monkeypatch, tmp_path):
        monkeypatch.setattr(ollama_routes, "_pool", _FakePool({}))
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: (m, {}),
        )
        r = app_client.delete("/api/delete?name=nope")
        assert r.status_code == 404

    def test_delete_409_when_loaded(self, app_client, monkeypatch, tmp_path):
        entry = _FakeEntry(
            engine=object(), model_path=str(tmp_path / "test-model")
        )
        monkeypatch.setattr(
            ollama_routes, "_pool", _FakePool({"test-model": entry})
        )
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: ("test-model", {}),
        )
        r = app_client.delete("/api/delete?name=test-model")
        assert r.status_code == 409

    def test_delete_removes_dir(self, app_client, monkeypatch, tmp_path):
        models_root = tmp_path / "models"
        model_dir = models_root / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        entry = _FakeEntry(model_path=str(model_dir))
        monkeypatch.setattr(
            ollama_routes, "_pool", _FakePool({"test-model": entry})
        )
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: ("test-model", {}),
        )
        monkeypatch.setattr(ollama_routes, "_models_root", lambda: models_root)
        r = app_client.delete("/api/delete?name=test-model")
        assert r.status_code == 200
        assert not model_dir.exists()


class TestCopyEndpoint:
    def test_copy_creates_symlink(self, app_client, monkeypatch, tmp_path):
        models_root = tmp_path / "models"
        src = models_root / "src-model"
        src.mkdir(parents=True)
        (src / "config.json").write_text("{}")
        entry = _FakeEntry(model_path=str(src))
        monkeypatch.setattr(
            ollama_routes, "_pool", _FakePool({"src-model": entry})
        )
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: ("src-model", {}),
        )
        monkeypatch.setattr(ollama_routes, "_models_root", lambda: models_root)
        r = app_client.post(
            "/api/copy", json={"source": "src-model", "destination": "alias-model"}
        )
        assert r.status_code == 200
        assert (models_root / "alias-model").is_symlink()

    def test_copy_rejects_path_in_destination(self, app_client, monkeypatch):
        monkeypatch.setattr(ollama_routes, "_pool", _FakePool({}))
        r = app_client.post(
            "/api/copy",
            json={"source": "a", "destination": "x/y"},
        )
        assert r.status_code == 400

    def test_copy_409_when_dest_exists(
        self, app_client, monkeypatch, tmp_path
    ):
        models_root = tmp_path / "models"
        src = models_root / "src-model"
        dest = models_root / "alias-model"
        src.mkdir(parents=True)
        dest.mkdir(parents=True)
        entry = _FakeEntry(model_path=str(src))
        monkeypatch.setattr(
            ollama_routes, "_pool", _FakePool({"src-model": entry})
        )
        monkeypatch.setattr(
            "fusion_mlx.server.resolve_model_with_profile",
            lambda m: ("src-model", {}),
        )
        monkeypatch.setattr(ollama_routes, "_models_root", lambda: models_root)
        r = app_client.post(
            "/api/copy", json={"source": "src-model", "destination": "alias-model"}
        )
        assert r.status_code == 409
