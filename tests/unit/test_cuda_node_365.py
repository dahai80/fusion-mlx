"""Unit tests for #365 Windows CUDA backend node + platform routing.

These run on Mac (no vLLM/CUDA) by stubbing the vLLM module in sys.modules.
Covers: platform detection, mDNS platform TXT record, CudaNodeConfig, and the
CUDA node's OpenAI routes (health / models / chat / completions).
"""

from __future__ import annotations

import sys
import types

import pytest

# --- platform detection -------------------------------------------------


def test_platform_enum_values():
    from fusion_mlx.cluster.platform import Platform

    assert Platform.MAC.value == "mac"
    assert Platform.WINDOWS_CUDA.value == "windows-cuda"
    assert str(Platform.WINDOWS_CUDA) == "windows-cuda"


def test_platform_env_override(monkeypatch):
    from fusion_mlx.cluster.platform import Platform, detect_platform

    monkeypatch.setenv("FUSION_PLATFORM", "windows-cuda")
    assert detect_platform() == Platform.WINDOWS_CUDA


def test_platform_env_override_mac(monkeypatch):
    from fusion_mlx.cluster.platform import detect_platform

    monkeypatch.setenv("FUSION_PLATFORM", "mac")
    assert str(detect_platform()) == "mac"


def test_platform_invalid_env_falls_back(monkeypatch):
    from fusion_mlx.cluster.platform import detect_platform

    monkeypatch.setenv("FUSION_PLATFORM", "totally-bogus")
    # Invalid value logs a warning and falls back to heuristics (mac on darwin).
    assert str(detect_platform()) in {"mac", "windows-cuda"}


# --- mDNS TXT platform record -------------------------------------------


def test_build_txt_records_includes_platform():
    from fusion_mlx.cluster.mdns import build_txt_records

    snapshot = {
        "node_id": "host:8000",
        "host": "host",
        "port": 8000,
        "platform": "windows-cuda",
        "models": [{"id": "Qwen/Qwen2.5-72B", "loaded": True}],
        "memory": {"available_percent": 50.0},
    }
    txt = build_txt_records(snapshot)
    assert txt["platform"] == "windows-cuda"
    assert txt["node_id"] == "host:8000"
    assert txt["models_csv"] == "Qwen/Qwen2.5-72B"


def test_build_txt_records_platform_defaults_mac():
    from fusion_mlx.cluster.mdns import build_txt_records

    txt = build_txt_records({"node_id": "h:1", "host": "h", "port": 1, "models": []})
    assert txt["platform"] == "mac"


# --- config field -------------------------------------------------------


def test_server_config_has_platform_field():
    from fusion_mlx.config import ServerConfig

    cfg = ServerConfig()
    assert cfg.platform is None  # auto-detect by default


# --- CUDA node app (vLLM stubbed) ---------------------------------------


@pytest.fixture
def stubbed_vllm(monkeypatch):
    """Inject a fake vllm module so create_cuda_app works without CUDA."""

    class _Out:
        def __init__(self, text, finish_reason):
            self.text = text
            self.finish_reason = finish_reason

    class _Output:
        def __init__(self):
            self.outputs = [_Out("hello from cuda", "stop")]

    class _Engine:
        def __init__(self, model):
            self.engine_args = types.SimpleNamespace(model=model)

        @classmethod
        def from_engine_args(cls, args):
            return cls(getattr(args, "model", "stub"))

        async def generate(self, prompt, sampling, request_id):
            yield _Output()

    class _SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    class _EngineArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    vllm = types.ModuleType("vllm")
    vllm.AsyncLLMEngine = _Engine
    engine_pkg = types.ModuleType("vllm.engine")
    engine_pkg.arg_utils = types.ModuleType("vllm.engine.arg_utils")
    engine_pkg.arg_utils.AsyncEngineArgs = _EngineArgs
    vllm.sampling_params = types.ModuleType("vllm.sampling_params")
    vllm.sampling_params.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", engine_pkg.arg_utils)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", vllm.sampling_params)
    return vllm


def test_create_cuda_app_routes(stubbed_vllm):
    from fastapi.testclient import TestClient

    from fusion_mlx.backends.cuda_node import CudaNodeConfig, create_cuda_app

    cfg = CudaNodeConfig(model="Qwen/Qwen2.5-72B-Instruct", cluster_advertise=False)
    app = create_cuda_app(cfg)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["platform"] == "windows-cuda"
        assert body["model"] == "Qwen/Qwen2.5-72B-Instruct"

        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "Qwen/Qwen2.5-72B-Instruct"

        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        )
        assert r.status_code == 200
        chat = r.json()
        assert chat["object"] == "chat.completion"
        assert chat["choices"][0]["message"]["content"] == "hello from cuda"

        r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 8})
        assert r.status_code == 200
        comp = r.json()
        assert comp["object"] == "text_completion"
        assert comp["choices"][0]["text"] == "hello from cuda"


def test_create_cuda_app_rejects_missing_messages(stubbed_vllm):
    from fastapi.testclient import TestClient

    from fusion_mlx.backends.cuda_node import CudaNodeConfig, create_cuda_app

    app = create_cuda_app(CudaNodeConfig(model="m", cluster_advertise=False, port=0))
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={})
        assert r.status_code == 400


def test_cuda_node_config_defaults():
    from fusion_mlx.backends.cuda_node import CudaNodeConfig

    cfg = CudaNodeConfig(model="m")
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.tensor_parallel_size == 1
    assert cfg.gpu_memory_utilization == 0.90
    assert cfg.cluster_advertise is True
    assert cfg.quantization is None


def test_import_vllm_raises_without_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    import importlib

    import fusion_mlx.backends.cuda_node.node as node_mod

    importlib.reload(node_mod)
    with pytest.raises(RuntimeError, match="vLLM"):
        node_mod._import_vllm()
