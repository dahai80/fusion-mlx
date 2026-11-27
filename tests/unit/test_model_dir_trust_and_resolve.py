
from fusion_mlx import server as server_mod
from fusion_mlx.config import ServerConfig
from fusion_mlx.pool.engine_pool import EnginePool


class _StubPool:
    def __init__(self, models):
        self._models = list(models)

    def list_models(self):
        return list(self._models)


def test_resolve_model_id_keeps_discovered_directory_name(monkeypatch):
    # model-dir discovery registers entries under directory names; the
    # dynamic alias resolver rewrites the request to the HF repo id from
    # the model config. When the alias target is not a discovered entry
    # but the raw id is, the raw id must win (multi-model pool 404 fix).
    monkeypatch.setitem(
        server_mod._server_state,
        "engine_pool",
        _StubPool(["kimi-linear-48b-a3b-4bit"]),
    )
    monkeypatch.setattr(
        "fusion_mlx.model_aliases.resolve_model",
        lambda mid: "mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit",
    )
    assert (
        server_mod.resolve_model_id("kimi-linear-48b-a3b-4bit")
        == "kimi-linear-48b-a3b-4bit"
    )


def test_resolve_model_id_keeps_alias_when_discovered(monkeypatch):
    # If the pool actually discovered the alias spelling (HF-cache layout),
    # alias resolution must keep working.
    monkeypatch.setitem(
        server_mod._server_state,
        "engine_pool",
        _StubPool(["mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit"]),
    )
    monkeypatch.setattr(
        "fusion_mlx.model_aliases.resolve_model",
        lambda mid: "mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit",
    )
    assert (
        server_mod.resolve_model_id("kimi-linear-48b-a3b-4bit")
        == "mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit"
    )


def test_resolve_model_id_without_pool_keeps_alias(monkeypatch):
    # No pool wired (unit context): existing alias behavior unchanged.
    monkeypatch.setitem(server_mod._server_state, "engine_pool", None)
    monkeypatch.setattr(
        "fusion_mlx.model_aliases.resolve_model",
        lambda mid: "mlx-community/Qwen3.8-27B-8bit",
    )
    assert (
        server_mod.resolve_model_id("qwen3.8-27b-8bit")
        == "mlx-community/Qwen3.8-27B-8bit"
    )


def test_server_config_has_trust_remote_code():
    cfg = ServerConfig(model_dir="/tmp/x", trust_remote_code=True)
    assert cfg.trust_remote_code is True
    assert ServerConfig().trust_remote_code is False


def test_engine_pool_accepts_default_trust_remote_code():
    pool = EnginePool(default_trust_remote_code=True)
    assert pool._default_trust_remote_code is True
    pool2 = EnginePool()
    assert pool2._default_trust_remote_code is False
