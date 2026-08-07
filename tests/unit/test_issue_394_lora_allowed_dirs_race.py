# Tests for #394: FUSION_LORA_ALLOWED_DIRS startup-order race.
# EnginePool caches _allowed_adapter_dirs at init time, but server.py's
# auto-add of ~/.fusion-mlx/adapters runs AFTER pool init (and joins dirs
# with ":"). Two fixes under test:
#   1. _resolve_allowed_adapter_dirs accepts both "," and ":" separators
#      (server.py joins with ":").
#   2. _validate_adapter_path re-resolves from env each call so a dir
#      auto-added to the env post-init takes effect without a restart.
#
# Importers/callers: EnginePool._validate_adapter_path is called by
# get_engine(adapter_path=...) and per-request /v1/chat/completions
# `adapters` field, plus serve_adapter. Affected API: LoRA adapter path
# validation. Schema: env var FUSION_LORA_ALLOWED_DIRS (comma/colon list).

import os

import pytest

from fusion_mlx.pool.engine_pool import AdapterPathError, EnginePool


@pytest.fixture
def _clean_env(monkeypatch):
    monkeypatch.delenv("FUSION_LORA_ALLOWED_DIRS", raising=False)
    yield


def test_resolve_accepts_colon_separator(_clean_env, monkeypatch):
    monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", "/tmp/a:/tmp/b")
    dirs = EnginePool._resolve_allowed_adapter_dirs()
    assert dirs == [os.path.realpath("/tmp/a"), os.path.realpath("/tmp/b")]


def test_resolve_accepts_comma_separator(_clean_env, monkeypatch):
    monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", "/tmp/a,/tmp/b")
    dirs = EnginePool._resolve_allowed_adapter_dirs()
    assert dirs == [os.path.realpath("/tmp/a"), os.path.realpath("/tmp/b")]


def test_resolve_accepts_mixed_separator(_clean_env, monkeypatch):
    monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", "/tmp/a:/tmp/b,/tmp/c")
    dirs = EnginePool._resolve_allowed_adapter_dirs()
    assert len(dirs) == 3


def test_resolve_empty_env_returns_empty(_clean_env):
    assert EnginePool._resolve_allowed_adapter_dirs() == []


def test_validate_re_resolves_env_after_init(_clean_env, monkeypatch, tmp_path):
    # Simulate the #394 race: pool init captures empty env, then server.py
    # auto-adds the adapters dir to the env. _validate_adapter_path must
    # pick up the post-init env value without needing a pool re-init.
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    adapter_file = adapter_dir / "my-adapter"
    adapter_file.mkdir()

    pool = EnginePool.__new__(EnginePool)
    # init-time cache: env was empty -> []
    monkeypatch.delenv("FUSION_LORA_ALLOWED_DIRS", raising=False)
    pool._allowed_adapter_dirs = EnginePool._resolve_allowed_adapter_dirs()
    assert pool._allowed_adapter_dirs == []

    # Now server.py auto-adds the dir to env (post-init), joining with ":"
    monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", str(adapter_dir))

    resolved = pool._validate_adapter_path(str(adapter_file))
    assert resolved == os.path.realpath(str(adapter_file))


def test_validate_rejects_when_env_still_empty(_clean_env, monkeypatch, tmp_path):
    pool = EnginePool.__new__(EnginePool)
    monkeypatch.delenv("FUSION_LORA_ALLOWED_DIRS", raising=False)
    pool._allowed_adapter_dirs = []

    with pytest.raises(AdapterPathError):
        pool._validate_adapter_path(str(tmp_path / "x"))


def test_validate_rejects_path_outside_allowed(_clean_env, monkeypatch, tmp_path):
    pool = EnginePool.__new__(EnginePool)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("FUSION_LORA_ALLOWED_DIRS", str(allowed))
    pool._allowed_adapter_dirs = []

    with pytest.raises(AdapterPathError):
        pool._validate_adapter_path(str(outside))
