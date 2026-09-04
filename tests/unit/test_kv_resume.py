# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disconnect KV two-end sync (kv_resume)."""

from __future__ import annotations

import os

import pytest

from fusion_mlx.service import kv_resume


class _FakeKVLayer:
    pass


def _fake_cache(n_layers: int = 2):
    return [_FakeKVLayer() for _ in range(n_layers)]


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_KV_CHECKPOINT_INTERVAL", "256")
    yield
    monkeypatch.delenv("FUSION_MLX_KV_CHECKPOINT_INTERVAL", raising=False)


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.delenv("FUSION_MLX_KV_CHECKPOINT_INTERVAL", raising=False)


@pytest.fixture
def temp_root(monkeypatch, tmp_path):
    root = str(tmp_path / "kv_checkpoints")
    monkeypatch.setattr(
        "fusion_mlx.runtime.disk_kv_checkpoint.get_default_root", lambda: root
    )
    return root


class TestInterval:
    def test_disabled_when_unset(self, disabled):
        assert kv_resume._interval() == 0

    def test_disabled_when_zero(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_KV_CHECKPOINT_INTERVAL", "0")
        assert kv_resume._interval() == 0

    def test_enabled_positive(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_KV_CHECKPOINT_INTERVAL", "128")
        assert kv_resume._interval() == 128

    def test_bad_value_disables(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_KV_CHECKPOINT_INTERVAL", "notanint")
        assert kv_resume._interval() == 0


class TestPersistDisabled:
    def test_noop_when_interval_zero(self, disabled):
        path = kv_resume.persist_request_kv(
            "req-1", _fake_cache(), 100, model_name="m"
        )
        assert path is None

    def test_noop_when_no_cache(self, enabled, temp_root):
        path = kv_resume.persist_request_kv(
            "req-1", None, 100, model_name="m"
        )
        assert path is None

    def test_noop_when_zero_tokens(self, enabled, temp_root):
        path = kv_resume.persist_request_kv(
            "req-1", _fake_cache(), 0, model_name="m"
        )
        assert path is None


class TestPersistLoadCleanup:
    def test_persist_writes_checkpoint_and_load_roundtrips(
        self, enabled, temp_root, monkeypatch
    ):
        written: dict[str, list] = {}

        def _fake_write(cache, *, root, req_hash, token_offset, **kwargs):
            import os

            dst_dir = os.path.join(root, req_hash)
            os.makedirs(dst_dir, exist_ok=True)
            body = os.path.join(
                dst_dir, f"checkpoint-{token_offset}.safetensors"
            )
            meta = os.path.join(dst_dir, f"checkpoint-{token_offset}.json")
            import json

            with open(body, "wb"):
                pass
            with open(meta, "w") as fh:
                json.dump(
                    {"token_offset": token_offset, "kv_dtype": "bf16", **kwargs.get("extra_metadata", {})},
                    fh,
                )
            written["cache"] = list(cache)
            return body

        monkeypatch.setattr(
            "fusion_mlx.runtime.disk_kv_checkpoint.write_checkpoint",
            _fake_write,
        )
        monkeypatch.setattr(
            "fusion_mlx.runtime.disk_kv_checkpoint.model_requires_full_checkpoint",
            lambda m: False,
        )

        cache = _fake_cache(3)
        path = kv_resume.persist_request_kv(
            "req-42", cache, 512, model_name="llama"
        )
        assert path is not None
        assert os.path.isfile(path)
        assert written["cache"] == cache

        from fusion_mlx.runtime.disk_kv_checkpoint import LoadedCheckpoint

        def _fake_load(p):
            return LoadedCheckpoint(
                cache=_fake_cache(3),
                token_offset=512,
                kv_dtype="bf16",
                requires_full_checkpoint=False,
                metadata={"source": "disconnect_persist"},
                path=p,
            )

        monkeypatch.setattr(
            "fusion_mlx.runtime.disk_kv_checkpoint.load_checkpoint", _fake_load
        )
        loaded = kv_resume.load_resumable_kv("req-42", model_name="llama")
        assert loaded is not None
        assert loaded.token_offset == 512

    def test_load_returns_none_when_no_checkpoint(self, enabled, temp_root):
        loaded = kv_resume.load_resumable_kv("missing", model_name="m")
        assert loaded is None

    def test_load_returns_none_when_disabled(self, disabled, temp_root):
        loaded = kv_resume.load_resumable_kv("anything", model_name="m")
        assert loaded is None

    def test_cleanup_removes_checkpoint_dir(
        self, enabled, temp_root, monkeypatch
    ):
        from fusion_mlx.runtime import disk_kv_checkpoint

        req_hash = disk_kv_checkpoint.request_hash("req-7", "llama")
        dst_dir = os.path.join(temp_root, req_hash)
        os.makedirs(dst_dir, exist_ok=True)
        body = os.path.join(dst_dir, "checkpoint-300.safetensors")
        meta = os.path.join(dst_dir, "checkpoint-300.json")
        with open(body, "wb"):
            pass
        with open(meta, "w") as fh:
            fh.write("{}")
        assert os.path.isdir(dst_dir)

        kv_resume.cleanup_resumable_kv("req-7", model_name="llama")
        assert not os.path.isdir(dst_dir)

    def test_cleanup_noop_when_disabled(self, disabled, temp_root):
        kv_resume.cleanup_resumable_kv("req-7", model_name="llama")


class TestNeverRaises:
    def test_persist_swallows_write_failure(self, enabled, temp_root, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(
            "fusion_mlx.runtime.disk_kv_checkpoint.write_checkpoint", _boom
        )
        monkeypatch.setattr(
            "fusion_mlx.runtime.disk_kv_checkpoint.model_requires_full_checkpoint",
            lambda m: False,
        )
        path = kv_resume.persist_request_kv(
            "req-1", _fake_cache(), 100, model_name="m"
        )
        assert path is None

    def test_load_swallows_scan_failure(self, enabled, monkeypatch):
        def _boom(root):
            raise RuntimeError("scan io error")

        monkeypatch.setattr(
            "fusion_mlx.runtime.disk_kv_checkpoint.scan_checkpoints", _boom
        )
        loaded = kv_resume.load_resumable_kv("req-1", model_name="m")
        assert loaded is None

    def test_cleanup_swallows_failure(self, enabled, temp_root, monkeypatch):
        def _boom(root, req_hash):
            raise RuntimeError("cleanup io error")

        monkeypatch.setattr(
            "fusion_mlx.runtime.disk_kv_checkpoint.cleanup_request", _boom
        )
        kv_resume.cleanup_resumable_kv("req-1", model_name="m")
