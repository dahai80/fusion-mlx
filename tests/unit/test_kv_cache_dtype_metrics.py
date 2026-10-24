# SPDX-License-Identifier: Apache-2.0
"""kv_cache_dtype observability on the JSON /metrics endpoint.

fusion surfaces the effective KV cache dtype via
``ServerMetrics.to_dict()["kv_cache_dtype"]`` (fusion-native JSON — NOT
rapid-mlx's Prometheus-text gauge). The resolver reads the
ServerConfig.kv_cache_dtype stash (set by cli_serve after the safelist
resolves --kv-cache-dtype), then SchedulerConfig.kv_cache_dtype for
programmatic callers, defaulting to bf16.
"""

from __future__ import annotations

from types import SimpleNamespace

from fusion_mlx.server_metrics import ServerMetrics, _resolve_kv_cache_dtype


def _cfg(*, kv_cache_dtype=None, scheduler_dtype=None):
    return SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        scheduler=SimpleNamespace(kv_cache_dtype=scheduler_dtype),
    )


class TestResolveStash:
    def test_stash_int4(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config", lambda: _cfg(kv_cache_dtype="int4")
        )
        assert _resolve_kv_cache_dtype() == "int4"

    def test_stash_int8(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config", lambda: _cfg(kv_cache_dtype="int8")
        )
        assert _resolve_kv_cache_dtype() == "int8"

    def test_stash_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype="bf16"),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_no_stash_defaults_to_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_unknown_stash_falls_back_to_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config", lambda: _cfg(kv_cache_dtype="fp8")
        )
        assert _resolve_kv_cache_dtype() == "bf16"


class TestResolveSchedulerFallback:
    def test_scheduler_dtype_used_when_stash_is_default(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, scheduler_dtype="int4"),
        )
        assert _resolve_kv_cache_dtype() == "int4"

    def test_scheduler_dtype_used_when_stash_is_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype="bf16", scheduler_dtype="int8"),
        )
        assert _resolve_kv_cache_dtype() == "int8"

    def test_stash_wins_over_scheduler(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype="int8", scheduler_dtype="int4"),
        )
        assert _resolve_kv_cache_dtype() == "int8"

    def test_scheduler_bf16_keeps_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, scheduler_dtype="bf16"),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_unknown_scheduler_dtype_falls_back_to_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, scheduler_dtype="fp8"),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_resolver_swallows_config_error_defaults_bf16(self, monkeypatch):
        def _boom():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("fusion_mlx.config.get_config", _boom)
        assert _resolve_kv_cache_dtype() == "bf16"


class TestToDictSurface:
    def test_to_dict_includes_kv_cache_dtype(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config", lambda: _cfg(kv_cache_dtype="int4")
        )
        d = ServerMetrics().to_dict()
        assert "kv_cache_dtype" in d
        assert d["kv_cache_dtype"] == "int4"

    def test_to_dict_matches_resolver(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, quant=True, bits=8),
        )
        assert ServerMetrics().to_dict()["kv_cache_dtype"] == _resolve_kv_cache_dtype()
