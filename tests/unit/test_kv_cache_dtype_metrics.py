# SPDX-License-Identifier: Apache-2.0
"""kv_cache_dtype observability on the JSON /metrics endpoint.

fusion surfaces the effective KV cache dtype via
``ServerMetrics.to_dict()["kv_cache_dtype"]`` (fusion-native JSON — NOT
rapid-mlx's Prometheus-text gauge). The resolver reads the
ServerConfig.kv_cache_dtype stash (set by cli_serve after the safelist
resolves --kv-cache-dtype / legacy --kv-cache-quantization), with a
legacy-bits fallback for programmatic callers, defaulting to bf16.
"""

from __future__ import annotations

from types import SimpleNamespace

from fusion_mlx.server_metrics import ServerMetrics, _resolve_kv_cache_dtype


def _cfg(*, kv_cache_dtype=None, quant=False, bits=8):
    scheduler = SimpleNamespace(
        kv_cache_quantization=quant,
        kv_cache_quantization_bits=bits,
    )
    return SimpleNamespace(kv_cache_dtype=kv_cache_dtype, scheduler=scheduler)


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

    def test_stash_bf16_no_legacy_signal(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype="bf16", quant=False),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_no_stash_defaults_to_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, quant=False),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_unknown_stash_falls_back_to_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config", lambda: _cfg(kv_cache_dtype="fp8")
        )
        assert _resolve_kv_cache_dtype() == "bf16"


class TestResolveLegacyFallback:
    def test_legacy_derives_int4_from_bits(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, quant=True, bits=4),
        )
        assert _resolve_kv_cache_dtype() == "int4"

    def test_legacy_derives_int8_from_bits(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, quant=True, bits=8),
        )
        assert _resolve_kv_cache_dtype() == "int8"

    def test_stash_wins_over_legacy(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype="int8", quant=True, bits=4),
        )
        assert _resolve_kv_cache_dtype() == "int8"

    def test_legacy_skipped_when_quantization_false(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, quant=False, bits=4),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_legacy_unknown_bits_leaves_bf16(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype=None, quant=True, bits=3),
        )
        assert _resolve_kv_cache_dtype() == "bf16"

    def test_legacy_fires_when_stash_is_bf16_default(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_mlx.config.get_config",
            lambda: _cfg(kv_cache_dtype="bf16", quant=True, bits=4),
        )
        assert _resolve_kv_cache_dtype() == "int4"

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
