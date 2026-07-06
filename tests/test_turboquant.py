# SPDX-License-Identifier: Apache-2.0
import logging

import mlx.core as mx
import numpy as np
import pytest

from fusion_mlx.turboquant import (
    MODELS_INCOMPATIBLE_WITH_TURBOQUANT,
    SKIP_REASON_MLA,
    SKIP_REASON_SLIDING,
    TURBOQUANT_MODES,
    is_incompatible_with_turboquant,
    resolve_turboquant_mode_default,
)
from fusion_mlx.turboquant_kv import (
    TurboQuantConfig,
    TurboQuantKVCache,
    auto_select_bits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TurboQuantConfig
# ---------------------------------------------------------------------------


class TestTurboQuantConfig:
    def test_valid_3bit(self):
        cfg = TurboQuantConfig(bits=3)
        assert cfg.bits == 3

    def test_valid_4bit(self):
        cfg = TurboQuantConfig(bits=4)
        assert cfg.bits == 4

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be 3 or 4"):
            TurboQuantConfig(bits=2)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size must be >= 1"):
            TurboQuantConfig(group_size=0)

    def test_defaults(self):
        cfg = TurboQuantConfig()
        assert cfg.bits == 3
        assert cfg.group_size == 32
        assert cfg.rotation_seed == 42
        assert cfg.mode == "v4"

    def test_k8v4_requires_bits4(self):
        with pytest.raises(ValueError, match="k8v4.*requires bits=4"):
            TurboQuantConfig(mode="k8v4", bits=3)

    def test_k8v4_k_bits_is_8(self):
        cfg = TurboQuantConfig(mode="k8v4", bits=4)
        assert cfg.k_bits == 8

    def test_v4_k_bits_is_none(self):
        cfg = TurboQuantConfig(mode="v4", bits=4)
        assert cfg.k_bits is None

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            TurboQuantConfig(mode="k4v4")

    def test_modes_enum(self):
        assert TURBOQUANT_MODES == ("v4", "k8v4")


# ---------------------------------------------------------------------------
# auto_select_bits
# ---------------------------------------------------------------------------


class TestAutoSelectBits:
    def test_large_head_dim(self):
        assert auto_select_bits(128) == 3

    def test_medium_head_dim(self):
        assert auto_select_bits(96) == 3

    def test_small_head_dim(self):
        assert auto_select_bits(64) == 4

    def test_tiny_head_dim(self):
        assert auto_select_bits(32) == 4


# ---------------------------------------------------------------------------
# Skip-list registry
# ---------------------------------------------------------------------------


class TestSkipList:
    @pytest.mark.parametrize(
        "model_name,expected_reason",
        [
            ("gemma-3-27b-it", SKIP_REASON_SLIDING),
            ("mlx-community/gemma3-9b", SKIP_REASON_SLIDING),
            ("openai/gpt-oss-120b", SKIP_REASON_SLIDING),
            ("gpt_oss_20b", SKIP_REASON_SLIDING),
            ("deepseek-ai/deepseek-v3", SKIP_REASON_MLA),
            ("deepseek_v4_lite", SKIP_REASON_MLA),
            ("kimi-k2.5-flash", SKIP_REASON_MLA),
            ("Kimi-K2.6-Preview", SKIP_REASON_MLA),
        ],
    )
    def test_skip_by_name_pattern(self, model_name, expected_reason):
        skip, reason = is_incompatible_with_turboquant(model_name=model_name)
        assert skip is True
        assert reason == expected_reason

    @pytest.mark.parametrize(
        "model_name",
        [
            "mlx-community/Qwen3-32B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "",
        ],
    )
    def test_compatible_models_pass(self, model_name):
        skip, reason = is_incompatible_with_turboquant(model_name=model_name)
        assert skip is False
        assert reason is None

    def test_skip_by_hf_config_sliding_window(self):
        skip, reason = is_incompatible_with_turboquant(
            model_name="future-model",
            hf_config={"sliding_window": 4096},
        )
        assert skip is True
        assert reason == SKIP_REASON_SLIDING

    def test_skip_by_alias_metadata_mla(self):
        skip, reason = is_incompatible_with_turboquant(
            model_name="generic", alias_metadata={"is_mla": True}
        )
        assert skip is True
        assert reason == SKIP_REASON_MLA

    def test_skip_by_model_type_deepseek_v3(self):
        skip, reason = is_incompatible_with_turboquant(
            model_name="generic", hf_config={"model_type": "deepseek_v3"}
        )
        assert skip is True
        assert reason == SKIP_REASON_MLA

    def test_skip_registry_has_documented_patterns(self):
        keys = list(MODELS_INCOMPATIBLE_WITH_TURBOQUANT.keys())
        joined = " ".join(keys).lower()
        for needle in ("gemma", "gpt", "deepseek", "kimi"):
            assert needle in joined, f"family {needle!r} missing from skip registry"


# ---------------------------------------------------------------------------
# TurboQuantKVCache — from_cache / dequantize roundtrip
# ---------------------------------------------------------------------------


def _make_real_kv(head_dim=128, seq_len=32, n_heads=8):
    from mlx_lm.models.cache import KVCache

    rng = np.random.RandomState(0)
    kv = KVCache()
    kv.keys = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    kv.values = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    kv.offset = seq_len
    return kv


class TestTurboQuantKVCache:
    def test_from_cache(self):
        kv = _make_real_kv()
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        assert tq.offset == 32
        assert tq.bits == 4.0
        assert tq.keys is not None
        assert tq.values is not None

    def test_dequantize_roundtrip_quality(self):
        kv = _make_real_kv()
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        k_deq, v_deq = tq.dequantize()

        orig_v = np.array(kv.values, dtype=np.float32).reshape(-1, 128)
        recon_v = np.array(v_deq, dtype=np.float32).reshape(-1, 128)
        cosines = np.sum(orig_v * recon_v, axis=-1) / (
            np.linalg.norm(orig_v, axis=-1) * np.linalg.norm(recon_v, axis=-1) + 1e-8
        )
        mean_cos = cosines.mean()
        assert mean_cos > 0.93, f"V cosine {mean_cos:.4f} < 0.93"

    def test_nbytes_positive(self):
        kv = _make_real_kv()
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        assert tq.nbytes > 0

    def test_nbytes_less_than_fp16(self):
        kv = _make_real_kv()
        fp16_total = kv.keys.nbytes + kv.values.nbytes
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        assert tq.nbytes < fp16_total

    def test_is_trimmable(self):
        kv = _make_real_kv()
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        assert tq.is_trimmable()

    def test_trim(self):
        kv = _make_real_kv()
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        tq.trim(10)
        assert tq.offset == 22

    def test_trim_all(self):
        kv = _make_real_kv()
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        tq.trim(100)
        assert tq.offset == 0

    def test_3bit_from_cache(self):
        kv = _make_real_kv()
        tq = TurboQuantKVCache.from_cache(kv, bits=3.0, seed=0)
        k_deq, v_deq = tq.dequantize()
        assert k_deq.shape == kv.keys.shape
        assert v_deq.shape == kv.values.shape

    def test_empty_cache(self):
        from mlx_lm.models.cache import KVCache

        kv = KVCache()
        kv.keys = None
        kv.values = None
        kv.offset = 0
        tq = TurboQuantKVCache(bits=4.0, seed=0)
        assert tq.keys is None
        assert tq.offset == 0


# ---------------------------------------------------------------------------
# Memory cache integration
# ---------------------------------------------------------------------------


class TestMemoryCacheIntegration:
    def _make_cache_list(self, n_layers=4, seq_len=32, n_heads=8, head_dim=128):
        from mlx_lm.models.cache import KVCache

        cache = []
        np.random.seed(0)
        for _ in range(n_layers):
            kv = KVCache()
            kv.keys = mx.array(
                np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
            )
            kv.values = mx.array(
                np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
            )
            kv.offset = seq_len
            cache.append(kv)
        return cache

    def test_compress_decompress_roundtrip(self):
        from fusion_mlx.memory_cache import (
            _turboquant_compress_cache,
            _turboquant_decompress_cache,
        )

        cache = self._make_cache_list()
        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)

        for layer in compressed:
            assert isinstance(layer, TurboQuantKVCache)

        decompressed = _turboquant_decompress_cache(compressed)

        for layer in decompressed:
            assert layer.keys is not None
            assert layer.values is not None

    def test_none_layers_passthrough(self):
        from fusion_mlx.memory_cache import (
            _turboquant_compress_cache,
            _turboquant_decompress_cache,
        )

        cache = [None, None]
        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)
        assert compressed == [None, None]

        decompressed = _turboquant_decompress_cache(compressed)
        assert decompressed == [None, None]

    def test_mixed_layers(self):
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache

        from fusion_mlx.memory_cache import _turboquant_compress_cache

        kv = KVCache()
        np.random.seed(0)
        kv.keys = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float16))
        kv.values = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float16))
        kv.offset = 32

        mamba = MagicMock()

        cache = [kv, mamba, None]
        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)

        assert isinstance(compressed[0], TurboQuantKVCache)
        assert compressed[1] is mamba
        assert compressed[2] is None


# ---------------------------------------------------------------------------
# resolve_turboquant_mode_default
# ---------------------------------------------------------------------------


class TestResolveTurboquantModeDefault:
    @staticmethod
    def _args(turboquant=None, quantization=False):
        from types import SimpleNamespace

        return SimpleNamespace(
            kv_cache_turboquant=turboquant,
            kv_cache_quantization=quantization,
        )

    def test_explicit_v4_overrides_default(self):
        assert (
            resolve_turboquant_mode_default(
                self._args(turboquant="v4"), model_name="qwen3.5-35b-8bit"
            )
            == "v4"
        )

    def test_explicit_k8v4(self):
        assert (
            resolve_turboquant_mode_default(
                self._args(turboquant="k8v4"), model_name="qwen3.5-35b-8bit"
            )
            == "k8v4"
        )

    def test_none_off_switch(self):
        assert (
            resolve_turboquant_mode_default(
                self._args(turboquant="none"), model_name="qwen3.5-35b-8bit"
            )
            is None
        )

    def test_legacy_quantization_suppresses_autoflip(self):
        assert (
            resolve_turboquant_mode_default(
                self._args(quantization=True),
                model_name="qwen3.5-35b-8bit",
            )
            is None
        )

    def test_unknown_model_returns_none(self):
        result = resolve_turboquant_mode_default(
            self._args(), model_name="some-unknown-model"
        )
        assert result is None or result in ("v4", "k8v4")
