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
)
from fusion_mlx.turboquant_kv import (
    TurboQuantConfig,
    TurboQuantKVCache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config validation (K8V4-specific)
# ---------------------------------------------------------------------------


class TestK8V4ConfigValidation:
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

    def test_default_mode_is_v4(self):
        cfg = TurboQuantConfig()
        assert cfg.mode == "v4"
        assert cfg.k_bits is None


# ---------------------------------------------------------------------------
# K8V4 TurboQuantKVCache via from_cache
# ---------------------------------------------------------------------------


def _make_real_kv(head_dim=128, seq_len=32, n_heads=8):
    from mlx_lm.models.cache import KVCache

    rng = np.random.RandomState(0)
    kv = KVCache()
    kv.keys = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    kv.values = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    kv.offset = seq_len
    return kv


class TestK8V4Cache:
    def test_k8v4_from_cache(self):
        kv = _make_real_kv(head_dim=128)
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        assert tq.offset == 32
        assert tq.bits == 4.0
        assert tq.keys is not None
        assert tq.values is not None

    def test_k8v4_dequantize_quality(self):
        kv = _make_real_kv(head_dim=128)
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        k_deq, v_deq = tq.dequantize()

        orig_k = np.array(kv.keys, dtype=np.float32).reshape(-1, 128)
        recon_k = np.array(k_deq, dtype=np.float32).reshape(-1, 128)
        k_cos = np.sum(orig_k * recon_k, axis=-1) / (
            np.linalg.norm(orig_k, axis=-1) * np.linalg.norm(recon_k, axis=-1) + 1e-8
        )
        assert k_cos.mean() > 0.93, f"K cosine {k_cos.mean():.4f} < 0.93"

        orig_v = np.array(kv.values, dtype=np.float32).reshape(-1, 128)
        recon_v = np.array(v_deq, dtype=np.float32).reshape(-1, 128)
        v_cos = np.sum(orig_v * recon_v, axis=-1) / (
            np.linalg.norm(orig_v, axis=-1) * np.linalg.norm(recon_v, axis=-1) + 1e-8
        )
        assert v_cos.mean() > 0.93, f"V cosine {v_cos.mean():.4f} < 0.93"

    def test_k8v4_trim(self):
        kv = _make_real_kv(head_dim=128)
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        tq.trim(10)
        assert tq.offset == 22

    def test_k8v4_nbytes(self):
        kv = _make_real_kv(head_dim=128)
        tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
        assert tq.nbytes > 0
        fp16_total = kv.keys.nbytes + kv.values.nbytes
        assert tq.nbytes < fp16_total


# ---------------------------------------------------------------------------
# Skip-list (K8V4-specific patterns)
# ---------------------------------------------------------------------------


class TestK8V4SkipList:
    @pytest.mark.parametrize(
        "model_name,expected_reason",
        [
            ("gemma-3-27b-it", SKIP_REASON_SLIDING),
            ("mlx-community/gemma3-9b", SKIP_REASON_SLIDING),
            ("openai/gpt-oss-120b", SKIP_REASON_SLIDING),
            ("deepseek-ai/deepseek-v3", SKIP_REASON_MLA),
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
            "",
        ],
    )
    def test_compatible_models_pass(self, model_name):
        skip, reason = is_incompatible_with_turboquant(model_name=model_name)
        assert skip is False
        assert reason is None

    def test_skip_registry_has_documented_patterns(self):
        keys = list(MODELS_INCOMPATIBLE_WITH_TURBOQUANT.keys())
        joined = " ".join(keys).lower()
        for needle in ("gemma", "gpt", "deepseek", "kimi"):
            assert needle in joined, f"family {needle!r} missing from skip registry"


# ---------------------------------------------------------------------------
# CLI flag — v4 / k8v4 mutual exclusion
# ---------------------------------------------------------------------------


import argparse


def _build_minimal_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--kv-cache-turboquant",
        nargs="?",
        const="v4",
        default=None,
        choices=["v4", "k8v4", "none"],
    )
    parser.add_argument("--kv-cache-quantization", action="store_true", default=False)
    return parser


class TestCLIFlag:
    def test_bare_flag_defaults_to_v4(self):
        ns = _build_minimal_parser().parse_args(["--kv-cache-turboquant"])
        assert ns.kv_cache_turboquant == "v4"

    def test_explicit_v4_value(self):
        ns = _build_minimal_parser().parse_args(["--kv-cache-turboquant", "v4"])
        assert ns.kv_cache_turboquant == "v4"

    def test_explicit_k8v4_value(self):
        ns = _build_minimal_parser().parse_args(["--kv-cache-turboquant", "k8v4"])
        assert ns.kv_cache_turboquant == "k8v4"

    def test_none_off_switch(self):
        ns = _build_minimal_parser().parse_args(["--kv-cache-turboquant", "none"])
        assert ns.kv_cache_turboquant == "none"

    def test_unknown_value_rejected(self):
        with pytest.raises(SystemExit):
            _build_minimal_parser().parse_args(["--kv-cache-turboquant", "bogus"])

    def test_off_when_unset(self):
        ns = _build_minimal_parser().parse_args([])
        assert ns.kv_cache_turboquant is None


# ---------------------------------------------------------------------------
# Codec dtype preservation
# ---------------------------------------------------------------------------


def test_codec_preserves_input_dtype():
    from mlx_lm.models.cache import KVCache

    seq, head_dim = 64, 128
    rng = np.random.RandomState(42)
    kv = KVCache()
    kv.keys = mx.array(rng.randn(1, 4, seq, head_dim).astype(np.float16))
    kv.values = mx.array(rng.randn(1, 4, seq, head_dim).astype(np.float16))
    kv.offset = seq

    tq = TurboQuantKVCache.from_cache(kv, bits=4.0, seed=0)
    k_deq, v_deq = tq.dequantize()
    # dequantize returns float32 (quantization math operates in f32)
    assert k_deq.dtype == mx.float32
    assert v_deq.dtype == mx.float32
