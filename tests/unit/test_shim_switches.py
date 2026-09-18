# SPDX-License-Identifier: Apache-2.0
"""PR-P: degradation-switch full verification (v2 doc §7 L1 gate).

Every shim feature switch must default OFF, respect "0"/"1" literally,
and stay independent (enabling one never enables another). Default OFF
is the L1 guarantee: with a clean environment the whole shim layer is
inert and inference runs the native mlx_lm/mlx_vlm path.
"""

import importlib

import pytest

FEATURE_SWITCHES = [
    ("FUSION_SHIM_ENABLED", "fusion_mlx.shim", "is_shim_enabled"),
    ("FUSION_ENGINE_RUNNER", "fusion_mlx.shim.fast", "is_engine_runner_enabled"),
    (
        "FUSION_SHIM_FUSED_RMSNORM",
        "fusion_mlx.shim.fused_ops",
        "is_fused_rmsnorm_enabled",
    ),
    ("FUSION_SHIM_FUSED_ROPE", "fusion_mlx.shim.fused_ops", "is_fused_rope_enabled"),
    (
        "FUSION_SHIM_TWO_LEVEL_KV",
        "fusion_mlx.custom_kernels.paged_kv_cow",
        "is_two_level_kv_enabled",
    ),
    (
        "FUSION_SHIM_TREE_MASK",
        "fusion_mlx.speculative.tree_mask",
        "is_tree_mask_enabled",
    ),
    ("FUSION_SHIM_QUANT_KV", "fusion_mlx.shim.quant_kv", "is_quant_kv_enabled"),
    ("FUSION_SHIM_IQ", "fusion_mlx.shim.mixed_quant", "is_iq_enabled"),
    ("FUSION_SHIM_ASFW", "fusion_mlx.migrate.gguf_loader", "is_asfw_enabled"),
    (
        "FUSION_SHIM_GRAMMAR_RING",
        "fusion_mlx.shim.grammar_ring",
        "is_grammar_ring_enabled",
    ),
    ("FUSION_SHIM_MOE", "fusion_mlx.shim.moe_dispatch", "is_moe_dispatch_enabled"),
    ("FUSION_SHIM_SSM", "fusion_mlx.shim.ssm_scan", "is_ssm_scan_enabled"),
]


def _accessor(mod_name, fn_name):
    return getattr(importlib.import_module(mod_name), fn_name)


def _all_accessors():
    return [(env, _accessor(m, f)) for env, m, f in FEATURE_SWITCHES]


@pytest.mark.parametrize(
    "env,mod,fn",
    FEATURE_SWITCHES,
    ids=[env for env, _, _ in FEATURE_SWITCHES],
)
class TestDefaultOff:
    def test_default_off(self, monkeypatch, env, mod, fn):
        monkeypatch.delenv(env, raising=False)
        assert _accessor(mod, fn)() is False

    def test_set_to_one(self, monkeypatch, env, mod, fn):
        monkeypatch.setenv(env, "1")
        assert _accessor(mod, fn)() is True

    def test_set_to_zero(self, monkeypatch, env, mod, fn):
        monkeypatch.setenv(env, "0")
        assert _accessor(mod, fn)() is False


class TestIndependence:
    def test_enabling_one_leaves_others_off(self, monkeypatch):
        for env, _, _ in FEATURE_SWITCHES:
            monkeypatch.delenv(env, raising=False)
        accessors = _all_accessors()
        for env, _ in accessors:
            monkeypatch.setenv(env, "1")
            enabled = {name for name, acc in accessors if acc()}
            assert enabled == {env}, f"enabling {env} leaked into {enabled - {env}}"
            monkeypatch.delenv(env)

    def test_clean_env_all_off(self, monkeypatch):
        for env, _, _ in FEATURE_SWITCHES:
            monkeypatch.delenv(env, raising=False)
        for _, acc in _all_accessors():
            assert acc() is False
