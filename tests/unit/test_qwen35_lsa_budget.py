# SPDX-License-Identifier: Apache-2.0
"""Tests for the LSA+TQ serving memory budget hard gate (P3 #4)."""

from __future__ import annotations

import pytest

from fusion_mlx.memory_monitor import MemoryMonitor
from fusion_mlx.patches.qwen35_lsa.budget import (
    arm_lsa_memory_budget,
    default_budget_bytes,
    tq_kv_bytes_per_token,
)


def test_tq_kv_bytes_per_token_qwen38_dims():
    # k8v4 packed: K indices 1 B/elem + V indices 0.5 B/elem + one fp16 norm
    # per head per array. Qwen3.8-27B full-attention: 4 kv heads, head_dim
    # 256, 16 full-attention layers -> (4*256*1.5 + 4*4) * 16 = 24832
    assert tq_kv_bytes_per_token(4, 256, 16) == 24832


def test_tq_kv_bytes_per_token_single_layer():
    assert tq_kv_bytes_per_token(2, 128, 1) == int((2 * 128 * 1.5 + 2 * 4) * 1)


def test_default_budget_bytes_fraction_of_ram(monkeypatch):
    import os

    monkeypatch.setattr(os, "sysconf", lambda name: 128 * 1024**3)
    assert default_budget_bytes() == int(128 * 1024**3 * 0.85)


def test_arm_wires_override_guard_and_ceiling():
    monitor = MemoryMonitor()
    monitor.set_model_info(
        num_layers=64, head_dim=256, num_kv_heads=4, num_kv_cache_layers=16
    )
    scheduler = type(
        "Sched",
        (),
        {
            "memory_monitor": monitor,
            "_memory_hard_limit_bytes": 0,
            "_prefill_memory_guard": False,
        },
    )()
    weights = 50 * 1024**3
    summary = arm_lsa_memory_budget(scheduler, weights_bytes=weights)

    assert summary is not None
    assert monitor._kv_bytes_per_token_override == 24832.0
    assert scheduler._prefill_memory_guard is True
    assert scheduler._memory_hard_limit_bytes == int(128 * 1024**3 * 0.85)
    expected_max_ctx = int((int(128 * 1024**3 * 0.85) - weights) // 24832)
    assert summary["max_context_tokens"] == expected_max_ctx


def test_arm_respects_existing_enforcer_ceiling():
    monitor = MemoryMonitor()
    monitor.set_model_info(
        num_layers=64, head_dim=256, num_kv_heads=4, num_kv_cache_layers=16
    )
    existing = 64 * 1024**3
    scheduler = type(
        "Sched",
        (),
        {
            "memory_monitor": monitor,
            "_memory_hard_limit_bytes": existing,
            "_prefill_memory_guard": False,
        },
    )()
    summary = arm_lsa_memory_budget(scheduler, weights_bytes=1)
    assert scheduler._memory_hard_limit_bytes == existing
    assert summary["ceiling_bytes"] == existing


def test_arm_noop_without_monitor():
    scheduler = type(
        "Sched",
        (),
        {
            "memory_monitor": None,
            "_memory_hard_limit_bytes": 0,
            "_prefill_memory_guard": False,
        },
    )()
    assert arm_lsa_memory_budget(scheduler, weights_bytes=0) is None
    assert scheduler._prefill_memory_guard is False


def test_arm_noop_without_model_info():
    scheduler = type(
        "Sched",
        (),
        {
            "memory_monitor": MemoryMonitor(),
            "_memory_hard_limit_bytes": 0,
            "_prefill_memory_guard": False,
        },
    )()
    assert arm_lsa_memory_budget(scheduler, weights_bytes=0) is None
    assert scheduler._prefill_memory_guard is False


def test_budget_max_context_covers_1m_on_128gb():
    # acceptance math: 128 GB machine, doc budget 0.85, model ~50 GB, TQ KV
    # 24832 B/token -> the 1M-token context must fit with headroom
    budget = int(128 * 1024**3 * 0.85)
    bpt = tq_kv_bytes_per_token(4, 256, 16)
    max_ctx = (budget - 50 * 1024**3) // bpt
    assert max_ctx >= 1_000_000, f"1M context does not fit: {max_ctx}"


def test_arm_extracts_dims_from_model_when_monitor_blank():
    # scheduler init runs before the model config is populated, so the
    # monitor often has no model info at engine-setup time - the arm must
    # fall back to extracting dims from the loaded model itself
    import types

    monitor = MemoryMonitor()
    config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(
            num_hidden_layers=64,
            num_key_value_heads=4,
            num_attention_heads=24,
            head_dim=256,
        )
    )
    model = types.SimpleNamespace(config=config)
    scheduler = type(
        "Sched",
        (),
        {
            "memory_monitor": monitor,
            "_memory_hard_limit_bytes": 0,
            "_prefill_memory_guard": False,
            "model": model,
        },
    )()
    summary = arm_lsa_memory_budget(scheduler, weights_bytes=50 * 1024**3)

    assert summary is not None
    assert monitor._kv_bytes_per_token_override == tq_kv_bytes_per_token(4, 256, 64)
    assert scheduler._prefill_memory_guard is True
    assert monitor._num_kv_cache_layers == 64  # no make_cache: full fallback


def test_arm_model_fallback_counts_kvcache_layers():
    # hybrid models carry fewer KVCache layers than total layers; the
    # fallback must count them via make_cache like sched_misc does
    import types

    from mlx_lm.models.cache import ArraysCache, KVCache

    monitor = MemoryMonitor()
    config = types.SimpleNamespace(
        num_hidden_layers=4,
        num_key_value_heads=2,
        head_dim=128,
    )
    model = types.SimpleNamespace(
        config=config,
        make_cache=lambda: [KVCache(), ArraysCache(1), KVCache(), ArraysCache(1)],
    )
    scheduler = type(
        "Sched",
        (),
        {
            "memory_monitor": monitor,
            "_memory_hard_limit_bytes": 0,
            "_prefill_memory_guard": False,
            "model": model,
        },
    )()
    summary = arm_lsa_memory_budget(scheduler, weights_bytes=0)

    assert summary is not None
    assert monitor._num_kv_cache_layers == 2
    assert summary["kv_bytes_per_token"] == (2 * 128 * 1.5 + 2 * 4) * 2


def test_arm_model_fallback_handles_dict_args():
    # mlx_vlm Model.args carries the HF config.json as plain dicts
    import types

    monitor = MemoryMonitor()
    model = types.SimpleNamespace(
        args={
            "model_type": "qwen3",
            "text_config": {
                "num_hidden_layers": 64,
                "num_key_value_heads": 4,
                "num_attention_heads": 24,
                "head_dim": 256,
            },
        }
    )
    scheduler = type(
        "Sched",
        (),
        {
            "memory_monitor": monitor,
            "_memory_hard_limit_bytes": 0,
            "_prefill_memory_guard": False,
            "model": model,
        },
    )()
    summary = arm_lsa_memory_budget(scheduler, weights_bytes=50 * 1024**3)

    assert summary is not None
    assert monitor._num_kv_heads == 4
    assert monitor._head_dim == 256
    assert summary["kv_bytes_per_token"] == tq_kv_bytes_per_token(4, 256, 64)
