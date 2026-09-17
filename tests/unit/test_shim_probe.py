"""Unit tests for fusion_mlx.shim PR-A (C++ skeleton + Tier-1 safety base).

These tests run headless (no MLX build required) by exercising the Python
fallback path when _ext is absent, and the native path when the shim
extension has been built inplace. conftest.py mocks mlx.core on Linux CI.
"""

from __future__ import annotations

import pytest

import fusion_mlx.shim as shim
from fusion_mlx.shim import fast


def test_master_switch_default_off(monkeypatch):
    monkeypatch.delenv("FUSION_SHIM_ENABLED", raising=False)
    assert shim.is_shim_enabled() is False


def test_master_switch_on(monkeypatch):
    monkeypatch.setenv("FUSION_SHIM_ENABLED", "1")
    assert shim.is_shim_enabled() is True


def test_native_available_matches_ext_import():
    # is_native_available must agree with whether _ext imported.
    assert fast.is_native_available() == (fast._ext is not None)


def test_hardware_probe_returns_dict_with_required_keys():
    probe = shim.hardware_probe()
    assert isinstance(probe, dict)
    for key in (
        "architecture",
        "gen",
        "has_bf16_mma",
        "has_fp8_mma",
        "gpu_core_count",
        "device_name",
    ):
        assert key in probe, f"missing key: {key}"
    assert isinstance(probe["gen"], int)
    assert isinstance(probe["has_bf16_mma"], bool)
    assert isinstance(probe["has_fp8_mma"], bool)


def test_hardware_probe_bf16_implies_gen_ge_3_or_marked_unavailable():
    # Either the probe found a real chip (gen>=3 => bf16) or it fell back
    # and reported a conservative value. The invariant: bf16_mma must not
    # be True when gen < 3 (M1/M2 have no hardware BF16 MMA).
    probe = shim.hardware_probe()
    if probe["has_bf16_mma"]:
        assert probe["gen"] >= 3, "bf16_mma=True requires gen>=3 (M3+)"


def test_hardware_probe_fp8_implies_gen_ge_4():
    probe = shim.hardware_probe()
    if probe["has_fp8_mma"]:
        assert probe["gen"] >= 4, "fp8_mma=True requires gen>=4 (M4+)"


def test_hardware_probe_is_idempotent():
    a = shim.hardware_probe()
    b = shim.hardware_probe()
    assert a["gen"] == b["gen"]
    assert a["has_bf16_mma"] == b["has_bf16_mma"]


def test_python_fallback_derives_gen_from_chip_string(monkeypatch):
    # Force the fallback path by clearing FUSION_SHIM_FORCE_CHIP and
    # exercising the pure-Python derivation directly.
    monkeypatch.setenv("FUSION_SHIM_FORCE_CHIP", "Apple M4 Pro")
    probe = fast._python_hardware_probe()
    assert probe["gen"] == 4
    assert probe["has_bf16_mma"] is True
    assert probe["has_fp8_mma"] is True


def test_python_fallback_m2_no_bf16(monkeypatch):
    monkeypatch.setenv("FUSION_SHIM_FORCE_CHIP", "Apple M2")
    probe = fast._python_hardware_probe()
    assert probe["gen"] == 2
    assert probe["has_bf16_mma"] is False
    assert probe["has_fp8_mma"] is False


def test_python_fallback_unknown_chip(monkeypatch):
    monkeypatch.setenv("FUSION_SHIM_FORCE_CHIP", "Apple Silicon")
    probe = fast._python_hardware_probe()
    assert probe["gen"] == 0
    assert probe["has_bf16_mma"] is False


def test_memory_sentinel_start_returns_bool():
    # Native: True/False. Fallback: False. Either way it must not raise.
    result = shim.start_memory_sentinel()
    assert isinstance(result, bool)
    # stop is idempotent
    shim.stop_memory_sentinel()


def test_memory_sentinel_callback_invocable_when_native(monkeypatch):
    # If native is available, a callback should be accepted. We cannot
    # force a pressure event, but start+stop with a callback must not
    # crash. Skip when native unavailable.
    if not fast.is_native_available():
        pytest.skip("native _ext not built — callback path is native-only")

    called = []

    def cb(level, name):
        called.append((level, name))

    ok = shim.start_memory_sentinel(cb)
    assert ok is True
    shim.stop_memory_sentinel()
    # No assertion on `called` — we cannot synthesize a pressure event.


def test_last_memory_pressure_returns_int():
    val = shim.last_memory_pressure()
    assert isinstance(val, int)
    assert 0 <= val <= 2


def test_last_error_accessors_return_correct_types():
    assert isinstance(shim.last_error_code(), int)
    assert isinstance(shim.last_error_message(), str)


def test_status_report_shape():
    s = shim.status()
    assert "shim_enabled" in s
    assert "native_available" in s
    assert "import_error" in s
    assert "native_symbols" in s
    assert isinstance(s["native_symbols"], list)


def test_degrade_on_missing_ext(monkeypatch):
    # Simulate the degrade path: force _ext=None and confirm the fallback
    # still serves a probe without raising.
    monkeypatch.setenv("FUSION_SHIM_FORCE_CHIP", "Apple M3 Max")
    saved = fast._ext
    fast._ext = None
    try:
        probe = shim.hardware_probe()
        assert probe["gen"] == 3
        assert probe["has_bf16_mma"] is True
        assert shim.start_memory_sentinel() is False  # no native sentinel
    finally:
        fast._ext = saved


def test_native_symbols_tuple_when_unavailable():
    if fast.is_native_available():
        assert len(fast.native_symbols()) > 0
    else:
        assert fast.native_symbols() == ()


def test_missing_symbols_empty_for_known_ops():
    # All PR-A symbols should resolve (either native or via the proxy) —
    # missing_symbols only reports native-side gaps.
    missing = fast.missing_symbols(("hardware_probe", "start_memory_sentinel"))
    if fast.is_native_available():
        assert missing == []
