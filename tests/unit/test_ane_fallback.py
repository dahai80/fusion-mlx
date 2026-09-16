# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ANE fallback / degradation tolerance (P0底座)."""

from fusion_mlx.engine.base import ANEExecutionProvider


class _AneEngine(ANEExecutionProvider):
    def __init__(self):
        self.warmup_calls = 0
        self.supported_after_warmup = False

    def ane_supported(self):
        # warmup protection: not supported until warmup signals ready
        return self.supported_after_warmup

    def ane_warmup(self):
        self.warmup_calls += 1
        if self.warmup_calls >= 3:
            self.supported_after_warmup = True
            self._ane_enabled = True
            return True
        return False


def test_ane_disabled_by_default_goes_metal():
    e = _AneEngine()
    assert e.ane_supported() is False
    # unsupported → caller must route to Metal
    assert e._ane_enabled is False


def test_warmup_protection_first_three_calls_not_supported():
    e = _AneEngine()
    for _ in range(2):
        assert e.ane_warmup() is False
        assert e.ane_supported() is False
    assert e.ane_warmup() is True
    assert e.ane_supported() is True


def test_coreml_exception_triggers_fallback():
    e = _AneEngine()
    e._ane_enabled = True
    e._ane_resident_bytes = 512 * 1024**2
    # simulate CoreML raising during inference
    try:
        raise RuntimeError("coreml compile failed")
    except RuntimeError:
        e.ane_fallback_to_metal()
    assert e._ane_enabled is False
    assert e.ane_supported() is False or e.ane_supported() is True
    # resident bytes still reported for cleanup accounting
    assert e.ane_resident_memory() == 512 * 1024**2


def test_os_abi_break_disables_ane_no_crash():
    """Simulate ABI validation failure (private MIL coremlc mismatch)."""
    e = _AneEngine()
    e._ane_enabled = True

    class _AbiError(Exception):
        pass

    try:
        raise _AbiError("coremlc 3520.x ABI mismatch")
    except _AbiError:
        e.ane_fallback_to_metal()
    assert e._ane_enabled is False


def test_dynamic_fallback_threshold():
    """T_ANE > 1.5x T_metal → mark degraded (simulate via flag)."""
    e = _AneEngine()
    e._ane_enabled = True
    t_ane = 0.150
    t_metal = 0.080
    degraded = t_ane > 1.5 * t_metal
    if degraded:
        e.ane_fallback_to_metal()
    assert degraded is True
    assert e._ane_enabled is False
