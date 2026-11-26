# SPDX-License-Identifier: Apache-2.0
"""#1722 ask 3: the custom tier's configured ceiling must constrain the
prefill abort path.

`_get_abort_limit_bytes()` deliberately excludes the tier-derived dynamic
ceiling — but the custom tier's ceiling is the operator's explicit number,
and the wired limit is already armed from it (OP-901). Excluding it made
`memory_guard_custom_ceiling_gb` silently fail to constrain prefill
aborts: the abort path kept using the machine-derived static/metal cap.
"""

from __future__ import annotations

from tests.unit.test_soft_busy_evict_1473 import _import_memory_enforcer

GB = 1024**3


def _make_enforcer(me, tier: str, custom_gb: float, static_gb: int, metal_gb: int):
    enforcer = me.ProcessMemoryEnforcer.__new__(me.ProcessMemoryEnforcer)
    enforcer._memory_guard_tier = tier
    enforcer._memory_guard_custom_ceiling_bytes = int(custom_gb * GB)
    enforcer._prefill_memory_guard = True
    enforcer._get_static_ceiling = lambda: static_gb * GB
    enforcer._get_effective_metal_cap_bytes = lambda: (
        metal_gb * GB if metal_gb > 0 else 0
    )
    return enforcer


class TestCustomCeilingAbortLimit:
    def test_custom_ceiling_constrains_abort_limit(self):
        me = _import_memory_enforcer()
        # Static 120GB, custom 100GB: abort line must follow the operator.
        enforcer = _make_enforcer(me, "custom", 100.0, 120, 0)
        assert enforcer._get_abort_limit_bytes() == 100 * GB

    def test_non_custom_tier_ignores_dynamic(self):
        me = _import_memory_enforcer()
        # Balanced tier: dynamic heuristic must not tighten the abort line
        # (pre-existing contract). Custom bytes stored but tier != custom.
        enforcer = _make_enforcer(me, "balanced", 50.0, 120, 0)
        assert enforcer._get_abort_limit_bytes() == 120 * GB

    def test_metal_cap_still_wins_when_lower(self):
        me = _import_memory_enforcer()
        enforcer = _make_enforcer(me, "custom", 132.0, 120, 100)
        assert enforcer._get_abort_limit_bytes() == 100 * GB

    def test_custom_disabled_falls_back_to_static(self):
        me = _import_memory_enforcer()
        enforcer = _make_enforcer(me, "custom", 0.0, 120, 0)
        assert enforcer._get_abort_limit_bytes() == 120 * GB

    def test_guard_disabled_returns_zero(self):
        me = _import_memory_enforcer()
        enforcer = _make_enforcer(me, "custom", 100.0, 120, 0)
        enforcer._prefill_memory_guard = False
        assert enforcer._get_abort_limit_bytes() == 0
