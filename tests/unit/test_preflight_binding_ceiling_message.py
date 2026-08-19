# SPDX-License-Identifier: Apache-2.0
"""Split from quarantined test_engine_preflight.py — only the
``TestRejectionMessageNamesBindingCeiling`` class survives because it
exercises ``Scheduler._preflight_memory_check`` +
``_format_rejection_message`` which ARE ported prod code.

The parent suite's 20 other tests require an unported engine-wrapper
preflight layer (``preflight_chat`` / ``preflight_completion`` /
``preflight_or_raise`` / ``_preflight_or_raise_with_eviction`` /
``preflight_eviction_request`` / ``_IMAGE_TOKEN_UPPER_BOUND_FALLBACK`` /
EngineCore add_request-cleanup-on-raise) — those stay quarantined in
``test_engine_preflight.py``.

Harness drift fixed here:
  * ``monkeypatch.setattr(scheduler_mod, "get_phys_footprint", ...)``
    failed (``get_phys_footprint`` is a local import inside
    ``sched_query``, NOT a ``fusion_mlx.scheduler`` package attribute).
    Repointed to ``fusion_mlx.scheduler.sched_query.get_phys_footprint``.
  * ``monkeypatch.setattr(sched, "_raise_prefill_eviction_if_available",
    ...)`` referenced a symbol that never shipped — removed. The ported
    ``_preflight_memory_check`` does not call any eviction helper before
    formatting the rejection, so the patch was dead isolation glue.
"""

from unittest.mock import MagicMock

from fusion_mlx.scheduler import Scheduler, SchedulerConfig


class _ModelConfig:
    def __init__(
        self,
        num_hidden_layers=32,
        num_key_value_heads=8,
        num_attention_heads=32,
        head_dim=192,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim


def _make_scheduler():
    model = MagicMock()
    model.layers = []
    model.config = _ModelConfig()
    del model.make_cache

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=2048,
        paged_cache_block_size=0,
    )
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


class TestRejectionMessageNamesBindingCeiling:
    """When a request is rejected, the message must name which of the
    three component ceilings (static / dynamic / metal_cap) is binding
    and steer the user to the right remedy.

    Without this discrimination operators on Pi-class hosts spent hours
    staring at a generic "reduce context length, free system memory, or
    loosen memory_guard_tier" message that didn't tell them which of
    their three knobs to actually turn. The most common confusion was a
    metal_cap-bound 413 on hosts where ``iogpu.wired_limit_mb`` had
    never been raised — the message told them to free system memory
    when no amount of freeing system memory would help.
    """

    def _arm_ceilings(
        self,
        sched,
        *,
        static: int,
        dynamic: int,
        metal_cap: int,
        tier: str = "balanced",
        hot_cache_reserved: int = 0,
    ) -> None:
        """Set the four propagated ceiling fields directly.

        Mirrors what ``ProcessMemoryEnforcer._propagate_memory_limit``
        does on a real run; the binding-aware message reads only these
        fields plus ``_memory_hard_limit_bytes``.
        """
        sched._prefill_memory_guard = True
        hard_limit = min(v for v in (static, dynamic, metal_cap) if v > 0)
        if hot_cache_reserved > 0:
            hard_limit = max(1, hard_limit - hot_cache_reserved)
        sched._memory_hard_limit_bytes = hard_limit
        sched._memory_static_ceiling_bytes = static
        sched._memory_dynamic_ceiling_bytes = dynamic
        sched._memory_metal_cap_bytes = metal_cap
        sched._memory_hot_cache_reserved_bytes = hot_cache_reserved
        sched._memory_guard_tier = tier
        # Set_model_info populated dims at scheduler construction; we
        # only need a non-zero peak estimate to drive the rejection
        # path, not exact bytes.

    def _force_rejection(self, sched, monkeypatch):
        """Mock the parts of the math we don't care about and call
        ``_preflight_memory_check`` so we can inspect the message it
        returns."""
        # Peak chosen larger than any ceiling tested below so the
        # rejection branch fires deterministically.
        sched.memory_monitor = MagicMock()
        sched.memory_monitor.estimate_prefill_peak_bytes.return_value = 512 * 1024**3

        import fusion_mlx.scheduler.sched_query as sched_query_mod

        monkeypatch.setattr(
            sched_query_mod, "get_phys_footprint", lambda: 0, raising=False
        )

        req = MagicMock()
        req.request_id = "binding-test"
        req.num_prompt_tokens = 65536
        req.cached_tokens = 0
        rej = sched._preflight_memory_check(req)
        assert rej is not None, "rejection branch must fire when peak > ceiling"
        return rej

    def test_metal_cap_binding_names_sysctl(self, monkeypatch):
        sched = _make_scheduler()
        self._arm_ceilings(
            sched, static=64 * 1024**3, dynamic=32 * 1024**3, metal_cap=16 * 1024**3
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert (
            "iogpu.wired_limit_mb" in rej.message
        ), f"metal_cap binding must steer user to the sysctl knob; got: {rej.message}"
        assert "metal_cap ceiling" in rej.message
        assert "caps Metal at 16.00 GB" in rej.message

    def test_dynamic_binding_under_custom_names_admin_setting(self, monkeypatch):
        sched = _make_scheduler()
        self._arm_ceilings(
            sched,
            static=64 * 1024**3,
            dynamic=16 * 1024**3,
            metal_cap=48 * 1024**3,
            tier="custom",
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "custom_ceiling_bytes" in rej.message, (
            "dynamic binding under custom tier must point at the admin "
            f"Memory setting, not 'close other apps'; got: {rej.message}"
        )
        assert "close other apps" not in rej.message.lower()

    def test_dynamic_binding_under_reclaim_tier_names_apps(self, monkeypatch):
        sched = _make_scheduler()
        # Static > dynamic, balanced tier: closing apps and/or raising
        # tier is what helps.
        self._arm_ceilings(
            sched,
            static=64 * 1024**3,
            dynamic=16 * 1024**3,
            metal_cap=48 * 1024**3,
            tier="balanced",
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "close other apps" in rej.message.lower(), (
            "dynamic binding on a reclaim tier should suggest closing "
            f"apps; got: {rej.message}"
        )
        assert "memory_guard_tier" in rej.message

    def test_hot_cache_reservation_preserves_binding_label(self, monkeypatch):
        sched = _make_scheduler()
        self._arm_ceilings(
            sched,
            static=64 * 1024**3,
            dynamic=32 * 1024**3,
            metal_cap=16 * 1024**3,
            hot_cache_reserved=2 * 1024**3,
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "metal_cap ceiling" in rej.message
        assert "effective ceiling" not in rej.message
        assert "caps Metal at 16.00 GB" in rej.message

    def test_static_binding_falls_back_to_generic_advice(self, monkeypatch):
        sched = _make_scheduler()
        # Static is the smallest non-zero ceiling.
        self._arm_ceilings(
            sched,
            static=16 * 1024**3,
            dynamic=64 * 1024**3,
            metal_cap=48 * 1024**3,
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "memory_guard_tier" in rej.message
        assert "iogpu.wired_limit_mb" not in rej.message
        assert "custom_ceiling_bytes" not in rej.message
