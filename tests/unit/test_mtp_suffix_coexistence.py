# SPDX-License-Identifier: Apache-2.0
# Tests for mtp + suffix coexistence: per-step mtp-handled flag + guard helper.
#
# Validates the foundation for mtp<->suffix per-request routing: when mtp
# owns a decode step (verify+accept inside GenerationBatch.next), the
# scheduler's _try_spec_decode guard (last_step_was_mtp) must suppress
# suffix/dflash/dspark/draft on that step to prevent double-spec cache
# corruption. These tests are self-contained (correct _fusion_mlx_*
# naming) and NOT in debt_modules.txt so they run in CI.

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

logger = logging.getLogger(__name__)


# --- last_step_was_mtp helper (pure, no mlx dependency) ---


def _bg_with_flag(stepped=None, has_gen_batch=True):
    if not has_gen_batch:
        return SimpleNamespace()
    gen_batch = SimpleNamespace()
    if stepped is not None:
        gen_batch._fusion_mlx_mtp_stepped = stepped
    return SimpleNamespace(_generation_batch=gen_batch)


class TestLastStepWasMtp:
    def test_true_when_stepped(self):
        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        assert last_step_was_mtp(_bg_with_flag(stepped=True)) is True

    def test_false_when_not_stepped(self):
        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        assert last_step_was_mtp(_bg_with_flag(stepped=False)) is False

    def test_false_when_no_generation_batch_attr(self):
        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        assert last_step_was_mtp(_bg_with_flag(has_gen_batch=False)) is False

    def test_false_when_generation_batch_none(self):
        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        bg = SimpleNamespace(_generation_batch=None)
        assert last_step_was_mtp(bg) is False

    def test_false_when_flag_missing(self):
        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        bg = SimpleNamespace(_generation_batch=SimpleNamespace())
        assert last_step_was_mtp(bg) is False

    def test_false_on_exception(self):
        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        class _Boom:
            def __getattr__(self, _name):
                raise RuntimeError("boom")

        # RuntimeError (not AttributeError) propagates out of getattr's
        # default; the helper's except-Exception must swallow it -> False.
        assert last_step_was_mtp(_Boom()) is False


# --- patched_next flag lifecycle (needs mlx_lm + patch applied) ---


def _fake_batch():
    return SimpleNamespace(
        uids=[0],
        _fusion_mlx_realign_rows=lambda: None,
    )


class _StopBeforeOriginal(Exception):
    pass


@pytest.fixture
def mtp_patched():
    from fusion_mlx.patches.mlx_lm_mtp import batch_generator

    if not batch_generator.apply():
        pytest.skip("batch_generator patch refused to apply (mlx_lm absent)")
    return batch_generator


class TestPatchedNextFlag:
    def test_success_sets_flag(self, mtp_patched, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        bg = mtp_patched
        monkeypatch.setattr(bg, "_is_mtp_batch_eligible", lambda _self: False)
        monkeypatch.setattr(bg, "_is_mtp_eligible", lambda _self: True)
        monkeypatch.setattr(bg, "_prepare_mtp_state_for_next", lambda _self: object())
        monkeypatch.setattr(bg, "_mtp_next", lambda _self, _state: ["mtp-result"])
        monkeypatch.setattr(bg, "_drop_mtp_state", lambda *a, **k: None)
        monkeypatch.setattr(bg, "_mark_standard_multirow_decode", lambda _self: None)

        batch = _fake_batch()
        result = GenerationBatch.next(batch)
        logger.debug(
            "mtp success path result=%r flag=%r", result, batch._fusion_mlx_mtp_stepped
        )
        assert result == ["mtp-result"]
        assert batch._fusion_mlx_mtp_stepped is True

    def test_fallback_leaves_flag_false(self, mtp_patched, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        bg = mtp_patched
        monkeypatch.setattr(bg, "_is_mtp_batch_eligible", lambda _self: False)
        monkeypatch.setattr(bg, "_is_mtp_eligible", lambda _self: True)
        monkeypatch.setattr(bg, "_prepare_mtp_state_for_next", lambda _self: object())

        def _raise_fallback(_self, _state):
            raise bg._MtpStepFallback("test fallback")

        monkeypatch.setattr(bg, "_mtp_next", _raise_fallback)
        monkeypatch.setattr(bg, "_drop_mtp_state", lambda *a, **k: None)

        def _stop(_self):
            raise _StopBeforeOriginal

        monkeypatch.setattr(bg, "_mark_standard_multirow_decode", _stop)

        batch = _fake_batch()
        with pytest.raises(_StopBeforeOriginal):
            GenerationBatch.next(batch)
        logger.debug(
            "mtp fallback path flag=%r (expected False)", batch._fusion_mlx_mtp_stepped
        )
        assert batch._fusion_mlx_mtp_stepped is False

    def test_flag_resets_each_call(self, mtp_patched, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        bg = mtp_patched
        monkeypatch.setattr(bg, "_is_mtp_batch_eligible", lambda _self: False)
        monkeypatch.setattr(bg, "_is_mtp_eligible", lambda _self: True)
        monkeypatch.setattr(bg, "_prepare_mtp_state_for_next", lambda _self: object())
        monkeypatch.setattr(bg, "_mtp_next", lambda _self, _state: ["mtp-result"])
        monkeypatch.setattr(bg, "_drop_mtp_state", lambda *a, **k: None)
        monkeypatch.setattr(bg, "_mark_standard_multirow_decode", lambda _self: None)

        batch = _fake_batch()
        GenerationBatch.next(batch)
        assert batch._fusion_mlx_mtp_stepped is True

        # Second call: not eligible. Flag must reset to False at the top of
        # patched_next. Short-circuit before original_next via the marker so
        # we never need a fully-formed GenerationBatch.
        monkeypatch.setattr(bg, "_is_mtp_eligible", lambda _self: False)

        def _stop(_self):
            raise _StopBeforeOriginal

        monkeypatch.setattr(bg, "_mark_standard_multirow_decode", _stop)
        with pytest.raises(_StopBeforeOriginal):
            GenerationBatch.next(batch)
        logger.debug(
            "reset path flag=%r (expected False after non-mtp step)",
            batch._fusion_mlx_mtp_stepped,
        )
        assert batch._fusion_mlx_mtp_stepped is False

    def test_batch_path_success_sets_flag(self, mtp_patched, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        bg = mtp_patched
        monkeypatch.setattr(bg, "_is_mtp_batch_eligible", lambda _self: True)
        monkeypatch.setattr(
            bg, "_prepare_mtp_batch_state_for_next", lambda _self: object()
        )
        monkeypatch.setattr(bg, "_mtp_batch_next", lambda _self, _state: ["mtp-batch"])
        monkeypatch.setattr(bg, "_drop_mtp_state", lambda *a, **k: None)
        monkeypatch.setattr(bg, "_mark_standard_multirow_decode", lambda _self: None)

        batch = _fake_batch()
        result = GenerationBatch.next(batch)
        logger.debug(
            "mtp batch path result=%r flag=%r", result, batch._fusion_mlx_mtp_stepped
        )
        assert result == ["mtp-batch"]
        assert batch._fusion_mlx_mtp_stepped is True


# --- guard-helper integration: flag drives last_step_was_mtp end-to-end ---


class TestFlagToGuardIntegration:
    def test_flagged_batch_seen_by_helper(self, mtp_patched, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        bg = mtp_patched
        monkeypatch.setattr(bg, "_is_mtp_batch_eligible", lambda _self: False)
        monkeypatch.setattr(bg, "_is_mtp_eligible", lambda _self: True)
        monkeypatch.setattr(bg, "_prepare_mtp_state_for_next", lambda _self: object())
        monkeypatch.setattr(bg, "_mtp_next", lambda _self, _state: ["mtp-result"])
        monkeypatch.setattr(bg, "_drop_mtp_state", lambda *a, **k: None)
        monkeypatch.setattr(bg, "_mark_standard_multirow_decode", lambda _self: None)

        batch = _fake_batch()
        GenerationBatch.next(batch)

        # The scheduler reads the flag off the batch_generator's
        # _generation_batch. Simulate that wiring: a batch_generator whose
        # _generation_batch is the batch that just ran mtp.
        sched_view = SimpleNamespace(_generation_batch=batch)
        assert last_step_was_mtp(sched_view) is True

    def test_unflagged_batch_not_seen_by_helper(self, mtp_patched, monkeypatch):
        from mlx_lm.generate import GenerationBatch

        from fusion_mlx.patches.mlx_lm_mtp import last_step_was_mtp

        bg = mtp_patched
        monkeypatch.setattr(bg, "_is_mtp_batch_eligible", lambda _self: False)
        monkeypatch.setattr(bg, "_is_mtp_eligible", lambda _self: True)
        monkeypatch.setattr(bg, "_prepare_mtp_state_for_next", lambda _self: object())

        def _raise_fallback(_self, _state):
            raise bg._MtpStepFallback("test fallback")

        monkeypatch.setattr(bg, "_mtp_next", _raise_fallback)
        monkeypatch.setattr(bg, "_drop_mtp_state", lambda *a, **k: None)

        def _stop(_self):
            raise _StopBeforeOriginal

        monkeypatch.setattr(bg, "_mark_standard_multirow_decode", _stop)

        batch = _fake_batch()
        with pytest.raises(_StopBeforeOriginal):
            GenerationBatch.next(batch)

        sched_view = SimpleNamespace(_generation_batch=batch)
        assert last_step_was_mtp(sched_view) is False
