# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dense-LLM batched-sampler fast-path installer.

QUARANTINED in ``tests/unit/debt_modules.txt``. The original dense-sampler
fast-path feature (``_install_dense_sampler_fastpath`` + the per-Scheduler
bounded-LRU ``_sampler_cache``) was removed during the Rapid-MLX migration
and replaced by the simpler module-level singleton ``get_or_create_fused_sampler``
(covered by ``test_sampler_fast_path.py``). ``conftest`` injects a no-op
shim so the import below resolves; the installer it exercises does not
exist in prod.

What remains here are the **install-safety / defensive-behavior** tests.
They assert that a no-op installer is safe — no spurious swap, restore-on-
exception, plain-closure wrapping, missing-``_generation_batch`` tolerance.
These pass legitimately against the shim (a no-op installer is genuinely
safe) and document the contract a future re-port of the homogeneous-batch
swap would need to honor. See issue #674 for the product decision and
the rationale for keeping the file quarantined rather than deleting it
wholesale or un-quarantining the shim-driven tests.

The tests stub a minimal ``GenerationBatch``-shaped object so they don't
have to load a model. Behaviors locked in:

1. Heterogeneous batch → leaves ``self.samplers`` untouched.
2. B=1 → no swap (degenerate case; identity-equality with empty rest is
   true but the patch must NOT engage since the fast-path savings only
   exist for B ≥ 2).
3. Swap is reversed after the call returns — even on exception — so the
   per-request samplers are restored for the NEXT step.
"""

from __future__ import annotations

import types

import pytest

from fusion_mlx.scheduler import _install_dense_sampler_fastpath


class _FakeGenBatch:
    """Stub matching mlx-lm ``GenerationBatch`` attributes the fast path touches."""

    def __init__(self, samplers, fallback):
        self.samplers = samplers
        self.fallback_sampler = fallback
        self.observed_samplers = None
        self.observed_fallback = None
        self.step_calls = 0
        self.raise_in_step: Exception | None = None

    def _step(self):
        self.step_calls += 1
        self.observed_samplers = list(self.samplers)
        self.observed_fallback = self.fallback_sampler
        if self.raise_in_step is not None:
            raise self.raise_in_step
        return ([0] * len(self.samplers), [None] * len(self.samplers))


class _FakeBatchGen:
    def __init__(self, gen_batch):
        self._generation_batch = gen_batch


def _install(gen_batch):
    _install_dense_sampler_fastpath(_FakeBatchGen(gen_batch))


def test_heterogeneous_batch_keeps_per_request_samplers():
    """Mixed sampler identities → patch is a no-op; mlx-lm's per-row loop runs."""
    s1 = lambda x: x  # noqa: E731
    s2 = lambda x: x  # noqa: E731
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[s1, s2, s1, s2], fallback=original_fallback)
    _install(gb)

    gb._step()

    # Slow path observed: samplers and fallback unchanged inside the call.
    assert gb.observed_samplers == [s1, s2, s1, s2]
    assert gb.observed_fallback is original_fallback


def test_b1_does_not_engage_fast_path():
    """B=1 is degenerate — patch must NOT swap (no perf upside, and
    swapping just adds attribute writes per step)."""
    sampler = lambda x: x  # noqa: E731
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[sampler], fallback=original_fallback)
    _install(gb)

    gb._step()

    assert gb.observed_samplers == [sampler]
    assert gb.observed_fallback is original_fallback


def test_homogeneous_with_first_none_does_not_engage():
    """If samplers[0] is None we cannot share — even if the rest match,
    ``None`` means mlx-lm will reach for ``fallback_sampler`` per row.
    The patch's identity-equality check is gated on ``first is not None``."""
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[None, None], fallback=original_fallback)
    _install(gb)

    gb._step()

    # mlx-lm's slow-path branch is `any(self.samplers)` — all-None already
    # takes the fast path naturally; we just must not synthesize a swap.
    assert gb.observed_samplers == [None, None]
    assert gb.observed_fallback is original_fallback


def test_swap_is_reversed_on_exception():
    """If ``_step`` raises, the per-request samplers MUST still be
    restored. Otherwise the next step would silently see the wrong
    sampling distribution for those requests.

    Uses ``pytest.raises`` so a regression that stops raising is caught —
    a bare ``try/except`` would let the test silently pass."""
    shared = lambda x: x  # noqa: E731
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[shared, shared], fallback=original_fallback)
    boom = RuntimeError("metal blew up")
    gb.raise_in_step = boom
    _install(gb)

    with pytest.raises(RuntimeError) as excinfo:
        gb._step()
    assert excinfo.value is boom

    # After the exception, swap must be reverted.
    assert gb.samplers == [shared, shared]
    assert gb.fallback_sampler is original_fallback


def test_install_is_safe_when_step_already_a_plain_closure():
    """SuffixDecoding writes ``gb._step = _suffix_step`` (a plain closure,
    not a bound method). The fast-path installer must wrap it without
    requiring ``__func__``."""
    shared = lambda x: x  # noqa: E731
    captured = {"called": False}

    def suffix_like_step():  # zero-arg closure, mimics _install_suffix_decoding
        captured["called"] = True
        return ([0, 0], [None, None])

    gb = _FakeGenBatch(samplers=[shared, shared], fallback=lambda x: x)
    gb._step = suffix_like_step  # type: ignore[method-assign]
    _install(gb)

    gb._step()

    assert captured["called"]
    # Outside the wrapped call samplers are restored to the per-request list.
    assert gb.samplers == [shared, shared]


def test_install_no_op_when_generation_batch_missing():
    """Defensive: older mlx-lm shapes without ``_generation_batch`` must
    not crash the installer — they just skip the fast path."""

    class _BareBatchGen:
        _generation_batch = None

    # Should not raise.
    _install_dense_sampler_fastpath(_BareBatchGen())

    class _NoStepBatch:
        pass

    class _BareBatchGen2:
        _generation_batch = _NoStepBatch()

    _install_dense_sampler_fastpath(_BareBatchGen2())


def test_method_type_wrapper_sees_correct_self():
    """The installed patch is bound via ``types.MethodType`` — ``self``
    inside the wrapper must be the actual generation_batch instance
    (not whatever was passed at install time)."""
    shared = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[shared, shared], fallback=lambda x: x)
    _install(gb)

    assert isinstance(gb._step, types.MethodType)
    assert gb._step.__self__ is gb
