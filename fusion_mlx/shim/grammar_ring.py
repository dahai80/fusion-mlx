# SPDX-License-Identifier: Apache-2.0
"""PR-M: ring-buffer prefetch of grammar bitmasks (v2 doc §3.4).

Per decode step the stock path fills the grammar bitmask inline
(``GrammarConstraintProcessor.__call__``): the CPU computes the DFA
bitmask, ``mx.array`` transfers it, and only then can sampling proceed —
CPU work directly on the critical path of every constrained token.

This module pipelines it: a single worker thread computes the *next*
step's bitmask into a preallocated ring slot immediately after the
current token is accepted. While the GPU runs the next forward pass the
CPU is already done; ``__call__`` consumes the ready slot. If the worker
has not finished (DFA step slower than the GPU), the caller falls back
to the inline path — never blocks long, never changes results (same
matcher, same bitmask bytes).

Thread-safety contract: the matcher is mutated only by the caller thread
(``accept_token``) and read only by the worker between "job submitted"
and "slot ready". The caller observes the slot (apply/wait) before its
next ``accept_token`` — this matches the mlx_lm loop order (logits
processors run during sampling, accept happens after the token is
known).

Degrade switches (default OFF for the new path):
  FUSION_SHIM_GRAMMAR_RING=1 — enable the prefetch ring; OFF → callers
  use the stock inline processor unchanged.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def is_grammar_ring_enabled() -> bool:
    return _env_on("FUSION_SHIM_GRAMMAR_RING")


def bitmask_width(vocab_size: int) -> int:
    return (vocab_size + 31) // 32


def apply_bitmask(bitmask, logits, vocab_size: int):
    # Same contract as xgrammar's apply_token_bitmask_mlx: bitmask word
    # i holds the allow-bits for tokens [32i, 32i+32); disallowed tokens
    # get -inf logits. Vectorized manual fallback (the xgrammar kernel
    # stays the fast path when importable).
    import mlx.core as mx

    flat = np.asarray(bitmask, dtype=np.int32).reshape(-1)
    n_words = min(flat.shape[0], (vocab_size + 31) // 32)
    bits = (
        (
            flat[:n_words, None].astype(np.uint32)
            >> np.arange(32, dtype=np.uint32)[None, :]
        )
        & 1
    ).astype(bool)
    bits = bits.reshape(-1)[:vocab_size]
    mask = np.where(bits, 0.0, float("-inf")).astype(np.float32)
    mask_mx = mx.array(mask)
    if logits.ndim == 2:
        return logits + mask_mx.reshape(1, -1)
    return logits + mask_mx


class GrammarMaskRing:
    """Pipelined bitmask producer wrapping a per-request matcher.

    ``wrapped`` is any object exposing the GrammarConstraintProcessor
    interface used by the scheduler: ``__call__(tokens, logits)``,
    ``accept_token(token_id)``, ``is_terminated``, ``matcher``. The ring
    intercepts bitmask production; ``__call__`` consumes a prefilled
    slot when one is ready and otherwise delegates to ``wrapped``
    unchanged.
    """

    def __init__(self, wrapped, vocab_size: int, depth: int = 2):
        if depth < 2:
            raise ValueError("grammar ring depth must be >= 2")
        self._wrapped = wrapped
        self._vocab_size = vocab_size
        self._width = bitmask_width(vocab_size)
        self._depth = depth
        self._slots = [
            np.full((1, self._width), -1, dtype=np.int32) for _ in range(depth)
        ]
        self._ready = [False] * depth
        self._steps = 0  # consumption epochs; slot = steps % depth
        self._job_epoch = -1  # epoch the in-flight job was submitted for
        self._terminated = False
        self._cv = threading.Condition()
        self._job_slot = -1
        self._job_in_flight = False
        self._stats = {"prefetched": 0, "inline": 0, "submits": 0}
        logger.debug(
            "GrammarMaskRing: depth=%d width=%d vocab=%d",
            depth,
            self._width,
            vocab_size,
        )

    # -- worker --------------------------------------------------------------

    def _run_job(self) -> None:
        slot = self._job_slot
        produced = False
        try:
            mask = self._compute_mask(slot)
            if mask is not None:
                np.copyto(self._slots[slot], mask.reshape(1, -1))
                produced = True
        except Exception:
            logger.debug(
                "grammar ring prefetch failed; slot left invalid", exc_info=True
            )
        with self._cv:
            self._ready[slot] = produced
            self._job_in_flight = False
            self._cv.notify_all()

    def _submit(self) -> None:
        with self._cv:
            if self._terminated or self._job_in_flight:
                return
            # _steps already points at the next consumption epoch (it was
            # advanced when the previous slot was consumed).
            self._job_epoch = self._steps
            slot = self._steps % self._depth
            self._job_slot = slot
            self._ready[slot] = False
            self._job_in_flight = True
            self._stats["submits"] += 1
        threading.Thread(target=self._run_job, daemon=True).start()

    def _wait_ready(self, slot: int, timeout: float = 0.05) -> bool:
        with self._cv:
            if self._ready[slot]:
                return True
            self._cv.wait(timeout)
            return self._ready[slot]

    # -- mask production (worker + inline fallback share this) ---------------

    def _compute_mask(self, slot: int) -> np.ndarray | None:
        # Mirror of GrammarConstraintProcessor's per-backend bitmask fill,
        # driven through the wrapped processor's matcher — identical bytes
        # to the inline path.
        m = self._wrapped.matcher
        backend = getattr(self._wrapped, "backend", None)
        name = str(getattr(backend, "name", backend) or "").upper()
        if "LLGUIDANCE" in name:
            bitmask = m.compute_bitmask()
            if bitmask is None:
                return None
            if isinstance(bitmask, bytes):
                return np.frombuffer(bitmask, dtype=np.int32)
            return np.asarray(bitmask, dtype=np.int32).reshape(-1)
        # xgrammar (default): fill into the target slot directly.
        m.fill_next_token_bitmask(self._slots[slot])
        return self._slots[slot]

    # -- public interface (matches GrammarConstraintProcessor) ----------------

    @property
    def is_terminated(self) -> bool:
        return self._terminated or self._wrapped.is_terminated

    @property
    def matcher(self):
        return self._wrapped.matcher

    @property
    def backend(self):
        return getattr(self._wrapped, "backend", None)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _drain_job(self) -> None:
        # Block until any in-flight worker has finished reading the
        # matcher. Callers that took the inline fallback may arrive here
        # while the worker is still running; matcher mutation must not
        # overlap the worker's read.
        with self._cv:
            while self._job_in_flight:
                self._cv.wait()

    def accept_token(self, token_id: int) -> None:
        self._drain_job()
        self._wrapped.accept_token(token_id)
        self._terminated = self._wrapped.is_terminated
        if not self._terminated:
            self._submit()

    def __call__(self, tokens, logits):
        if self._terminated:
            return logits
        slot = self._steps % self._depth
        # Consume only a slot whose ready flag belongs to THIS epoch —
        # stale flags from abandoned (timed-out) jobs never match.
        if self._job_epoch == self._steps and self._wait_ready(slot):
            self._ready[slot] = False
            self._steps += 1
            self._stats["prefetched"] += 1
            return apply_bitmask(self._slots[slot], logits, self._vocab_size)
        # Slot not ready (or no job for this epoch): stock inline path.
        self._stats["inline"] += 1
        out = self._wrapped(tokens, logits)
        self._steps += 1
        return out

    def stop(self) -> None:
        self._drain_job()
        self._terminated = True


def wrap_processor(processor, vocab_size: int, depth: int = 2):
    """Return a ring-wrapped processor when enabled, else the original.

    Degrade convention: with FUSION_SHIM_GRAMMAR_RING unset the behavior
    is byte-identical to the stock path.
    """
    if not is_grammar_ring_enabled():
        return processor
    logger.info("grammar ring enabled: wrapping %s", type(processor).__name__)
    return GrammarMaskRing(processor, vocab_size, depth=depth)
