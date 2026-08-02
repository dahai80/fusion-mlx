# SPDX-License-Identifier: Apache-2.0
"""Constraint evaluation for speculative-decode routing.

Centralised constraint predicates used by SpecRouteEntry.evaluate_constraints()
and the /v1/spec/resolve API. Each constraint is a string token that maps to
a predicate over RouteSignals. New constraints only need an entry in
_CONSTRAINT_EVALUATORS.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .auto_router import RouteSignals

logger = logging.getLogger(__name__)


def _not_moe(signals: RouteSignals) -> bool:
    return not signals.is_moe


def _not_recurrent(signals: RouteSignals) -> bool:
    return not signals.is_recurrent


def _make_min_bits(threshold: int) -> Callable[[RouteSignals], bool]:
    def _check(signals: RouteSignals) -> bool:
        if signals.quant_bits is None:
            return True
        return signals.quant_bits >= threshold
    _check.__name__ = f"min_bits_{threshold}"
    return _check


def evaluate_constraints(
    constraints: tuple[str, ...], signals: RouteSignals
) -> bool:
    for c in constraints:
        fn = _CONSTRAINT_EVALUATORS.get(c)
        if fn is not None:
            if not fn(signals):
                logger.debug("constraint %s FAILED for %s", c, signals.model_family)
                return False
            continue
        if c.startswith("min_bits:"):
            threshold = int(c.split(":")[1])
            if not _make_min_bits(threshold)(signals):
                logger.debug("constraint %s FAILED for %s", c, signals.model_family)
                return False
            continue
        logger.warning("unknown constraint %r — treating as passed", c)
    return True


_CONSTRAINT_EVALUATORS: dict[str, Callable[[RouteSignals], bool]] = {
    "not_moe": _not_moe,
    "not_recurrent": _not_recurrent,
}
