# SPDX-License-Identifier: Apache-2.0
"""ENG-13 (#0909 audit): template-method base for speculative-decode
eligibility checks.

The four strategy modules (dflash/dflash2/dspark/dfly) each shipped a
near-identical quartet of module-level functions — ``report``,
``eligible_aliases``, ``check``, ``have_runtime`` — differing only in
the strategy label, the feature-flag attribute, the probed runtime
module, the unavailable exception class, and the per-gate reason strings.
~200 lines of copy-paste, with the ``eligible_aliases`` body verbatim
identical across all four.

This base absorbs the identical parts (``eligible_aliases`` wholesale;
``check`` and ``have_runtime`` as template methods driven by small
subclass hooks) and leaves ``report`` — the genuine per-strategy override
point — to each subclass. Subclass modules instantiate a checker and
re-bind the module-level function names (``report = _checker.report``,
etc.) so existing import sites (``from ...eligibility import check``)
keep working unchanged.
"""

import importlib
import logging
from dataclasses import dataclass

from fusion_mlx.model_aliases import AliasProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseEligibilityReport:
    alias: str | None
    reasons: tuple[str, ...]


class BaseEligibilityChecker:
    # Subclass hooks (override) -------------------------------------------
    _STRATEGY_LABEL: str = "Spec"

    def report(self, profile: AliasProfile, alias: str | None = None):
        raise NotImplementedError

    def _unavailable_exc(self) -> type[RuntimeError]:
        raise NotImplementedError

    def _runtime_module(self) -> str:
        raise NotImplementedError

    def _empty_eligible_suffix(self) -> str:
        return (
            f"No aliases currently pass every {self._STRATEGY_LABEL} gate. "
            f"Run `fusion-mlx info <alias>` to inspect per-alias "
            f"{self._STRATEGY_LABEL} status."
        )

    # Template methods (shared) -------------------------------------------
    def eligible_aliases(self) -> list[str]:
        try:
            from fusion_mlx.model_aliases import list_profiles

            return sorted(
                p.name for p in list_profiles().values() if not self.report(p).reasons
            )
        except Exception as e:
            logger.debug("eligible_aliases failed: %s", e)
            return []

    def check(
        self,
        profile: AliasProfile,
        alias: str | None = None,
        *,
        _eligible_fn=None,
    ) -> None:
        r = self.report(profile, alias=alias)
        if not r.reasons:
            return
        label = self._STRATEGY_LABEL
        header = (
            f"{label} unavailable for {alias!r}" if alias else f"{label} unavailable"
        )
        bullet = "\n  - ".join(r.reasons)
        # _eligible_fn lets the module-level wrapper pass its own
        # ``eligible_aliases`` name (resolved at call time) so tests that
        # monkeypatch ``eligibility.eligible_aliases`` take effect — a plain
        # ``self.eligible_aliases`` bound method would bypass the patch.
        eligible = (_eligible_fn or self.eligible_aliases)()
        if eligible:
            suffix = (
                f"Eligible aliases today: {', '.join(eligible)}. Run "
                f"`fusion-mlx info <alias>` to inspect per-alias {label} status."
            )
        else:
            suffix = self._empty_eligible_suffix()
        raise self._unavailable_exc()(f"{header}:\n  - {bullet}\n\n{suffix}")

    def have_runtime(self) -> bool:
        try:
            spec = importlib.util.find_spec(self._runtime_module())
            return spec is not None
        except (ImportError, AttributeError, ModuleNotFoundError):
            return False
