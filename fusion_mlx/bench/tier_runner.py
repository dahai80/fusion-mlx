# SPDX-License-Identifier: Apache-2.0
"""Bench tier runner — standardized validation tiers.

FUNC-P1-1 (#0907 audit): ``--tier`` advertised a smoke/speed/harness tier
dispatcher in the CLI help and README, but the implementation was a bare
``raise NotImplementedError`` that dumped a traceback on the user. A fake
implementation that fabricates bench numbers would be worse (silent wrong
results — Rule 12), so this module surfaces a clear, user-actionable CLI
error instead of a traceback, and the README marks the feature as
not-yet-implemented.

The CLI handlers (cli_serve.py _run_tier_submit_flow / bench_command) catch
``TierRunnerUnavailable`` and print a one-line message + return a non-zero
exit code, so ``--tier`` fails visibly without a stack trace.
"""

import logging

logger = logging.getLogger(__name__)


class TierRunnerUnavailable(RuntimeError):
    """Raised when the tier dispatcher is not implemented in this build."""


_TIER_HELP = (
    "bench --tier is not implemented in this build. "
    "Use `fusion-mlx bench <model> --num-prompts N` for a local speed "
    "benchmark, or `fusion-mlx bench <model> --submit` for the community "
    "benchmark submission flow."
)


def run_tier(*args, **kwargs):
    logger.warning("Bench tier runner not available in this build")
    raise TierRunnerUnavailable(_TIER_HELP)
