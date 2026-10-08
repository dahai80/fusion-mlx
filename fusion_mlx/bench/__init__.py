"""Benchmark module.

run_benchmark is a stub — use 'fusion-mlx bench' CLI for actual
benchmarking. The flywheel API accepts an explicit runner function.
"""

import logging

logger = logging.getLogger(__name__)


class BenchmarkRunnerUnavailable(RuntimeError):
    pass


def run_benchmark(model: str, **kwargs) -> dict:
    # #1012: was returning {"tokens_per_second": 0} — fake data that callers
    # silently consumed instead of failing. Raise so callers know no runner
    # is wired (use the CLI or pass an explicit runner to flywheel).
    raise BenchmarkRunnerUnavailable(
        f"run_benchmark('{model}') has no runner wired — "
        "use 'fusion-mlx bench' CLI or pass an explicit runner to flywheel()"
    )
