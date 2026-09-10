"""Benchmark module.

run_benchmark is a stub — use 'fusion-mlx bench' CLI for actual
benchmarking. The flywheel API accepts an explicit runner function.
"""

import logging

logger = logging.getLogger(__name__)


class BenchmarkRunnerUnavailable(RuntimeError):
    pass


def run_benchmark(model: str, **kwargs) -> dict:
    return {"tokens_per_second": 0}
