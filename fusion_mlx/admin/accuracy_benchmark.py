"""Accuracy benchmark admin module (stub for test migration)."""

import logging

logger = logging.getLogger(__name__)

_queue: list = []
_results: list = []
_running: bool = False
_current_model: str | None = None
_current_bench_id: str | None = None


class AccuracyBenchmarkRequest:
    def __init__(self, **kwargs):
        self.model_id = kwargs.get("model_id", "")
        self.benchmarks = kwargs.get("benchmarks", {})


def add_to_queue(request):
    _queue.append(request)


def get_queue_status():
    return {
        "queue": [q.__dict__ if hasattr(q, "__dict__") else q for q in _queue],
        "running": _running,
        "current_model": _current_model,
        "current_bench_id": _current_bench_id,
    }


def remove_from_queue(idx: int) -> bool:
    if 0 <= idx < len(_queue):
        _queue.pop(idx)
        return True
    return False


def get_accumulated_results():
    return _results


def reset_accumulated_results():
    global _results
    _results = []


def cancel_queue():
    global _running, _current_model, _current_bench_id
    _queue.clear()
    _running = False
    _current_model = None
    _current_bench_id = None


def get_run(bench_id: str):
    for r in _results:
        if isinstance(r, dict) and r.get("bench_id") == bench_id:
            return r
    return None


def start_next_from_queue(engine_pool=None):
    global _running
    if not _queue or _running:
        return
    _running = True
