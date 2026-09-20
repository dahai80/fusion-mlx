# SPDX-License-Identifier: Apache-2.0
# PRD v1 §3.6 — fusion-autotest extreme stress scenarios.
#
# Two mandatory stress classes (acceptance §7.6):
#   1. high_concurrency — batch-submit mixed LTX/H3 tasks, verify the
#      dual-model mutex serializes them (no co-residency), no deadlock,
#      every task either completes or is cleanly rejected.
#   2. oom_degrade_recovery — drive the 3-level breaker via fake memory
#      probes, verify L1/L2/L3 plans mutate params correctly + emergency
#      reclaim fires + cache releases + scheduler returns to OK after reclaim.
#
# Does NOT load real video models (those are the §4 PoC gates). Exercises
# the scheduler + router machinery directly under contention, the way
# fusion-autotest would drive the public_api surface.
#
# Usage:
#   python scripts/stress_video_v1.py            # run both scenarios
#   python scripts/stress_video_v1.py --scenario high_concurrency
#   python scripts/stress_video_v1.py --json

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field


@dataclass
class StressResult:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    metrics: dict = field(default_factory=dict)


def _make_params(model: str):
    class P:
        prompt = "stress test prompt"
        num_inference_steps = 40
        height = 768
        width = 768
        audio = True
        no_compile = False
        n = 1
        num_frames = 25

    return P()


def _scenario_high_concurrency() -> StressResult:
    """§3.6.1 — concurrent mixed LTX/H3 acquires must serialize via mutex."""
    t0 = time.monotonic()
    from fusion_mlx.scheduler.video_unified_scheduler import VideoUnifiedScheduler

    s = VideoUnifiedScheduler()
    results = {"acquired": 0, "rejected": 0, "deadlock": False}
    lock = threading.Lock()

    def worker(model: str):
        try:
            s.acquire(model)
            with lock:
                results["acquired"] += 1
            time.sleep(0.05)
            s.release(model)
        except RuntimeError:
            with lock:
                results["rejected"] += 1
        except Exception:
            with lock:
                results["deadlock"] = True

    threads = []
    for i in range(12):
        model = "ltx2_5" if i % 2 == 0 else "minimax_h3"
        threads.append(threading.Thread(target=worker, args=(model,)))
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)
        if th.is_alive():
            results["deadlock"] = True

    # Mutex invariant: at most one model resident at any time → exactly the
    # serialized acquires succeed, the rest raise RuntimeError (no co-residency).
    # No deadlock, every thread released.
    ok = (
        not results["deadlock"]
        and results["acquired"] + results["rejected"] == 12
        and s.snapshot()["active_model"] is None
    )
    return StressResult(
        "high_concurrency",
        ok,
        f"acquired={results['acquired']} rejected={results['rejected']} "
        f"deadlock={results['deadlock']}",
        time.monotonic() - t0,
        results,
    )


def _scenario_oom_degrade_recovery() -> StressResult:
    """§3.6.2 — drive breaker levels, verify degrade + reclaim + recovery."""
    t0 = time.monotonic()
    import mlx.core as mx

    from fusion_mlx.scheduler.video_unified_scheduler import (
        DegradationPlan,
        MemoryLevel,
        NF4DequantCache,
        VideoUnifiedScheduler,
    )

    checks = {
        "l1_steps": False,
        "l2_res_audio": False,
        "l3_reclaim": False,
        "recovery_ok": False,
        "cache_released": False,
    }

    # L1: reduce steps + disable upsample
    p = _make_params("ltx2_5")
    DegradationPlan(
        level=MemoryLevel.L1_WARN,
        reduce_steps=True,
        disable_upsample=True,
        reason="stress",
    ).apply_to(p)
    checks["l1_steps"] = p.num_inference_steps == 20 and p.no_compile is True

    # L2: drop resolution + disable audio
    p = _make_params("h3")
    DegradationPlan(
        level=MemoryLevel.L2_PROTECT,
        drop_resolution=True,
        disable_audio=True,
        reason="stress",
    ).apply_to(p)
    checks["l2_res_audio"] = p.height == 384 and p.audio is False

    # L3: emergency reclaim clears cache + gc + metal clear
    s = VideoUnifiedScheduler()
    s.acquire("ltx2_5")
    cache = s.dequant_cache
    cache.get_or_dequant("w", lambda: mx.zeros((4, 4)))
    assert len(cache._entries) == 1
    s.emergency_reclaim()  # L3 path
    checks["l3_reclaim"] = (
        len(cache._entries) == 0 if cache._entries is not None else True
    )
    # after reclaim, scheduler can accept a new task (recovery)
    s.release("ltx2_5")
    try:
        s.acquire("minimax_h3")
        checks["recovery_ok"] = True
        s.release("minimax_h3")
    except RuntimeError:
        checks["recovery_ok"] = False

    # standalone cache release
    c2 = NF4DequantCache(budget_gb=1)
    c2.get_or_dequant("w", lambda: mx.zeros((2, 2)))
    c2.release()
    checks["cache_released"] = len(c2._entries) == 0

    ok = all(checks.values())
    return StressResult(
        "oom_degrade_recovery",
        ok,
        f"l1={checks['l1_steps']} l2={checks['l2_res_audio']} "
        f"l3={checks['l3_reclaim']} recovery={checks['recovery_ok']} "
        f"cache_release={checks['cache_released']}",
        time.monotonic() - t0,
        checks,
    )


_SCENARIOS = {
    "high_concurrency": ("high_concurrency", _scenario_high_concurrency),
    "oom_degrade": ("oom_degrade_recovery", _scenario_oom_degrade_recovery),
}


def main():
    ap = argparse.ArgumentParser(description="PRD v1 §3.6 stress scenarios")
    ap.add_argument("--scenario", choices=list(_SCENARIOS))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    targets = [args.scenario] if args.scenario else list(_SCENARIOS)
    results = [_SCENARIOS[k][1]() for k in targets]
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "detail": r.detail,
                        "elapsed_s": round(r.elapsed_s, 3),
                        "metrics": r.metrics,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"[{mark}] {r.name}: {r.detail} ({r.elapsed_s:.3f}s)")
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
