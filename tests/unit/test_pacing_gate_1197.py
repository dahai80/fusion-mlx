# SPDX-License-Identifier: Apache-2.0
"""#1197 pacing gate: token bucket + AIMD controller + wiring contracts."""

import asyncio
import time

import pytest

from fusion_mlx import pacing
from fusion_mlx.pacing import (
    get_pacing_gate,
    pace_acquire_async,
    reset_pacing_gate,
)


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    reset_pacing_gate()
    for cls in pacing.CLASSES:
        monkeypatch.delenv(f"FUSION_MLX_PACE_{cls.upper()}_RATE", raising=False)
        monkeypatch.delenv(f"FUSION_MLX_PACE_{cls.upper()}_BURST", raising=False)
    monkeypatch.delenv("FUSION_MLX_PACING", raising=False)
    yield
    reset_pacing_gate()


def test_bucket_burst_then_throttle():
    gate = get_pacing_gate()
    bucket = gate.bucket("image")
    bucket.rate = 0.0001  # effectively no refill
    bucket.burst = 1.0
    bucket.tokens = 1.0
    assert gate.try_acquire_counted("image") is True  # burst token
    assert gate.try_acquire_counted("image") is False  # bucket dry


def test_bucket_refills_over_time():
    gate = get_pacing_gate()
    gate.bucket("laya").rate = 200.0
    gate.bucket("laya").tokens = 0.0
    assert gate.try_acquire_counted("laya") is False
    time.sleep(0.02)  # 200/s -> 4 tokens
    assert gate.try_acquire_counted("laya") is True


def test_disabled_gate_always_admits(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_PACING", "off")
    reset_pacing_gate()
    gate = get_pacing_gate()
    assert gate.enabled() is False
    gate.bucket("image").rate = 0.0001
    for _ in range(10):
        assert gate.try_acquire_counted("image") is True


def test_env_rate_override(monkeypatch):
    monkeypatch.setenv("FUSION_MLX_PACE_IMAGE_RATE", "0.0001")
    reset_pacing_gate()
    gate = get_pacing_gate()
    assert gate.bucket("image").rate == pytest.approx(0.0001)


def test_acquire_bounded_wait_recovers():
    gate = get_pacing_gate()
    gate.bucket("video").rate = 50.0
    gate.bucket("video").tokens = 0.0
    t0 = time.monotonic()
    assert gate.acquire("video", timeout_s=0.1) is True
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0  # waited for refill, not forever


def test_acquire_timeout_counts_throttled():
    gate = get_pacing_gate()
    gate.bucket("image").rate = 0.0001
    gate.bucket("image").tokens = 0.0
    assert gate.acquire("image", timeout_s=0.05) is False
    snap = gate.snapshot()["classes"]["image"]
    assert snap["throttled"] >= 1


def test_async_acquire_never_blocks_loop_long():
    gate = get_pacing_gate()
    gate.bucket("llm").rate = 30.0
    gate.bucket("llm").tokens = 0.0

    async def main():
        t0 = asyncio.get_running_loop().time()
        ok = await pace_acquire_async("llm", timeout_s=0.08)
        return ok, asyncio.get_running_loop().time() - t0

    ok, elapsed = asyncio.run(main())
    assert ok is True  # 30/s refills within 80ms
    assert elapsed < 1.0


def test_aimd_decrease_on_sustained_bad_signal():
    gate = get_pacing_gate()
    initial = gate.bucket("image").rate
    # llm reports bad quality: image is an aggressor, must shrink after
    # the hysteresis streak; llm's own bucket untouched.
    for _ in range(pacing._BAD_STREAK_THRESHOLD):
        gate.update_signal("llm", 0.5)
    assert gate.bucket("image").rate < initial
    # victim untouched (llm bucket rate == its initial)
    assert gate.bucket("llm").rate == pytest.approx(
        gate.snapshot()["classes"]["llm"]["initial_rate"]
    )


def test_aimd_single_noise_sample_does_not_throttle():
    gate = get_pacing_gate()
    initial = gate.bucket("video").rate
    gate.update_signal("llm", 0.5)  # one bad sample only
    gate.update_signal("llm", 1.0)  # immediately healthy again
    assert gate.bucket("video").rate == pytest.approx(initial)


def test_aimd_floor_respected():
    gate = get_pacing_gate()
    floor = pacing._FLOOR_FRACTION * gate.snapshot()["classes"]["image"]["initial_rate"]
    for _ in range(50):
        gate.update_signal("llm", 0.5)
        # bypass cooldown for the test by resetting the timer
        gate._last_adjust = 0.0
    assert gate.bucket("image").rate >= floor * 0.999


def test_aimd_increase_on_recovery():
    gate = get_pacing_gate()
    initial = gate.snapshot()["classes"]["image"]["initial_rate"]
    for _ in range(5):
        gate.update_signal("llm", 0.5)
        gate._last_adjust = 0.0
    shrunk = gate.bucket("image").rate
    assert shrunk < initial
    gate._bad_streak["llm"] = 0
    gate._last_adjust = 0.0
    gate.update_signal("llm", 1.0)
    assert gate.bucket("image").rate > shrunk


def test_aimd_convergence_no_oscillation():
    # Square-wave signal: bad for 3 samples, good for 3, repeated. The
    # controller must converge DOWNWARD (protect the victim) and never
    # oscillate above the initial budget.
    gate = get_pacing_gate()
    initial = gate.snapshot()["classes"]["image"]["initial_rate"]
    rates = []
    for cycle in range(4):
        for _ in range(3):
            gate.update_signal("llm", 0.5)
            gate._last_adjust = 0.0
        for _ in range(3):
            gate.update_signal("llm", 1.0)
            gate._last_adjust = 0.0
        rates.append(gate.bucket("image").rate)
    assert all(r <= initial + 1e-9 for r in rates)
    assert rates[-1] <= rates[0]  # net downward under a 50% duty wave


def test_snapshot_shape():
    snap = get_pacing_gate().snapshot()
    assert snap["enabled"] is True
    for cls in ("llm", "image", "video", "laya"):
        info = snap["classes"][cls]
        for key in ("rate_per_s", "burst", "tokens", "submissions", "throttled"):
            assert key in info


def test_metrics_render():
    from fusion_mlx.routes_internal.metrics import render_prometheus_metrics

    gate = get_pacing_gate()
    gate.try_acquire_counted("laya")
    out = render_prometheus_metrics()
    assert 'pacing_submissions_total{class="laya"}' in out


def test_llm_loop_gate_wired():
    # Source-level: the engine loop must gate BEFORE arming the step
    # deadline (a deferred submission must never leave a stale armed
    # deadline — watchdog false-poison hazard).
    import inspect

    from fusion_mlx import engine_core

    src = inspect.getsource(engine_core)
    gate_idx = src.index('pace_acquire_async("llm"')
    arm_idx = src.index("_llm_step_deadline = time.monotonic() + step_window")
    assert gate_idx < arm_idx, "llm pacing gate must precede deadline arming"


def test_throttle_surfaces_explicit_error():
    # Wiring contract: a throttled route submission raises an explicit
    # error (503 via HTTPException for laya, RuntimeError for engines) —
    # never a silent drop.
    import inspect

    from fusion_mlx.api import decisions_routes
    from fusion_mlx.engines import image_gen, video

    laya_src = inspect.getsource(decisions_routes)
    assert 'pace_acquire_async("laya"' in laya_src
    assert "503" in laya_src
    img_src = inspect.getsource(image_gen)
    assert 'pace_acquire_async("image"' in img_src
    vid_src = inspect.getsource(video)
    assert 'pace_acquire_async("video"' in vid_src


def test_spike_all_submissions_outcome():
    # DoD chaos shape (spike): 60 concurrent paced submissions on a small
    # bucket — every one either admitted or counted-throttled (explicit
    # outcome), none hang.
    gate = get_pacing_gate()
    gate.bucket("image").rate = 5.0
    gate.bucket("image").burst = 3.0
    gate.bucket("image").tokens = 3.0

    async def main():
        async def one():
            return await pace_acquire_async("image", timeout_s=0.05)

        return await asyncio.wait_for(
            asyncio.gather(*[one() for _ in range(60)]), timeout=10.0
        )

    results = asyncio.run(main())
    assert len(results) == 60
    admitted = sum(1 for r in results if r)
    snap = gate.snapshot()["classes"]["image"]
    assert admitted + snap["throttled"] >= 60
    assert admitted >= 3  # burst admits at minimum


def test_sustained_mixed_load_rates_visible():
    # DoD chaos shape (sustained mixed load): llm+image+video submissions
    # for a sustained window — all classes show non-zero submissions in
    # the snapshot, none silently starved to zero admissions.
    gate = get_pacing_gate()

    async def main():
        async def cls_loop(cls, n):
            ok = 0
            for _ in range(n):
                if await pace_acquire_async(cls, timeout_s=0.01):
                    ok += 1
                await asyncio.sleep(0.001)
            return ok

        return await asyncio.wait_for(
            asyncio.gather(
                cls_loop("llm", 30),
                cls_loop("image", 10),
                cls_loop("video", 5),
            ),
            timeout=15.0,
        )

    llm_ok, img_ok, vid_ok = asyncio.run(main())
    # None silently starved: every class keeps a healthy admission share
    # under the sustained mixed load (image rate 8/s admits its share of
    # a ~1s window; exact split is timing-dependent, the floor is not).
    assert llm_ok >= 25
    assert img_ok >= 4
    assert vid_ok >= 2
    snap = gate.snapshot()["classes"]
    assert snap["llm"]["submissions"] >= 25
    assert snap["image"]["submissions"] >= 4
    assert snap["video"]["submissions"] >= 2
