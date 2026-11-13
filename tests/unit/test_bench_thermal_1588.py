import asyncio

import pytest

from fusion_mlx.bench.thermal import (
    THERMAL_IDLE_RECOMMENDED_S,
    ThermalTracker,
    record_idle,
)


class TestRecordIdle:
    def test_zero_is_noop(self):
        assert asyncio.run(record_idle(0)) == 0.0

    def test_negative_is_noop(self):
        assert asyncio.run(record_idle(-5)) == 0.0

    def test_records_elapsed_and_logs(self):
        lines = []
        elapsed = asyncio.run(record_idle(0.05, log=lines.append))
        assert elapsed >= 0.05
        assert any("idling GPU" in ln for ln in lines)
        assert any("idle done" in ln for ln in lines)


def test_recommended_idle_is_ten_minutes():
    assert THERMAL_IDLE_RECOMMENDED_S == 600


class TestThermalTracker:
    def test_summary_empty_when_no_samples(self):
        t = ThermalTracker(lambda: None)
        s = t.summary()
        assert s["idle_before_s"] == 0.0
        assert s["first_window_ms_per_step"] is None
        assert s["drift_ratio"] is None

    def test_summary_reflects_samples(self):
        values = iter([7.2, 7.2, 14.5])

        def get():
            try:
                return next(values)
            except StopIteration:
                return 14.5

        t = ThermalTracker(get)
        t._samples = [7.2, 7.2, 14.5]
        s = t.summary()
        assert s["first_window_ms_per_step"] == 7.2
        assert s["last_window_ms_per_step"] == 14.5
        assert s["drift_ratio"] == round(14.5 / 7.2, 3)

    def test_note_idle_surfaces_in_summary(self):
        t = ThermalTracker(lambda: None)
        t.note_idle(600.0)
        assert t.summary()["idle_before_s"] == 600.0

    def test_sampler_survives_reader_exception(self):
        def boom():
            raise RuntimeError("scheduler gone")

        t = ThermalTracker(boom, interval_s=0.2)

        async def bench():
            await asyncio.sleep(0.5)
            return "ok"

        result, summary = asyncio.run(_run(t, bench))
        assert result == "ok"
        assert summary["first_window_ms_per_step"] is None

    def test_run_with_sampling_collects_during_bench(self):
        seq = []

        def get():
            seq.append(7.0)
            return 7.0

        t = ThermalTracker(get, interval_s=0.2)

        async def bench():
            await asyncio.sleep(0.5)
            return 42

        result, summary = asyncio.run(_run(t, bench))
        assert result == 42
        assert summary["first_window_ms_per_step"] == 7.0
        assert len(seq) > 0

    def test_format_summary_contains_fields(self):
        t = ThermalTracker(lambda: None)
        t._samples = [7.2, 14.5]
        t.note_idle(600)
        text = t.format_summary()
        assert "idle_before=600s" in text
        assert "drift_ratio=" in text


async def _run(tracker, bench_coro_factory):
    result = await tracker.run_with_sampling(bench_coro_factory())
    return result, tracker.summary()


@pytest.mark.parametrize("first,last", [(0.0, 10.0), (7.2, 0.0)])
def test_summary_guard_division(first, last):
    t = ThermalTracker(lambda: None)
    t._samples = [first, last] if first > 0 else []
    s = t.summary()
    if first <= 0:
        assert s["drift_ratio"] is None
