import logging
from types import SimpleNamespace

import pytest

from fusion_mlx.request import Request, SamplingParams, queue_wait_seconds
from fusion_mlx.scheduler.sched_response import _log_completion_summary


def _make_request():
    return Request(
        request_id="req-1717",
        prompt="hi",
        sampling_params=SamplingParams(max_tokens=4),
    )


class TestQueueWaitSeconds:
    def test_none_before_insert(self):
        req = _make_request()
        assert queue_wait_seconds(req) is None

    def test_positive_after_admit_time(self):
        req = _make_request()
        req._admit_time = req.arrival_time + 113.0
        # approx: monotonic-clock float addition can lose a ULP at this
        # magnitude (112.99999999999977 on CI).
        assert queue_wait_seconds(req) == pytest.approx(113.0)

    def test_clamps_negative(self):
        req = _make_request()
        req._admit_time = req.arrival_time - 5.0
        assert queue_wait_seconds(req) == 0.0


class TestInsertWaitWarning:
    def test_warning_logged_when_insert_delayed(self, caplog):
        from fusion_mlx import engine_core

        req = _make_request()
        req._admit_time = req.arrival_time + 113.0
        with caplog.at_level(logging.WARNING, logger="fusion_mlx.engine_core"):
            # Reproduce the post-insert check exactly as add_request does.
            qw = engine_core.queue_wait_seconds(req)
            if qw is not None and qw > engine_core._INSERT_QUEUE_WAIT_WARN_S:
                engine_core.logger.warning(
                    "add_request: id=%s queue wait %.1fs before scheduler "
                    "insert (single-worker executor busy with in-flight "
                    "step; #1717)",
                    req.request_id,
                    qw,
                )
        assert "queue wait 113.0s" in caplog.text
        assert "#1717" in caplog.text

    def test_no_warning_when_insert_fast(self):
        from fusion_mlx import engine_core

        req = _make_request()
        req._admit_time = req.arrival_time + 0.01
        qw = engine_core.queue_wait_seconds(req)
        assert qw is not None
        assert qw <= engine_core._INSERT_QUEUE_WAIT_WARN_S


class TestFirstTokenQueueWaitLog:
    def test_threshold_constant_exists(self):
        from fusion_mlx.scheduler import sched_response

        assert sched_response._QUEUE_WAIT_LOG_S == 1.0


class TestCompletionSummaryQueueWait:
    def _summary_line(self, req, caplog):
        with caplog.at_level(
            logging.INFO, logger="fusion_mlx.scheduler.sched_response"
        ):
            _log_completion_summary(
                req,
                SimpleNamespace(
                    prompt_tokens=25,
                    completion_tokens=9,
                    cached_tokens=0,
                    finish_reason="stop",
                ),
                model_name="sft",
            )
        assert caplog.records
        return caplog.records[-1].getMessage()

    def test_queue_wait_surfaced(self, caplog):
        req = _make_request()
        req.first_token_at = req.arrival_time + 120.0
        req.generation_started_at = req.arrival_time + 113.0
        req._admit_time = req.arrival_time + 113.0
        line = self._summary_line(req, caplog)
        assert "queue_wait=113.00s" in line
        assert "ttft=" in line

    def test_queue_wait_zero_when_unknown(self, caplog):
        req = _make_request()
        line = self._summary_line(req, caplog)
        assert "queue_wait=0.00s" in line
