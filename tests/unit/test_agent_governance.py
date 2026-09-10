# SPDX-License-Identifier: Apache-2.0
"""G4 Agent governance unit tests."""

import pytest

from fusion_mlx.agents.governance import (
    AgentGovernor,
    GovernorLimitExceeded,
    RunStatus,
    get_governor,
)


@pytest.fixture
def gov():
    return AgentGovernor(max_steps=3, timeout_s=1.0, token_budget=100)


class TestStartAndFinish:
    def test_start_run_returns_running(self, gov):
        run = gov.start_run("g1")
        assert run.graph_id == "g1"
        assert run.status == RunStatus.RUNNING
        assert run.steps == 0

    def test_finish_sets_completed(self, gov):
        run = gov.start_run("g1")
        gov.finish(run)
        assert run.status == RunStatus.COMPLETED

    def test_finish_with_error_sets_failed(self, gov):
        run = gov.start_run("g1")
        gov.finish(run, error="boom")
        assert run.status == RunStatus.FAILED
        assert run.error == "boom"

    def test_get_run(self, gov):
        run = gov.start_run("g1")
        assert gov.get_run(run.run_id) is run
        assert gov.get_run("nope") is None


class TestStepCap:
    def test_check_step_ok_under_limit(self, gov):
        run = gov.start_run("g1")
        run.steps = 2
        gov.check_step(run)

    def test_check_step_exceeds(self, gov):
        run = gov.start_run("g1")
        run.steps = 3
        with pytest.raises(GovernorLimitExceeded) as exc:
            gov.check_step(run)
        assert exc.value.status == RunStatus.STEPS_EXCEEDED

    def test_zero_max_steps_disables(self):
        gov = AgentGovernor(max_steps=0, timeout_s=0, token_budget=0)
        run = gov.start_run("g1")
        run.steps = 9999
        gov.check_step(run)


class TestTimeout:
    def test_timeout_triggers(self, gov):
        import time

        run = gov.start_run("g1")
        run.started_at = time.time() - 2.0
        with pytest.raises(GovernorLimitExceeded) as exc:
            gov.check_step(run)
        assert exc.value.status == RunStatus.TIMEOUT


class TestTokenBudget:
    def test_record_usage_accumulates(self, gov):
        run = gov.start_run("g1")
        gov.record_usage(run, {"total_tokens": 30})
        assert run.steps == 1
        assert run.tokens_used == 30

    def test_record_usage_exceeds_budget(self, gov):
        run = gov.start_run("g1")
        gov.record_usage(run, {"total_tokens": 50})
        with pytest.raises(GovernorLimitExceeded) as exc:
            gov.record_usage(run, {"total_tokens": 60})
        assert exc.value.status == RunStatus.TOKEN_BUDGET_EXCEEDED
        assert run.tokens_used == 110

    def test_record_usage_sums_prompt_completion(self, gov):
        run = gov.start_run("g1")
        gov.record_usage(run, {"prompt_tokens": 10, "completion_tokens": 20})
        assert run.tokens_used == 30

    def test_zero_budget_disables(self):
        gov = AgentGovernor(max_steps=0, timeout_s=0, token_budget=0)
        run = gov.start_run("g1")
        gov.record_usage(run, {"total_tokens": 999999})
        assert run.status == RunStatus.RUNNING


class TestKillSwitch:
    def test_cancel_sets_cancelled(self, gov):
        run = gov.start_run("g1")
        assert gov.cancel(run.run_id) is True
        assert run.cancel_requested is True
        assert run.status == RunStatus.CANCELLED

    def test_cancel_nonexistent(self, gov):
        assert gov.cancel("nope") is False

    def test_cancel_finished_run_fails(self, gov):
        run = gov.start_run("g1")
        gov.finish(run)
        assert gov.cancel(run.run_id) is False

    def test_check_step_raises_on_cancel(self, gov):
        run = gov.start_run("g1")
        gov.cancel(run.run_id)
        with pytest.raises(GovernorLimitExceeded) as exc:
            gov.check_step(run)
        assert exc.value.status == RunStatus.CANCELLED


class TestListAndPurge:
    def test_list_runs(self, gov):
        gov.start_run("g1")
        gov.start_run("g2")
        runs = gov.list_runs()
        assert len(runs) == 2

    def test_purge_removes_old_finished(self, gov):
        import time

        run = gov.start_run("g1")
        gov.finish(run)
        run.started_at = time.time() - 7200
        purged = gov.purge(max_age_s=3600)
        assert purged == 1

    def test_purge_keeps_running(self, gov):
        import time

        run = gov.start_run("g1")
        run.started_at = time.time() - 7200
        purged = gov.purge(max_age_s=3600)
        assert purged == 0


class TestSingleton:
    def test_get_governor_returns_same(self):
        g1 = get_governor()
        g2 = get_governor()
        assert g1 is g2
