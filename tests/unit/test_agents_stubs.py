# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fusion_mlx.agents adapter + testing surface.

The agents subsystem was ported from Rapid-MLX in issue #442 (replacing the
prior stubs). These tests verify the real public contract: the adapter
functions return non-empty strings and accept an AgentProfile, and
AgentTestRunner constructs against a profile instead of raising
NotImplementedError. Deep behaviour (render_config shape, profile listing)
is covered by test_codex_profile.py.
"""

from __future__ import annotations

import pytest

from fusion_mlx.agents import adapter, get_profile, testing


@pytest.fixture(scope="module")
def codex_profile():
    profile = get_profile("codex")
    assert profile is not None, "codex profile must load (port #442)"
    return profile


class TestAdapterSurface:
    def test_get_adapter_returns_none(self):
        # Compatibility shim — kept None so legacy callers don't crash.
        assert adapter.get_adapter() is None
        assert adapter.get_adapter("any", "args", kw=True) is None

    def test_get_setup_instructions_returns_nonempty(self, codex_profile):
        instructions = adapter.get_setup_instructions(
            codex_profile,
            base_url="http://localhost:8000/v1",
            model_id="qwen3.6-27b-4bit",
        )
        assert isinstance(instructions, str)
        assert instructions != ""
        assert "fusion-mlx" in instructions

    def test_setup_agent_config_env_profile_returns_summary(self):
        # env-type profile (openhands) returns export instructions, not None.
        profile = get_profile("openhands")
        assert profile is not None
        summary = adapter.setup_agent_config(
            profile,
            base_url="http://localhost:8000/v1",
            model_id="qwen3.5-9b-4bit",
        )
        assert isinstance(summary, str)
        assert "export" in summary

    def test_logger_defined(self):
        assert adapter.logger is not None


class TestTestingSurface:
    def test_agent_test_runner_constructs(self, codex_profile):
        # Real runner constructs against a profile (no NotImplementedError).
        runner = testing.AgentTestRunner(
            codex_profile,
            base_url="http://localhost:9999/v1",  # unused; not calling run()
            model_id="qwen3.6-27b-4bit",
        )
        assert runner.profile is codex_profile
        assert runner.model_id == "qwen3.6-27b-4bit"
        # build_test_plan returns a list of test names (server is down so
        # e2e is skipped, but the api-test names must still be present).
        plan = runner.build_test_plan()
        assert isinstance(plan, list)
        assert "plain_chat" in plan

    def test_logger_defined(self):
        assert testing.logger is not None
