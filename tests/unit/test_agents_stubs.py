# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.agents stubs (adapter.py + testing.py).

Both modules are intentional stubs ("not available in this build") — verify
the stub contract: adapter functions return None/""/noop, testing functions
raise NotImplementedError. Aims at 100% line coverage of both stub files.
"""

from __future__ import annotations

import pytest

from fusion_mlx.agents import adapter, testing


class TestAdapterStub:
    def test_get_adapter_returns_none(self):
        assert adapter.get_adapter() is None
        assert adapter.get_adapter("any", "args", kw=True) is None

    def test_get_setup_instructions_returns_empty(self):
        assert adapter.get_setup_instructions("codex") == ""
        assert adapter.get_setup_instructions("") == ""

    def test_setup_agent_config_noop(self):
        # returns None (noop); should not raise
        assert adapter.setup_agent_config("codex") is None
        adapter.setup_agent_config("claude", model="x", priority=5)

    def test_logger_defined(self):
        assert adapter.logger is not None


class TestTestingStub:
    def test_run_agent_test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not available"):
            testing.run_agent_test()
        with pytest.raises(NotImplementedError):
            testing.run_agent_test("arg", kw="val")

    def test_agent_test_runner_init_raises(self):
        with pytest.raises(NotImplementedError, match="not available"):
            testing.AgentTestRunner()
        with pytest.raises(NotImplementedError):
            testing.AgentTestRunner("arg", kw="val")

    def test_logger_defined(self):
        assert testing.logger is not None
