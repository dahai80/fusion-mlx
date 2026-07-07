# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.agents.__init__ stub."""

from __future__ import annotations

from fusion_mlx import agents


class TestAgentsStub:
    def test_get_profile_returns_none(self):
        assert agents.get_profile("codex") is None
        assert agents.get_profile("") is None

    def test_list_profiles_returns_empty(self):
        assert agents.list_profiles() == []

    def test_logger_defined(self):
        assert agents.logger is not None
