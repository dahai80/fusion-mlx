# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.agents.__init__ registry surface.

The agents subsystem was ported from Rapid-MLX in issue #442 (replacing the
prior stubs). Verify the real contract: get_profile returns an AgentProfile
for a known name (None for unknown), and list_profiles returns a non-empty
list of loaded profiles.
"""

from __future__ import annotations

from fusion_mlx import agents


class TestAgentsRegistry:
    def test_get_profile_returns_profile_for_known_name(self):
        profile = agents.get_profile("codex")
        assert profile is not None
        assert profile.name == "codex"

    def test_get_profile_returns_none_for_unknown_name(self):
        assert agents.get_profile("does-not-exist") is None
        assert agents.get_profile("") is None

    def test_list_profiles_returns_nonempty(self):
        profiles = agents.list_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) >= 1
        # list_profiles is sorted by stars desc — verify the sort invariant.
        stars = [p.stars or 0 for p in profiles]
        assert stars == sorted(stars, reverse=True)

    def test_logger_defined(self):
        assert agents.logger is not None
