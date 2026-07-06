# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #974 -- ``scripts/release_check_m3.sh``
must thread ``$PORT`` into ``FUSION_MLX_BASE_URL`` (and OpenAI-SDK
conventional siblings) so G7 SDK integration tests hit the gauntlet
server, not whatever default port their env-var defaults resolve to.

Migrated from Rapid-MLX. The ``scripts/release_check_m3.sh`` script
has NOT been migrated to fusion-mlx. All tests are skipped with a
clear reason.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

_SKIP_REASON = (
    "scripts/release_check_m3.sh has not been migrated to fusion-mlx. "
    "Re-enable when the release check script is added with FUSION_MLX_BASE_URL "
    "port threading."
)


@pytest.mark.skip(reason=_SKIP_REASON)
def test_prelude_exports_base_url_from_port():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_prelude_default_port_matches_hardcoded_probes():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_script_asserts_g7_env_matches_port():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_every_integration_base_url_env_is_covered():
    pass
