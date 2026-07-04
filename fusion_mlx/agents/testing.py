# SPDX-License-Identifier: Apache-2.0
"""Agent testing stub — not available in this build."""

import logging

logger = logging.getLogger(__name__)


def run_agent_test(*args, **kwargs):
    raise NotImplementedError("Agent testing not available in this build")


class AgentTestRunner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Agent testing not available in this build")
