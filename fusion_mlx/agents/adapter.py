# SPDX-License-Identifier: Apache-2.0
"""Agent adapter stub — not available in this build."""

import logging

logger = logging.getLogger(__name__)


def get_adapter(*args, **kwargs):
    return None


def get_setup_instructions(agent_name: str) -> str:
    return ""


def setup_agent_config(agent_name: str, **kwargs) -> None:
    pass
