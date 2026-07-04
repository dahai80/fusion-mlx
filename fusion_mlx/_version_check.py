# SPDX-License-Identifier: Apache-2.0
"""Version check stub — no remote version check in this build."""

import logging

logger = logging.getLogger(__name__)


def check_for_update(*args, **kwargs):
    logger.debug("Version check skipped (stub)")


def print_staleness_warning_if_any():
    pass


def prompt_upgrade_if_available():
    pass
