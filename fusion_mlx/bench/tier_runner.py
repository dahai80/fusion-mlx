# SPDX-License-Identifier: Apache-2.0
"""Bench tier runner stub — not available in this build."""

import logging

logger = logging.getLogger(__name__)


def run_tier(*args, **kwargs):
    logger.warning("Bench tier runner not available (stub)")
    raise NotImplementedError("Bench tier runner not available in this build")
