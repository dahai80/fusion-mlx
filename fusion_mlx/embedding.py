# SPDX-License-Identifier: Apache-2.0
"""Embedding stub — no embedding model support in this build."""

import logging
import sys

logger = logging.getLogger(__name__)


def require_mlx_embeddings_or_exit():
    logger.error("Embedding models are not available in this build")
    sys.exit(1)
