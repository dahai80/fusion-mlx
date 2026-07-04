# SPDX-License-Identifier: Apache-2.0
"""Embedding stub — no embedding model support in this build."""

import logging
import sys

logger = logging.getLogger(__name__)


def require_mlx_embeddings_or_exit():
    print("Embedding models are not available in this build", file=sys.stderr)
    sys.exit(1)
