# SPDX-License-Identifier: Apache-2.0
"""Reranker model re-exports from native implementations."""

import logging

logger = logging.getLogger(__name__)

from .xlm_roberta import Model as XLMRobertaModel  # noqa: F401, E402
from .xlm_roberta import ModelArgs as XLMRobertaModelArgs  # noqa: F401, E402
