# SPDX-License-Identifier: Apache-2.0
"""Reranker model re-exports from native implementations."""

import logging

from .xlm_roberta import Model as XLMRobertaModel
from .xlm_roberta import ModelArgs as XLMRobertaModelArgs

logger = logging.getLogger(__name__)
