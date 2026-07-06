# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    type: str
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list | None = None
    finish_reason: str | None = None
    tool_calls_detected: bool = False
    metadata: dict = field(default_factory=dict)
