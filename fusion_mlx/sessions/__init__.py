# SPDX-License-Identifier: Apache-2.0
"""Per-session token usage tracking (issue #226).

In-memory, process-wide session stats aggregator. No persistence.
"""

from .tracker import (
    SessionStats,
    SessionTracker,
    get_session_tracker,
    record_chat_session,
    reset_session_tracker_for_tests,
    set_session_max_context,
)

__all__ = [
    "SessionStats",
    "SessionTracker",
    "get_session_tracker",
    "record_chat_session",
    "reset_session_tracker_for_tests",
    "set_session_max_context",
]
