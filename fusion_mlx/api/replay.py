# SPDX-License-Identifier: Apache-2.0
"""Deterministic replay provider (#1518) — free, zero-GPU model for
pipeline E2E structural runs.

Modeled after the t3code "Replay (dev)" provider: the ``replay-dev``
model id answers every request with deterministic output — the last
user message is echoed verbatim, and when the last user message
contains the ``[REPLAY_JSON]`` marker the reply is a canned
schema-shaped JSON object instead. No engine, no weights, no GPU; the
request never reaches the executor.

The catalog marks it ``capabilities=["test"]`` so chat-default pickers
(cowork#131-style tier filters) never select it in production paths,
and ``resolve_default_chat_model`` can never land on it (it resolves
from the alias/registry config, where replay-dev is not registered).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

REPLAY_MODEL_ID = "replay-dev"
REPLAY_MODEL_ALIASES = {"replay", "replay-dev", "replay_dev"}
REPLAY_JSON_MARKER = "[REPLAY_JSON]"


def is_replay_model(model: str | None) -> bool:
    normalized = (model or "").strip().lower()
    return normalized in REPLAY_MODEL_ALIASES


def _last_user_text(messages: list[Any]) -> str:
    for msg in reversed(messages or []):
        role = getattr(msg, "role", None)
        if role != "user":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        # Content-part list — concatenate text parts.
        parts = []
        for part in content or []:
            text = getattr(part, "text", None) or (
                part.get("text") if isinstance(part, dict) else None
            )
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


def replay_content(messages: list[Any]) -> str:
    # Deterministic reply: JSON marker → canned schema-shaped object;
    # otherwise verbatim echo of the last user message.
    last_text = _last_user_text(messages)
    if REPLAY_JSON_MARKER in last_text:
        payload = {
            "echo": last_text.replace(REPLAY_JSON_MARKER, "").strip(),
            "message_count": len(messages or []),
            "replay": True,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return last_text
