# SPDX-License-Identifier: Apache-2.0
"""Wire-level constants shared across api / service layers."""

REASONING_CUTOFF_SENTINEL = "[truncated — reasoning incomplete; raise max_tokens]"
RESCUE_TAIL_LENGTH = 200


def is_rescue_payload(content: str | None) -> bool:
    if not content:
        return False
    if content == REASONING_CUTOFF_SENTINEL:
        return True
    prefix = REASONING_CUTOFF_SENTINEL + "\n\n"
    return content.startswith(prefix) and len(content) > len(prefix)
