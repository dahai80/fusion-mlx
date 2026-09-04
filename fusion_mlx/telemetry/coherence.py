# SPDX-License-Identifier: Apache-2.0
import logging

logger = logging.getLogger(__name__)

_GARBLE_REPEAT_RATIO = 0.7


def is_empty(completion_tokens: int) -> bool:
    return int(completion_tokens) <= 0


def is_abnormally_short(text: str, completion_tokens: int, threshold: int = 3) -> bool:
    if is_empty(completion_tokens):
        return False
    return int(completion_tokens) <= threshold


def looks_like_garbage(text: str, completion_tokens: int) -> bool:
    if is_empty(completion_tokens):
        return False
    if not text or not text.strip():
        logger.debug(
            "coherence: empty text with tokens=%d -> garbage", completion_tokens
        )
        return True
    if len(text) < 4:
        return True
    most_common_char = max(set(text), key=text.count)
    ratio = text.count(most_common_char) / len(text)
    if ratio >= _GARBLE_REPEAT_RATIO:
        logger.debug(
            "coherence: repetition ratio %.2f >= %.2f -> garbage",
            ratio,
            _GARBLE_REPEAT_RATIO,
        )
        return True
    return False
