# SPDX-License-Identifier: Apache-2.0
import logging

logger = logging.getLogger(__name__)


class IncrementalDecoder:
    __slots__ = ("_tokenizer", "_skip_special_tokens", "_token_ids", "_prev_text")

    def __init__(self, tokenizer, skip_special_tokens: bool = False):
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._token_ids: list[int] = []
        self._prev_text: str = ""

    def add_token(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        full_text = self._tokenizer.decode(
            self._token_ids, skip_special_tokens=self._skip_special_tokens
        )
        delta = full_text[len(self._prev_text) :]
        if "�" in delta:
            return ""
        self._prev_text = full_text
        return delta

    def get_full_text(self) -> str:
        if not self._token_ids:
            return ""
        return self._tokenizer.decode(
            self._token_ids, skip_special_tokens=self._skip_special_tokens
        )

    @property
    def token_ids(self) -> list[int]:
        return self._token_ids

    @property
    def prev_text(self) -> str:
        return self._prev_text

    def reset(self):
        self._token_ids.clear()
        self._prev_text = ""
