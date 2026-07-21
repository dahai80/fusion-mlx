# SPDX-License-Identifier: Apache-2.0
"""Model registry: tracks loaded ModelEntry instances for /v1/models listing
and effective-parser surfacing (CLI override / auto-detect live state)."""

import logging

logger = logging.getLogger(__name__)


class ModelEntry:
    def __init__(
        self,
        *,
        engine,
        model_name,
        model_path=None,
        aliases=None,
        tool_call_parser=None,
        reasoning_parser=None,
        is_mllm=False,
        max_tokens=4096,
    ):
        self.engine = engine
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.aliases = set(aliases or [])
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.is_mllm = is_mllm
        self.max_tokens = max_tokens

    def matches(self, name):
        # Strict-bool contract: returns True/False only. Callers gate with
        # ``is True`` to reject truthy-non-bool returns (codex r2 blocker).
        if name == self.model_name:
            return True
        if name in self.aliases:
            return True
        return False


class ModelRegistry:
    def __init__(self, **kwargs):
        self._entries: list[ModelEntry] = []
        self._default: ModelEntry | None = None

    def add(self, entry, is_default=False):
        self._entries.append(entry)
        if is_default:
            self._default = entry
        logger.info(
            "model_registry: added entry model=%s default=%s total=%d",
            entry.model_name,
            is_default,
            len(self._entries),
        )

    def get_entry(self, name):
        for entry in self._entries:
            if entry.matches(name) is True:
                return entry
        return self._default

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)
