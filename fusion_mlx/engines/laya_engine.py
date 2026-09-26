# SPDX-License-Identifier: Apache-2.0
"""Laya typed-decision engine for fusion-mlx.

Wraps the vendored laya_mlx Agent (ModernBERT encoder + RL decision head)
behind the BaseNonStreamingEngine interface so /v1/decisions can load and
serve it like other specialty engines. The Laya checkpoint layout is
non-standard (encoder/config.json + rl_agent_config.json, no top-level
config.json), so discovery is via explicit model id, not the normal
EnginePool HF-config scan.

mlx-serve parity: POST /v1/decisions {model, state, questions} -> answers
with type/confidence/action.act_probability + choice|score|noul.
"""

import logging
import threading
from typing import Any

from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_DEFAULT_LAYA_MODEL = "aac6fef/laya-multilingual-mlx"


class LayaDecisionEngine(BaseNonStreamingEngine):
    _non_streaming_engine = True

    def __init__(self, model_name: str = _DEFAULT_LAYA_MODEL):
        super().__init__()
        self._model_name = model_name
        self._agent: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._agent is not None:
            return
        logger.info("Starting Laya decision engine: %s", self._model_name)

        def _load_sync():
            from .laya import Agent

            return Agent(self._model_name, dtype="float16")

        self._agent = await self._run_via_executor(
            _load_sync, executor_name="llm", timeout=180.0
        )
        logger.info("Laya decision engine loaded: %s", self._model_name)

    async def stop(self) -> None:
        if self._agent is None:
            return
        self._agent = None
        await self._teardown_cache(executor_name="llm", timeout=5.0)

    async def decide(
        self, state: str | dict, questions: dict[str, dict]
    ) -> dict[str, Any]:
        if self._agent is None:
            raise RuntimeError("Laya engine not started. Call start() first.")
        activity_id = self._begin_activity(
            "decide", detail="Laya typed decision", total_items=len(questions)
        )
        try:

            def _predict_sync():
                return self._agent.predict(state, questions)

            result = await self._run_via_executor(
                _predict_sync, executor_name="llm", timeout=60.0
            )
            return result
        finally:
            self._end_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._agent is not None,
        }

    def __repr__(self) -> str:
        status = "running" if self._agent is not None else "stopped"
        return f"<LayaDecisionEngine model={self._model_name} status={status}>"
