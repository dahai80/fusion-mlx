# SPDX-License-Identifier: Apache-2.0
"""NER engine wrapping GLiNER for named entity recognition."""

import asyncio
import gc
import logging
from dataclasses import dataclass, field
from typing import Any

from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class NEROutput:
    entities: list[list[dict[str, Any]]]
    total_tokens: int


class MLXNERModel:
    _NER_ARCHITECTURES = frozenset({
        "SpaModel",
        "GLiNERModel",
    })

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _validate_architecture(self) -> None:
        import json
        from pathlib import Path

        model_path = Path(self._model_name)
        config_path = model_path / "config.json"
        if not config_path.exists():
            return
        try:
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            if not architectures:
                return
            arch = architectures[0]
            if arch not in self._NER_ARCHITECTURES:
                raise ValueError(
                    f"Model architecture '{arch}' is not a supported NER model. "
                    f"Supported: {sorted(self._NER_ARCHITECTURES)}"
                )
        except (json.JSONDecodeError, OSError):
            pass

    def load(self) -> None:
        if self._loaded:
            return
        self._validate_architecture()
        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError(
                "gliner is required for NER. Install with: pip install gliner"
            )
        logger.info("Loading NER model: %s", self._model_name)
        self._model = GLiNER.from_pretrained(self._model_name)
        self._model.eval()
        self._loaded = True
        logger.info("NER model loaded: %s", self._model_name)

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> list[dict[str, Any]]:
        if not self._loaded:
            raise RuntimeError("NER model not loaded. Call load() first.")
        return self._model.predict_entities(
            text,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )

    def batch_predict_entities(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> list[list[dict[str, Any]]]:
        if not self._loaded:
            raise RuntimeError("NER model not loaded. Call load() first.")
        return self._model.batch_predict_entities(
            texts,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXNERModel model={self._model_name} status={status}>"


class NEREngine(BaseNonStreamingEngine):
    _non_streaming_engine = True

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model: MLXNERModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return
        logger.info("Starting NER engine: %s", self._model_name)
        self._model = MLXNERModel(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        loop = asyncio.get_running_loop()
        from ..engine_core import get_executor

        await asyncio.wait_for(
            loop.run_in_executor(get_executor("llm"), self._model.load),
            timeout=120.0,
        )

    async def stop(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from ..engine_core import get_executor
        from ..scheduler.helpers import _safe_clear_cache_for_non_llm

        await asyncio.wait_for(
            loop.run_in_executor(get_executor("llm"), _safe_clear_cache_for_non_llm),
            timeout=5.0,
        )

    async def ner(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> NEROutput:
        if not texts:
            return NEROutput(entities=[], total_tokens=0)
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")
        activity_id = self._begin_activity(
            "ner", detail="NER extraction", total_items=len(texts)
        )
        try:
            loop = asyncio.get_running_loop()
            from ..engine_core import get_executor

            def _ner_sync():
                if len(texts) == 1:
                    result = self._model.predict_entities(
                        texts[0],
                        labels,
                        threshold=threshold,
                        flat_ner=flat_ner,
                        multi_label=multi_label,
                    )
                    return [result]
                return self._model.batch_predict_entities(
                    texts,
                    labels,
                    threshold=threshold,
                    flat_ner=flat_ner,
                    multi_label=multi_label,
                )

            entities = await asyncio.wait_for(
                loop.run_in_executor(get_executor("llm"), _ner_sync),
                timeout=60.0,
            )
            total_tokens = sum(
                len(text.split()) + len(labels) for text in texts
            )
            return NEROutput(entities=entities, total_tokens=total_tokens)
        finally:
            self._end_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None and self._model.loaded,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<NEREngine model={self._model_name} status={status}>"
