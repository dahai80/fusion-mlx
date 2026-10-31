# SPDX-License-Identifier: Apache-2.0
"""#1468: embedding/reranker must run on the dedicated "embed" executor.

A wedged llm-side eval (stuck QuantizedMatmul on the single-worker llm
pool) previously starved every embedding until a restart — load, teardown,
embed, and even the post-embed activity cleanup all queued behind it.
"""

import asyncio
import concurrent.futures

from fusion_mlx.engine.base import BaseNonStreamingEngine
from fusion_mlx.engine_core import _executor_config
from fusion_mlx.engines.embedding import EmbeddingEngine
from fusion_mlx.engines.reranker import RerankerEngine


class _RecordingExecutor:
    def __init__(self):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.calls = []

    def get(self, name):
        self.calls.append(name)
        return self._pool

    def shutdown(self):
        self._pool.shutdown(wait=True)


def test_embed_executor_registered():
    assert "embed" in _executor_config
    cfg = _executor_config["embed"]
    assert cfg["max_workers"] == 1
    assert cfg["prefix"] == "mlx-embed"


def test_embedding_engine_uses_embed_pool():
    assert EmbeddingEngine._mlx_executor_name == "embed"
    rec = _RecordingExecutor()
    engine = EmbeddingEngine("stub-embed")

    async def _run():
        engine._model = _StubEmbedModel()
        await engine.start()
        await engine.embed(["hello"])
        await engine.stop()
        return list(rec.calls)

    import fusion_mlx.engines.embedding as emb_mod

    orig = emb_mod.get_executor
    emb_mod.get_executor = rec.get
    try:
        calls = asyncio.run(_run())
    finally:
        emb_mod.get_executor = orig
        rec.shutdown()
    assert calls, "expected executor calls"
    assert all(name == "embed" for name in calls), calls


def test_reranker_engine_uses_embed_pool():
    assert RerankerEngine._mlx_executor_name == "embed"
    rec = _RecordingExecutor()
    engine = RerankerEngine("stub-rerank")

    async def _run():
        engine._model = _StubRerankModel()
        await engine.start()
        await engine.rerank("q", ["d1", "d2"])
        await engine.stop()
        return list(rec.calls)

    import fusion_mlx.engines.reranker as rer_mod

    orig = rer_mod.get_executor
    rer_mod.get_executor = rec.get
    try:
        calls = asyncio.run(_run())
    finally:
        rer_mod.get_executor = orig
        rec.shutdown()
    assert calls, "expected executor calls"
    assert all(name == "embed" for name in calls), calls


def test_finish_activity_honors_executor_name():

    class _EmbedOnEmbedPool(BaseNonStreamingEngine):
        _mlx_executor_name = "embed"

        @property
        def model_name(self):
            return "stub"

        async def start(self):
            pass

        async def stop(self):
            pass

    rec = _RecordingExecutor()
    engine = _EmbedOnEmbedPool()

    async def _run():
        activity_id = engine._begin_activity("embedding", detail="t")
        await engine._finish_activity(activity_id)
        return list(rec.calls)

    import fusion_mlx.engine.base as base_mod

    orig = base_mod.get_executor
    base_mod.get_executor = rec.get
    try:
        calls = asyncio.run(_run())
    finally:
        base_mod.get_executor = orig
        rec.shutdown()
    assert calls == ["embed"], calls


def test_finish_activity_default_pool_unchanged():
    class _DefaultEngine(BaseNonStreamingEngine):
        @property
        def model_name(self):
            return "stub"

        async def start(self):
            pass

        async def stop(self):
            pass

    rec = _RecordingExecutor()
    engine = _DefaultEngine()

    async def _run():
        activity_id = engine._begin_activity("embedding", detail="t")
        await engine._finish_activity(activity_id)
        return list(rec.calls)

    import fusion_mlx.engine.base as base_mod

    orig = base_mod.get_executor
    base_mod.get_executor = rec.get
    try:
        calls = asyncio.run(_run())
    finally:
        base_mod.get_executor = orig
        rec.shutdown()
    assert calls == ["llm"], calls


class _StubEmbedModel:
    class _Inner:
        def __init__(self):
            self.loaded = True

        def embed(self, inputs, max_length=512, padding=True, truncation=True):
            from fusion_mlx.engines.embedding import EmbeddingOutput

            n = len(inputs) if isinstance(inputs, list) else 1
            return EmbeddingOutput(
                embeddings=[[0.0, 1.0] for _ in range(n)],
                total_tokens=2 * n,
                dimensions=2,
            )

    def __init__(self):
        self._model = self._Inner()
        self.loaded = False

    def load(self):
        self.loaded = True

    def embed(self, inputs, max_length=512, padding=True, truncation=True):
        return self._model.embed(inputs, max_length, padding, truncation)


class _StubRerankModel:
    def __init__(self, *args, **kwargs):
        self.loaded = False

    def load(self):
        self.loaded = True

    def rerank(self, query, documents, max_length=None):
        from fusion_mlx.engines.reranker import RerankOutput

        return RerankOutput(
            scores=[0.5] * len(documents),
            indices=list(range(len(documents))),
            total_tokens=4,
        )
