# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from fusion_mlx.cache.probe_core import probe_prompt_cache
from fusion_mlx.harness_bridge import PrefixCacheAdvisor, RemoteTokenCounter

MODEL_ID = "test-model"
MODEL_NAME = "/models/test-model"


# ---------------------------------------------------------------------------
# Radix trie fakes for probe_core
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(self, key=None, block_id=None):
        self.key = key
        self.block_id = block_id
        self.children = {}


class _FakeBlock:
    def __init__(self, block_hash):
        self.block_hash = block_hash


def _make_radix_cache(token_ids, block_size, cached_blocks):
    # Build a trie carrying the first `cached_blocks` full blocks of
    # token_ids, mirroring RadixPrefixCache's keying: children are stored
    # under the block's first token id and hold the full block tuple.
    root = _FakeNode()
    paged = SimpleNamespace(allocated_blocks={})
    node = root
    for i in range(cached_blocks):
        start = i * block_size
        key = tuple(token_ids[start : start + block_size])
        child = _FakeNode(key=key, block_id=100 + i)
        node.children[key[0]] = child
        paged.allocated_blocks[100 + i] = _FakeBlock(block_hash=bytes([i]))
        node = child

    def block_slices(tokens):
        n = len(tokens)
        for start in range(0, n - n % block_size, block_size):
            yield start, start + block_size, tuple(tokens[start : start + block_size])

    return SimpleNamespace(
        _root=root,
        _block_slices=block_slices,
        paged_cache=paged,
        block_size=block_size,
    )


def _make_pool(tokenizer, scheduler):
    entry = SimpleNamespace(
        engine=SimpleNamespace(
            _tokenizer=tokenizer,
            _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler)),
            _apply_chat_template=lambda msgs, tools, **kw: "rendered prompt",
        )
    )
    return SimpleNamespace(_entries={MODEL_ID: entry})


def _probe(token_ids, block_size, cached_blocks, total_override=None):
    tokenizer = SimpleNamespace(encode=lambda text: list(token_ids))
    scheduler = SimpleNamespace(
        block_aware_cache=_make_radix_cache(token_ids, block_size, cached_blocks),
        paged_ssd_cache_manager=None,
        paged_cache_manager=SimpleNamespace(model_name=MODEL_NAME),
        config=SimpleNamespace(paged_cache_block_size=block_size),
    )
    pool = _make_pool(tokenizer, scheduler)
    return probe_prompt_cache(pool, MODEL_ID, [{"role": "user", "content": "hello"}])


class TestProbeCoreRadixWalk:
    def test_full_ram_match(self):
        result = _probe(list(range(8)), 4, cached_blocks=2)
        assert result["blocks_ram"] == 2
        assert result["ram_hit_tokens"] == 8
        assert result["cached_prefix_tokens"] == 8
        assert result["cache_index"] == "radix"

    def test_partial_ram_match_stops_at_first_miss(self):
        result = _probe(list(range(12)), 4, cached_blocks=1)
        assert result["blocks_ram"] == 1
        assert result["cached_prefix_tokens"] == 4

    def test_no_ram_cache_is_all_cold(self):
        result = _probe(list(range(8)), 4, cached_blocks=0)
        assert result["blocks_ram"] == 0
        assert result["cached_prefix_tokens"] == 0
        assert result["blocks_cold"] == 2

    def test_partial_last_block_counts_as_cold(self):
        # 10 tokens / block_size 4 -> 2 full + 1 partial; only full blocks
        # can live in the trie, so the partial block is always cold.
        result = _probe(list(range(10)), 4, cached_blocks=2)
        assert result["total_blocks"] == 3
        assert result["blocks_ram"] == 2
        assert result["blocks_cold"] == 1
        assert result["cached_prefix_tokens"] == 8


class TestProbeCoreErrors:
    def test_unknown_model_raises_404(self):
        pool = SimpleNamespace(_entries={})
        with pytest.raises(HTTPException) as exc_info:
            probe_prompt_cache(pool, MODEL_ID, [{"role": "user", "content": "x"}])
        assert exc_info.value.status_code == 404

    def test_unloaded_model_reports_not_loaded(self):
        pool = SimpleNamespace(_entries={MODEL_ID: SimpleNamespace(engine=None)})
        result = probe_prompt_cache(pool, MODEL_ID, [{"role": "user", "content": "x"}])
        assert result["model_loaded"] is False


# ---------------------------------------------------------------------------
# harness routes
# ---------------------------------------------------------------------------


def _make_app():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fusion_mlx.api import harness_routes
    from fusion_mlx.middleware.auth import verify_api_key

    app = FastAPI()
    app.include_router(harness_routes.router)
    app.dependency_overrides[verify_api_key] = lambda: True
    return TestClient(app)


def _route_tokenizer(token_ids):
    tok = SimpleNamespace(encode=lambda text, **kw: list(token_ids))
    return tok


class TestHarnessRoutes:
    def test_tokens_count_uses_default_loaded_model(self):
        app = _make_app()
        pool = SimpleNamespace(
            _entries={
                MODEL_ID: SimpleNamespace(
                    engine=SimpleNamespace(_tokenizer=_route_tokenizer([1, 2, 3]))
                ),
                "unloaded": SimpleNamespace(engine=None),
            }
        )
        with patch("fusion_mlx.api.harness_routes._get_engine_pool", return_value=pool):
            response = app.post("/harness/v1/tokens/count", json={"texts": ["a", "bb"]})
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == MODEL_ID
        assert data["counts"] == [3, 3]

    def test_tokens_count_requested_model_not_loaded_is_409(self):
        app = _make_app()
        pool = SimpleNamespace(_entries={"idle": SimpleNamespace(engine=None)})
        with patch("fusion_mlx.api.harness_routes._get_engine_pool", return_value=pool):
            response = app.post(
                "/harness/v1/tokens/count",
                json={"texts": ["a"], "model_id": "idle"},
            )
        assert response.status_code == 409

    def test_cache_probe_returns_counts(self):
        app = _make_app()
        token_ids = list(range(8))
        scheduler = SimpleNamespace(
            block_aware_cache=_make_radix_cache(token_ids, 4, cached_blocks=1),
            paged_ssd_cache_manager=None,
            paged_cache_manager=SimpleNamespace(model_name=MODEL_NAME),
            config=SimpleNamespace(paged_cache_block_size=4),
        )
        pool = _make_pool(_route_tokenizer(token_ids), scheduler)
        with patch("fusion_mlx.api.harness_routes._get_engine_pool", return_value=pool):
            response = app.post(
                "/harness/v1/cache/probe",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["cached_prefix_tokens"] == 4
        assert data["blocks_ram"] == 1


# ---------------------------------------------------------------------------
# bridge objects
# ---------------------------------------------------------------------------


def _mock_transport(routes):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        payload = json.loads(request.content) if request.content else {}
        route = routes.get(path)
        if route is None:
            return httpx.Response(404, json={"detail": "not found"})
        return route(payload)

    return httpx.MockTransport(handler)


class TestRemoteTokenCounter:
    def test_count_texts_batches_in_one_request(self):
        calls = []

        def route(payload):
            calls.append(payload)
            return httpx.Response(200, json={"model_id": MODEL_ID, "counts": [3, 5]})

        counter = RemoteTokenCounter(
            "http://engine",
            transport=_mock_transport({"/harness/v1/tokens/count": route}),
        )
        assert counter.count_texts(["a", "bb"]) == [3, 5]
        assert len(calls) == 1

    def test_count_text_single(self):
        counter = RemoteTokenCounter(
            "http://engine",
            transport=_mock_transport(
                {
                    "/harness/v1/tokens/count": lambda p: httpx.Response(
                        200, json={"model_id": MODEL_ID, "counts": [7]}
                    )
                }
            ),
        )
        assert counter.count_text("hello") == 7


class TestPrefixCacheAdvisor:
    def _messages(self, n_rounds):
        messages = [{"role": "system", "content": "sys"}]
        for i in range(n_rounds):
            messages.append({"role": "user", "content": f"u{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})
        return messages

    def _routes(self, cached_tokens, total_tokens, counts):
        return {
            "/harness/v1/cache/probe": lambda p: httpx.Response(
                200,
                json={
                    "model_id": MODEL_ID,
                    "total_tokens": total_tokens,
                    "cached_prefix_tokens": cached_tokens,
                    "block_size": 256,
                },
            ),
            "/harness/v1/tokens/count": lambda p: httpx.Response(
                200, json={"model_id": MODEL_ID, "counts": counts}
            ),
        }

    def test_no_cached_prefix_means_no_opinion(self):
        advisor = PrefixCacheAdvisor(
            "http://engine",
            transport=_mock_transport(self._routes(0, 4000, [100] * 10)),
        )
        assert advisor.suggest_keep_rounds(self._messages(4)) == 0

    def test_probe_failure_returns_zero(self):
        advisor = PrefixCacheAdvisor(
            "http://engine",
            transport=_mock_transport({}),
        )
        assert advisor.suggest_keep_rounds(self._messages(4)) == 0

    def test_suggests_minimal_keep_meeting_budget(self):
        # 3 rounds; the last round alone (~7400 of 10100 tokens) exceeds
        # the halving budget, so the smallest valid keep is 2.
        counts = [200] + [500] * 4 + [4000, 4000]
        advisor = PrefixCacheAdvisor(
            "http://engine",
            transport=_mock_transport(self._routes(3000, 10100, counts)),
        )
        keep = advisor.suggest_keep_rounds(self._messages(3))
        assert keep == 2

    def test_keep_never_exceeds_rounds_minus_one(self):
        # One round dominates the prompt, so no keep value fits the
        # halving budget; the advisor falls back to dropping one round.
        counts = [200] + [100] * 4 + [4000, 4000]
        advisor = PrefixCacheAdvisor(
            "http://engine",
            min_cached_tokens=10,
            transport=_mock_transport(self._routes(3000, 8400, counts)),
        )
        keep = advisor.suggest_keep_rounds(self._messages(3))
        assert keep == 2

    def test_count_failure_returns_zero(self):
        routes = {
            "/harness/v1/cache/probe": lambda p: httpx.Response(
                200,
                json={
                    "model_id": MODEL_ID,
                    "total_tokens": 4200,
                    "cached_prefix_tokens": 3000,
                    "block_size": 256,
                },
            )
        }
        advisor = PrefixCacheAdvisor("http://engine", transport=_mock_transport(routes))
        assert advisor.suggest_keep_rounds(self._messages(4)) == 0

    def test_leading_assistant_round_is_first_round(self):
        # Messages that start (after system) with a non-user message form
        # the first round, mirroring harness _split_rounds.
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "greeting"},
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ]
        starts = PrefixCacheAdvisor._round_start_indexes(
            PrefixCacheAdvisor.__new__(PrefixCacheAdvisor), messages
        )
        assert starts == [1, 2]
