# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from fusion_mlx.cache.paged_cache import compute_block_hash
import fusion_mlx.admin.stats as admin_stats


BLOCK_SIZE = 4
MODEL_ID = "test-model"
MODEL_NAME = "/models/test-model"


def _make_request(model_id=MODEL_ID, messages=None):
    return admin_stats.CacheProbeRequest(
        model_id=model_id,
        messages=messages or [{"role": "user", "content": "hello"}],
    )


def _make_tokenizer(token_ids):
    tok = MagicMock(spec=[])
    tok.apply_chat_template = MagicMock(return_value="rendered prompt")
    tok.encode = MagicMock(return_value=token_ids)
    return tok


def _compute_hashes(token_ids, block_size=BLOCK_SIZE, model_name=MODEL_NAME):
    hashes = []
    parent = b""
    for start in range(0, len(token_ids), block_size):
        block_tokens = token_ids[start : start + block_size]
        h = compute_block_hash(parent, block_tokens, model_name=model_name)
        hashes.append(h)
        parent = h
    return hashes


def _make_ssd_index(known_hashes):
    idx = MagicMock(spec=[])
    idx.contains = MagicMock(side_effect=lambda h: h in known_hashes)
    return idx


def _make_engine_entry(
    tokenizer,
    scheduler,
    has_apply_chat_template=True,
):
    engine_ns = SimpleNamespace(
        _tokenizer=tokenizer,
        _engine=SimpleNamespace(
            engine=SimpleNamespace(scheduler=scheduler),
        ),
    )
    if has_apply_chat_template:
        engine_ns._apply_chat_template = lambda msgs, tools, **kw: "rendered prompt"
    return SimpleNamespace(engine=engine_ns)


def _make_scheduler(
    ssd_hot=None,
    ssd_index=None,
    model_name=MODEL_NAME,
    block_size=BLOCK_SIZE,
):
    return SimpleNamespace(
        block_aware_cache=SimpleNamespace(block_size=block_size),
        paged_ssd_cache_manager=SimpleNamespace(
            _hot_cache=ssd_hot or {},
            _index=ssd_index or _make_ssd_index(set()),
        ),
        paged_cache_manager=SimpleNamespace(model_name=model_name),
        config=SimpleNamespace(paged_cache_block_size=block_size),
    )


def _pool_with(entries):
    pool = MagicMock(spec=[])
    pool._entries = entries
    return pool


class TestCacheProbeErrors:
    def test_engine_pool_not_initialized(self):
        with patch.object(admin_stats, "_get_engine_pool", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_stats.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 503

    def test_model_not_found(self):
        pool = _pool_with({})
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_stats.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 404

    def test_model_not_loaded(self):
        pool = _pool_with({MODEL_ID: SimpleNamespace(engine=None)})
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )
        assert result["model_loaded"] is False
        assert "reason" in result

    def test_no_tokenizer(self):
        entry = SimpleNamespace(
            engine=SimpleNamespace(
                _tokenizer=None,
                _engine=SimpleNamespace(
                    engine=SimpleNamespace(scheduler=_make_scheduler()),
                ),
            )
        )
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_stats.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 400

    def test_scheduler_unavailable(self):
        tokenizer = _make_tokenizer([1, 2, 3])
        entry = SimpleNamespace(
            engine=SimpleNamespace(
                _tokenizer=tokenizer,
                _engine=None,
            )
        )
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_stats.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 500

    def test_block_size_unavailable(self):
        tokenizer = _make_tokenizer([1, 2, 3])
        scheduler = SimpleNamespace(
            block_aware_cache=None,
            paged_ssd_cache_manager=None,
            paged_cache_manager=None,
            config=SimpleNamespace(paged_cache_block_size=0),
        )
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_stats.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 500

    def test_empty_tokens(self):
        tokenizer = _make_tokenizer([])
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )
        assert result["model_loaded"] is True
        assert result["total_tokens"] == 0
        assert result["total_blocks"] == 0


class TestCacheProbeClassification:
    def test_all_blocks_cold(self):
        token_ids = list(range(12))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert result["total_tokens"] == 12
        assert result["total_blocks"] == 3
        assert result["blocks_ssd_hot"] == 0
        assert result["blocks_ssd_disk"] == 0
        assert result["blocks_cold"] == 3
        assert result["ssd_hit_tokens"] == 0
        assert result["cold_tokens"] == 12

    def test_all_blocks_ssd_disk(self):
        token_ids = list(range(8))
        hashes = _compute_hashes(token_ids)
        ssd_index = _make_ssd_index(set(hashes))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_hot"] == 0
        assert result["blocks_ssd_disk"] == 2
        assert result["blocks_cold"] == 0
        assert result["ssd_hit_tokens"] == 8
        assert result["cold_tokens"] == 0

    def test_all_blocks_ssd_hot(self):
        token_ids = list(range(8))
        hashes = _compute_hashes(token_ids)
        ssd_hot = {h: {} for h in hashes}
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_hot=ssd_hot)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_hot"] == 2
        assert result["blocks_ssd_disk"] == 0
        assert result["blocks_cold"] == 0

    def test_mixed_hot_and_disk(self):
        token_ids = list(range(8))
        hashes = _compute_hashes(token_ids)
        ssd_hot = {hashes[0]: {}}
        ssd_index = _make_ssd_index({hashes[1]})
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_hot=ssd_hot, ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_hot"] == 1
        assert result["blocks_ssd_disk"] == 1
        assert result["blocks_cold"] == 0

    def test_partial_prefix_hit(self):
        token_ids = list(range(12))
        hashes = _compute_hashes(token_ids)
        ssd_index = _make_ssd_index({hashes[0], hashes[1]})
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_disk"] == 2
        assert result["blocks_cold"] == 1
        assert result["ssd_hit_tokens"] == 8
        assert result["cold_tokens"] == 4

    def test_gap_in_prefix_makes_rest_cold(self):
        token_ids = list(range(12))
        hashes = _compute_hashes(token_ids)
        ssd_index = _make_ssd_index({hashes[0], hashes[2]})
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_disk"] == 1
        assert result["blocks_cold"] == 2

    def test_partial_last_block(self):
        token_ids = list(range(10))
        hashes = _compute_hashes(token_ids)
        ssd_index = _make_ssd_index(set(hashes))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert result["total_tokens"] == 10
        assert result["total_blocks"] == 3
        assert result["blocks_ssd_disk"] == 3
        assert result["blocks_cold"] == 0
        assert result["ssd_hit_tokens"] == 10
        assert result["cold_tokens"] == 0


class TestCacheProbeResponseShape:
    EXPECTED_FIELDS = {
        "model_id",
        "model_loaded",
        "total_tokens",
        "block_size",
        "total_blocks",
        "blocks_ssd_hot",
        "blocks_ssd_disk",
        "blocks_cold",
        "ssd_hit_tokens",
        "cold_tokens",
    }

    def test_loaded_response_fields(self):
        token_ids = list(range(4))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert set(result.keys()) == self.EXPECTED_FIELDS

    def test_no_dead_fields(self):
        token_ids = list(range(4))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_stats, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_stats.probe_cache(_make_request(), is_admin=True)
            )

        assert "blocks_ram" not in result
        assert "ram_hit_tokens" not in result
        assert "prefix_index_hits" not in result
