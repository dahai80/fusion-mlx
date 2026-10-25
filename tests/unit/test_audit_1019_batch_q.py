# SPDX-License-Identifier: Apache-2.0
"""Regression tests for audit 1019 L-series findings (batch Q).

L14 - route_guard builds its 403 body in one place.
L15 - single-block and batch free share one teardown (hash + orphan cascade).
L16 - the duplicate _cow_copy_block helper is gone.
L17 - SSD block writes go through one persist path and one queue put.
"""

import ast
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _source(rel):
    return (_REPO / rel).read_text()


def _in_free_queue(queue, block_id):
    node = queue.fake_head.next_free_block
    while node is not None and node is not queue.fake_tail:
        if node.block_id == block_id:
            return True
        node = node.next_free_block
    return False


class TestRouteGuardRejectBody:
    def test_single_json_dumps_site(self):
        tree = ast.parse(_source("fusion_mlx/middleware/route_guard.py"))
        calls = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "dumps"
        ]
        assert len(calls) == 1, f"json.dumps built at lines {calls}"

    def test_reject_helper_emits_error_envelope(self):
        import asyncio
        import json

        from fusion_mlx.middleware.route_guard import RouteGuardMiddleware

        sent = []

        async def send(message):
            sent.append(message)

        mw = RouteGuardMiddleware(app=None)
        asyncio.run(
            mw._send_reject(
                send,
                code="missing_route",
                message="Missing X-Fusion-Route header",
                hint="FUSION_ROUTE_WARN_ONLY=true",
            )
        )

        start, body = sent
        assert start["status"] == 403
        payload = json.loads(body["body"])
        assert payload == {
            "error": {
                "message": "Missing X-Fusion-Route header",
                "code": "missing_route",
                "hint": "FUSION_ROUTE_WARN_ONLY=true",
            }
        }
        headers = dict(start["headers"])
        assert headers[b"content-length"] == str(len(body["body"])).encode()


class TestPagedCacheFreePaths:
    def _manager(self, initial_blocks=4, max_blocks=8):
        from fusion_mlx.cache.paged_cache import PagedCacheManager

        return PagedCacheManager(
            block_size=4,
            max_blocks=max_blocks,
            model_name="test-model",
            initial_blocks=initial_blocks,
        )

    def test_shared_block_survives_both_paths(self):
        manager = self._manager()
        block = manager.allocate_block()
        block.ref_count = 2
        hash_value = b"h" * 32
        block.block_hash = hash_value
        block.token_count = 4
        manager.cached_block_hash_to_block.insert(hash_value, block)

        assert manager.free_block(block.block_id) is False
        assert block.ref_count == 1
        assert block.block_id in manager.allocated_blocks
        assert manager.cached_block_hash_to_block.get_block(hash_value) is block

        manager.free_blocks_batch([block])
        assert block.ref_count == 0
        assert block.block_id not in manager.allocated_blocks
        assert manager.cached_block_hash_to_block.get_block(hash_value) is None
        assert _in_free_queue(manager.free_block_queue, block.block_id)

    def test_batch_frees_only_zero_ref_blocks(self):
        manager = self._manager()
        held = manager.allocate_block()
        empty = manager.allocate_block()
        held.ref_count = 2
        empty.ref_count = 1

        manager.free_blocks_batch([held, empty])

        assert held.ref_count == 1
        assert held.block_id in manager.allocated_blocks
        assert empty.block_id not in manager.allocated_blocks
        assert _in_free_queue(manager.free_block_queue, empty.block_id)
        assert _in_free_queue(manager.free_block_queue, held.block_id) is False

    def test_null_block_is_never_freed(self):
        manager = self._manager()
        block = manager.allocate_block()
        block.is_null = True

        assert manager.free_block(block.block_id) is False
        assert block.block_id in manager.allocated_blocks

    def test_stats_stay_consistent_across_paths(self):
        manager = self._manager()
        first = manager.allocate_block()
        second = manager.allocate_block()
        first.token_count = 4
        second.token_count = 4
        before_allocated = manager.stats.allocated_blocks
        before_tokens = manager.stats.total_tokens_cached

        manager.free_block(first.block_id)
        manager.free_blocks_batch([second])

        assert manager.stats.allocated_blocks == before_allocated - 2
        assert manager.stats.total_tokens_cached == before_tokens - 8


class TestCowCopyHelperRemoved:
    def test_helper_is_gone(self):
        from fusion_mlx.cache.paged_cache import PagedCacheManager

        assert not hasattr(PagedCacheManager, "_cow_copy_block")
        assert "def _cow_copy_block" not in _source("fusion_mlx/cache/paged_cache.py")


class TestSsdWritePathsDeduped:
    def test_one_safetensors_write_site(self):
        tree = ast.parse(_source("fusion_mlx/cache/paged_ssd_cache.py"))
        sites = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_write_safetensors_no_mx"
        ]
        assert len(sites) == 1, f"write sites at lines {sites}"

    def test_one_queue_put_site(self):
        tree = ast.parse(_source("fusion_mlx/cache/paged_ssd_cache.py"))
        puts = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "put"
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "_write_queue"
        ]
        assert len(puts) == 2, f"_write_queue.put at lines {puts}"

    def test_persist_failure_removes_temp_file(self, tmp_path, monkeypatch):
        from fusion_mlx.cache import paged_ssd_cache
        from fusion_mlx.cache.paged_ssd_cache import (
            PagedSSDBlockMetadata,
            PagedSSDCacheManager,
        )

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd", max_size_bytes=1024**3
        )
        block_hash = b"q" * 32
        block_metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=manager._get_file_path(block_hash),
            file_size=0,
            token_count=4,
            created_at=0.0,
            last_access=0.0,
            num_layers=1,
            model_name="test-model",
            block_size=4,
            cache_signature="sig",
        )

        def boom(path, *args, **kwargs):
            pathlib.Path(path).write_bytes(b"partial")
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(paged_ssd_cache, "_write_safetensors_no_mx", boom)
        try:
            with pytest.raises(OSError):
                manager._persist_block_file(block_hash, [], {}, block_metadata)
        finally:
            manager.close()

        temp = manager._get_file_path(block_hash).with_suffix(".tmp")
        assert not temp.exists()
        assert not manager._get_file_path(block_hash).exists()

    def test_try_queue_write_reports_full(self, tmp_path):
        import queue as queue_mod

        from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

        class _FullQueue:
            # #1390: filling the real queue raced the live writer thread
            # (it drained the sentinel first, so the put succeeded and the
            # assertion flipped). Exercise the Full policy directly.
            def put(self, item, timeout=None):
                raise queue_mod.Full

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd", max_size_bytes=1024**3
        )
        try:
            assert manager._try_queue_write(b"a" * 32, [], {}, None) is True
            manager._write_queue.get_nowait()
            manager._write_queue = _FullQueue()
            assert manager._try_queue_write(b"b" * 32, [], {}, None) is False
            assert manager._stats["ssd_write_drops"] == 0
        finally:
            manager._write_queue = queue_mod.Queue()
            manager.close()

    def test_malformed_queue_item_does_not_kill_writer(self, tmp_path):
        # #1390: the unpack used to sit outside the writer's try/except, so
        # a single bad item killed the thread and every later SSD write was
        # queued forever with no listener.
        import time

        from fusion_mlx.cache.paged_ssd_cache import PagedSSDCacheManager

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd", max_size_bytes=1024**3
        )
        try:
            manager._write_queue.put("not-a-queue-item")
            deadline = time.time() + 5.0
            while manager._write_queue.unfinished_tasks and time.time() < deadline:
                time.sleep(0.01)
            assert manager._write_queue.unfinished_tasks == 0
            assert manager._writer_thread.is_alive()
            assert manager._stats["errors"] >= 1
        finally:
            manager.close()
