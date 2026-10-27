# SPDX-License-Identifier: Apache-2.0

import json
import logging
import struct
from pathlib import Path
from types import SimpleNamespace

from fusion_mlx.cache.paged_ssd_cache import (
    PagedSSDBlockMetadata,
    PagedSSDCacheManager,
)
from fusion_mlx.scheduler import sched_misc

FINGERPRINT = "5142ce673d4782f8063c09738dd1f2cbd63aea9b214f1d24a8b5cc5949e6de84"
RAW_PATH = "/Users/dahai/.fusion-mlx/models/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caa"


def _meta(
    block_hash: bytes, model_name: str, num_layers: int = 64
) -> PagedSSDBlockMetadata:
    return PagedSSDBlockMetadata(
        block_hash=block_hash,
        file_path=Path("/tmp/none.safetensors"),
        file_size=1024,
        token_count=2048,
        created_at=0.0,
        last_access=1000.0,
        num_layers=num_layers,
        model_name=model_name,
        block_size=2048,
        cache_signature="",
    )


def _manager(
    tmp_path: Path, model_name: str, num_layers: int = 64
) -> PagedSSDCacheManager:
    return PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd_cache",
        max_size_bytes=1 << 30,
        expected_model_name=model_name,
        expected_num_layers=num_layers,
        expected_block_size=2048,
    )


def _fake_scheduler(config_model_name: str, paged_model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(model_name=config_model_name),
        paged_cache_manager=SimpleNamespace(model_name=paged_model_name),
    )


def test_expected_ssd_model_name_prefers_paged_cache_fingerprint():
    fake = _fake_scheduler(RAW_PATH, FINGERPRINT)
    assert sched_misc._expected_ssd_model_name(fake) == FINGERPRINT


def test_expected_ssd_model_name_falls_back_to_config_path():
    fake = _fake_scheduler(RAW_PATH, "")
    assert sched_misc._expected_ssd_model_name(fake) == RAW_PATH


def test_expected_ssd_model_name_without_paged_cache():
    fake = SimpleNamespace(config=SimpleNamespace(model_name=RAW_PATH))
    assert sched_misc._expected_ssd_model_name(fake) == RAW_PATH


def test_production_ssd_path_passes_fingerprint_not_raw_path(tmp_path: Path):
    captured = {}

    class _Manager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def set_paged_ssd_cache_manager(self, manager):
            pass

    fake = SimpleNamespace(
        config=SimpleNamespace(
            model_name=RAW_PATH,
            paged_ssd_cache_dir=str(tmp_path / "ssd_cache"),
            paged_ssd_cache_max_size=1 << 30,
            hot_cache_max_size=0,
            hot_cache_only=False,
            paged_cache_block_size=2048,
            boundary_prefix_persist=False,
            boundary_prefix_max_bytes=0,
        ),
        paged_cache_manager=SimpleNamespace(
            model_name=FINGERPRINT,
            set_paged_ssd_cache_manager=lambda manager: None,
        ),
        block_aware_cache=SimpleNamespace(
            expected_num_layers=64,
            set_paged_ssd_cache_manager=lambda manager: None,
        ),
        memory_monitor=None,
        _activate_tiered_cache_manager=lambda: None,
        _activate_rollback_manager=lambda: None,
        _format_bytes=lambda value: str(value),
    )
    fake._expected_ssd_model_name = lambda: sched_misc._expected_ssd_model_name(fake)

    original_manager = sched_misc.PagedSSDCacheManager
    original_store = sched_misc.BoundarySnapshotSSDStore
    sched_misc.PagedSSDCacheManager = _Manager
    sched_misc.BoundarySnapshotSSDStore = None
    try:
        sched_misc._init_tiered_cache(fake)
    finally:
        sched_misc.PagedSSDCacheManager = original_manager
        sched_misc.BoundarySnapshotSSDStore = original_store

    assert captured["expected_model_name"] == FINGERPRINT
    assert captured["expected_model_name"] != RAW_PATH


def test_fingerprint_block_is_compatible_after_restart(tmp_path: Path):
    mgr = _manager(tmp_path, FINGERPRINT)
    assert mgr._is_compatible_block(_meta(b"aa" * 10, FINGERPRINT)) is True


def test_raw_path_stamp_is_not_compatible_with_fingerprint_identity(tmp_path: Path):
    mgr = _manager(tmp_path, FINGERPRINT)
    assert mgr._is_compatible_block(_meta(b"bb" * 10, RAW_PATH)) is False


def test_reclaim_skips_blocks_of_live_foreign_model(tmp_path: Path):
    other = "3f94a26097d6c656"
    live = _manager(tmp_path, other)
    mgr = _manager(tmp_path, FINGERPRINT)
    try:
        foreign = _meta(b"cc" * 10, other)
        mgr._incompatible_index.add(foreign)
        assert mgr._pick_reclaimable_incompatible() is None
    finally:
        mgr.close()
        live.close()


def test_reclaim_allows_foreign_model_without_live_manager(tmp_path: Path):
    mgr = _manager(tmp_path, FINGERPRINT)
    foreign = _meta(b"cd" * 10, "3f94a26097d6c656")
    mgr._incompatible_index.add(foreign)
    picked = mgr._pick_reclaimable_incompatible()
    assert picked is not None
    assert picked.block_hash == foreign.block_hash


def test_reclaim_prefers_own_fingerprint_over_live_foreign(tmp_path: Path):
    other = "3f94a26097d6c656"
    live = _manager(tmp_path, other)
    mgr = _manager(tmp_path, FINGERPRINT)
    try:
        foreign = _meta(b"dd" * 10, other)
        foreign.last_access = 10.0
        own = _meta(b"ee" * 10, FINGERPRINT)
        own.last_access = 20.0
        mgr._incompatible_index.add(foreign)
        mgr._incompatible_index.add(own)
        picked = mgr._pick_reclaimable_incompatible()
        assert picked is not None
        assert picked.block_hash == own.block_hash
    finally:
        mgr.close()
        live.close()


def test_reclaim_allows_legacy_blocks_without_model_name(tmp_path: Path):
    mgr = _manager(tmp_path, FINGERPRINT)
    legacy = _meta(b"ff" * 10, "")
    mgr._incompatible_index.add(legacy)
    picked = mgr._pick_reclaimable_incompatible()
    assert picked is not None
    assert picked.block_hash == legacy.block_hash


def test_enforce_keeps_blocks_of_live_foreign_model(tmp_path: Path):
    cache_dir = tmp_path / "ssd_cache"
    other = "3f94a26097d6c656"
    live = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1 << 30,
        expected_model_name=other,
        expected_num_layers=64,
        expected_block_size=2048,
    )
    mgr = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1,
        expected_model_name=FINGERPRINT,
        expected_num_layers=64,
        expected_block_size=2048,
    )
    try:
        foreign = _meta(b"11" * 10, other)
        file_path = mgr._get_file_path(foreign.block_hash)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x" * 64)
        foreign.file_path = file_path
        mgr._incompatible_index.add(foreign)

        mgr._enforce_incompatible_at_startup()

        assert file_path.exists()
        assert mgr._incompatible_index.count == 1
    finally:
        mgr.close()
        live.close()


def test_enforce_unlinks_own_incompatible_blocks(tmp_path: Path):
    cache_dir = tmp_path / "ssd_cache"
    mgr = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1,
        expected_model_name=FINGERPRINT,
        expected_num_layers=64,
        expected_block_size=2048,
    )
    own = _meta(b"22" * 10, FINGERPRINT)
    file_path = mgr._get_file_path(own.block_hash)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"x" * 64)
    own.file_path = file_path
    mgr._incompatible_index.add(own)

    mgr._enforce_incompatible_at_startup()

    assert not file_path.exists()
    assert mgr._incompatible_index.count == 0


def test_enforce_logs_when_only_live_foreign_blocks_remain(tmp_path: Path):
    cache_dir = tmp_path / "ssd_cache"
    other = "3f94a26097d6c656"
    live = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1 << 30,
        expected_model_name=other,
        expected_num_layers=64,
        expected_block_size=2048,
    )
    mgr = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1,
        expected_model_name=FINGERPRINT,
        expected_num_layers=64,
        expected_block_size=2048,
    )
    mgr._incompatible_index.add(_meta(b"33" * 10, other))
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    module_logger = logging.getLogger("fusion_mlx.cache.paged_ssd_cache")
    handler = _Collect()
    module_logger.addHandler(handler)
    previous_level = module_logger.level
    module_logger.setLevel(logging.INFO)
    try:
        mgr._enforce_incompatible_at_startup()
    finally:
        module_logger.removeHandler(handler)
        module_logger.setLevel(previous_level)
        mgr.close()
        live.close()

    assert any("belong to another live model" in r.getMessage() for r in records)


def test_real_scan_reuses_blocks_after_restart(tmp_path: Path):
    cache_dir = tmp_path / "ssd_cache"
    writer = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1 << 30,
        expected_model_name=FINGERPRINT,
        expected_num_layers=64,
        expected_block_size=2048,
    )
    block_hash = b"44" * 10
    file_path = writer._get_file_path(block_hash)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "fusion_cache_format_version": "3",
        "block_hash": block_hash.hex(),
        "token_count": "2048",
        "num_layers": "64",
        "model_name": FINGERPRINT,
        "block_size": "2048",
        "cache_signature": "",
        "created_at": "0",
        "last_access": "1000",
    }
    payload = json.dumps({"__metadata__": metadata}).encode("utf-8")
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        f.write(b"\0" * 64)

    reader = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1 << 30,
        expected_model_name=FINGERPRINT,
        expected_num_layers=64,
        expected_block_size=2048,
    )
    assert reader._index.count == 1
    assert reader._index.get(block_hash) is not None
