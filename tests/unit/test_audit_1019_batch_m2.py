# SPDX-License-Identifier: Apache-2.0
# Regression tests for the second M-series batch of
# audit/fusion-mlx-codeing-audit-report-1019.md:
#   M7 VideoWriter release + scratch cleanup, M8 local RNG keys,
#   M18 orphan purge counted apart from real evictions.

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from fusion_mlx.cache.paged_cache import PagedCacheManager

REPO_ROOT = Path(__file__).resolve().parents[2]
FUSION_MLX = REPO_ROOT / "fusion_mlx"


class TestOrphanPurgeNotCountedAsEviction:
    # M18: _evict_lru_blocks_locked used to add cascade-purged hash entries to
    # the evicted count, so callers believed they reclaimed more blocks than
    # they did and stopped retrying with the pool still full.

    def _manager_with_chain(self):
        mgr = PagedCacheManager(block_size=4, max_blocks=8, initial_blocks=8)
        parent = mgr.blocks[1]
        child = mgr.blocks[2]
        parent.block_hash = b"P" * 32
        parent.token_count = 4
        child.block_hash = b"C" * 32
        child.parent_hash = parent.block_hash
        child.token_count = 4
        mgr.cached_block_hash_to_block.insert(parent.block_hash, parent)
        mgr.cached_block_hash_to_block.insert(child.block_hash, child)
        return mgr, parent, child

    def test_cascade_purge_not_added_to_evicted_count(self):
        mgr, parent, child = self._manager_with_chain()
        child_hash = child.block_hash

        evicted = mgr._evict_lru_blocks_locked(1)

        # Only the parent block was actually freed.
        assert evicted == 1
        assert parent.block_hash is None
        # The orphan's stale hash entry is gone, its block was NOT evicted.
        assert mgr.cached_block_hash_to_block.get_block(child_hash) is None
        assert child.block_hash is None
        assert child.ref_count == 0

    def test_no_cascade_leaves_count_unchanged(self):
        mgr = PagedCacheManager(block_size=4, max_blocks=8, initial_blocks=8)
        block = mgr.blocks[1]
        block.block_hash = b"L" * 32
        block.token_count = 4
        mgr.cached_block_hash_to_block.insert(block.block_hash, block)

        assert mgr._evict_lru_blocks_locked(1) == 1


class _ExplodingWriter:
    instances: list[_ExplodingWriter] = []

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.released = False
        _ExplodingWriter.instances.append(self)

    def write(self, frame):
        raise RuntimeError("encoder died")

    def release(self):
        self.released = True


class TestWriteMp4ReleasesWriter:
    # M7: a failed frame write must still release the cv2 writer and remove the
    # scratch file.

    @pytest.fixture(autouse=True)
    def _reset(self):
        _ExplodingWriter.instances.clear()

    def _patch(self, monkeypatch):
        import cv2

        monkeypatch.setattr(cv2, "VideoWriter", _ExplodingWriter)
        monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *a: 0)

    def test_failure_path_releases_writer(self, monkeypatch):
        from fusion_mlx.video.ltx2_5 import generate as gen

        self._patch(monkeypatch)
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="encoder died"):
            gen._write_mp4(frames, 24, None)

        assert len(_ExplodingWriter.instances) == 1
        writer = _ExplodingWriter.instances[0]
        assert writer.released is True
        assert not Path(writer.path).exists()


class TestLocalRngKeys:
    # M8: the video generators must not reseed the process-global MLX/numpy RNG.

    VIDEO_GENERATORS = [
        "video/cogvideox/generate.py",
        "video/wan2/generate.py",
        "video/wan2/stage.py",
    ]

    @pytest.mark.parametrize("rel_path", VIDEO_GENERATORS)
    def test_no_global_seeding(self, rel_path):
        src = (FUSION_MLX / rel_path).read_text()
        assert not re.search(r"\bmx\.random\.seed\s*\(", src)
        assert not re.search(r"\bnp\.random\.seed\s*\(", src)

    @pytest.mark.parametrize("rel_path", VIDEO_GENERATORS)
    def test_noise_draws_use_local_key(self, rel_path):
        src = (FUSION_MLX / rel_path).read_text()
        draws = re.findall(r"mx\.random\.normal\([^)]*\)", src)
        assert draws, f"no mx.random.normal call found in {rel_path}"
        for call in draws:
            assert "key=" in call, f"unkeyed draw in {rel_path}: {call}"

    def test_cogvideox_vae_encode_accepts_key(self):
        from fusion_mlx.video.cogvideox.vae import AutoencoderKLCogVideoX

        params = AutoencoderKLCogVideoX.encode.__code__.co_varnames[
            : AutoencoderKLCogVideoX.encode.__code__.co_argcount
        ]
        assert "key" in params


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
