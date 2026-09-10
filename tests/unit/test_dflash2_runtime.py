# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DFlash2 in-target drafter + runtime (mocked dflash pkg).

No real model load: monkeypatches ``dflash.model_mlx`` so the drafter
constructs from fakes. The drafter loads ONLY the draft (no target) and
binds to an external target via bind().
"""

from __future__ import annotations

import logging
import types

import pytest

logger = logging.getLogger(__name__)


def _install_fake_dflash_model_mlx(monkeypatch):
    fake = types.ModuleType("dflash.model_mlx")

    class _FakeDraft:
        def __init__(self, repo):
            self.repo = repo
            self.bound = None
            self.config = types.SimpleNamespace(
                mask_token_id=999,
                target_layer_ids=[5, 19, 33, 47, 61],
                layer_types=("full_attention",) * 6,
                sliding_window=None,
                hidden_size=5120,
                block_size=5,
            )

        def bind(self, target):
            self.bound = target
            return self

        def make_cache(self):
            return [types.SimpleNamespace(offset=0)]

        def leaf_modules(self):
            return {}

        def update_modules(self, modules):
            pass

        def propose(self, block, hidden, cache, temperature, logits_start=0):
            import mlx.core as mx

            bs = block.shape[1] - logits_start
            tokens = mx.array([[100 + i for i in range(bs)]], dtype=mx.uint32)
            indices = None
            probs = None
            return tokens, indices, probs

    def _fake_load_draft(repo):
        return _FakeDraft(repo)

    def _fake_patch_model(model, layer_ids):
        model._hidden_states = [None] * len(layer_ids)

    def _fake_trim_recent_cache(cache, n):
        for c in cache:
            if hasattr(c, "offset"):
                c.offset = max(0, c.offset - n)

    fake.load_draft = _fake_load_draft
    fake._patch_model = _fake_patch_model
    fake._trim_recent_cache = _fake_trim_recent_cache
    fake.DFlash2DraftModel = _FakeDraft
    fake.snapshot_download = lambda _id, **_kw: "/fake/path"
    monkeypatch.setitem(__import__("sys").modules, "dflash.model_mlx", fake)
    return fake


def test_drafter_loads_only_draft(monkeypatch):
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2InTargetDrafter

    _install_fake_dflash_model_mlx(monkeypatch)
    drafter = DFlash2InTargetDrafter(
        draft_repo="z-lab/Qwen3.8-27B-DFlash2",
        block_size=5,
    )
    assert drafter.loaded
    assert drafter.block_size == 5
    assert drafter.mask_id == 999
    assert drafter._bound is False
    assert drafter._target is None


def test_drafter_bind_installs_hooks(monkeypatch):
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2InTargetDrafter

    _install_fake_dflash_model_mlx(monkeypatch)
    drafter = DFlash2InTargetDrafter("d", block_size=5)
    target = types.SimpleNamespace()
    drafter.bind(target)
    assert drafter._bound is True
    assert drafter._target is target
    assert drafter.draft.bound is target
    assert hasattr(target, "_hidden_states")


def test_drafter_reset_creates_cache(monkeypatch):
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2InTargetDrafter

    _install_fake_dflash_model_mlx(monkeypatch)
    drafter = DFlash2InTargetDrafter("d", block_size=5)
    assert drafter._draft_cache is None
    drafter.reset()
    assert drafter._draft_cache is not None
    assert drafter._last_hidden is None


def test_drafter_rejects_invalid_block_size():
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2InTargetDrafter

    with pytest.raises(ValueError, match="block_size"):
        DFlash2InTargetDrafter("d", block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        DFlash2InTargetDrafter("d", block_size=9)


def test_drafter_rejects_empty_repo():
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2InTargetDrafter

    with pytest.raises(ValueError, match="draft_repo"):
        DFlash2InTargetDrafter("", block_size=5)


def test_drafter_rejects_bad_draft_bits():
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2InTargetDrafter

    with pytest.raises(ValueError, match="draft_bits"):
        DFlash2InTargetDrafter("d", block_size=5, draft_bits=3)
    with pytest.raises(ValueError, match="draft_bits"):
        DFlash2InTargetDrafter("d", block_size=5, draft_bits=16)


def test_load_runtime_builds_runtime(monkeypatch):
    from fusion_mlx.speculative.dflash2 import DFlash2Runtime, load_runtime

    _install_fake_dflash_model_mlx(monkeypatch)
    rt = load_runtime("d", block_size=5)
    assert isinstance(rt, DFlash2Runtime)
    assert rt.draft_repo == "d"
    assert rt.block_size == 5
    assert rt.drafter is not None
    assert rt.drafter.block_size == 5


def test_load_runtime_rejects_bad_block_size():
    from fusion_mlx.speculative.dflash2 import load_runtime

    with pytest.raises(ValueError, match="block_size"):
        load_runtime("d", block_size=20)
    with pytest.raises(ValueError, match="block_size"):
        load_runtime("d", block_size=0)


def test_load_runtime_rejects_bad_draft_bits():
    from fusion_mlx.speculative.dflash2 import load_runtime

    with pytest.raises(ValueError, match="draft_bits"):
        load_runtime("d", block_size=5, draft_bits=3)
    with pytest.raises(ValueError, match="draft_bits"):
        load_runtime("d", block_size=5, draft_bits=16)


def test_load_runtime_default_draft_bits(monkeypatch):
    from fusion_mlx.speculative.dflash2 import load_runtime

    _install_fake_dflash_model_mlx(monkeypatch)
    rt = load_runtime("d", block_size=5)
    assert rt.drafter is not None
    rt_bf16 = load_runtime("d", block_size=5, draft_bits=None)
    assert rt_bf16.drafter is not None


def test_load_runtime_rejects_empty_repo():
    from fusion_mlx.speculative.dflash2 import load_runtime

    with pytest.raises(ValueError, match="draft_repo"):
        load_runtime("", block_size=5)


def test_runtime_accept_lens_telemetry():
    from fusion_mlx.speculative.dflash2.runtime import DFlash2Runtime

    rt = DFlash2Runtime()
    assert rt.accept_lens_snapshot() == []
    rt.record_accept(3.5)
    rt.record_accept(None)
    rt.record_accept(2.0)
    snap = rt.accept_lens_snapshot()
    assert snap == [3.5, 2.0]
    rt.reset_accept_lens()
    assert rt.accept_lens_snapshot() == []
    rt.record_accept(0.0)
    assert rt.accept_lens_snapshot() == []


def test_have_runtime_returns_bool():
    from fusion_mlx.speculative.dflash2 import have_runtime

    assert isinstance(have_runtime(), bool)
