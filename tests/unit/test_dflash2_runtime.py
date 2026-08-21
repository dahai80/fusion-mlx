# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DFlash2 runtime + generator bridge (mocked dflash pkg).

No real model load: monkeypatches ``dflash.model_mlx`` so the generator
constructs from fakes and stream_from_tokens yields scripted tokens.
"""

from __future__ import annotations

import logging
import types

import pytest

logger = logging.getLogger(__name__)


def _install_fake_dflash_model_mlx(monkeypatch, token_blocks):
    fake = types.ModuleType("dflash.model_mlx")

    class _FakeDraft:
        def __init__(self, repo):
            self.repo = repo
            self.bound = None

        def bind(self, target):
            self.bound = target

    class _FakeResp:
        def __init__(self, toks):
            self.tokens = toks
            self.accepted = len(toks)

    def _fake_load(repo):
        return (types.SimpleNamespace(repo=repo), types.SimpleNamespace(repo=repo))

    def _fake_load_draft(repo):
        return _FakeDraft(repo)

    def _fake_stream_generate(target, draft, tokenizer, prompt, **kw):
        for block in token_blocks:
            yield _FakeResp(block)

    fake.load = _fake_load
    fake.load_draft = _fake_load_draft
    fake.stream_generate = _fake_stream_generate
    fake.DFlash2DraftModel = _FakeDraft
    monkeypatch.setitem(__import__("sys").modules, "dflash.model_mlx", fake)
    return fake


def test_generator_stream_yields_all_tokens(monkeypatch):
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2Generator

    _install_fake_dflash_model_mlx(monkeypatch, [[1, 2, 3], [4, 5], [6]])
    gen = DFlash2Generator(
        target_repo="mlx-community/Qwen3.8-27B-4bit",
        draft_repo="z-lab/Qwen3.8-27B-DFlash2",
        block_size=5,
    )
    out = list(gen.stream_from_tokens([10, 11], max_new_tokens=100))
    assert out == [1, 2, 3, 4, 5, 6]


def test_generator_respects_max_new_tokens(monkeypatch):
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2Generator

    _install_fake_dflash_model_mlx(monkeypatch, [[1, 2, 3, 4, 5], [6, 7, 8]])
    gen = DFlash2Generator("t", "d", block_size=5)
    out = list(gen.stream_from_tokens([0], max_new_tokens=4))
    assert out == [1, 2, 3, 4]


def test_generator_binds_draft_to_target(monkeypatch):
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2Generator

    _install_fake_dflash_model_mlx(monkeypatch, [[1]])
    gen = DFlash2Generator("t", "d", block_size=5)
    assert gen.draft.bound is gen.target


def test_generator_rejects_invalid_block_size():
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2Generator

    with pytest.raises(ValueError, match="block_size"):
        DFlash2Generator("t", "d", block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        DFlash2Generator("t", "d", block_size=6)


def test_generator_rejects_empty_repos():
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2Generator

    with pytest.raises(ValueError, match="target_repo"):
        DFlash2Generator("", "d")
    with pytest.raises(ValueError, match="draft_repo"):
        DFlash2Generator("t", "")


def test_generator_stream_validates_args(monkeypatch):
    from fusion_mlx.speculative.dflash2.engine.generator import DFlash2Generator

    _install_fake_dflash_model_mlx(monkeypatch, [[1]])
    gen = DFlash2Generator("t", "d", block_size=5)
    with pytest.raises(ValueError, match="max_new_tokens"):
        list(gen.stream_from_tokens([0], max_new_tokens=0))
    with pytest.raises(ValueError, match="temperature"):
        list(gen.stream_from_tokens([0], temperature=-0.1))


def test_load_runtime_builds_runtime(monkeypatch):
    from fusion_mlx.speculative.dflash2 import DFlash2Runtime, load_runtime

    _install_fake_dflash_model_mlx(monkeypatch, [[1, 2]])
    rt = load_runtime("t", "d", block_size=5)
    assert isinstance(rt, DFlash2Runtime)
    assert rt.target_repo == "t"
    assert rt.draft_repo == "d"
    assert rt.block_size == 5
    assert rt.generator is not None


def test_load_runtime_rejects_bad_block_size():
    from fusion_mlx.speculative.dflash2 import load_runtime

    with pytest.raises(ValueError, match="block_size"):
        load_runtime("t", "d", block_size=20)
    with pytest.raises(ValueError, match="block_size"):
        load_runtime("t", "d", block_size=0)


def test_load_runtime_rejects_empty_repos():
    from fusion_mlx.speculative.dflash2 import load_runtime

    with pytest.raises(ValueError, match="target_repo"):
        load_runtime("", "d")
    with pytest.raises(ValueError, match="draft_repo"):
        load_runtime("t", "")


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
