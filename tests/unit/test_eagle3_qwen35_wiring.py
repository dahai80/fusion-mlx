"""resolve_text_inner wrapper navigation + qwen3.5 registry entry.

Regression guard for the single-hop navigation bug: eagle3/medusa wiring
gated on hasattr(model, "model") and read model.model.layers directly, so
the qwen3_5 multimodal wrapper (top-level exposes .language_model) silently
disabled spec decode and HiddenStateCapture installed nothing.
"""

import mlx.nn as nn

from fusion_mlx.speculative.eagle3.speculator import EAGLE3_DRAFT_MODELS
from fusion_mlx.speculative.hidden_capture import HiddenStateCapture, resolve_text_inner


class _Layer(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def __call__(self, x, *a, **kw):
        return x


class _Decoder:
    def __init__(self, n):
        self.layers = [_Layer(i) for i in range(n)]


class _LlamaStyle:
    def __init__(self, n):
        self.model = _Decoder(n)


class _TextModel:
    def __init__(self, n):
        self.model = _Decoder(n)


class _Qwen35Style:
    def __init__(self, n):
        self.language_model = _TextModel(n)


class _Bare:
    def __init__(self, n):
        self.layers = [_Layer(i) for i in range(n)]


def test_resolve_llama_style():
    m = _LlamaStyle(32)
    inner = resolve_text_inner(m)
    assert inner is m.model
    assert len(inner.layers) == 32


def test_resolve_qwen35_multimodal():
    m = _Qwen35Style(32)
    inner = resolve_text_inner(m)
    assert inner is m.language_model.model
    assert len(inner.layers) == 32


def test_resolve_bare_decoder():
    m = _Bare(8)
    inner = resolve_text_inner(m)
    assert inner is m


def test_resolve_unknown_returns_none():
    class _Empty:
        pass

    assert resolve_text_inner(_Empty()) is None
    assert resolve_text_inner(None) is None


def test_capture_installs_on_qwen35_wrapper():
    m = _Qwen35Style(32)
    hc = HiddenStateCapture(m, layer_ids=[1, 15, 28])
    hc.install()
    assert hc.installed
    layers = resolve_text_inner(m).layers
    for idx in (1, 15, 28):
        assert layers[idx].tag == idx
        assert hasattr(layers[idx], "_original")
    hc.uninstall()
    assert resolve_text_inner(m).layers[15].tag == 15


def test_qwen35_head_registered():
    entry = EAGLE3_DRAFT_MODELS.get("qwen3.5-4b")
    assert entry is not None
    assert entry["target_family"] == "qwen3"
    assert entry["draft_vocab_size"] == 50000
    assert "Qwen3.5-4B-Eagle3" in entry["hf_path"]
