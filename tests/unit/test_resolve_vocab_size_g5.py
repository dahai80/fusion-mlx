# SPDX-License-Identifier: Apache-2.0
"""Regression for G-5 (#0912 audit): strict ``response_format: json_schema``
was not enforced — the model ran unconstrained and returned free text with
200 OK instead of schema-conformant JSON.

Root cause: ``resolve_vocab_size`` (utils/tokenizer.py) only checked
``model.config`` and ``model.vocab_size``. Recent mlx-lm ``Model`` objects
expose neither — config lives on ``model.args`` (a ``ModelArgs`` dataclass
with ``vocab_size``). The engine already read ``model.args`` for
``model_type`` (engines/batched.py:188) but the vocab resolver did not.

Effect: ``_compile_grammar_for_request`` created an llguidance LLMatcher
with ``vocab_size=None`` (log: "LLMatcher created (vocab_size=None)") and
returned it as ``compiled_grammar`` (not None) → chat.py's R12-4 post-gen
validation was skipped (``if strict_enforcement_enabled() and
compiled_grammar is None``). Then the scheduler's
``GrammarConstraintProcessor`` also called ``resolve_vocab_size`` → None →
logged "Cannot determine vocab_size; skipping grammar constraint" and did
NOT attach the logits processor. Net: grammar compiled but never applied;
strict json_schema silently dropped to 200-OK-with-violating-output.

Fix: ``resolve_vocab_size`` falls back to ``model.args.vocab_size``.
"""

from __future__ import annotations

from types import SimpleNamespace


class _ModelArgs:
    def __init__(self, vocab_size=151936, model_type="qwen3"):
        self.vocab_size = vocab_size
        self.model_type = model_type


class _ModelWithConfig:
    def __init__(self, vocab_size=32000):
        self.config = SimpleNamespace(vocab_size=vocab_size)


class _ModelWithArgs:
    def __init__(self, vocab_size=151936):
        self.args = _ModelArgs(vocab_size=vocab_size)


class _ModelWithDictConfig:
    def __init__(self):
        self.config = {"vocab_size": 50257, "model_type": "gpt2"}


class _ModelWithTextConfig:
    def __init__(self):
        self.config = SimpleNamespace(
            vocab_size=None, text_config=SimpleNamespace(vocab_size=128256)
        )


class _ModelBare:
    pass


def test_resolve_from_config_attr():
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(_ModelWithConfig(32000)) == 32000


def test_resolve_from_dict_config():
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(_ModelWithDictConfig()) == 50257


def test_resolve_from_text_config_nested():
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(_ModelWithTextConfig()) == 128256


def test_resolve_from_model_args_fallback():
    """mlx-lm Model has no ``.config`` — falls back to ``.args.vocab_size``.

    This is the G-5 fix: pre-fix this returned None.
    """
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(_ModelWithArgs(151936)) == 151936


# ── #1153: 多模态模型 args.text_config 路径 ──────────────────────
class _MultimodalModelArgs:
    """qwen3_5 等多模态模型: args.vocab_size=None, 真值嵌在 args.text_config。"""

    def __init__(self, text_config):
        self.vocab_size = None
        self.model_type = "qwen3_5"
        self.text_config = text_config


class _MultimodalModelDictTextConfig:
    """args.text_config 是 dict (mlx-lm qwen3_5 实际形态)。"""

    def __init__(self):
        self.config = None
        self.args = _MultimodalModelArgs(
            {"vocab_size": 248320, "model_type": "qwen3_5_text"}
        )


class _MultimodalModelObjTextConfig:
    """args.text_config 是对象 (防御其他 mlx-lm 变体)。"""

    def __init__(self):
        self.config = None
        self.args = _MultimodalModelArgs(SimpleNamespace(vocab_size=151936))


def test_resolve_multimodal_args_dict_text_config():
    """#1153: 多模态模型 args.vocab_size=None → 须从 args.text_config (dict) 取。

    Pre-fix: 返回 None → scheduler 跳过 grammar 约束 → 模型自由输出。
    """
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(_MultimodalModelDictTextConfig()) == 248320


def test_resolve_multimodal_args_obj_text_config():
    """#1153: args.text_config 为对象时同样须解析。"""
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(_MultimodalModelObjTextConfig()) == 151936


def test_resolve_multimodal_args_text_config_missing_vocab():
    """args.text_config 存在但无 vocab_size → 返回 None (不误报)。"""
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    m = _MultimodalModelDictTextConfig()
    m.args.text_config = {"model_type": "qwen3_5_text"}
    assert resolve_vocab_size(m) is None


def test_resolve_returns_none_when_unavailable():
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(_ModelBare()) is None


def test_resolve_none_model():
    from fusion_mlx.utils.tokenizer import resolve_vocab_size

    assert resolve_vocab_size(None) is None
