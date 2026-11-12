# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #477 — MTP injection + install on VLM/hybrid models.

Adapted to the current architecture (the original upstream module
``patches.qwen3_next_mtp`` never existed in fusion-mlx; the guards moved):

1. ``speculative.mtp.qwen3_5_inject._resolve_inner_text_model`` — VLM
   checkpoints nest the LLM under ``model.language_model`` (with the outer
   ``model.args`` carrying vision config). Injection must resolve the inner
   text model or bail out cleanly rather than patch the outer VLM class and
   deferred-crash on the next forward (codex round-1 P1 on #477).

2. ``scheduler._install_mtp_vendored`` — gate-miss paths must log a clear
   warning and return False without touching the generator. The original
   trigger was a hybrid Gated-DeltaNet BatchGenerator (e.g. Qwen3.6-35B-A3B)
   routed through ``--force-spec-decode`` crashing with AttributeError
   (shipped in <=0.6.66, issue #477).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace


# ----------------------------------------------------------------------
# _resolve_inner_text_model — VLM/text-only resolution in inject_mtp_support
# ----------------------------------------------------------------------
def _llm_args_ns(**overrides):
    """SimpleNamespace shaped like a populated ``ModelArgs``."""
    defaults = {
        "hidden_size": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "full_attention_interval": 4,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "tie_word_embeddings": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_inner_text_model_returns_model_for_text_only():
    """Text-only path: ``model.model`` + ``model.args`` present — the model
    resolves to itself and injection proceeds normally."""
    from fusion_mlx.speculative.mtp.qwen3_5_inject import _resolve_inner_text_model

    model = SimpleNamespace(
        model=SimpleNamespace(layers=[]), args=_llm_args_ns(hidden_size=4096)
    )
    assert _resolve_inner_text_model(model) is model


def test_resolve_inner_text_model_prefers_language_model_for_vlm():
    """VLM checkpoint: ``model.language_model`` carries the inner text model
    (outer args hold vision config) — resolution must pick the inner LLM so
    injection patches the right class. Pins codex round-1 P1 on #477."""
    from fusion_mlx.speculative.mtp.qwen3_5_inject import _resolve_inner_text_model

    inner_lm = SimpleNamespace(
        model=SimpleNamespace(layers=[]), args=_llm_args_ns(hidden_size=3584)
    )
    model = SimpleNamespace(
        args=SimpleNamespace(
            text_config={"hidden_size": 3584}, vision_config={"hidden_size": 1280}
        ),
        language_model=inner_lm,
    )
    assert _resolve_inner_text_model(model) is inner_lm


def test_resolve_inner_text_model_none_when_language_model_is_none():
    """Defensive — ``language_model`` attr exists but is None (text-only
    branch of a multimodal class) and there is no usable ``model``:
    resolution returns None so the inject bail-out fires instead of
    pretending a text model exists."""
    from fusion_mlx.speculative.mtp.qwen3_5_inject import _resolve_inner_text_model

    model = SimpleNamespace(args=SimpleNamespace(), language_model=None)
    assert _resolve_inner_text_model(model) is None


def test_resolve_inner_text_model_language_model_wins_when_both_present():
    """Both shapes present: the inner ``language_model`` wins — injecting
    MTP onto the outer wrapper class would reference fields only the inner
    LLM has (``self.model.embed_tokens`` / ``self.lm_head``)."""
    from fusion_mlx.speculative.mtp.qwen3_5_inject import _resolve_inner_text_model

    inner_lm = SimpleNamespace(
        model=SimpleNamespace(layers=[]), args=_llm_args_ns(hidden_size=1024)
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[]),
        args=_llm_args_ns(hidden_size=2048),
        language_model=inner_lm,
    )
    assert _resolve_inner_text_model(model) is inner_lm


def test_inject_mtp_support_bails_out_on_unresolvable_model(caplog):
    """End-to-end bail-out: a model with neither ``language_model`` nor
    ``model + args`` must return False, log the skip warning, and never
    build an MTP module."""
    from fusion_mlx.speculative.mtp.qwen3_5_inject import inject_mtp_support

    with caplog.at_level(logging.WARNING, logger="fusion_mlx.speculative.mtp.qwen3_5_inject"):
        result = inject_mtp_support(SimpleNamespace())
    assert result is False
    assert "skipping MTP injection" in caplog.text


# ----------------------------------------------------------------------
# _install_mtp_vendored gate-miss paths (issue #477 Issue 2 equivalent)
# ----------------------------------------------------------------------
def test_install_mtp_vendored_skips_without_generation_batch(caplog):
    """A BatchGenerator without ``_generation_batch`` (mlx-lm version
    mismatch, or a hybrid generator routing through its own step flow)
    must return False, log a clear warning, and leave the object untouched
    — no AttributeError, no partial patching."""
    from fusion_mlx.scheduler._mtp_vendored import _install_mtp_vendored

    class _HybridGen:
        """Stand-in for the Qwen3.6 hybrid BatchGenerator — no
        ``_generation_batch``, no ``_step``."""

    bg = _HybridGen()
    model = SimpleNamespace()

    with caplog.at_level(logging.WARNING, logger="fusion_mlx.scheduler._mtp_vendored"):
        result = _install_mtp_vendored(bg, model=model)

    assert result is False
    assert not hasattr(bg, "_step"), "generator must remain untouched"
    assert "_generation_batch" in caplog.text


def test_install_mtp_vendored_skips_when_model_lacks_mtp_protocol(caplog):
    """Generator shape is fine but the model never went through MTP
    injection (no ``mtp_forward`` / ``make_mtp_cache`` / ``mtp``): install
    must refuse with the dispatch warning rather than crash later inside
    the wrapper."""
    from fusion_mlx.scheduler._mtp_vendored import _install_mtp_vendored

    gb = SimpleNamespace(_step=lambda: None)
    bg = SimpleNamespace(_generation_batch=gb)
    model = SimpleNamespace()  # no mtp protocol attributes

    with caplog.at_level(logging.WARNING, logger="fusion_mlx.scheduler._mtp_vendored"):
        result = _install_mtp_vendored(bg, model=model)

    assert result is False
    assert "mtp_forward" in caplog.text
    assert gb._step.__name__ == "<lambda>", "gb._step must remain unpatched"


def test_install_mtp_vendored_installs_on_compatible_gen_and_model():
    """Pin existing behavior: a generator with ``_generation_batch`` plus a
    model carrying the MTP protocol gets its ``gb._step`` re-bound to the
    wrapper and the install returns True — guards against the gate being
    too eager and breaking the normal (non-hybrid) path."""
    from fusion_mlx.scheduler._mtp_vendored import _install_mtp_vendored

    orig_step = lambda: None  # noqa: E731
    gb = SimpleNamespace(_step=orig_step)
    bg = SimpleNamespace(_generation_batch=gb)
    model = SimpleNamespace(
        mtp_forward=lambda *a, **k: None,
        make_mtp_cache=lambda *a, **k: None,
        mtp=SimpleNamespace(),
    )

    result = _install_mtp_vendored(bg, model=model)

    assert result is True
    assert gb._step is not orig_step, "gb._step was not re-bound by the installer"
    assert callable(gb._step)
    assert bg._mtp_vendored_stats is not None
