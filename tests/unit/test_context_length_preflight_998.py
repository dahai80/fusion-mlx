# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #998.

#998: ``/v1/chat/completions`` (and ``/v1/messages``) hung ~168s then
returned an empty 500 for prompts exceeding the model's context window.
Root cause: the route-layer preflight used ``get_max_context_window``
(server config ``max_context_window``), which is unset in standalone
deploys → the 85% heuristic was skipped entirely. The
``enforce_context_length`` helper — which uses the model's REAL
``max_position_embeddings`` via ``get_model_max_context`` — was dead
code (defined, never wired). Over-context prompts reached prefill and
either OOM'd or hit a position-out-of-bounds → uncaught ``RuntimeError``
→ generic 500.

Fix (#998): wired ``enforce_context_length`` into the chat (non-stream +
stream) and anthropic (non-stream + stream) paths, raising a clean 400
``context_length_exceeded`` BEFORE the engine call. Also guarded
``enforce_context_length`` against ``max_context <= 0`` (the latent bug
that made it unsafe to wire — it would have rejected every request when
model config was unavailable).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from fusion_mlx.service.helpers import (
    enforce_context_length,
    enforce_context_length_for_messages,
    get_model_max_context,
)


class _StubArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubModel:
    def __init__(self, args=None, config=None):
        if args is not None:
            self.args = args
        if config is not None:
            self.config = config


class _StubTokenizer:
    # 1 token per character — deterministic, easy to reason about.
    def __init__(self, bos_token=None):
        self.bos_token = bos_token

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text)))


class _StubEngine:
    is_mllm = False

    def __init__(self, max_position_embeddings=32768):
        self._model = _StubModel(
            args=_StubArgs(max_position_embeddings=max_position_embeddings)
        )
        self._tokenizer = _StubTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        # Concatenate all message content — length = token count.
        parts = []
        for m in messages:
            c = m.get("content", "") if isinstance(m, dict) else str(m)
            parts.append(c)
        return "".join(parts)


class TestEnforceContextLength:
    # The helper at the heart of the #998 fix.
    def test_under_context_passes(self):
        engine = _StubEngine(max_position_embeddings=32768)
        # 100 prompt + 50 gen = 150 << 32768 -> no raise.
        enforce_context_length(engine, prompt_tokens=100, max_tokens=50)

    def test_over_context_raises_400(self):
        engine = _StubEngine(max_position_embeddings=32768)
        with pytest.raises(HTTPException) as exc_info:
            enforce_context_length(
                engine, prompt_tokens=32000, max_tokens=1000
            )
        assert exc_info.value.status_code == 400
        detail = exc_info.value.detail
        err = detail.get("error", detail) if isinstance(detail, dict) else {}
        assert err.get("code") == "context_length_exceeded", detail
        assert "32768" in err.get("message", ""), err

    def test_prompt_plus_max_tokens_over_raises(self):
        # prompt=32000 + max_tokens=1000 = 33000 > 32768 -> raise even
        # though prompt alone fits.
        engine = _StubEngine(max_position_embeddings=32768)
        with pytest.raises(HTTPException) as exc_info:
            enforce_context_length(
                engine, prompt_tokens=32000, max_tokens=1000
            )
        assert exc_info.value.status_code == 400

    def test_exact_fit_passes(self):
        # prompt + max_tokens == max_context exactly -> passes (<=).
        engine = _StubEngine(max_position_embeddings=32768)
        enforce_context_length(
            engine, prompt_tokens=32768, max_tokens=0
        )

    def test_max_context_zero_skips(self):
        # #998 latent bug: max_context<=0 must skip, not reject all.
        # (This is why enforce_context_length was never wired — wiring it
        # without this guard would 400 every request when model config
        # is unavailable.) get_model_max_context normally falls back to a
        # 4M default, so patch it to 0 to exercise the guard directly.
        engine = _StubEngine(max_position_embeddings=32768)
        with patch(
            "fusion_mlx.service.helpers.get_model_max_context",
            return_value=0,
        ):
            enforce_context_length(engine, prompt_tokens=99999, max_tokens=99999)


class TestEnforceContextLengthForMessages:
    # The messages-level wrapper (tokenizes via build_prompt + counts).
    def test_over_context_messages_raise_400(self):
        engine = _StubEngine(max_position_embeddings=100)
        # build_prompt concatenates content; 200 chars = 200 tokens > 100.
        messages = [{"role": "user", "content": "x" * 200}]
        with pytest.raises(HTTPException) as exc_info:
            enforce_context_length_for_messages(
                engine, messages, max_tokens=10
            )
        assert exc_info.value.status_code == 400

    def test_under_context_messages_pass(self):
        engine = _StubEngine(max_position_embeddings=10000)
        messages = [{"role": "user", "content": "short prompt"}]
        result = enforce_context_length_for_messages(
            engine, messages, max_tokens=50
        )
        # Returns the prompt token count on success.
        assert result is not None and result > 0

    def test_mllm_engine_skips(self):
        # MLLM engines bypass the check (vision tokens not text-countable).
        engine = _StubEngine()
        engine.is_mllm = True
        messages = [{"role": "user", "content": "x" * 999999}]
        result = enforce_context_length_for_messages(
            engine, messages, max_tokens=999999
        )
        assert result is None


class TestGetModelMaxContext:
    # Confirms the source of truth is model.args.max_position_embeddings
    # (NOT server config) — the key insight behind the #998 fix.
    def test_reads_model_args(self):
        engine = _StubEngine(max_position_embeddings=4096)
        assert get_model_max_context(engine) == 4096

    def test_fallback_when_unset(self):
        # When max_position_embeddings is absent, get_model_max_context
        # falls back to _FALLBACK_MAX_CONTEXT_TOKENS (a large default) —
        # NOT 0. This is why the #998 guard (max_context <= 0) is
        # defensive: it only fires if the fallback is also overridden.
        engine = _StubEngine(max_position_embeddings=0)
        del engine._model.args.max_position_embeddings
        assert get_model_max_context(engine) > 0

    def test_nested_text_config(self):
        # VLM-style nested config (text_config.max_position_embeddings).
        engine = _StubEngine()
        del engine._model.args.max_position_embeddings
        engine._model.args.text_config = _StubArgs(max_position_embeddings=8192)
        assert get_model_max_context(engine) == 8192


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
