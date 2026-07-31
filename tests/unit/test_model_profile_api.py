# SPDX-License-Identifier: Apache-2.0
"""Tests for model:profile API syntax resolution."""

import pytest


def test_resolve_model_with_profile_no_colon():
    from fusion_mlx.server import resolve_model_with_profile

    model_id, overrides = resolve_model_with_profile("qwen3")
    assert model_id == "qwen3"
    assert overrides == {}


def test_resolve_model_with_profile_no_settings_manager():
    from fusion_mlx.server import resolve_model_with_profile, _server_state

    saved = _server_state.pop("settings_manager", None)
    try:
        model_id, overrides = resolve_model_with_profile("qwen3:creative")
        assert model_id == "qwen3"
        assert overrides == {}
    finally:
        if saved is not None:
            _server_state["settings_manager"] = saved


def test_resolve_model_with_profile_alias():
    from fusion_mlx.server import resolve_model_with_profile

    model_id, overrides = resolve_model_with_profile("fusion-mlx/qwen3")
    assert model_id == "qwen3"
    assert overrides == {}


def test_build_sampling_params_openai_with_profile():
    from unittest.mock import MagicMock
    from fusion_mlx.api.openai_routes import _build_sampling_params

    req = MagicMock()
    req.max_tokens = None
    req.temperature = None
    req.top_p = None
    req.top_k = 0
    req.min_p = 0.0
    req.presence_penalty = None
    req.frequency_penalty = None
    req.stop = None
    req.stop_token_ids = None
    req.logprobs = False
    req.top_logprobs = None

    overrides = {"temperature": 0.3, "top_p": 0.5, "max_tokens": 4096}
    sampling = _build_sampling_params(req, profile_overrides=overrides)
    assert sampling.temperature == 0.3
    assert sampling.top_p == 0.5
    assert sampling.max_tokens == 4096


def test_build_sampling_params_openai_request_takes_precedence():
    from unittest.mock import MagicMock
    from fusion_mlx.api.openai_routes import _build_sampling_params

    req = MagicMock()
    req.max_tokens = 1024
    req.temperature = 0.9
    req.top_p = 0.95
    req.top_k = 0
    req.min_p = 0.0
    req.presence_penalty = None
    req.frequency_penalty = None
    req.stop = None
    req.stop_token_ids = None
    req.logprobs = False
    req.top_logprobs = None

    overrides = {"temperature": 0.3, "top_p": 0.5, "max_tokens": 4096}
    sampling = _build_sampling_params(req, profile_overrides=overrides)
    assert sampling.temperature == 0.9
    assert sampling.top_p == 0.95
    assert sampling.max_tokens == 1024


def test_build_sampling_params_anthropic_with_profile():
    from unittest.mock import MagicMock
    from fusion_mlx.api.anthropic_routes import _build_sampling_params

    req = MagicMock(spec=["max_tokens", "temperature", "top_p", "stop_sequences"])
    req.max_tokens = None
    req.temperature = None
    req.top_p = None
    req.stop_sequences = None

    overrides = {"temperature": 0.2, "top_p": 0.8, "max_tokens": 8192}
    sampling = _build_sampling_params(req, profile_overrides=overrides)
    assert sampling.temperature == 0.2
    assert sampling.top_p == 0.8
    assert sampling.max_tokens == 8192


def test_build_sampling_params_anthropic_request_takes_precedence():
    from unittest.mock import MagicMock
    from fusion_mlx.api.anthropic_routes import _build_sampling_params

    req = MagicMock(spec=["max_tokens", "temperature", "top_p", "stop_sequences"])
    req.max_tokens = 512
    req.temperature = 1.0
    req.top_p = 0.99
    req.stop_sequences = None

    overrides = {"temperature": 0.2, "top_p": 0.8, "max_tokens": 8192}
    sampling = _build_sampling_params(req, profile_overrides=overrides)
    assert sampling.temperature == 1.0
    assert sampling.top_p == 0.99
    assert sampling.max_tokens == 512
