def _config_with_min_tokens(value):
    from fusion_mlx.config import get_config, reset_config

    reset_config()
    cfg = get_config()
    cfg.default_min_tokens = value
    return cfg


def _teardown_function(function):
    from fusion_mlx.config import reset_config

    reset_config()


def test_server_config_has_default_min_tokens_zero():
    from fusion_mlx.config import ServerConfig

    assert ServerConfig().default_min_tokens == 0


def _magic_request():
    from unittest.mock import MagicMock

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
    # MagicMock attribute is not an int -> treated as "not set by request"
    req.min_tokens = MagicMock()
    return req


def test_build_sampling_params_uses_server_default_min_tokens():
    from fusion_mlx.api.openai import _build_sampling_params

    _config_with_min_tokens(2)
    sampling = _build_sampling_params(_magic_request(), profile_overrides={})
    assert sampling.min_tokens == 2


def test_build_sampling_params_profile_min_tokens_beats_server_default():
    from fusion_mlx.api.openai import _build_sampling_params

    _config_with_min_tokens(2)
    sampling = _build_sampling_params(
        _magic_request(), profile_overrides={"min_tokens": 3}
    )
    assert sampling.min_tokens == 3


def test_build_sampling_params_request_min_tokens_beats_all():
    from fusion_mlx.api.openai import _build_sampling_params

    _config_with_min_tokens(2)
    req = _magic_request()
    req.min_tokens = 5
    sampling = _build_sampling_params(req, profile_overrides={"min_tokens": 3})
    assert sampling.min_tokens == 5


def test_build_sampling_params_zero_default_keeps_fast_path():
    from fusion_mlx.api.openai import _build_sampling_params

    _config_with_min_tokens(0)
    sampling = _build_sampling_params(_magic_request(), profile_overrides={})
    assert sampling.min_tokens == 0


def test_cli_value_flags_include_default_min_tokens():
    from fusion_mlx.cli_commands import _SERVE_VALUE_FLAGS

    assert "--default-min-tokens" in _SERVE_VALUE_FLAGS


def test_sched_response_helper_matches_greedy_single_token_stop():
    from unittest.mock import MagicMock

    from fusion_mlx.scheduler.sched_response import (
        _is_near_tie_greedy_stop,
    )

    req = MagicMock()
    req.num_output_tokens = 0
    req.sampling_params.temperature = 0.0
    assert _is_near_tie_greedy_stop(req, "stop") is True

    req.num_output_tokens = 1
    assert _is_near_tie_greedy_stop(req, "stop") is True


def test_sched_response_helper_rejects_other_cases():
    from unittest.mock import MagicMock

    from fusion_mlx.scheduler.sched_response import (
        _is_near_tie_greedy_stop,
    )

    req = MagicMock()
    req.num_output_tokens = 1
    req.sampling_params.temperature = 0.7
    assert _is_near_tie_greedy_stop(req, "stop") is False

    req.sampling_params.temperature = 0.0
    req.num_output_tokens = 5
    assert _is_near_tie_greedy_stop(req, "stop") is False
    assert _is_near_tie_greedy_stop(req, "length") is False
