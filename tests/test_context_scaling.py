from fusion_mlx.api.context_scaling import (
    compute_scale_factor,
    get_context_scaling_settings,
    is_claude_code_request,
    scale_usage,
)


class TestIsClaudeCodeRequest:
    def test_empty_headers(self):
        assert not is_claude_code_request({})

    def test_normal_user_agent(self):
        assert not is_claude_code_request({"user-agent": "Mozilla/5.0"})

    def test_claude_code_user_agent(self):
        assert is_claude_code_request({"user-agent": "claude-code/1.0"})

    def test_claude_code_x_client(self):
        assert is_claude_code_request({"x-client": "claude-code"})

    def test_claude_code_case_insensitive(self):
        assert is_claude_code_request({"user-agent": "Claude-Code/2.0"})

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_CONTEXT_SCALING", "1")
        assert is_claude_code_request({})

    def test_env_override_false(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_CONTEXT_SCALING", raising=False)
        assert not is_claude_code_request({"user-agent": "curl/8.0"})


class TestGetContextScalingSettings:
    def test_empty_settings(self):
        enabled, target = get_context_scaling_settings({})
        assert not enabled
        assert target == 128000

    def test_enabled_default_target(self):
        enabled, target = get_context_scaling_settings(
            {"claude_code": {"context_scaling_enabled": True}}
        )
        assert enabled
        assert target == 128000

    def test_enabled_custom_target(self):
        enabled, target = get_context_scaling_settings(
            {
                "claude_code": {
                    "context_scaling_enabled": True,
                    "target_context_size": 200000,
                }
            }
        )
        assert enabled
        assert target == 200000

    def test_invalid_target_falls_back(self):
        enabled, target = get_context_scaling_settings(
            {
                "claude_code": {
                    "context_scaling_enabled": True,
                    "target_context_size": "bad",
                }
            }
        )
        assert enabled
        assert target == 128000

    def test_negative_target_falls_back(self):
        enabled, target = get_context_scaling_settings(
            {
                "claude_code": {
                    "context_scaling_enabled": True,
                    "target_context_size": -1,
                }
            }
        )
        assert enabled
        assert target == 128000


class TestComputeScaleFactor:
    def test_32k_to_128k(self):
        assert compute_scale_factor(32000, 128000) == 0.25

    def test_equal_no_scale(self):
        assert compute_scale_factor(128000, 128000) is None

    def test_larger_no_scale(self):
        assert compute_scale_factor(200000, 128000) is None

    def test_zero_model_context(self):
        assert compute_scale_factor(0, 128000) is None

    def test_zero_target(self):
        assert compute_scale_factor(32000, 0) is None


class TestScaleUsage:
    def test_basic_scaling(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 70,
        }
        scaled = scale_usage(usage, 0.25)
        assert scaled["input_tokens"] == 25
        assert scaled["output_tokens"] == 50
        assert scaled["cache_creation_input_tokens"] == 7
        assert scaled["cache_read_input_tokens"] == 17

    def test_factor_ge1_noop(self):
        usage = {"input_tokens": 100}
        assert scale_usage(usage, 1.0) == usage
        assert scale_usage(usage, 0.0) == usage

    def test_missing_keys(self):
        usage = {"input_tokens": 100}
        scaled = scale_usage(usage, 0.5)
        assert scaled["input_tokens"] == 50

    def test_zero_values_unchanged(self):
        usage = {"input_tokens": 0, "cache_read_input_tokens": 0}
        scaled = scale_usage(usage, 0.5)
        assert scaled["input_tokens"] == 0
        assert scaled["cache_read_input_tokens"] == 0
